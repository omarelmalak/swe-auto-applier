"""Parse application forms: minify DOM, extract fields, detect Next vs Submit."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from playwright.sync_api import Page

from core.browser import launch_page

NavigationKind = Literal["next", "submit", "unknown"]

_KEEP_ATTRS = {
    "id",
    "name",
    "type",
    "placeholder",
    "aria-label",
    "aria-labelledby",
    "aria-describedby",
    "aria-required",
    "for",
    "required",
    "value",
    "checked",
    "selected",
    "disabled",
    "readonly",
    "multiple",
    "accept",
    "autocomplete",
    "role",
    "href",
    "action",
    "method",
    "maxlength",
    "min",
    "max",
    "step",
    "pattern",
    "rows",
    "cols",
}

_CONTROL_TAGS = {"input", "textarea", "select", "button"}

_NEXT_PATTERNS = (
    r"\bnext\b",
    r"\bcontinue\b",
    r"\bsave\s+(&|and)\s+continue\b",
    r"\bproceed\b",
    r"\bgo\s+to\s+next\b",
)
_SUBMIT_PATTERNS = (
    r"\bsubmit(\s+application)?\b",
    r"\bapply(\s+now)?\b",
    r"\bsend\s+application\b",
    r"\bfinish\b",
    r"\bcomplete\s+application\b",
)


@dataclass
class FormField:
    """One interactive control on the current application step."""

    tag: str
    selector: str
    input_type: str | None = None
    name: str | None = None
    element_id: str | None = None
    label: str | None = None
    # Surrounding question text (fieldset legend / group heading), distinct from
    # the option label (e.g. label="He/Him", context="What are your pronouns?").
    context: str | None = None
    placeholder: str | None = None
    aria_label: str | None = None
    options: list[str] = field(default_factory=list)
    # Parallel to options for collapsed radio/checkbox groups (executor remap).
    option_selectors: list[str] = field(default_factory=list)
    # Internal: DOM option-list container id for collapsing (not for Cohere).
    group_key: str | None = None
    required: bool = False


@dataclass
class NavigationTarget:
    """Primary navigation control for the current step."""

    kind: NavigationKind
    selector: str | None = None
    text: str | None = None
    candidates: list[dict[str, str]] = field(default_factory=list)


@dataclass
class FormParseResult:
    """Structured parse of the current form step."""

    url: str
    minified_html: str
    fields: list[FormField]
    navigation: NavigationTarget
    form_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        fields_out: list[dict[str, Any]] = []
        for f in self.fields:
            payload = asdict(f)
            # Drop empty / internal-only noise from CLI JSON.
            fields_out.append(
                {
                    k: v
                    for k, v in payload.items()
                    if v not in (None, "", [], False) and k != "group_key"
                }
            )
        return {
            "url": self.url,
            "form_count": self.form_count,
            "navigation": asdict(self.navigation),
            "fields": fields_out,
            "minified_html": self.minified_html,
        }


def parse_form_html(html: str, *, url: str = "") -> FormParseResult:
    """
    Pure HTML parse path (no browser). Used by unit tests and by page parsers.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "path", "meta", "link"]):
        tag.decompose()

    forms = soup.find_all("form")
    scope: Tag | BeautifulSoup
    if forms:
        # Prefer the largest form (ATS pages often wrap chrome in tiny forms).
        scope = max(forms, key=lambda f: len(f.find_all(_CONTROL_TAGS)))
        form_count = len(forms)
    else:
        scope = soup
        form_count = 0

    fields = _collapse_option_groups(_extract_fields(scope, soup))
    navigation = detect_navigation(soup)
    minified = minify_form_html(scope)

    return FormParseResult(
        url=url,
        minified_html=minified,
        fields=fields,
        navigation=navigation,
        form_count=form_count,
    )


def minify_form_html(scope: Tag | BeautifulSoup) -> str:
    """Keep structure + useful attrs; drop class/style noise and empty wrappers."""
    node = BeautifulSoup(str(scope), "html.parser")
    root = node if isinstance(scope, BeautifulSoup) else (node.find(True) or node)

    for tag in list(node.find_all(True)):
        # Parent decompose() can leave detached nodes with attrs=None in the snapshot.
        if getattr(tag, "attrs", None) is None or tag.name is None:
            continue

        # Drop tracking/hidden noise attributes; keep allowlisted ones.
        attrs = dict(tag.attrs or {})
        tag.attrs = {}
        for key, value in attrs.items():
            key_l = key.lower()
            if key_l in _KEEP_ATTRS or key_l.startswith("aria-") or key_l.startswith("data-qa"):
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                tag[key] = value

        # Remove empty non-control leaves.
        if (
            tag.name not in _CONTROL_TAGS
            and tag.name not in {"form", "label", "option", "optgroup", "fieldset", "legend"}
            and not tag.find(_CONTROL_TAGS)
            and not tag.get_text(strip=True)
        ):
            tag.decompose()

    text = str(root)
    text = re.sub(r">\s+<", "><", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def detect_navigation(soup: BeautifulSoup | Tag) -> NavigationTarget:
    """
    Classify the primary CTA as next | submit | unknown.

    Prefer explicit Next/Continue over Submit when both appear (multi-step wizards
    often keep a disabled submit or secondary apply link in the chrome).
    """
    candidates: list[dict[str, Any]] = []

    for el in soup.find_all(["button", "input", "a"]):
        if not isinstance(el, Tag):
            continue
        if el.get("disabled") is not None or el.get("aria-disabled") == "true":
            continue

        text = _control_text(el)
        if not text and el.name == "input" and el.get("type") in {"submit", "button", "image"}:
            text = str(el.get("value") or el.get("aria-label") or "")

        kind = _classify_nav_text(text, el)
        if kind == "unknown" and not text:
            continue
        if kind == "unknown":
            # Keep bare type=submit as submit even with generic labels.
            if el.name == "input" and (el.get("type") or "").lower() == "submit":
                kind = "submit"
            else:
                continue

        selector = _best_selector(el)
        candidates.append(
            {
                "kind": kind,
                "selector": selector,
                "text": text.strip()[:120],
            }
        )

    if not candidates:
        return NavigationTarget(kind="unknown")

    next_hits = [c for c in candidates if c["kind"] == "next"]
    submit_hits = [c for c in candidates if c["kind"] == "submit"]

    # Multi-step: Next wins if present.
    chosen = next_hits[-1] if next_hits else submit_hits[-1] if submit_hits else candidates[-1]
    kind: NavigationKind = chosen["kind"]  # type: ignore[assignment]
    return NavigationTarget(
        kind=kind,
        selector=chosen.get("selector"),
        text=chosen.get("text"),
        candidates=[{"kind": c["kind"], "selector": c["selector"], "text": c["text"]} for c in candidates],
    )


def parse_page(page: Page) -> FormParseResult:
    """Parse the currently loaded Playwright page."""
    # Wait briefly for a form or any input to appear (SPAs / Greenhouse iframes vary).
    try:
        page.wait_for_selector("form, input, textarea, select", timeout=15_000)
    except Exception:
        pass
    html = page.content()
    return parse_form_html(html, url=page.url)


def parse_url(url: str, *, headless: bool = True) -> FormParseResult:
    """Navigate to a URL and parse the application form on the page."""
    _validate_url(url)
    with launch_page(headless=headless) as (_pw, _browser, page):
        page.goto(url, wait_until="domcontentloaded")
        # Allow late hydration of ATS widgets.
        page.wait_for_timeout(1500)
        return parse_page(page)


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected an http(s) URL, got: {url!r}")


def _control_text(el: Tag) -> str:
    if el.name == "input":
        return str(el.get("value") or el.get("aria-label") or el.get("placeholder") or "")
    parts: list[str] = []
    for child in el.descendants:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name in {"script", "style"}:
            continue
    text = " ".join(p.strip() for p in parts if p.strip())
    if not text:
        text = str(el.get("aria-label") or el.get("title") or "")
    return re.sub(r"\s+", " ", text).strip()


def _classify_nav_text(text: str, el: Tag) -> NavigationKind:
    blob = text.lower().strip()
    type_attr = (el.get("type") or "").lower()

    for pattern in _NEXT_PATTERNS:
        if re.search(pattern, blob, flags=re.IGNORECASE):
            return "next"
    for pattern in _SUBMIT_PATTERNS:
        if re.search(pattern, blob, flags=re.IGNORECASE):
            return "submit"

    if type_attr == "submit" and blob in {"", "submit"}:
        return "submit"
    return "unknown"


def _extract_fields(scope: Tag | BeautifulSoup, full_doc: BeautifulSoup) -> list[FormField]:
    fields: list[FormField] = []
    seen_selectors: set[str] = set()

    for el in scope.find_all(["input", "textarea", "select"]):
        if not isinstance(el, Tag):
            continue
        input_type = (el.get("type") or ("textarea" if el.name == "textarea" else "text")).lower()
        if input_type in {"hidden", "submit", "button", "reset", "image"}:
            continue

        selector = _best_selector(el)
        if selector in seen_selectors:
            continue
        seen_selectors.add(selector)

        label = _resolve_label(el, full_doc)
        option_container = (
            _option_list_container(el)
            if input_type in {"radio", "checkbox"}
            else None
        )
        context = _resolve_question_context(
            el,
            full_doc,
            option_label=label,
            scope=option_container,
        )
        options: list[str] = []
        if el.name == "select":
            options = [
                (opt.get_text(" ", strip=True) or str(opt.get("value") or "")).strip()
                for opt in el.find_all("option")
                if isinstance(opt, Tag)
            ]
            options = [o for o in options if o]

        required = el.has_attr("required") or el.get("aria-required") == "true"
        fields.append(
            FormField(
                tag=el.name,
                selector=selector,
                input_type=input_type if el.name == "input" else el.name,
                name=el.get("name"),
                element_id=el.get("id"),
                label=label,
                context=context,
                placeholder=el.get("placeholder"),
                aria_label=el.get("aria-label"),
                options=options,
                group_key=(f"dom:{id(option_container)}" if option_container else None),
                required=required,
            )
        )

    # Contenteditable surfaces (rare but used by some ATS rich-text boxes).
    for el in scope.find_all(attrs={"contenteditable": True}):
        if not isinstance(el, Tag):
            continue
        selector = _best_selector(el)
        if selector in seen_selectors:
            continue
        seen_selectors.add(selector)
        label = _resolve_label(el, full_doc)
        fields.append(
            FormField(
                tag=el.name,
                selector=selector,
                input_type="contenteditable",
                name=el.get("name"),
                element_id=el.get("id"),
                label=label,
                context=_resolve_question_context(el, full_doc, option_label=label),
                aria_label=el.get("aria-label"),
                required=el.get("aria-required") == "true",
            )
        )

    return fields


def _looks_like_question_text(text: str | None) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    cleaned = text.strip()
    if "?" in cleaned:
        return True
    return bool(_QUESTION_LEAD_RE.match(cleaned))


def _control_census(parent: Tag) -> tuple[int, int]:
    """Return (option_control_count, other_control_count) under parent."""
    option_controls = 0
    other_controls = 0
    for node in parent.find_all(["input", "textarea", "select"]):
        if not isinstance(node, Tag):
            continue
        if node.name in {"textarea", "select"}:
            other_controls += 1
            continue
        t = (node.get("type") or "text").lower()
        if t in {"radio", "checkbox"}:
            option_controls += 1
        elif t not in {"hidden", "submit", "button", "reset", "image"}:
            other_controls += 1
    return option_controls, other_controls


def _fieldset_question_text(fieldset: Tag) -> str | None:
    """
    Question prompt for a fieldset.

    HTML often uses <legend>; many apps use a plain <label>/<heading> as the
    first non-control child instead.
    """
    legend = fieldset.find("legend", recursive=False)
    if legend:
        text = legend.get_text(" ", strip=True)
        if text:
            return text
    for child in fieldset.children:
        if not isinstance(child, Tag):
            continue
        if child.name not in {"label", "legend", "h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "span"}:
            continue
        if child.find(["input", "textarea", "select"]):
            continue
        text = child.get_text(" ", strip=True)
        if text and len(text) >= 2:
            return text
    return None


def _option_list_container(el: Tag) -> Tag | None:
    """
    Tightest ancestor that wraps only this radio/checkbox group.

    Site-agnostic HTML rules:
    - fieldset
    - role=radiogroup|group|checkboxgroup
    - for radios only: nearest parent with 2+ radios and no other inputs

    Checkboxes are not merged via a generic parent — adjacent single
    confirm/agree boxes often share a section without being one question.
    """
    input_type = (el.get("type") or "").lower()
    for parent in el.parents:
        if not isinstance(parent, Tag) or parent.name in {"form", "body", "html"}:
            break
        if parent.name == "fieldset":
            return parent
        role = (parent.get("role") or "").lower()
        if role in {"radiogroup", "group", "checkboxgroup"}:
            return parent
        if input_type != "radio":
            continue
        option_controls, other_controls = _control_census(parent)
        if other_controls > 0:
            break
        if option_controls >= 2:
            return parent
    return None


def _option_group_key(field: FormField) -> tuple[str, ...] | None:
    """
    Collapse key for one multiple-choice question.

    1) Shared DOM option-list container (general)
    2) Shared HTML name on radios (general)
    Never group by question text alone — leaked prompts merge neighbors.
    """
    kind = (field.input_type or "").lower()
    if kind not in {"radio", "checkbox"}:
        return None

    if field.group_key:
        return ("dom", kind, field.group_key)

    if kind == "radio" and field.name:
        return ("name", kind, field.name)

    return None


def _collapse_option_groups(fields: list[FormField]) -> list[FormField]:
    """
    Merge radios/checkboxes that belong to the same question into one field
    with options[] (one prompt entry, not one per choice).
    """
    groups: dict[tuple[str, ...], list[FormField]] = {}
    order: list[tuple[str, ...] | None] = []
    singles: list[FormField] = []

    for field in fields:
        key = _option_group_key(field)
        if key is None:
            order.append(None)
            singles.append(field)
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(field)

    collapsed: list[FormField] = []
    single_iter = iter(singles)
    emitted: set[tuple[str, ...]] = set()

    for item in order:
        if item is None:
            collapsed.append(next(single_iter))
            continue
        if item in emitted:
            continue
        emitted.add(item)
        members = groups[item]
        if len(members) < 2:
            collapsed.extend(members)
            continue

        options: list[str] = []
        option_selectors: list[str] = []
        for member in members:
            opt = (member.label or member.aria_label or "").strip()
            if not opt:
                opt = member.name or member.selector
            options.append(opt)
            option_selectors.append(member.selector)

        head = members[0]
        # Prefer a question-shaped prompt; otherwise keep fieldset titles like
        # "Gender" / "Degree Type" (still useful, just not interrogative).
        question = next(
            (m.context for m in members if _looks_like_question_text(m.context)),
            None,
        ) or next((m.context for m in members if m.context), None)
        names = {m.name for m in members if m.name}
        collapsed.append(
            FormField(
                tag=head.tag,
                selector=head.selector,
                input_type=head.input_type,
                name=names.pop() if len(names) == 1 else None,
                element_id=head.element_id,
                # Question becomes the field label; options hold the choices.
                label=question or head.label,
                context=question,
                placeholder=head.placeholder,
                aria_label=head.aria_label,
                options=options,
                option_selectors=option_selectors,
                group_key=head.group_key,
                required=any(m.required for m in members),
            )
        )

    return collapsed


def _resolve_label(el: Tag, doc: BeautifulSoup) -> str | None:
    el_id = el.get("id")
    if el_id:
        lab = doc.find("label", attrs={"for": el_id})
        if lab:
            text = lab.get_text(" ", strip=True)
            if text:
                return text

    parent_label = el.find_parent("label")
    if parent_label:
        text = parent_label.get_text(" ", strip=True)
        if text:
            return text

    aria = el.get("aria-label")
    if aria:
        return str(aria).strip()

    labelled_by = el.get("aria-labelledby")
    if labelled_by:
        parts: list[str] = []
        for ref in str(labelled_by).split():
            node = doc.find(id=ref)
            if node:
                parts.append(node.get_text(" ", strip=True))
        joined = " ".join(p for p in parts if p)
        if joined:
            return joined

    return None


_QUESTION_LEAD_RE = re.compile(
    r"^(what|which|how|why|when|where|who|do|does|did|are|is|was|were|"
    r"have|has|had|can|could|would|will|please|select|choose|indicate|"
    r"describe|tell|list|enter|provide|share|confirm)\b",
    re.IGNORECASE,
)


def _normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _prompt_from_node(node: Tag, *, option_label: str | None) -> str | None:
    """
    Extract a question/prompt from a nearby node.

    If the node is a pure text container, use its text. If it also wraps
    controls, prefer an early heading/label/legend that is not an option row.
    """
    option_norm = _normalize_prompt_text(option_label or "").lower()

    def usable(raw: str) -> str | None:
        cleaned = _normalize_prompt_text(raw)
        if not cleaned or len(cleaned) < 3:
            return None
        lower = cleaned.lower()
        if option_norm and lower == option_norm:
            return None
        if option_norm and option_norm in lower and len(cleaned) <= len(option_norm) + 2:
            return None
        # Whole-section dumps are useless as "context".
        if len(cleaned) > 280:
            return None
        # Option rows often look like short noun phrases; keep them out.
        if cleaned.count(" ") <= 2 and "?" not in cleaned and not _QUESTION_LEAD_RE.match(cleaned):
            if option_norm and lower == option_norm:
                return None
        return cleaned

    controls = node.find_all(["input", "textarea", "select"])
    if not controls:
        return usable(node.get_text(" ", strip=True))

    # Prefer dedicated prompt nodes inside a mixed container.
    for tag_name in ("legend", "h1", "h2", "h3", "h4", "h5", "h6", "label", "p", "span", "div"):
        for child in node.find_all(tag_name, recursive=True):
            if not isinstance(child, Tag):
                continue
            if child.find(["input", "textarea", "select"]):
                continue
            # Skip nodes nested under an option label that wraps a control.
            parent_label = child.find_parent("label")
            if parent_label is not None and parent_label.find(["input", "textarea", "select"]):
                continue
            hit = usable(child.get_text(" ", strip=True))
            if hit:
                return hit
    return None


def _score_prompt_candidate(text: str) -> int:
    """Higher is better — prefer real questions over stray nearby labels."""
    score = 0
    if "?" in text:
        score += 50
    if _QUESTION_LEAD_RE.match(text):
        score += 30
    # Prefer concise prompts; penalize blobs.
    score += max(0, 40 - abs(len(text) - 60) // 4)
    return score


def _collect_previous_prompts(
    node: Tag,
    *,
    option_label: str | None,
    into: list[str],
    sibling_limit: int = 8,
) -> None:
    sibling = node.previous_sibling
    steps = 0
    while sibling is not None and steps < sibling_limit:
        steps += 1
        if isinstance(sibling, Tag):
            # Don't steal prompts from a neighboring field that has its own controls.
            opts, others = _control_census(sibling)
            if opts or others:
                sibling = sibling.previous_sibling
                continue
            prompt = _prompt_from_node(sibling, option_label=option_label)
            if prompt:
                into.append(prompt)
        elif isinstance(sibling, NavigableString):
            text = str(sibling).strip()
            if text:
                into.append(text)
        sibling = sibling.previous_sibling


def _resolve_question_context(
    el: Tag,
    doc: BeautifulSoup,
    *,
    option_label: str | None,
    scope: Tag | None = None,
) -> str | None:
    """
    Find the question/prompt around a control (not the option text itself).

    When `scope` is the option-list container, only search around that block so
    a later question (visa / internships) cannot inherit an earlier prompt
    (pronouns) from higher up the form.
    """
    trusted: list[str] = []
    nearby: list[str] = []

    fieldset = el.find_parent("fieldset")
    if fieldset:
        heading = _fieldset_question_text(fieldset)
        if heading:
            trusted.append(heading)

    # Aria / role labels on the scoped container and its parents (short climb).
    start = scope or el
    climb_parents = [start] if scope is not None else []
    if scope is not None and isinstance(scope.parent, Tag):
        climb_parents.append(scope.parent)
    if scope is None:
        for parent in el.parents:
            if not isinstance(parent, Tag) or parent.name in {"form", "body", "html"}:
                break
            climb_parents.append(parent)
            if len(climb_parents) >= 4:
                break

    for parent in climb_parents:
        if not isinstance(parent, Tag):
            continue
        role = (parent.get("role") or "").lower()
        labelled_by = parent.get("aria-labelledby")
        if labelled_by:
            parts = []
            for ref in str(labelled_by).split():
                node = doc.find(id=ref)
                if node:
                    parts.append(node.get_text(" ", strip=True))
            if parts:
                nearby.append(" ".join(parts))
        aria = parent.get("aria-label")
        if aria:
            nearby.append(str(aria).strip())
        if role in {"group", "radiogroup", "checkboxgroup"}:
            prompt = _prompt_from_node(parent, option_label=option_label)
            if prompt:
                nearby.append(prompt)

    # Previous siblings of the option-list container (or a short climb from el).
    if scope is not None:
        _collect_previous_prompts(scope, option_label=option_label, into=nearby)
        if isinstance(scope.parent, Tag) and scope.parent.name not in {"form", "body", "html"}:
            _collect_previous_prompts(scope.parent, option_label=option_label, into=nearby)
            for child in scope.parent.children:
                if child is scope:
                    break
                if isinstance(child, Tag):
                    opts, others = _control_census(child)
                    if opts or others:
                        continue
                    prompt = _prompt_from_node(child, option_label=option_label)
                    if prompt:
                        nearby.append(prompt)
    else:
        node: Tag | None = el
        for _ in range(4):
            if node is None or not isinstance(node, Tag):
                break
            parent = node.parent
            if not isinstance(parent, Tag) or parent.name in {"form", "body", "html"}:
                break
            _collect_previous_prompts(node, option_label=option_label, into=nearby)
            for child in parent.children:
                if child is node:
                    break
                if isinstance(child, Tag):
                    opts, others = _control_census(child)
                    if opts or others:
                        continue
                    prompt = _prompt_from_node(child, option_label=option_label)
                    if prompt:
                        nearby.append(prompt)
            node = parent

    option_norm = _normalize_prompt_text(option_label or "").lower()

    def _rank(texts: list[str], *, require_question_shape: bool) -> str | None:
        ranked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for text in texts:
            cleaned = _normalize_prompt_text(text)
            if not cleaned or len(cleaned) < 3:
                continue
            lower = cleaned.lower()
            if option_norm and lower == option_norm:
                continue
            if option_norm and option_norm in lower and len(cleaned) <= len(option_norm) + 2:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            if require_question_shape and "?" not in cleaned and not _QUESTION_LEAD_RE.match(
                cleaned
            ):
                continue
            ranked.append((_score_prompt_candidate(cleaned), cleaned[:300]))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    return _rank(trusted, require_question_shape=False) or _rank(
        nearby, require_question_shape=True
    )


def _best_selector(el: Tag) -> str:
    el_id = el.get("id")
    if el_id:
        eid = str(el_id)
        # Bare #id only when it is a valid CSS identifier (no leading digit).
        if re.match(r"^[A-Za-z_][\w\-\:\.]*$", eid):
            return f"#{eid}"
        if re.match(r"^[\w\-\:\.]+$", eid):
            return f'{el.name}[id="{_css_escape(eid)}"]'

    name = el.get("name")
    if name:
        input_type = (el.get("type") or "").lower()
        value = el.get("value")
        # Radios/checkboxes often share name — disambiguate by value so every
        # option is kept, then collapsed into one question with options[].
        if (
            input_type in {"radio", "checkbox"}
            and value is not None
            and str(value) != ""
        ):
            return (
                f'{el.name}[name="{_css_escape(str(name))}"]'
                f'[value="{_css_escape(str(value))}"]'
            )
        return f'{el.name}[name="{_css_escape(str(name))}"]'

    aria = el.get("aria-label")
    if aria:
        return f'{el.name}[aria-label="{_css_escape(str(aria))}"]'

    placeholder = el.get("placeholder")
    if placeholder:
        return f'{el.name}[placeholder="{_css_escape(str(placeholder))}"]'

    type_attr = el.get("type")
    if type_attr:
        return f'{el.name}[type="{_css_escape(str(type_attr))}"]'

    return el.name


def _css_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def dump_debug_artifacts(url: str, out_dir: str | Path, *, headless: bool = True) -> Path:
    """
    Open a page and write raw HTML + the same parsed field map the router uses.

    Writes: page.html, parsed_fields.json, minified_form.html
    """
    import json
    from pathlib import Path as _Path

    dest = _Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _validate_url(url)

    with launch_page(headless=headless) as (_pw, _browser, page):
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        try:
            page.wait_for_selector("form, input, textarea, select, fieldset", timeout=15_000)
        except Exception:
            pass
        page.wait_for_timeout(1500)
        html = page.content()
        (dest / "page.html").write_text(html, encoding="utf-8")
        result = parse_form_html(html, url=page.url)

    payload = result.to_dict()
    (dest / "minified_form.html").write_text(
        payload.pop("minified_html", "") or "", encoding="utf-8"
    )
    (dest / "parsed_fields.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse a job application page into minified form HTML + field map."
    )
    parser.add_argument("url", help="Application form URL")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window (default is headless)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the JSON result",
    )
    parser.add_argument(
        "--html-out",
        type=str,
        default=None,
        help="Optional path to write only the minified HTML",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Write page.html + parsed_fields.json (same field map the router uses)",
    )
    args = parser.parse_args(argv)

    if args.debug_dir:
        dest = dump_debug_artifacts(
            args.url, args.debug_dir, headless=not args.headed
        )
        print(f"Wrote debug artifacts to {dest}/", flush=True)
        result = parse_form_html(
            (dest / "page.html").read_text(encoding="utf-8"),
            url=args.url,
        )
    else:
        result = parse_url(args.url, headless=not args.headed)
    payload = result.to_dict()

    summary = {
        "url": result.url,
        "form_count": result.form_count,
        "field_count": len(result.fields),
        "navigation": asdict(result.navigation),
        "fields": [
            {
                "selector": f.selector,
                "label": f.label,
                "type": f.input_type,
                "required": f.required,
            }
            for f in result.fields
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nminified_html_chars={len(result.minified_html)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {args.out}")

    if args.html_out:
        with open(args.html_out, "w", encoding="utf-8") as f:
            f.write(result.minified_html)
            f.write("\n")
        print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
