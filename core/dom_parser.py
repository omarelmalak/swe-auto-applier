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
    placeholder: str | None = None
    aria_label: str | None = None
    options: list[str] = field(default_factory=list)
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
        return {
            "url": self.url,
            "form_count": self.form_count,
            "navigation": asdict(self.navigation),
            "fields": [asdict(f) for f in self.fields],
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

    fields = _extract_fields(scope, soup)
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
                placeholder=el.get("placeholder"),
                aria_label=el.get("aria-label"),
                options=options,
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
        fields.append(
            FormField(
                tag=el.name,
                selector=selector,
                input_type="contenteditable",
                name=el.get("name"),
                element_id=el.get("id"),
                label=_resolve_label(el, full_doc),
                aria_label=el.get("aria-label"),
                required=el.get("aria-required") == "true",
            )
        )

    return fields


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


def _best_selector(el: Tag) -> str:
    el_id = el.get("id")
    if el_id and re.match(r"^[A-Za-z_][\w\-\:\.]*$", str(el_id)):
        return f"#{el_id}"

    name = el.get("name")
    if name:
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
    args = parser.parse_args(argv)

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
