"""Map parsed form fields to fill actions via FAISS RAG + Cohere chat."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import cohere
import httpx
from cohere.types import JsonObjectResponseFormatV2

import config
from core.dom_parser import FormField, FormParseResult, parse_url
from core.knowledge_base import KnowledgeBase, SearchResult

ActionKind = Literal["fill", "select", "upload", "check", "skip", "unknown"]

ChatFn = Callable[[str, str], str]

# Small batches keep Cohere JSON-mode responses fast enough to avoid read timeouts.
FIELD_BATCH_SIZE = 5

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+\d[\d\-\s().]{8,}\d|\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_\-/]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_\-]+", re.I)

SYSTEM_PROMPT = """You are the form-filling brain for AutoApply AI.
You receive a SMALL BATCH of form fields plus resume/Q&A evidence for those fields.
Output a JSON object with key "actions": an array of field actions. No markdown.

Each action object MUST use this shape:
{
  "selector": "<css selector from the field list>",
  "action": "fill" | "select" | "upload" | "check" | "skip" | "unknown",
  "value": "<string value, or empty if skip/unknown>",
  "reason": "<short optional note, required when action is unknown>"
}

Action meanings:
- fill / select / check / upload: you have enough evidence to answer.
- skip: intentionally leave blank (honeypot, optional marketing opt-in, N/A, or
  unused extra experience slot). Do NOT use skip when you simply do not know.
- unknown: the question is real but evidence is missing/ambiguous. The human will
  be asked to supply this before submit. Always include a brief reason.

Rules:
- Each field lists evidence: [{chunk_id, score}, ...]. score is similarity for
  THAT field's query only. Resolve chunk_id in the shared "evidence" map for text —
  that is the ONLY resume/Q&A text you may use for that field.
- If evidence is empty or does not answer the question, action "unknown"
  (not skip) with a reason like "no work-auth info in resume".
- Never invent employers, dates, GPAs, visas, salary, or contact info.
- Prefer the exact selector strings provided in the field list.
- For email/phone/name/links: copy verbatim from contact evidence.
- For select fields: value MUST be one of the listed options (exact visible text).
- For file / resume / CV upload inputs: you will not see them (handled locally).
- For radio/checkbox GROUPS (one field with an options list): emit ONE action for
  the whole group. action "select" (radio) or "check" (checkbox). value MUST be
  the exact option text (or comma-separated option texts for multi-select
  checkboxes). Do NOT emit one action per option.
- For a lone checkbox with no options list: action "check" with value "true".

Work-experience blocks (title / start / end / description):
- Job title: copy verbatim from the matching resume role.
- Start/end dates: use the resume dates; adapt only the format the control needs
  (e.g. YYYY-MM, separate month/year selects). Do not invent ranges.
- Description / responsibilities textareas: COPY the resume bullet points VERBATIM.
  Preserve wording and bullet structure (join bullets with newlines in one textarea).
  Do NOT paraphrase, shorten, or rewrite to sound more corporate.
- If the form has N experience slots, map the N most recent resume roles in order.
  Leave unused slots as skip — never invent jobs.
- If snippets truncate a role, prefer the fullest experience snippet for that employer.
"""


@dataclass
class FillAction:
    selector: str
    action: ActionKind
    value: str = ""
    label: str | None = None
    context: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v is not None and v != ""}


def summarize_missing_info(actions: Sequence[FillAction]) -> list[FillAction]:
    """Fields that need human input before submit (for CLI + Discord)."""
    return [a for a in actions if a.action == "unknown"]


def format_missing_info_report(actions: Sequence[FillAction]) -> str:
    missing = summarize_missing_info(actions)
    if not missing:
        return "No fields flagged as missing info."
    lines = [f"MISSING INFO — {len(missing)} field(s) need your input:"]
    for action in missing:
        title = action.context or action.label or action.selector
        reason = action.reason or "model could not answer from resume/Q&A evidence"
        lines.append(f"  - {title}")
        lines.append(f"    selector: {action.selector}")
        lines.append(f"    reason: {reason}")
    return "\n".join(lines)


def build_field_query(field: FormField) -> str:
    parts = [
        field.context,
        field.label,
        field.placeholder,
        field.aria_label,
        field.name if field.name and not _looks_like_opaque_id(field.name) else None,
        field.input_type,
    ]
    return " ".join(p for p in parts if p).strip() or field.selector


_OPAQUE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_GENERIC_PLACEHOLDERS = {
    "start typing...",
    "start typing",
    "type here",
    "search",
    "select...",
    "select",
    "choose",
}


def _looks_like_opaque_id(value: str | None) -> bool:
    if not value:
        return False
    # Generated ids (uuid / uuid_uuid) are not human field names.
    head = value.split("_", 1)[0]
    return bool(_OPAQUE_ID_RE.match(head))


def _looks_like_question(text: str | None) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    cleaned = text.strip()
    if "?" in cleaned:
        return True
    return bool(
        re.match(
            r"^(what|which|how|why|when|where|who|do|does|did|are|is|was|were|"
            r"have|has|had|can|could|would|will|please|select|choose|indicate|"
            r"describe|tell|list|enter|provide|share|confirm)\b",
            cleaned,
            re.IGNORECASE,
        )
    )


def is_contact_field(field: FormField) -> bool:
    """Identity fields we fill from the resume contact line — never send to Cohere."""
    blob = _field_text_blob(field)
    if field.input_type in {"email", "tel"}:
        return True
    if re.search(r"\b(email|phone|mobile|tel)\b", blob):
        return True
    if "linkedin" in blob or "github" in blob:
        return True
    if re.search(r"\b(full\s*name|first\s*name|last\s*name|legal\s*name)\b", blob):
        return True
    return False


def field_is_answerable(field: FormField) -> bool:
    """
    True only if we have enough human-readable semantics to ask an LLM.

    UUID-only checkboxes, 'Start typing...' autocomplete boxes, and option
    radios with no surrounding question are not sent to Cohere.
    """
    # Collapsed option groups are answerable via their question + options.
    if field.options and (field.context or field.label):
        return True
    if field.aria_label and len(field.aria_label.strip()) >= 3:
        return True
    if field.label and len(field.label.strip()) >= 2:
        # Option-only labels ("He/Him", "Yes") need a question context for radios/checkboxes.
        if (field.input_type or "").lower() in {"checkbox", "radio"}:
            return bool(field.context) or bool(field.options)
        return True
    # Context alone only counts when it looks like a real question — page/job
    # titles must not make unlabeled "Start typing..." boxes look answerable.
    if _looks_like_question(field.context):
        return True
    placeholder = (field.placeholder or "").strip().lower()
    if placeholder and placeholder not in _GENERIC_PLACEHOLDERS:
        return True
    if field.name and not _looks_like_opaque_id(field.name):
        # Readable name like "LinkedIn" / "Product" without a question still isn't
        # enough for the LLM — treat via classify_opaque_field instead.
        if (field.input_type or "").lower() in {"checkbox", "radio"}:
            return False
        return True
    return False


def classify_opaque_field(field: FormField) -> FillAction:
    """
    Decide skip vs unknown for controls we refuse to send to Cohere.

    - Completely unlabeled UUID noise → skip
    - Has a human option/name (He/Him, LinkedIn, Product) but missing the
      surrounding question → unknown so Discord/CLI asks you
    """
    human_hint = next(
        (
            t
            for t in (field.label, field.name, field.aria_label, field.placeholder)
            if t and not _looks_like_opaque_id(t) and t.strip().lower() not in _GENERIC_PLACEHOLDERS
        ),
        None,
    )
    if human_hint:
        return FillAction(
            selector=field.selector,
            action="unknown",
            value="",
            label=field.label,
            context=field.context,
            reason=(
                f"saw option/name {human_hint!r} but not the question text "
                "(DOM missing fieldset/legend context) — needs human"
            ),
        )
    return FillAction(
        selector=field.selector,
        action="skip",
        value="",
        label=field.label,
        context=field.context,
        reason="unlabeled/opaque control",
    )


def expand_option_group_action(action: FillAction, field: FormField) -> list[FillAction]:
    """
    Map a group-level select/check (value = option text) onto concrete controls.

    Collapsed radio/checkbox fields use the first option's selector in the prompt;
    the model returns that selector + chosen option label(s). Remap to the real
    option selector(s) so Playwright can click the right input.
    """
    if action.action in {"skip", "unknown", "upload", "fill"}:
        return [action]
    if not field.options or not field.option_selectors:
        return [action]
    if len(field.options) != len(field.option_selectors):
        return [action]

    raw = (action.value or "").strip()
    if not raw or raw.lower() in {"true", "false", "1", "0", "yes", "no"}:
        # Model fell back to old checkbox style — keep as-is on the head selector.
        return [action]

    chosen = [part.strip() for part in raw.split(",") if part.strip()]
    option_map = {
        opt.strip().lower(): sel
        for opt, sel in zip(field.options, field.option_selectors, strict=True)
        if opt.strip()
    }
    kind = (field.input_type or "").lower()
    expanded: list[FillAction] = []
    for label in chosen:
        sel = option_map.get(label.lower())
        if sel is None:
            continue
        expanded.append(
            FillAction(
                selector=sel,
                action="check" if kind in {"radio", "checkbox"} else action.action,
                value="true",
                label=field.label,
                context=field.context,
                reason=action.reason,
            )
        )
        if kind == "radio":
            break
    if not expanded:
        # Unknown option text — keep unknown so HITL can fix it.
        return [
            FillAction(
                selector=field.selector,
                action="unknown",
                value=raw,
                label=field.label,
                context=field.context,
                reason=action.reason or f"option {raw!r} not in {field.options}",
            )
        ]
    return expanded


@dataclass
class FieldEvidence:
    """Per-field FAISS hits (not a global top-N dump)."""

    field: FormField
    hits: list[SearchResult]


@dataclass
class RetrievalPlan:
    """
    One Cohere call still fills the whole step, but evidence is field-scoped.

    `evidence` is the deduped map of chunks referenced by any field's hits.
    A contact-only form might only include chunk 0; a full work-history page
    may include many role chunks — that's intentional, not a fixed top-5.
    """

    field_evidence: list[FieldEvidence]
    evidence: dict[int, SearchResult]


def build_retrieval_plan(
    fields: Sequence[FormField],
    kb: KnowledgeBase,
    *,
    per_field_k: int = 2,
    min_score: float = 0.28,
    verbose: bool = False,
) -> RetrievalPlan:
    """
    For each field, search FAISS with that field's label/placeholder query.

    Chunking matters here: "Email" should attach the contact chunk via
    evidence_ids, while "Job title" attaches an experience role — without
    forcing a global top-5 that re-assembles most of a tiny resume.
    """
    field_evidence: list[FieldEvidence] = []
    evidence: dict[int, SearchResult] = {}

    for field in fields:
        if _is_resume_upload_field(field):
            continue
        # Contact identity is filled locally from the resume contact line —
        # never FAISS / Cohere (avoids email→experience chunk noise).
        if is_contact_field(field):
            continue
        if not field_is_answerable(field):
            continue
        query = build_field_query(field)
        hits = kb.search(query, k=per_field_k, min_score=min_score) if query else []
        field_evidence.append(FieldEvidence(field=field, hits=hits))
        for hit in hits:
            prev = evidence.get(hit.chunk.id)
            if prev is None or hit.score > prev.score:
                evidence[hit.chunk.id] = hit

    if verbose:
        print("-" * 72, flush=True)
        print("FAISS RETRIEVAL (per-field evidence)", flush=True)
        print(
            f"  fields={len(field_evidence)}  unique_evidence_chunks={len(evidence)}  "
            f"(per_field_k={per_field_k}, min_score={min_score})",
            flush=True,
        )
        for item in field_evidence:
            q = build_field_query(item.field)
            preview = q if len(q) <= 70 else q[:67] + "..."
            if item.hits:
                hit_s = ", ".join(f"id={h.chunk.id}:{h.score:.3f}" for h in item.hits)
            else:
                hit_s = "(none — model should skip)"
            print(f"  {item.field.selector}: {preview!r}", flush=True)
            print(f"     -> {hit_s}", flush=True)
        print("  evidence map:", flush=True)
        for chunk_id, hit in sorted(evidence.items(), key=lambda kv: -kv[1].score):
            head = hit.chunk.text.split("\n", 1)[0][:70]
            print(
                f"    id={chunk_id} score={hit.score:.4f} [{hit.chunk.source}] {head!r}",
                flush=True,
            )
        print("-" * 72, flush=True)

    return RetrievalPlan(field_evidence=field_evidence, evidence=evidence)


def _is_resume_upload_field(field: FormField) -> bool:
    if (field.input_type or "").lower() == "file":
        return True
    blob = " ".join(
        p.lower()
        for p in (field.label, field.name, field.aria_label, field.placeholder)
        if p
    )
    return any(tok in blob for tok in ("resume", "cv", "curriculum vitae")) and (
        "upload" in blob or "attach" in blob or "file" in blob
    )


def _compact(obj: dict[str, Any]) -> dict[str, Any]:
    """Drop null/empty noise so Cohere prompts stay small."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if value is None or value == "" or value == []:
            continue
        out[key] = value
    return out


def format_user_prompt(
    form: FormParseResult,
    plan: RetrievalPlan,
    *,
    batch_index: int = 1,
    batch_total: int = 1,
) -> str:
    fields_payload = [
        _compact(
            {
                "selector": item.field.selector,
                "tag": item.field.tag,
                "type": item.field.input_type,
                "label": item.field.label,
                "context": item.field.context,
                "placeholder": item.field.placeholder,
                "aria_label": item.field.aria_label,
                "name": (
                    item.field.name
                    if item.field.name and not _looks_like_opaque_id(item.field.name)
                    else None
                ),
                "required": item.field.required or None,
                "options": item.field.options or None,
                # score = cosine similarity for THIS field's query vs that chunk
                "evidence": [
                    {"chunk_id": h.chunk.id, "score": round(h.score, 4)}
                    for h in item.hits
                ],
            }
        )
        for item in plan.field_evidence
    ]
    # Shared text bodies only — scores live on the per-field evidence links above.
    evidence_payload = {
        str(chunk_id): {
            "source": hit.chunk.source,
            "text": hit.chunk.text,
        }
        for chunk_id, hit in sorted(plan.evidence.items())
    }
    payload = _compact(
        {
            "url": form.url,
            "batch": {"index": batch_index, "total": batch_total},
            "navigation": asdict(form.navigation) if batch_index == 1 else None,
            "fields": fields_payload,
            "evidence": evidence_payload,
        }
    )
    return (
        "Produce the JSON fill plan for this field batch.\n"
        "Each field has evidence [{chunk_id, score}] for that field's own FAISS "
        "query; look up chunk text in the shared evidence map. "
        "Only use that field's linked chunks. No raw HTML, no file paths.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _field_text_blob(field: FormField) -> str:
    return " ".join(
        p.lower()
        for p in (
            field.label,
            field.name,
            field.aria_label,
            field.placeholder,
            field.input_type,
            field.selector,
        )
        if p
    )


def try_local_contact_fill(field: FormField, contact_text: str) -> FillAction | None:
    """
    Fill obvious contact fields without Cohere (email/phone/links/name).
    """
    if not contact_text.strip():
        return None
    blob = _field_text_blob(field)

    def _action(value: str) -> FillAction:
        return FillAction(
            selector=field.selector,
            action="fill",
            value=value,
            label=field.label,
        )

    if field.input_type == "email" or re.search(r"\bemail\b", blob):
        match = _EMAIL_RE.search(contact_text)
        return _action(match.group(0)) if match else None

    if field.input_type == "tel" or re.search(r"\b(phone|mobile|tel)\b", blob):
        match = _PHONE_RE.search(contact_text)
        return _action(re.sub(r"\s+", " ", match.group(0)).strip()) if match else None

    if "linkedin" in blob:
        match = _LINKEDIN_RE.search(contact_text)
        return _action(match.group(0)) if match else None

    if "github" in blob:
        match = _GITHUB_RE.search(contact_text)
        return _action(match.group(0)) if match else None

    if re.search(r"\b(full\s*name|first\s*name|last\s*name|legal\s*name)\b", blob):
        # First non-empty line of the contact chunk is usually the name.
        first_line = next(
            (ln.strip() for ln in contact_text.splitlines() if ln.strip()),
            "",
        )
        if not first_line or "@" in first_line or first_line.lower().startswith("http"):
            return None
        if "first" in blob and "last" not in blob:
            return _action(first_line.split()[0])
        if "last" in blob and "first" not in blob:
            parts = first_line.split()
            return _action(parts[-1]) if len(parts) >= 2 else None
        return _action(first_line)

    return None


def contact_text_from_kb(kb: KnowledgeBase) -> str:
    """
    Resume contact line for local email/phone/link fills.

    Prefer the chunk that literally contains an email — do NOT use FAISS here.
    Semantic search for "email" often returns experience roles on short resumes.
    """
    chunks = getattr(kb, "chunks", None) or []
    for chunk in chunks:
        if getattr(chunk, "source", None) == "resume" and "@" in chunk.text:
            return chunk.text
    for chunk in chunks:
        if "@" in chunk.text:
            return chunk.text
    if chunks:
        return chunks[0].text
    return ""


def _batched(items: Sequence[FieldEvidence], size: int) -> list[list[FieldEvidence]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _plan_for_batch(batch: Sequence[FieldEvidence]) -> RetrievalPlan:
    evidence: dict[int, SearchResult] = {}
    for item in batch:
        for hit in item.hits:
            prev = evidence.get(hit.chunk.id)
            if prev is None or hit.score > prev.score:
                evidence[hit.chunk.id] = hit
    return RetrievalPlan(field_evidence=list(batch), evidence=evidence)


def print_llm_request(system: str, user: str, *, model: str) -> None:
    """Print exactly what we send to Cohere (sizes + full prompts)."""
    approx_tokens = max(1, (len(system) + len(user)) // 4)
    # Best-effort breakdown of the structured user JSON payload.
    field_count = 0
    evidence_count = 0
    nav = None
    try:
        # User prompt is preamble + JSON; isolate the JSON object.
        json_start = user.find("{")
        if json_start >= 0:
            payload = json.loads(user[json_start:])
            field_count = len(payload.get("fields") or [])
            evidence_count = len(payload.get("evidence") or {})
            nav = payload.get("navigation")
    except Exception:  # noqa: BLE001 - debug printer must never break routing
        pass

    sep = "=" * 72
    print(sep, flush=True)
    print("COHERE CHAT REQUEST", flush=True)
    print(f"  model:            {model}", flush=True)
    print(f"  system_chars:     {len(system)}", flush=True)
    print(f"  user_chars:       {len(user)}", flush=True)
    print(f"  total_chars:      {len(system) + len(user)}", flush=True)
    print(f"  approx_tokens(~): {approx_tokens}  (chars/4 heuristic)", flush=True)
    print(f"  fields:           {field_count}", flush=True)
    print(f"  evidence_chunks:  {evidence_count}", flush=True)
    if nav is not None:
        print(f"  navigation:       {nav}", flush=True)
    print(sep, flush=True)
    print("----- SYSTEM PROMPT -----", flush=True)
    print(system, flush=True)
    print("----- USER PROMPT -----", flush=True)
    print(user, flush=True)
    print(sep, flush=True)


def make_cohere_chat(
    api_key: str | None = None,
    model: str | None = None,
    *,
    timeout_seconds: float = 120.0,
    verbose: bool = True,
) -> ChatFn:
    key = api_key if api_key is not None else config.COHERE_API_KEY
    if not key:
        raise ValueError(
            "COHERE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = cohere.ClientV2(
        api_key=key,
        timeout=httpx.Timeout(timeout_seconds, connect=20.0),
    )
    model_name = model or config.COHERE_CHAT_MODEL

    def _chat(system: str, user: str) -> str:
        if verbose:
            print_llm_request(system, user, model=model_name)
        last_exc: Exception | None = None
        for attempt in range(1, 3):
            try:
                print(f"Sending request to Cohere (attempt {attempt}/2)…", flush=True)
                response = client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                    response_format=JsonObjectResponseFormatV2(type="json_object"),
                )
                return _extract_chat_text(response)
            except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
                last_exc = exc
                print(f"Cohere timeout on attempt {attempt}: {exc}", flush=True)
        assert last_exc is not None
        raise last_exc

    return _chat


def _extract_chat_text(response: Any) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if not content:
        raise RuntimeError("Cohere chat returned empty content.")
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    joined = "\n".join(parts).strip()
    if not joined:
        raise RuntimeError("Cohere chat returned no text blocks.")
    return joined


def parse_actions_json(
    raw: str,
    *,
    known_selectors: set[str] | None = None,
) -> list[FillAction]:
    """Parse model JSON into validated FillAction list."""
    data = _loads_json_object(raw)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("actions")
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("Expected JSON with an 'actions' array.")

    allowed: set[str] = {"fill", "select", "upload", "check", "skip", "unknown"}
    actions: list[FillAction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        selector = str(item.get("selector", "")).strip()
        action = str(item.get("action", "unknown")).strip().lower()
        value = item.get("value", "")
        if value is None:
            value = ""
        value = str(value)
        reason = item.get("reason")
        reason_s = str(reason).strip() if reason else None
        if not selector or action not in allowed:
            continue
        if known_selectors is not None and selector not in known_selectors:
            # Allow minor drift only if it's clearly a provided selector variant.
            continue
        if action in {"fill", "select", "upload"} and not value.strip():
            action = "unknown"
            reason_s = reason_s or "model returned an empty value"
        if action == "unknown" and not reason_s:
            reason_s = "insufficient evidence"
        actions.append(
            FillAction(
                selector=selector,
                action=action,  # type: ignore[arg-type]
                value=value,
                reason=reason_s,
            )
        )
    return actions


def _loads_json_object(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def route_form(
    form: FormParseResult,
    *,
    kb: KnowledgeBase | None = None,
    chat_fn: ChatFn | None = None,
    resume_pdf_path: Path | None = None,
    verbose_retrieval: bool = True,
    batch_size: int = FIELD_BATCH_SIZE,
) -> list[FillAction]:
    """
    Route one form step.

    Efficient path:
    1) file uploads + contact fields filled locally (no Cohere)
    2) remaining fields sent to Cohere in small batches with only their evidence
    """
    if not form.fields:
        return []

    knowledge = kb or KnowledgeBase()
    knowledge.ensure_loaded()
    chat = chat_fn or make_cohere_chat()
    resume_path = resume_pdf_path or config.RESUME_PDF_PATH

    contact_text = contact_text_from_kb(knowledge)
    actions: list[FillAction] = []

    # Contact fields: local regex fill only — never Cohere / FAISS.
    for field in form.fields:
        if _is_resume_upload_field(field):
            continue
        if not is_contact_field(field):
            continue
        local = try_local_contact_fill(field, contact_text)
        if local is not None:
            actions.append(local)
            if verbose_retrieval:
                print(
                    f"LOCAL FILL {local.selector} -> {local.value!r}",
                    flush=True,
                )
        else:
            actions.append(
                FillAction(
                    selector=field.selector,
                    action="unknown",
                    value="",
                    label=field.label,
                    context=field.context,
                    reason="contact field but value not found on resume contact line",
                )
            )
            if verbose_retrieval:
                print(
                    f"LOCAL UNKNOWN {field.selector} (no contact value) — not sent to Cohere",
                    flush=True,
                )

    plan = build_retrieval_plan(
        form.fields,
        knowledge,
        verbose=verbose_retrieval,
    )
    needs_llm: list[FieldEvidence] = list(plan.field_evidence)
    handled = {a.selector for a in actions}

    # Opaque/unlabeled controls never enter the retrieval plan.
    # Human-readable options without a question → unknown (you decide).
    # Pure UUID noise → skip.
    for field in form.fields:
        if _is_resume_upload_field(field):
            continue
        if field.selector in handled:
            continue
        if any(item.field.selector == field.selector for item in needs_llm):
            continue
        if not field_is_answerable(field):
            opaque = classify_opaque_field(field)
            actions.append(opaque)
            if verbose_retrieval:
                kind = "AUTO UNKNOWN" if opaque.action == "unknown" else "AUTO SKIP"
                print(
                    f"{kind} {field.selector} "
                    f"({opaque.reason}) — not sent to Cohere",
                    flush=True,
                )

    batches = _batched(needs_llm, batch_size) if needs_llm else []
    if verbose_retrieval:
        print(
            f"Routing: local={len(actions)}  llm_fields={len(needs_llm)}  "
            f"batches={len(batches)} (size={batch_size})",
            flush=True,
        )

    for batch_index, batch in enumerate(batches, start=1):
        sub_plan = _plan_for_batch(batch)
        known = {item.field.selector for item in batch}
        user_prompt = format_user_prompt(
            form,
            sub_plan,
            batch_index=batch_index,
            batch_total=len(batches),
        )
        if verbose_retrieval:
            print(
                f"--- Cohere batch {batch_index}/{len(batches)} "
                f"({len(batch)} fields, {len(sub_plan.evidence)} evidence chunks) ---",
                flush=True,
            )
        raw = chat(SYSTEM_PROMPT, user_prompt)
        batch_actions = parse_actions_json(raw, known_selectors=known)
        meta_by_selector = {
            item.field.selector: item.field for item in batch
        }
        resolved: list[FillAction] = []
        for action in batch_actions:
            field = meta_by_selector.get(action.selector)
            if field is not None:
                action.label = field.label
                action.context = field.context
                resolved.extend(expand_option_group_action(action, field))
            else:
                resolved.append(action)
        batch_actions = resolved
        present = {a.selector for a in batch_actions}
        # Group fields are represented by the head selector; expanded option
        # selectors also count as present for that question.
        for item in batch:
            field = item.field
            covered = field.selector in present or any(
                sel in present for sel in (field.option_selectors or [])
            )
            if not covered:
                batch_actions.append(
                    FillAction(
                        selector=field.selector,
                        action="unknown",
                        value="",
                        label=field.label,
                        context=field.context,
                        reason="model omitted this field",
                    )
                )
        actions.extend(batch_actions)

    # Any answerable non-file field never seen -> unknown (needs human).
    present = {a.selector for a in actions}
    for field in form.fields:
        if _is_resume_upload_field(field):
            continue
        if field.selector not in present:
            actions.append(
                FillAction(
                    selector=field.selector,
                    action="unknown" if field_is_answerable(field) else "skip",
                    value="",
                    label=field.label,
                    context=field.context,
                    reason=(
                        "not covered by local fill or Cohere batches"
                        if field_is_answerable(field)
                        else "unlabeled/opaque control"
                    ),
                )
            )

    # Deterministic resume upload — never ask the LLM for a filesystem path.
    for field in form.fields:
        if not _is_resume_upload_field(field):
            continue
        actions.append(
            FillAction(
                selector=field.selector,
                action="upload",
                value=str(resume_path),
                label=field.label,
            )
        )
    return actions


def route_url(
    url: str,
    *,
    headless: bool = True,
    kb: KnowledgeBase | None = None,
    chat_fn: ChatFn | None = None,
) -> tuple[FormParseResult, list[FillAction]]:
    form = parse_url(url, headless=headless)
    actions = route_form(form, kb=kb, chat_fn=chat_fn)
    return form, actions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG + Cohere router: map an apply URL's form fields to fill actions."
    )
    parser.add_argument("url", help="Application form URL")
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not dump the full Cohere request prompts to the terminal",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the JSON fill plan",
    )
    args = parser.parse_args(argv)

    form = parse_url(args.url, headless=not args.headed)
    actions = route_form(
        form,
        chat_fn=make_cohere_chat(verbose=not args.quiet),
        verbose_retrieval=not args.quiet,
    )
    missing = summarize_missing_info(actions)
    payload = {
        "url": form.url,
        "navigation": asdict(form.navigation),
        "field_count": len(form.fields),
        "missing_info_count": len(missing),
        "missing_info": [a.to_dict() for a in missing],
        "actions": [a.to_dict() for a in actions],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(flush=True)
    print(format_missing_info_report(actions), flush=True)

    if args.out:
        path = Path(args.out)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
