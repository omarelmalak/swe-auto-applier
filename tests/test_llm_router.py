"""Unit tests for LLM fill-plan parsing (no live Cohere / browser)."""

from __future__ import annotations

from pathlib import Path

from core.dom_parser import FormField, FormParseResult, NavigationTarget
from core.knowledge_base import DocumentChunk, SearchResult
from core.llm_router import (
    SYSTEM_PROMPT,
    FillAction,
    build_field_query,
    build_retrieval_plan,
    classify_opaque_field,
    contact_text_from_kb,
    expand_option_group_action,
    field_is_answerable,
    format_missing_info_report,
    format_user_prompt,
    is_contact_field,
    parse_actions_json,
    route_form,
    summarize_missing_info,
    try_local_contact_fill,
)


def test_system_prompt_requires_verbatim_bullets() -> None:
    assert "VERBATIM" in SYSTEM_PROMPT or "verbatim" in SYSTEM_PROMPT.lower()
    assert "bullet" in SYSTEM_PROMPT.lower()
    assert "paraphrase" in SYSTEM_PROMPT.lower() or "rewrite" in SYSTEM_PROMPT.lower()
    assert "chunk_id" in SYSTEM_PROMPT
    assert "unknown" in SYSTEM_PROMPT


def test_parse_unknown_action_and_missing_report() -> None:
    raw = json_actions(
        [
            {
                "selector": "#auth",
                "action": "unknown",
                "value": "",
                "reason": "no work authorization in resume",
            },
            {"selector": "#email", "action": "fill", "value": "a@b.com"},
        ]
    )
    actions = parse_actions_json(raw, known_selectors={"#auth", "#email"})
    missing = summarize_missing_info(actions)
    assert len(missing) == 1
    assert missing[0].selector == "#auth"
    assert "work authorization" in (missing[0].reason or "")
    report = format_missing_info_report(actions)
    assert "MISSING INFO" in report
    assert "#auth" in report


def test_parse_actions_json_filters_unknown_selectors() -> None:
    raw = json_actions(
        [
            {"selector": "#email", "action": "fill", "value": "a@b.com"},
            {"selector": "#evil", "action": "fill", "value": "nope"},
            {"selector": "#auth", "action": "select", "value": "Yes"},
            {"selector": "#notes", "action": "fill", "value": ""},
        ]
    )
    actions = parse_actions_json(raw, known_selectors={"#email", "#auth", "#notes"})
    by_sel = {a.selector: a for a in actions}
    assert by_sel["#email"].action == "fill"
    assert by_sel["#auth"].action == "select"
    assert by_sel["#notes"].action == "unknown"  # empty fill value => needs human
    assert "#evil" not in by_sel


def test_parse_actions_json_accepts_bare_array() -> None:
    raw = '[{"selector":"#email","action":"fill","value":"x@y.com"}]'
    actions = parse_actions_json(raw, known_selectors={"#email"})
    assert actions == [FillAction(selector="#email", action="fill", value="x@y.com")]


def test_build_field_query_uses_label() -> None:
    field = FormField(
        tag="input",
        selector="#email",
        input_type="email",
        label="Email Address",
        placeholder="you@example.com",
    )
    q = build_field_query(field)
    assert "Email Address" in q
    assert "you@example.com" in q


def test_retrieval_is_per_field_not_global_top_n() -> None:
    hubspot = DocumentChunk(id=3, source="resume", text="HubSpot Backend Intern")
    scotia = DocumentChunk(id=5, source="resume", text="Scotiabank Frontend Intern")
    skills = DocumentChunk(id=1, source="resume", text="Skills\nPython TypeScript")

    class FakeKB:
        def search(self, query: str, k: int = 3, min_score=None):
            q = query.lower()
            if "skill" in q:
                hits = [SearchResult(chunk=skills, score=0.55)]
            elif "job" in q or "title" in q or "company" in q:
                hits = [
                    SearchResult(chunk=hubspot, score=0.48),
                    SearchResult(chunk=scotia, score=0.41),
                ]
            else:
                hits = []
            return [h for h in hits if min_score is None or h.score >= min_score][:k]

    fields = [
        FormField(tag="input", selector="#email", input_type="email", label="Email"),
        FormField(tag="input", selector="#skills", input_type="text", label="Skills"),
        FormField(tag="input", selector="#title", input_type="text", label="Job title"),
    ]
    plan = build_retrieval_plan(fields, FakeKB(), min_score=0.28)  # type: ignore[arg-type]

    by_sel = {item.field.selector: item for item in plan.field_evidence}
    # Contact fields are excluded from FAISS entirely (local fill path).
    assert "#email" not in by_sel
    assert [h.chunk.id for h in by_sel["#skills"].hits] == [1]
    assert [h.chunk.id for h in by_sel["#title"].hits] == [3, 5]
    # Unique evidence includes whatever fields referenced — no global top-5 trim.
    assert set(plan.evidence) == {1, 3, 5}

    prompt = format_user_prompt(
        FormParseResult(
            url="https://example.com",
            minified_html="",
            fields=fields,
            navigation=NavigationTarget(kind="submit"),
        ),
        plan,
    )
    compact = prompt.replace("\n", "").replace(" ", "")
    assert '"chunk_id":1' in compact
    assert '"score":0.55' in compact
    assert "knowledge_snippets" not in prompt
    assert "HubSpot" in prompt and "Python TypeScript" in prompt
    assert "email@x.com" not in prompt


def test_garbage_fields_are_not_answerable() -> None:
    assert not field_is_answerable(
        FormField(
            tag="input",
            selector='input[placeholder="Start typing..."]',
            input_type="text",
            placeholder="Start typing...",
        )
    )
    # Page/job title must not make unlabeled autocomplete look answerable.
    assert not field_is_answerable(
        FormField(
            tag="input",
            selector='input[placeholder="Start typing..."]',
            input_type="text",
            placeholder="Start typing...",
            context="Software Engineer, New Grad (Dec 2026)",
        )
    )
    assert not field_is_answerable(
        FormField(
            tag="input",
            selector='input[name="1d50878f-e8f2-4a9c-80e9-531e24a190a3"]',
            input_type="checkbox",
            name="1d50878f-e8f2-4a9c-80e9-531e24a190a3",
        )
    )
    assert not field_is_answerable(
        FormField(
            tag="input",
            selector='input[name="pronouns"]',
            input_type="radio",
            name="pronouns",
            label="He/Him",
        )
    )
    assert field_is_answerable(
        FormField(
            tag="input",
            selector='input[name="pronouns"]',
            input_type="radio",
            name="pronouns",
            label="He/Him",
            context="What are your pronouns?",
        )
    )


def test_expand_option_group_maps_label_to_selector() -> None:
    field = FormField(
        tag="input",
        selector="#pronouns-0",
        input_type="radio",
        label="What are your pronouns?",
        context="What are your pronouns?",
        options=["He/Him", "She/Her", "They/Them"],
        option_selectors=["#pronouns-0", "#pronouns-1", "#pronouns-2"],
    )
    expanded = expand_option_group_action(
        FillAction(selector="#pronouns-0", action="select", value="They/Them"),
        field,
    )
    assert len(expanded) == 1
    assert expanded[0].selector == "#pronouns-2"
    assert expanded[0].action == "check"
    assert expanded[0].value == "true"


def test_opaque_option_without_question_is_unknown_not_skip() -> None:
    he_him = classify_opaque_field(
        FormField(
            tag="input",
            selector='input[name="pronouns"]',
            input_type="radio",
            name="pronouns",
            label="He/Him",
        )
    )
    assert he_him.action == "unknown"
    assert "He/Him" in (he_him.reason or "")

    uuid_box = classify_opaque_field(
        FormField(
            tag="input",
            selector='input[name="1d50878f-e8f2-4a9c-80e9-531e24a190a3"]',
            input_type="checkbox",
            name="1d50878f-e8f2-4a9c-80e9-531e24a190a3",
        )
    )
    assert uuid_box.action == "skip"


def test_local_contact_fill_email_phone() -> None:
    contact = "Omar El Malak\n+1 905-609-6644 omarelmalak28@gmail.com linkedin.com/in/omarelmalak"
    email = FormField(tag="input", selector="#email", input_type="email", label="Email")
    phone = FormField(tag="input", selector="#phone", input_type="tel", label="Phone")
    other = FormField(tag="textarea", selector="#why", input_type="textarea", label="Why us?")

    email_action = try_local_contact_fill(email, contact)
    phone_action = try_local_contact_fill(phone, contact)
    assert email_action is not None and email_action.value == "omarelmalak28@gmail.com"
    assert phone_action is not None and "905" in phone_action.value
    assert try_local_contact_fill(other, contact) is None


def test_route_form_batches_llm_calls() -> None:
    fields = [
        FormField(
            tag="textarea",
            selector=f"#q{i}",
            input_type="textarea",
            label=f"Open question {i}",
        )
        for i in range(12)
    ]
    form = FormParseResult(
        url="https://example.com/apply",
        minified_html="",
        fields=fields,
        navigation=NavigationTarget(kind="submit", selector="button", text="Submit"),
    )

    class FakeKB:
        def ensure_loaded(self) -> None:
            return None

        def search(self, query: str, k: int = 3, min_score=None):
            return []

    calls: list[str] = []

    def fake_chat(system: str, user: str) -> str:
        calls.append(user)
        return '{"actions": []}'

    actions = route_form(
        form,
        kb=FakeKB(),  # type: ignore[arg-type]
        chat_fn=fake_chat,
        verbose_retrieval=False,
        batch_size=5,
    )
    assert len(calls) == 3  # 12 fields / batch 5 => 3 Cohere calls
    assert all('"batch"' in c for c in calls)
    assert len(actions) == 12
    # Omitted fields become unknown (need human), not silent skip.
    assert all(a.action == "unknown" for a in actions)


def test_contact_fields_never_sent_to_cohere() -> None:
    form = FormParseResult(
        url="https://example.com/apply",
        minified_html="<form></form>",
        fields=[
            FormField(tag="input", selector="#email", input_type="email", label="Email"),
            FormField(tag="input", selector="#phone", input_type="tel", label="Phone"),
            FormField(
                tag="input",
                selector="#li",
                input_type="text",
                label="LinkedIn Profile",
            ),
            FormField(
                tag="textarea",
                selector="#why",
                input_type="textarea",
                label="Why do you want this role?",
            ),
            FormField(
                tag="input",
                selector="#resume",
                input_type="file",
                label="Upload resume",
                name="resume",
            ),
        ],
        navigation=NavigationTarget(kind="submit", selector="button", text="Submit"),
    )

    contact = DocumentChunk(
        id=0,
        text="Omar El Malak\n+1 905-609-6644 omar@example.com linkedin.com/in/omar",
        source="resume",
    )

    class FakeKB:
        chunks = [contact]

        def ensure_loaded(self) -> None:
            return None

        def search(self, query: str, k: int = 3, min_score=None):
            # Deliberately return unrelated experience — contact path must ignore this.
            return [
                SearchResult(
                    chunk=DocumentChunk(
                        id=3,
                        text="Experience\nBackend Intern at HubSpot",
                        source="resume",
                    ),
                    score=0.99,
                )
            ]

    calls: list[str] = []

    def fake_chat(system: str, user: str) -> str:
        calls.append(user)
        assert "#email" not in user
        assert "#phone" not in user
        assert "#li" not in user
        assert "#resume" not in user
        assert "#why" in user
        assert "minified_html" not in user
        assert "resume_pdf_path" not in user
        return json_actions(
            [{"selector": "#why", "action": "fill", "value": "I like building tools."}]
        )

    actions = route_form(
        form,
        kb=FakeKB(),  # type: ignore[arg-type]
        chat_fn=fake_chat,
        resume_pdf_path=Path("data/resume.pdf"),
        verbose_retrieval=False,
    )
    by_sel = {a.selector: a for a in actions}
    assert by_sel["#email"].action == "fill" and by_sel["#email"].value == "omar@example.com"
    assert by_sel["#phone"].action == "fill" and "905" in by_sel["#phone"].value
    assert by_sel["#li"].action == "fill" and "linkedin.com/in/omar" in by_sel["#li"].value
    assert by_sel["#why"].value == "I like building tools."
    assert by_sel["#resume"].action == "upload"
    assert len(calls) == 1


def test_contact_text_prefers_email_chunk_not_faiss() -> None:
    class FakeKB:
        chunks = [
            DocumentChunk(id=0, text="Omar\nomar@example.com", source="resume"),
            DocumentChunk(id=3, text="Experience\nBackend Intern", source="resume"),
        ]

    assert "omar@example.com" in contact_text_from_kb(FakeKB())  # type: ignore[arg-type]
    assert is_contact_field(
        FormField(tag="input", selector="#email", input_type="email", label="Email")
    )


def json_actions(items: list[dict]) -> str:
    import json

    return json.dumps({"actions": items})
