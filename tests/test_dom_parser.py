"""Unit tests for form minify + Next/Submit navigation detection (no browser)."""

from __future__ import annotations

from core.dom_parser import detect_navigation, minify_form_html, parse_form_html
from bs4 import BeautifulSoup


SINGLE_PAGE_SUBMIT = """
<html><body>
  <form id="app">
    <label for="email">Email</label>
    <input id="email" name="email" type="email" required class="big-noise" style="color:red" />
    <label for="phone">Phone</label>
    <input id="phone" name="phone" type="tel" placeholder="555-0100" />
    <label for="auth">Work authorization</label>
    <select id="auth" name="work_auth">
      <option value="">Select</option>
      <option value="yes">Yes</option>
      <option value="no">No</option>
    </select>
    <button type="submit" class="btn">Submit Application</button>
  </form>
  <script>window.tracking = true;</script>
</body></html>
"""

MULTI_STEP_NEXT = """
<html><body>
  <form>
    <label for="first_name">First name</label>
    <input id="first_name" name="first_name" />
    <button type="button">Continue</button>
    <a href="/careers">Cancel</a>
  </form>
</body></html>
"""

BOTH_NEXT_AND_SUBMIT = """
<html><body>
  <form>
    <input name="q1" />
    <button type="button">Next</button>
    <button type="submit">Submit</button>
  </form>
</body></html>
"""


def test_parse_extracts_fields_and_strips_noise() -> None:
    result = parse_form_html(SINGLE_PAGE_SUBMIT, url="https://example.com/apply")
    assert result.form_count == 1
    assert len(result.fields) == 3

    selectors = {f.selector for f in result.fields}
    assert "#email" in selectors
    assert "#phone" in selectors
    assert "#auth" in selectors

    email = next(f for f in result.fields if f.selector == "#email")
    assert email.label == "Email"
    assert email.required is True
    assert email.input_type == "email"

    auth = next(f for f in result.fields if f.selector == "#auth")
    assert "Yes" in auth.options
    assert "No" in auth.options

    assert "script" not in result.minified_html.lower()
    assert "big-noise" not in result.minified_html
    assert "color:red" not in result.minified_html
    assert 'name="email"' in result.minified_html
    assert "Submit Application" in result.minified_html or "submit" in result.minified_html.lower()


def test_navigation_submit_single_page() -> None:
    result = parse_form_html(SINGLE_PAGE_SUBMIT)
    assert result.navigation.kind == "submit"
    assert result.navigation.selector is not None
    assert "submit" in (result.navigation.text or "").lower()


def test_navigation_next_multi_step() -> None:
    result = parse_form_html(MULTI_STEP_NEXT)
    assert result.navigation.kind == "next"
    assert "continue" in (result.navigation.text or "").lower()


def test_navigation_prefers_next_when_both_present() -> None:
    result = parse_form_html(BOTH_NEXT_AND_SUBMIT)
    assert result.navigation.kind == "next"
    kinds = {c["kind"] for c in result.navigation.candidates}
    assert "next" in kinds and "submit" in kinds


def test_detect_navigation_unknown_without_cta() -> None:
    soup = BeautifulSoup("<form><input name='x' /></form>", "html.parser")
    nav = detect_navigation(soup)
    assert nav.kind == "unknown"
    assert nav.selector is None


def test_minify_keeps_labels_and_options() -> None:
    soup = BeautifulSoup(SINGLE_PAGE_SUBMIT, "html.parser")
    form = soup.find("form")
    assert form is not None
    mini = minify_form_html(form)
    assert "<label" in mini
    assert "<option" in mini
    assert "Email" in mini


def test_radio_captures_question_context_not_just_option() -> None:
    html = """
    <form>
      <fieldset>
        <legend>What are your pronouns?</legend>
        <label><input type="radio" name="pronouns" value="he" /> He/Him</label>
        <label><input type="radio" name="pronouns" value="she" /> She/Her</label>
      </fieldset>
      <input type="checkbox" name="1d50878f-e8f2-4a9c-80e9-531e24a190a3" />
    </form>
    """
    result = parse_form_html(html)
    radios = [f for f in result.fields if f.input_type == "radio"]
    assert len(radios) == 1
    assert radios[0].context == "What are your pronouns?"
    assert radios[0].options == ["He/Him", "She/Her"]
    assert len(radios[0].option_selectors) == 2


def test_fieldset_label_heading_used_as_question_without_legend() -> None:
    """Many apps put the question in a fieldset > label, not <legend>."""
    html = """
    <form>
      <fieldset>
        <label>Will you require sponsorship for employment visa status?</label>
        <div><label><input type="radio" name="v1" value="opt" /> OPT</label></div>
        <div><label><input type="radio" name="v2" value="h1b" /> H1B</label></div>
      </fieldset>
      <fieldset>
        <label>How many prior internships have you had?</label>
        <div><label><input type="radio" name="i0" value="0" /> 0</label></div>
        <div><label><input type="radio" name="i1" value="1" /> 1</label></div>
      </fieldset>
    </form>
    """
    result = parse_form_html(html)
    radios = [f for f in result.fields if f.input_type == "radio"]
    assert len(radios) == 2
    by_opts = {tuple(f.options): f for f in radios}
    assert "sponsorship" in (by_opts[("OPT", "H1B")].context or "").lower()
    assert "internships" in (by_opts[("0", "1")].context or "").lower()


def test_neighboring_radio_questions_stay_separate() -> None:
    """Adjacent multiple-choice blocks on one form must not collapse together."""
    html = """
    <form>
      <h1>Application</h1>
      <div>
        <div>What pronouns would you like our team to use when addressing you?</div>
        <div>
          <label>
            <input type="radio" id="aaaaaaa1-aaaa-aaaa-aaaa-aaaaaaaaaaa1_bbbbbbb1-bbbb-bbbb-bbbb-bbbbbbbbbbb1"
                   name="aaaaaaa1-aaaa-aaaa-aaaa-aaaaaaaaaaa1_bbbbbbb1-bbbb-bbbb-bbbb-bbbbbbbbbbb1" />
            He/Him
          </label>
          <label>
            <input type="radio" id="aaaaaaa1-aaaa-aaaa-aaaa-aaaaaaaaaaa1_bbbbbbb2-bbbb-bbbb-bbbb-bbbbbbbbbbb2"
                   name="aaaaaaa1-aaaa-aaaa-aaaa-aaaaaaaaaaa1_bbbbbbb2-bbbb-bbbb-bbbb-bbbbbbbbbbb2" />
            She/Her
          </label>
        </div>
      </div>
      <div>
        <div>Will you now or at any time in the future require sponsorship for employment visa status (e.g. H1B, OPT)?</div>
        <div>
          <label>
            <input type="radio" id="ccccccc1-cccc-cccc-cccc-ccccccccccc1_ddddddd1-dddd-dddd-dddd-ddddddddddd1"
                   name="ccccccc1-cccc-cccc-cccc-ccccccccccc1_ddddddd1-dddd-dddd-dddd-ddddddddddd1" />
            OPT
          </label>
          <label>
            <input type="radio" id="ccccccc1-cccc-cccc-cccc-ccccccccccc1_ddddddd2-dddd-dddd-dddd-ddddddddddd2"
                   name="ccccccc1-cccc-cccc-cccc-ccccccccccc1_ddddddd2-dddd-dddd-dddd-ddddddddddd2" />
            H1B
          </label>
        </div>
      </div>
      <div>
        <div>How many prior internships have you had?</div>
        <div>
          <label>
            <input type="radio" id="eeeeeee1-eeee-eeee-eeee-eeeeeeeeeee1_fffffff1-ffff-ffff-ffff-fffffffffff1"
                   name="eeeeeee1-eeee-eeee-eeee-eeeeeeeeeee1_fffffff1-ffff-ffff-ffff-fffffffffff1" />
            0
          </label>
          <label>
            <input type="radio" id="eeeeeee1-eeee-eeee-eeee-eeeeeeeeeee1_fffffff2-ffff-ffff-ffff-fffffffffff2"
                   name="eeeeeee1-eeee-eeee-eeee-eeeeeeeeeee1_fffffff2-ffff-ffff-ffff-fffffffffff2" />
            1
          </label>
        </div>
      </div>
    </form>
    """
    result = parse_form_html(html)
    radios = [f for f in result.fields if f.input_type == "radio"]
    assert len(radios) == 3
    by_options = {tuple(f.options): f for f in radios}
    assert ("He/Him", "She/Her") in by_options
    assert ("OPT", "H1B") in by_options
    assert ("0", "1") in by_options
    assert "pronouns" in (by_options[("He/Him", "She/Her")].context or "").lower()
    assert "sponsorship" in (by_options[("OPT", "H1B")].context or "").lower()
    assert "internships" in (by_options[("0", "1")].context or "").lower()


def test_checkbox_group_in_fieldset_stays_one_question() -> None:
    """Multi-select checkboxes under one fieldset collapse; neighbors stay apart."""
    html = """
    <form>
      <fieldset>
        <label>What pronouns would you like our team to use when addressing you?</label>
        <div><label><input type="checkbox" name="p_he" /> He/Him</label></div>
        <div><label><input type="checkbox" name="p_she" /> She/Her</label></div>
        <div><label><input type="checkbox" name="p_they" /> They/Them</label></div>
      </fieldset>
      <fieldset>
        <label>How did you hear about this role?</label>
        <div><label><input type="checkbox" name="src_li" /> LinkedIn</label></div>
        <div><label><input type="checkbox" name="src_friend" /> Friend</label></div>
      </fieldset>
      <!-- lone confirm boxes must NOT merge with each other -->
      <div><label><input type="checkbox" name="c1" /> I can relocate</label></div>
      <div><label><input type="checkbox" name="c2" /> I am authorized to work</label></div>
    </form>
    """
    result = parse_form_html(html)
    checkboxes = [f for f in result.fields if f.input_type == "checkbox"]
    grouped = [f for f in checkboxes if f.options]
    lone = [f for f in checkboxes if not f.options]
    assert len(grouped) == 2
    assert {tuple(f.options) for f in grouped} == {
        ("He/Him", "She/Her", "They/Them"),
        ("LinkedIn", "Friend"),
    }
    assert len(lone) == 2


def test_minify_handles_nested_empty_wrappers() -> None:
    """Deeply nested empty wrappers must not crash minify."""
    html = """
    <form>
      <div><div><div></div></div></div>
      <label for="email">Email</label>
      <input id="email" name="email" />
      <div class="spacer"><span></span></div>
      <button type="submit">Submit</button>
    </form>
    """
    result = parse_form_html(html)
    assert any(f.selector == "#email" for f in result.fields)
    assert result.navigation.kind == "submit"
    assert "email" in result.minified_html
