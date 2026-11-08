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


def test_minify_handles_nested_empty_wrappers() -> None:
    """Ashby-like trees with nested empty divs must not crash minify."""
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
