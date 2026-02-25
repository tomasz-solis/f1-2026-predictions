from __future__ import annotations

from src.dashboard import layout


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_brand_asset_path_and_page_icon_resolution(tmp_path, patcher):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    icon = second / "mark.png"
    icon.write_bytes(b"png-bytes")

    patcher.setattr(layout, "BRAND_ASSET_DIRS", (first, second))
    patcher.setattr(layout, "BRAND_FAVICON_FILE", "mark.png")

    assert layout._brand_asset_path("mark.png") == icon
    assert layout._page_icon() == str(icon)

    icon.unlink()
    assert layout._page_icon() == "F1"


def test_header_alignment_normalization(patcher):
    patcher.setattr(layout, "BRAND_HEADER_ALIGNMENT", "center")
    assert layout._header_alignment() == "center"

    patcher.setattr(layout, "BRAND_HEADER_ALIGNMENT", "left")
    assert layout._header_alignment() == "left"

    patcher.setattr(layout, "BRAND_HEADER_ALIGNMENT", "invalid")
    assert layout._header_alignment() == "left"


def test_configure_page_and_render_global_styles_call_streamlit(patcher):
    config_calls = []
    markdown_calls = []

    patcher.setattr(layout.st, "set_page_config", lambda **kwargs: config_calls.append(kwargs))
    patcher.setattr(
        layout.st,
        "markdown",
        lambda body, unsafe_allow_html=False: markdown_calls.append((body, unsafe_allow_html)),
    )
    patcher.setattr(layout, "_page_icon", lambda: "ICON")

    layout.configure_page()
    layout.render_global_styles()

    assert config_calls[0]["page_title"] == layout.BRAND_PAGE_TITLE
    assert config_calls[0]["page_icon"] == "ICON"
    assert markdown_calls[0][1] is True
    assert "<style>" in markdown_calls[0][0]


def test_build_asset_data_uri_supports_png_and_svg(tmp_path):
    png = tmp_path / "logo.png"
    svg = tmp_path / "logo.svg"
    png.write_bytes(b"abc")
    svg.write_bytes(b"<svg/>")

    layout._build_asset_data_uri.cache_clear()
    png_uri = layout._build_asset_data_uri(str(png))
    svg_uri = layout._build_asset_data_uri(str(svg))

    assert png_uri.startswith("data:image/png;base64,")
    assert svg_uri.startswith("data:image/svg+xml;base64,")
    assert layout._build_asset_data_uri(str(png)) == png_uri


def test_render_header_renders_logo_when_asset_exists(tmp_path, patcher):
    logo = tmp_path / "wordmark.png"
    logo.write_bytes(b"image")
    output: list[str] = []

    patcher.setattr(layout, "_brand_asset_path", lambda filename: logo)
    patcher.setattr(layout, "_build_asset_data_uri", lambda path: "data:image/png;base64,abc")
    patcher.setattr(layout, "_header_alignment", lambda: "center")
    patcher.setattr(
        layout.st,
        "markdown",
        lambda body, unsafe_allow_html=False: output.append(body),
    )

    layout.render_header()

    assert "brand-shell--center" in output[0]
    assert '<img class="brand-logo"' in output[0]


def test_render_header_falls_back_to_text_when_logo_missing(tmp_path, patcher):
    missing = tmp_path / "missing.png"
    output: list[str] = []

    patcher.setattr(layout, "_brand_asset_path", lambda filename: missing)
    patcher.setattr(layout, "_header_alignment", lambda: "left")
    patcher.setattr(
        layout.st,
        "markdown",
        lambda body, unsafe_allow_html=False: output.append(body),
    )

    layout.render_header()

    assert "brand-shell--left" in output[0]
    assert "main-header" in output[0]
    assert '<img class="brand-logo"' not in output[0]


def test_render_sidebar_returns_page_and_logging_toggle(patcher):
    calls = {
        "markdown": [],
        "segmented_label": None,
        "segmented_options": None,
        "radio_called": False,
    }

    def _segmented_control(label, options, **_kwargs):
        calls["segmented_label"] = label
        calls["segmented_options"] = list(options)
        return "Team Comparison"

    def _radio(*_args, **_kwargs):
        calls["radio_called"] = True
        return "Prediction"

    patcher.setattr(layout.st, "segmented_control", _segmented_control)
    patcher.setattr(layout.st, "radio", _radio)
    patcher.setattr(layout.st, "expander", lambda *args, **kwargs: _Context())
    patcher.setattr(layout.st, "checkbox", lambda *args, **kwargs: True)
    patcher.setattr(
        layout.st,
        "markdown",
        lambda body, **_kwargs: calls["markdown"].append(body),
    )

    page, enable_logging = layout.render_sidebar()

    assert page == "Team Comparison"
    assert enable_logging is True
    assert calls["segmented_label"] == "Navigation"
    assert calls["segmented_options"] == layout.NAVIGATION_PAGES
    assert calls["radio_called"] is False
    assert any("Model Version" in text for text in calls["markdown"])


def test_navigation_pages_match_dashboard_order():
    assert layout.NAVIGATION_PAGES == [
        "Prediction",
        "Team Comparison",
        "Model & Learning",
        "Contact",
    ]


def test_custom_css_keeps_streamlit_spinner_visible():
    assert '[data-testid="stSpinner"] { display: none !important; }' not in layout._CUSTOM_CSS
