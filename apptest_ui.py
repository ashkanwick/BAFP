"""
Regression harness for app.py -- the UI contract the lazy-load rewrite had to hold.

The rewrite introduced three real bugs: a session_state ordering violation, a
ModuleNotFoundError for `src` under AppTest, and an f-string that printed the
literal Path(predictor.model_name).name. This harness is what caught them and
what stops them returning. Layer 1 exercises the pure helpers via `import app`
(safe now that top-level main() is guarded by `if __name__ == "__main__"`);
Layer 2 drives the real element tree through streamlit.testing.v1.AppTest, which
runs app.py the way `streamlit run` does (it sets __name__ to "__main__" --
verified empirically). Run `python apptest_ui.py`; exits non-zero on failure.
No model load, no network -- prediction is asserted structurally.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# `streamlit run` and the AppTest loader put this dir on sys.path; a plain
# `python apptest_ui.py` does not -- without this, `import app` (and its
# `from src.visualize import ...`) raises ModuleNotFoundError.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_FAILURES: list[str] = []
_PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion. Non-fatal: the harness runs to completion."""
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  \u2713 {name}")
    else:
        _FAILURES.append(f"{name}{(' -- ' + detail) if detail else ''}")
        print(f"  \u2717 {name}{(' -- ' + detail) if detail else ''}")


# ---- Layer 1: pure helpers -------------------------------------------------
def test_pure_helpers(app) -> None:
    print("[1] pure helpers")

    # fa() escapes HTML so a pasted comment cannot inject markup.
    escaped = app.fa('<img src=x onerror="alert(1)"> & <b>')
    check("fa escapes angle brackets", "<img" not in escaped and "&lt;img" in escaped)
    check("fa escapes ampersand", "&amp;" in escaped and " & " not in escaped)

    # ltr() wraps with bidi isolates (U+2066 ... U+2069) so a timestamp inside
    # RTL prose is not split around the colon.
    isolated = app.ltr("2026-09-28 02:54")
    check("ltr adds LRI/PDI isolates", isolated.startswith("\u2066") and isolated.endswith("\u2069"))
    check("ltr preserves inner text", isolated[1:-1] == "2026-09-28 02:54")

    # _probability_chart is a horizontal bar figure, x-range padded above max.
    figure = app._probability_chart({"positive": 0.7, "negative": 0.2, "neutral": 0.1})
    check("probability chart is horizontal", figure.data[0].orientation == "h")
    check("probability chart keeps 3 classes", len(figure.data[0].y) == 3)
    check("probability chart pads x-axis", figure.layout.xaxis.range[1] > 0.7)


# ---- Layer 2: AppTest DOM + session state ----------------------------------
def test_apptest_dom() -> None:
    print("[2] AppTest DOM + session state")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    at.run()
    check("landing page renders with no exceptions", not at.exception,
          "; ".join(e.value for e in at.exception) if at.exception else "")

    buttons = at.button
    predict_buttons = [b for b in buttons if "پیش‌بینی کن" in b.label]
    check("predict button present", len(predict_buttons) == 1)
    if predict_buttons:
        # disabled while empty -> get_predictor (cache_resource) never runs on render.
        check("predict button disabled while text empty", predict_buttons[0].disabled)

    # Example pills fill the widget-owned text area via on_click, never by
    # assigning the key after instantiation -- the ordering bug the harness caught.
    # (b.key is None for buttons Streamlit auto-keys, so guard before .startswith.)
    example_buttons = [b for b in buttons if (b.key or "").startswith("example_")]
    check("example pills rendered", len(example_buttons) == _EXAMPLES_LEN,
          f"expected {_EXAMPLES_LEN}, saw {len(example_buttons)}")
    if example_buttons:
        example_buttons[0].click().run()
        check("firing an example raises no exception", not at.exception,
              "; ".join(e.value for e in at.exception) if at.exception else "")
        text_area = at.text_area[0] if at.text_area else None
        check("example fills user_text", bool(text_area and text_area.value.strip()))



# ---- Layer 3: metadata source ----------------------------------------------
def test_metadata_source() -> None:
    print("[3] checkpoint metadata source")

    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    check("app.py reads training_meta.json", "training_meta.json" in source)
    check("no literal Path(predictor.model_name) in source",
          "Path(predictor.model_name)" not in source)

    meta_path = PROJECT_ROOT / "models" / "parsbert-sentiment-3class" / "training_meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # _model_status does Path(str(meta["base_model"])).name -- a bare id
        # (bert-fa-base-uncased), never a stray "Path(...)" wrapper.
        base_name = Path(str(meta.get("base_model", ""))).name
        check("base_model name is a bare id",
              bool(base_name) and "Path(" not in base_name and "\\" not in base_name,
              repr(base_name))
    else:
        check("training_meta.json present (base checkpoint)", False, f"{meta_path} not found")


_EXAMPLES_LEN = 3  # app.EXAMPLES has three tones; re-synced from the module in main().


def main() -> int:
    print("=" * 68)
    print("app.py regression harness")
    print("=" * 68)

    # Import is safe only because main() is guarded in app.py. A future edit that
    # un-guards it would launch the UI here instead of failing this assertion.
    import app  # noqa: E402  (late import, after sys.path setup)

    global _EXAMPLES_LEN
    _EXAMPLES_LEN = len(app.EXAMPLES)

    test_pure_helpers(app)
    test_apptest_dom()
    test_metadata_source()

    print("-" * 68)
    print(f"{_PASSED} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        print("\nFailures:")
        for failure in _FAILURES:
            print(f"  - {failure}")
        return 1
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

