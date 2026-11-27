"""Presentation pieces for the demo: styles, determination slips and the consensus line.

A determination slip is the small label botanists attach to a herbarium sheet to
record who identified a specimen and as what. Here each checkpoint attaches one
to the uploaded photo, so disagreement between models is visible at a glance.
"""

from __future__ import annotations

import html
from collections import Counter

# Categorical slots 1 and 2 of the validated chart palette, stepped per theme.
# Keep in sync with chartCategoricalColors in .streamlit/config.toml.
ARCH_COLORS = {
    "light": {"ResNet18": "#257de0", "ViT-Small": "#f47230"},
    "dark": {"ResNet18": "#338ced", "ViT-Small": "#e76624"},
}

CSS = """
<style>
:root {
  --pd-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --pd-display: "Bricolage Grotesque", system-ui, -apple-system, "Segoe UI", sans-serif;
}
.pd-eyebrow {
  font-family: var(--pd-mono); font-size: 0.75rem; letter-spacing: 0.08em;
  text-transform: uppercase; opacity: 0.72; margin: 0 0 0.25rem;
}
.pd-lede { font-size: 1.05rem; max-width: 46rem; opacity: 0.86; margin: 0.25rem 0 0.5rem; }
.pd-slip {
  position: relative; border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
  border-top: 3px solid var(--pd-arch); border-radius: 3px; padding: 0.7rem 0.85rem 0.75rem;
  background: color-mix(in srgb, currentColor 3%, transparent);
}
.pd-slip__by {
  font-family: var(--pd-mono); font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase;
  display: flex; align-items: center; gap: 0.4rem; opacity: 0.8; margin: 0;
}
.pd-slip__dot { width: 0.55rem; height: 0.55rem; border-radius: 50%; background: var(--pd-arch); flex: none; }
.pd-slip__name {
  font-family: var(--pd-display); font-weight: 700; font-size: 1.35rem; line-height: 1.15;
  margin: 0.45rem 0 0.55rem; letter-spacing: -0.01em;
}
.pd-slip--featured .pd-slip__name { font-size: clamp(1.8rem, 3.4vw, 2.6rem); }
.pd-slip__bar {
  height: 6px; border-radius: 3px; background: color-mix(in srgb, currentColor 12%, transparent);
  overflow: hidden;
}
.pd-slip__bar > span { display: block; height: 100%; border-radius: 3px; background: var(--pd-arch); }
.pd-slip__meta {
  font-family: var(--pd-mono); font-size: 0.72rem; margin: 0.45rem 0 0; opacity: 0.78;
  display: flex; flex-wrap: wrap; gap: 0.2rem 0.9rem;
}
.pd-slip__meta b { font-weight: 500; }
.pd-slip--row { display: flex; flex-direction: column; }
@media (min-width: 640px) { .pd-slip--row { min-height: 12rem; } }  /* align slips side by side */
.pd-slip--row .pd-slip__meta { margin-top: auto; padding-top: 0.45rem; }
.pd-slip--error { --pd-arch: color-mix(in srgb, currentColor 40%, transparent); }
.pd-consensus {
  font-family: var(--pd-display); font-size: 1.15rem; font-weight: 600; line-height: 1.35;
  border-left: 3px solid currentColor; padding: 0.15rem 0 0.15rem 0.85rem; margin: 0.25rem 0 0.5rem;
  max-width: 52rem;
}
.pd-key { font-family: var(--pd-mono); font-size: 0.75rem; opacity: 0.85; display: flex; gap: 1.1rem; flex-wrap: wrap; }
.pd-key span { display: inline-flex; align-items: center; gap: 0.35rem; }
.pd-key i { width: 0.7rem; height: 0.7rem; border-radius: 50%; border: 2px solid var(--pd-arch); display: inline-block; }
.pd-key i.filled { background: var(--pd-arch); }
.pd-key i.line { width: 1.4rem; height: 0; border-radius: 0; border-width: 2px 0 0; }
.pd-key i.dashed { border-top-style: dashed; }
</style>
"""


def _attr(value: str) -> str:
    return html.escape(value, quote=True)


def slip_html(*, model: str, color: str, determination: str, confidence: float,
              meta: list[str], featured: bool = False) -> str:
    """One determination slip. All text is escaped; ``color`` must be a hex value."""
    if not (color.startswith("#") and len(color) in (4, 7)):
        raise ValueError(f"Expected a hex colour, got {color!r}")
    width = max(0.0, min(1.0, confidence)) * 100
    items = "".join(f"<span>{html.escape(item)}</span>" for item in meta)
    kind = "pd-slip pd-slip--featured" if featured else "pd-slip pd-slip--row"
    return (
        f'<div class="{kind}" style="--pd-arch:{color}" role="group" '
        f'aria-label="{_attr(model)} determination: {_attr(determination)}">'
        f'<p class="pd-slip__by"><span class="pd-slip__dot"></span>Det. {html.escape(model)}</p>'
        f'<p class="pd-slip__name">{html.escape(determination)}</p>'
        f'<div class="pd-slip__bar" role="img" aria-label="{confidence:.0%} model probability">'
        f'<span style="width:{width:.1f}%"></span></div>'
        f'<p class="pd-slip__meta"><b>{confidence:.0%}</b>{items}</p></div>'
    )


def error_slip_html(model: str, message: str) -> str:
    return (
        f'<div class="pd-slip pd-slip--row pd-slip--error" role="group" aria-label="{_attr(model)} unavailable">'
        f'<p class="pd-slip__by"><span class="pd-slip__dot"></span>Det. {html.escape(model)}</p>'
        f'<p class="pd-slip__name">Not loaded</p><p class="pd-slip__meta">{html.escape(message)}</p></div>'
    )


def key_html(color: str, items: list[tuple[str, str]]) -> str:
    """A text legend: items are (glyph, label) with glyph in filled/hollow/line/dashed."""
    glyphs = {"filled": '<i class="filled"></i>', "hollow": "<i></i>",
              "line": '<i class="line"></i>', "dashed": '<i class="line dashed"></i>'}
    spans = "".join(f"<span>{glyphs[g]}{html.escape(label)}</span>" for g, label in items)
    return f'<div class="pd-key" style="--pd-arch:{color}">{spans}</div>'


def consensus(determinations: list[tuple[str | None, str]]) -> str:
    """Summarise (training regime, class) pairs from every checkpoint in one sentence.

    Training is "Lab only", "Field-adapted" or None for unknown checkpoints. The
    sentence reports agreement or the split between regimes; it never says which
    model is right.
    """
    if not determinations:
        return ""
    classes = [name for _, name in determinations]
    if len(set(classes)) == 1:
        count = len(classes)
        return (f"All {count} checkpoints agree: {classes[0]}." if count > 1
                else f"One checkpoint: {classes[0]}.")

    def lean(regime: str) -> str | None:
        votes = Counter(name for training, name in determinations if training == regime)
        if not votes:
            return None
        (top, top_count), *rest = votes.most_common()
        return None if rest and rest[0][1] == top_count else top

    lab, adapted = lean("Lab only"), lean("Field-adapted")
    if lab and adapted and lab != adapted:
        return (f"Lab-only checkpoints say {lab}; field-adapted checkpoints say {adapted}. "
                "This is the lab-to-field disagreement the study measures.")
    counts = Counter(classes).most_common()
    listed = ", ".join(f"{name} ({n})" for name, n in counts)
    return f"The checkpoints disagree: {listed}."
