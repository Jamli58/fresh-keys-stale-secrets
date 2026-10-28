"""Figure 1 (threat-model pipeline), vector style-2 (editorial hand-drawn) redraw: SVG -> PDF (rsvg-convert + gs).

Layout, regions, I1 colour tokens, trust boundary and the five arrows follow the accepted AI render
(figure1_academic_figure_workflow/style2_editorial_hand_drawn/, an academic-figure-workflow render) and its FigureSpec; every element is
drawn as vector geometry and every label as real text, so the PDF stays sharp at any zoom.
Headings: Chalkboard SE Bold (rounded hand-lettered); labels: Avenir Next Medium. Each label is measured with the
actual font and shrunk if it would overflow its box. TIFS version: printed at 0.9 x 3.5 in column width, so
labels are 40 units (about 6.3 pt) instead of 33, and the prompt shows the croissant example of Section I.

    uv run --no-project --with pillow python gen_fig1_pipeline_style2_vector.py   (needs rsvg-convert and gs)
"""

import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
NAME = "fig1_intro_threat_model"
OUT = HERE  # the SVG and the PDF are written to figures/
W, H = 1448, 1086
PRINT_IN = 0.9 * 3.5

CHALK = ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 2, "Chalkboard SE", 700)
AVENIR = ("/System/Library/Fonts/Avenir Next.ttc", 5, "Avenir Next", 500)
AVENIR_IT = ("/System/Library/Fonts/Avenir Next.ttc", 6, "Avenir Next", 500)

INK = "#24323D"
ARROW = "#334155"
I1 = {  # soft_fill, dark_outline, title_text, icon_accent (palettes.md, I1 Illustrated Zones)
    "green": ("#F3FBF0", "#3B7D23", "#2F681D", "#4E8D36"),
    "blue": ("#EDF4FB", "#194166", "#163E64", "#2E6B9E"),
    "peach": ("#FBE3D6", "#A94417", "#9E3F13", "#C65A22"),
    "purple": ("#F5ECF5", "#77206E", "#65185E", "#8B3B83"),
    "gold": ("#FBF1D1", "#856B1B", "#66500F", "#9B7B1C"),
}

parts: list[str] = []
sizes: dict[str, float] = {}


def width(text: str, face, size: float) -> float:
    path, index, _, _ = face
    return ImageFont.truetype(path, round(size), index=index).getlength(text)


def fit(text: str, face, size: float, max_w: float) -> float:
    while width(text, face, size) > max_w and size > 12:
        size -= 1
    return size


def text(x, y, s, face=AVENIR, size=34, fill=INK, anchor="start", max_w=None, italic=False):
    if max_w is not None:
        size = fit(s, face, size, max_w)
    sizes[s] = size
    family, weight = face[2], face[3]
    style = ' font-style="italic"' if italic else ""
    parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-weight="{weight}" font-size="{size}"{style} '
                 f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def rich(x, y, runs, size=34, fill=INK, anchor="start"):
    """runs: list of (string, kind) with kind in {'', 'sub', 'it'}; one <text> with tspans."""
    spans, lowered = [], False
    for s, kind in runs:
        back = f' dy="{-size * 0.22:.1f}"' if lowered and kind != "sub" else ""
        lowered = kind == "sub"
        if kind == "sub":
            spans.append(f'<tspan font-size="{size * 0.68:.1f}" dy="{size * 0.22:.1f}">{escape(s)}</tspan>')
        elif kind == "it":
            spans.append(f'<tspan font-style="italic"{back}>{escape(s)}</tspan>')
        else:
            spans.append(f"<tspan{back}>{escape(s)}</tspan>")
    parts.append(f'<text x="{x}" y="{y}" font-family="Avenir Next" font-weight="500" font-size="{size}" '
                 f'fill="{fill}" text-anchor="{anchor}">{"".join(spans)}</text>')


def rect(x, y, w, h, rx, fill, stroke, sw=3.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}/>')


def path(d, fill="none", stroke=INK, sw=4, extra=""):
    parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{extra}/>')


def arrow(x1, y1, x2, y2):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ARROW}" stroke-width="4.5" '
                 f'stroke-linecap="round" marker-end="url(#head)"/>')


def document(x, y, w, h, fold=34, fill="#FFFFFF"):
    path(f"M{x + 14},{y} H{x + w - fold} L{x + w},{y + fold} V{y + h - 14} Q{x + w},{y + h} {x + w - 14},{y + h} "
         f"H{x + 14} Q{x},{y + h} {x},{y + h - 14} V{y + 14} Q{x},{y} {x + 14},{y} Z", fill=fill, sw=4)
    path(f"M{x + w - fold},{y} V{y + fold} H{x + w}", sw=3.5)


def bubble(x, y, w, h, fill="#FFFFFF", tail=True):
    tail_d = (f"H{x + 70} L{x + 44},{y + h + 30} L{x + 48},{y + h}" if tail else "")
    path(f"M{x + 14},{y} H{x + w - 30} L{x + w},{y + 30} V{y + h - 14} Q{x + w},{y + h} {x + w - 14},{y + h} "
         f"{tail_d} H{x + 14} Q{x},{y + h} {x},{y + h - 14} V{y + 14} Q{x},{y} {x + 14},{y} Z", fill=fill, sw=4)
    path(f"M{x + w - 30},{y} V{y + 30} H{x + w}", sw=3.5)


def padlock(cx, cy, accent):
    rect(cx - 17, cy - 6, 34, 28, 6, accent, INK, sw=3.5)
    path(f"M{cx - 10},{cy - 6} V{cy - 16} A10,10 0 0 1 {cx + 10},{cy - 16} V{cy - 6}", sw=3.5)
    parts.append(f'<circle cx="{cx}" cy="{cy + 6}" r="3.5" fill="{INK}"/>')


def mask(cx, cy, s=1.0, accent=None):
    d = (f"M{cx - 62 * s},{cy - 24 * s} Q{cx - 30 * s},{cy - 38 * s} {cx},{cy - 18 * s} "
         f"Q{cx + 30 * s},{cy - 38 * s} {cx + 62 * s},{cy - 24 * s} Q{cx + 64 * s},{cy + 18 * s} {cx + 30 * s},{cy + 26 * s} "
         f"Q{cx + 10 * s},{cy + 28 * s} {cx},{cy + 12 * s} Q{cx - 10 * s},{cy + 28 * s} {cx - 30 * s},{cy + 26 * s} "
         f"Q{cx - 64 * s},{cy + 18 * s} {cx - 62 * s},{cy - 24 * s} Z")
    path(d, fill="#FFFFFF", sw=4.5)
    for side in (-1, 1):
        ex = cx + side * 31 * s
        path(f"M{ex - 15 * s},{cy - 3 * s} Q{ex},{cy - 15 * s} {ex + 15 * s},{cy - 3 * s} "
             f"Q{ex},{cy + 8 * s} {ex - 15 * s},{cy - 3 * s} Z", fill=accent or INK, sw=3)


def person(cx, cy):
    parts.append(f'<circle cx="{cx}" cy="{cy - 9}" r="9" fill="none" stroke="{INK}" stroke-width="3.5"/>')
    path(f"M{cx - 16},{cy + 17} Q{cx - 16},{cy + 2} {cx},{cy + 2} Q{cx + 16},{cy + 2} {cx + 16},{cy + 17}", sw=3.5)


def magnifier(cx, cy, r=28):
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{INK}" stroke-width="5"/>')
    path(f"M{cx - r * 0.72},{cy + r * 0.72} L{cx - r * 1.55},{cy + r * 1.55}", sw=8)


def bell(x0, x1, base, peak, accent, band=(0.5, 0.62)):
    xm = (x0 + x1) / 2
    d = (f"M{x0},{base} C{x0 + (xm - x0) * 0.55},{base} {xm - (xm - x0) * 0.42},{peak} {xm},{peak} "
         f"C{xm + (x1 - xm) * 0.42},{peak} {x1 - (x1 - xm) * 0.55},{base} {x1},{base}")
    bx0, bx1 = x0 + (x1 - x0) * band[0], x0 + (x1 - x0) * band[1]
    parts.append(f'<clipPath id="under"><path d="{d} Z"/></clipPath>')
    parts.append(f'<rect x="{bx0}" y="{peak}" width="{bx1 - bx0}" height="{base - peak}" fill="{accent}" '
                 f'fill-opacity="0.28" clip-path="url(#under)"/>')
    path(d, stroke=accent, sw=4.5)
    path(f"M{x0 - 8},{base} H{x1 + 8}", stroke=INK, sw=3.5)


def build() -> str:
    g, b, p, pu, go = I1["green"], I1["blue"], I1["peach"], I1["purple"], I1["gold"]
    # regions and trust boundary
    rect(16, 26, 540, 1034, 46, g[0], g[1], sw=4)
    rect(592, 26, 840, 1034, 46, b[0], b[1], sw=4)
    path("M574,30 V1056", stroke=INK, sw=3.5, extra=' stroke-dasharray="12,10"')
    text(286, 110, "Firm (trusted)", CHALK, 60, g[2], "middle", max_w=500)
    text(1012, 110, "Cloud (honest-but-curious)", CHALK, 60, b[2], "middle", max_w=800)

    # SME prompt
    document(50, 150, 470, 300)
    text(78, 204, "Firm prompt", CHALK, 40, INK, max_w=320)
    path("M80,232 H300", sw=3)
    text(84, 290, "croissant price  3.20", AVENIR, 40, max_w=420)
    text(84, 338, "unit cost  1.10", AVENIR, 40, max_w=420)
    path("M84,364 H420", sw=3)
    padlock(98, 408, g[3])
    text(128, 420, "secret: margin, cost", AVENIR, 40, max_w=372)
    arrow(250, 452, 250, 506)

    # Numeric disguise (one real subcard)
    rect(36, 514, 508, 300, 22, p[0], p[1], sw=4)
    text(60, 566, "Numeric disguise", CHALK, 42, p[2], max_w=330)
    mask(462, 552, s=0.62, accent=p[3])
    lines = [[("D1  shared scaling ×", ""), ("k", "it")],
             [("D1′  per-field ×", ""), ("k", "it"), ("f", "sub")],
             [("D2  order-preserving ", ""), ("h", "it")],
             [("D3o  Prεεmpt noise", "")]]
    for i, runs in enumerate(lines):
        rich(62, 626 + i * 50, runs, size=40)
    sizes["D2  order-preserving h"] = 40
    for line in ("D1  shared scaling ×k", "D1′  per-field ×kf", "D2  order-preserving h", "D3o  Prεεmpt noise"):
        assert width(line, AVENIR, 40) < 470, f"disguise line overflows: {line}"

    # secret regime toggle
    arrow(250, 866, 250, 822)
    rect(36, 872, 470, 156, 18, "#FFFFFF", INK, sw=4)
    parts.append(f'<circle cx="84" cy="924" r="17" fill="#FFFFFF" stroke="{INK}" stroke-width="4"/>')
    parts.append(f'<circle cx="84" cy="924" r="9.5" fill="{g[3]}"/>')
    parts.append(f'<circle cx="84" cy="984" r="17" fill="#FFFFFF" stroke="{INK}" stroke-width="4"/>')
    text(118, 937, "fresh secret per query", AVENIR, 40, max_w=370)
    text(118, 997, "one reused secret", AVENIR, 40, max_w=370)

    # Cloud LLM provider (hero): stack of linked query bubbles, thread to one firm account
    text(650, 214, "Cloud LLM provider", CHALK, 44, INK, max_w=420)
    for k in range(5):
        x, y = 646 + 17 * k, 244 + 16 * k
        bubble(x, y, 262, 176, tail=(k == 4))
    x, y = 646 + 68, 244 + 64
    for j, ln in enumerate((200, 200, 170, 190)):
        path(f"M{x + 30},{y + 42 + 30 * j} H{x + 30 + ln}", sw=3.5)
    path("M832,486 C826,540 820,575 820,606", sw=4)
    path("M862,486 C866,540 872,575 872,606", sw=4)
    rect(778, 606, 136, 62, 14, "#D9E7F5", INK, sw=4)
    person(846, 637)
    text(846, 726, "links queries by account", AVENIR, 40, INK, "middle", max_w=400)

    # disguised queries cross the trust boundary
    arrow(544, 637, 770, 637)
    rich(588, 586, [("q", "it"), ("1", "sub"), (", ", ""), ("q", "it"), ("2", "sub"), (", …, ", ""), ("q", "it"), ("n", "sub")],
         size=34)

    # Attacker (A1+) and result
    arrow(976, 440, 1060, 532)
    text(1024, 462, "linked", AVENIR, 40, INK)
    rect(1060, 540, 352, 316, 24, pu[0], pu[1], sw=4)
    text(1236, 592, "Attacker (A1+)", CHALK, 42, pu[2], "middle", max_w=320)
    magnifier(1124, 650, r=27)
    bell(1182, 1392, 700, 616, pu[3])
    text(1236, 748, "Bayesian solver", AVENIR, 40, INK, "middle", max_w=330)
    text(1236, 792, "+ sector prior", AVENIR, 40, INK, "middle", max_w=330)
    text(1236, 836, "+ LLM reads prose", AVENIR, 40, INK, "middle", max_w=340)
    arrow(1236, 858, 1236, 916)
    rect(1048, 924, 376, 92, 18, go[0], go[1], sw=4)
    text(1236, 962, "recovered margin,", AVENIR, 40, go[2], "middle", max_w=350)
    text(1236, 1004, "unit cost", AVENIR, 40, go[2], "middle", max_w=350)

    defs = (f'<defs><marker id="head" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="5.2" markerHeight="5.2" '
            f'orient="auto-start-reverse"><path d="M1,1 L11,6 L1,11 Z" fill="{ARROW}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>{defs}' + "".join(parts) + "</svg>")


def main() -> None:
    svg = HERE / f"{NAME}.svg"
    svg.write_text(build())
    pdf_w_pt = PRINT_IN * 72
    subprocess.run(["rsvg-convert", "-f", "pdf", "-w", f"{pdf_w_pt:.1f}pt", "-o", str(OUT / f"{NAME}.pdf"), str(svg)],
                   check=True)
    raw = OUT / f"{NAME}_text.pdf"
    (OUT / f"{NAME}.pdf").rename(raw)
    subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dNoOutputFonts", "-sDEVICE=pdfwrite",
                    f"-sOutputFile={OUT / f'{NAME}.pdf'}", str(raw)], check=True)
    raw.unlink()
    smallest = min(sizes.values())
    print(f"wrote {NAME}.svg/.pdf; smallest label {smallest:.0f} units = "
          f"{smallest / W * PRINT_IN * 72:.1f} pt at {PRINT_IN:.2f} in print width")


if __name__ == "__main__":
    main()
