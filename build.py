#!/usr/bin/env python3
"""Render book.md with pandoc into one self-contained HTML page."""
import pathlib, subprocess

here = pathlib.Path(__file__).parent


def rect(x, y, w, h, cls, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="{cls}" {extra}/>'


def text(x, y, s, cls="dg-text", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def dispatch_svg():
    out = ['<svg viewBox="0 0 700 300" role="img" aria-label="A screen covered by a grid of 16 by 16 workgroups; one workgroup is enlarged to show its threads.">']
    ox, oy, c = 20, 40, 44
    cols, rows = 7, 4
    sw, sh = 6.45 * c, 3.55 * c
    out.append(text(ox, 24, "the dispatch: 7 × 4 workgroups", "dg-label"))
    # screen
    out.append(rect(ox, oy, sw, sh, "dg-accent-soft", 'fill-opacity="0.55"'))
    for j in range(rows):
        for i in range(cols):
            hl = (i, j) == (3, 1)
            out.append(rect(ox + i * c, oy + j * c, c, c, "dg-muted" if not hl else "dg-line",
                            f'fill="{"var(--accent)" if hl else "none"}" fill-opacity="{0.35 if hl else 0}" stroke-width="{2 if hl else 1}"'))
    out.append(rect(ox, oy, sw, sh, "dg-line", 'fill="none" stroke-width="2"'))
    out.append(text(ox + sw - 4, oy + sh - 8, "screen", "dg-text", "end"))
    out.append(text(ox, oy + rows * c + 22, "edge workgroups hang past the screen,", "dg-label"))
    out.append(text(ox, oy + rows * c + 38, "which is why every shader starts with a bounds check", "dg-label"))
    # zoom
    zx, zy, zc, n = 420, 40, 26, 8
    hx, hy = ox + 3 * c, oy + 1 * c
    out.append(f'<line x1="{hx + c}" y1="{hy}" x2="{zx}" y2="{zy}" class="dg-muted" stroke-dasharray="3 3"/>')
    out.append(f'<line x1="{hx + c}" y1="{hy + c}" x2="{zx}" y2="{zy + n * zc}" class="dg-muted" stroke-dasharray="3 3"/>')
    for j in range(n):
        for i in range(n):
            hl = (i, j) == (2, 1)
            out.append(rect(zx + i * zc, zy + j * zc, zc, zc, "dg-muted",
                            f'fill="{"var(--accent)" if hl else "var(--panel)"}" stroke-width="1"'))
    out.append(rect(zx, zy, n * zc, n * zc, "dg-line", 'fill="none" stroke-width="2"'))
    out.append(text(zx, 24, "workgroup (3, 1): 16 × 16 threads, 8 × 8 drawn", "dg-label"))
    out.append(text(zx, zy + n * zc + 22, "thread with local_invocation_id (2, 1)", "dg-label"))
    out.append(text(20, 286, "global_invocation_id = workgroup_id × 16 + local_invocation_id  →  (3, 1) × 16 + (2, 1) = (50, 17)", "dg-text"))
    out.append("</svg>")
    return "".join(out)


def tree_svg():
    vals = [[3, 1, 4, 1, 5, 9, 2, 6], [8, 10, 6, 7], [14, 17], [31]]
    labels = ["load", "stride 4", "stride 2", "stride 1"]
    ox, oy, bw, bh, gap, rowh = 120, 20, 58, 30, 8, 66
    out = ['<svg viewBox="0 0 700 290" role="img" aria-label="Tree reduction of eight values: each step, the first half of the threads add in the value one stride away.">']
    for r, row in enumerate(vals):
        y = oy + r * rowh
        out.append(text(ox - 16, y + bh / 2 + 4, labels[r], "dg-label", "end"))
        for i in range(8):
            x = ox + i * (bw + gap)
            active = i < len(row)
            out.append(rect(x, y, bw, bh, "dg-muted" if not active else "dg-line",
                            f'rx="4" fill="{"var(--accent-soft)" if active else "none"}" stroke-width="1"'))
            if active:
                out.append(text(x + bw / 2, y + bh / 2 + 4, row[i], "dg-text", "middle"))
        if r > 0:
            stride = len(row)
            py = oy + (r - 1) * rowh + bh
            for i in range(stride):
                x = ox + i * (bw + gap) + bw / 2
                x2 = ox + (i + stride) * (bw + gap) + bw / 2
                out.append(f'<line x1="{x}" y1="{py}" x2="{x}" y2="{y}" class="dg-line" stroke-width="1.2"/>')
                out.append(f'<line x1="{x2}" y1="{py}" x2="{x + 6}" y2="{y}" class="dg-line" stroke-width="1.2" stroke-opacity="0.55"/>')
            by = y - (rowh - bh) / 2 + 1
            out.append(f'<line x1="{ox - 8}" y1="{by - 12}" x2="{ox + 8 * (bw + gap)}" y2="{by - 12}" stroke="var(--accent)" stroke-dasharray="5 4" stroke-width="1"/>')
    out.append(text(ox + 8 * (bw + gap) - gap, oy + rowh - 30 + 4, "workgroupBarrier()", "dg-label", "end"))
    out.append("</svg>")
    return "".join(out)


html = subprocess.run(
    ["pandoc", "book-1.md", "book-2.md","--from", "markdown", "--to", "html5",
     "--template", "template.html", "--toc", "--toc-depth=2",
     "--mathml", "--no-highlight", "--metadata", "pagetitle=Thread by Thread"],
    cwd=here, check=True, capture_output=True, text=True).stdout
html = html.replace("<!--SVG:dispatch-->", dispatch_svg()).replace("<!--SVG:tree-->", tree_svg())
# Fonts are self-hosted (fonts/, deployed next to the page) so readers' browsers never contact Google.
html = html.replace("/*FONTS_CSS*/", (here / "fonts" / "fonts.css").read_text())
html = html.replace("/*BOOK_CSS*/", (here / "book.css").read_text())
html = html.replace("/*TOY_JS*/", (here / "toy.js").read_text())
(here / "thread-by-thread.html").write_text(html)
print(f"wrote thread-by-thread.html ({len(html)//1024} KB)")
