# Thread by Thread

A hands-on introduction to GPU programming with WebGPU compute shaders, in 17 live exercises.

## Build

Requires Python 3 and pandoc (tested with 3.1).

    python3 build.py

This renders `book-1.md` and `book-2.md` through `template.html`, inlines `book.css` and `toy.js`,
and writes one self-contained page: `thread-by-thread.html`.

## View

WebGPU only works on secure pages. Open the HTML file directly, or serve the folder and use
`http://localhost:PORT` (not a LAN IP address):

    python3 -m http.server 8000   # then open http://localhost:8000/thread-by-thread.html

## Files

- `book-1.md`, `book-2.md`: the text. Each exercise is a fenced div:

      ::: {.toy #ex01 data-label="Exercise 1" data-title="Gradient"}
      ```wgsl
      starter code
      ```
      ```{.wgsl .solution}
      solution code
      ```
      :::

  Diagrams are inserted at `<!--SVG:dispatch-->` and `<!--SVG:tree-->` by `build.py`.
- `toy.js`: the runtime and editor. It turns each `.toy` div into a live console and implements the
  compute.toys-style prelude (`time`, `mouse`, `screen`, `passLoad`/`passStore`, `#storage`,
  `#workgroup_count`, `#dispatch_count`).
- `book.css`: styles.
- `template.html`: the pandoc template, including the header shader.
- `test.py`: compiles and runs every starter and solution in headless Chromium, and saves the rendered
  solutions as PNGs. It needs `pip install playwright` and `python3 -m playwright install chromium`.

      python3 test.py shots/
