
# [Start here]{.part} How this page works

Each exercise on this page is a small live GPU program. The picture on the left is drawn by the code on the right, and the code runs on your graphics card, once per pixel, every frame.

- **Edit the code** and it recompiles about half a second after you stop typing. Press **Ctrl+Enter** (**⌘+Enter** on a Mac) to compile right away.
- **Errors** show up under the editor with their line number. Click one to jump to that line. When the code doesn't compile, the last working version keeps running.
- **Show solution** swaps in a worked answer. **Back to my code** brings yours back. Try each exercise before you peek.
- **Your edits are saved in this browser**, so you can close the tab and come back later. **Reset code** restores the starting code, and Ctrl+Z undoes that.
- **Restart** sets the clock and frame counter back to zero and clears any buffers. Stateful exercises need this after you change how they initialize.

The strip under each exercise lists what the GPU runs every frame: each **entry point**, in order, with how many workgroups it launches. You'll learn what that means in a minute.

The playground follows the conventions of [compute.toys](https://compute.toys), a public WebGPU shader playground. Almost everything you write here can be pasted into compute.toys and will run unchanged. When you outgrow this page, that's the place to go next.

## Why this feels hard

If shaders feel harder than they look, that's normal. The syntax of WGSL, the WebGPU Shading Language, looks like a small, stripped-down Rust or C. But the language isn't the hard part. The execution model is:

- **You write the program for one thread out of thousands.** Your code runs once per pixel, in parallel, and each copy can't see what the others are doing. Instead of "loop over the image," you have to ask "what does *this one* pixel compute?"
- **Debugging is primitive.** There's no `print` and no debugger that steps through code. You debug by turning intermediate values into colors and looking at them.
- **The math is assumed.** Coordinate spaces, vectors and interpolation are everywhere.
- **Compute shaders add parallel programming.** Workgroups, shared memory, barriers and race conditions all show up by the end of this page.

People who say shaders are easy have usually internalized the parallel mental model and forgotten how long it took. The exercises below build that model one idea at a time.

# [Part 0]{.part} The mental model

## One thread per pixel

A **compute shader** is a function that the GPU runs many times at once. Each running copy is called an **invocation** (or a thread). You don't call it in a loop. Instead you **dispatch** it: you tell the GPU how many copies to launch, and it runs them in parallel.

Invocations are launched in fixed-size batches called **workgroups**. The line

```wgsl
@compute @workgroup_size(16, 16)
```

says each workgroup is a 16 × 16 square of 256 threads. To cover a 640 × 480 screen, the playground launches 40 × 30 workgroups. Each thread learns its position through **built-in inputs**:

| Built-in | Type | Meaning |
|---|---|---|
| `global_invocation_id` | `vec3u` | This thread's position in the whole dispatch. For a screen, it's the pixel. |
| `local_invocation_id` | `vec3u` | Position inside its own workgroup, from `(0, 0)` to `(15, 15)`. |
| `workgroup_id` | `vec3u` | Which workgroup this thread belongs to. |

They're related by one formula:

::: {.diagram .wide}
<figure class="diagram"><!--SVG:dispatch--><figcaption>Each workgroup covers a 16 × 16 tile of pixels. The global ID is the workgroup's corner plus the thread's offset inside it.</figcaption></figure>
:::

For the first half of this page, only `global_invocation_id` matters: it tells each thread which pixel it owns. The local and workgroup IDs become important in Part VI, when threads in a workgroup start cooperating.

## Anatomy of the starting shader

Here is the program every exercise starts from. It's the same default shader compute.toys gives you. Read through it, then try changing the numbers on the `col` line.

::: {.toy #ex00 data-label="Warm-up" data-title="The default shader"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0., 2., 4.));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

Line by line:

**`fn main_image(@builtin(global_invocation_id) id: vec3u)`** declares the entry point. `@builtin(global_invocation_id)` asks the GPU to fill in `id` with this thread's global position. `vec3u` is a vector of three unsigned 32-bit integers. Only `.x` and `.y` matter for a 2D image.

**`textureDimensions(screen)`** returns the size of the output image in pixels. `screen` is a **storage texture** the playground provides. Each thread writes its pixel into it with `textureStore`.

**The bounds check** exists because the dispatch is rounded up to whole workgroups. A screen 650 pixels wide needs 41 workgroups of 16, which is 656 threads per row, so six threads per row have no pixel. Returning early keeps them from writing outside the image.

**`fragCoord`** rebuilds the pixel coordinate from the thread ID. The name comes from Shadertoy's GLSL, where fragment shaders get `fragCoord` for free. Three things happen in that line:

1. `id.x` and `id.y` say which pixel this thread owns.
2. The `.5` moves to the **center** of the pixel. Pixel 0 covers the interval from 0 to 1, so its center is 0.5. Fragment shaders use the same convention, so this avoids half-pixel errors later.
3. `screen_size.y - id.y` **flips y**. In textures, row 0 is at the top and y grows downward. In math and in Shadertoy, y grows upward. After the flip, the top row gets `fragCoord.y = H - 0.5` and the bottom row gets `0.5`.

The `f32(...)` conversions are needed because WGSL never mixes integers and floats silently. You always convert explicitly.

**`uv`** divides by the screen size, turning pixel units into the range 0 to 1, with (0, 0) at the bottom left. `uv` means the same thing at any resolution, which is why most effects are written in terms of it.

**`col`** is the color. `uv.xyx` is a **swizzle**: it builds a `vec3f` from the components `x`, `y`, `x`. Adding a scalar to a vector adds it to every component.

**`pow(col, vec3f(2.2))`** converts the color into **linear** light. The screen texture stores linear values, and the display converts them back. If you want a value of 0.5 to *look* like middle grey, apply this line. Every exercise keeps it.

**`textureStore(screen, id.xy, vec4f(col, 1.))`** writes the pixel. The fourth component is alpha. Notice that storing uses `id.xy`, the raw texture coordinate with y pointing down, not `fragCoord`.

## What the playground gives you

These names are always available. They mirror the compute.toys prelude.

| Name | What it is |
|---|---|
| `screen` | The output image, `texture_storage_2d<rgba16float, write>`. Write with `textureStore(screen, coord, color)`. |
| `time.elapsed` | Seconds since start (`f32`). |
| `time.frame` | Frame counter (`u32`). It's 0 on the first frame after a restart. |
| `time.delta` | Seconds since the last frame. |
| `mouse.pos` | Mouse position in pixels (`vec2i`), with the origin at the top left like `id`. |
| `mouse.click` | 1 while the button is held on the canvas, else 0. |
| `passLoad`, `passStore` | Read and write scratch images between entry points. See Part V. |
| `#storage name type` | Declares a storage buffer, such as `#storage data array<f32>`. See Part VI. |
| `#workgroup_count name x y z` | Sets how many workgroups an entry point launches. |
| `#dispatch_count name n` | Runs an entry point `n` times in a row each frame. `dispatch.id` tells you which run it is. |

Every `@compute` function in your file is an entry point, and all of them run each frame in the order they appear. If you don't give one a `#workgroup_count`, it gets enough workgroups to cover the screen.

# [Part I]{.part} Coordinates and color

## [1]{.num} Gradient

A **vector constructor** can take a mix of smaller vectors and scalars, as long as the component count adds up. `vec3f(uv, 0.0)` takes the two components of `uv` and appends a zero:

```wgsl
let a = vec3f(uv, 0.0);          // (uv.x, uv.y, 0.0)
let b = vec3f(0.2);              // (0.2, 0.2, 0.2): one value fills every slot
let c = vec4f(a.zy, 1.0, 1.0);   // swizzles work anywhere
```

Since `uv` runs from 0 to 1 across the screen, using it as a color is the simplest way to *see* a coordinate system. This is the first debugging tool you'll use over and over: when you're not sure what a value is doing, draw it.

::: task
Make red increase from left to right and green increase from bottom to top. The bottom-left corner should be black and the top-right corner yellow. If the green is brightest at the bottom, your y is flipped.
:::

::: {.toy #ex01 data-label="Exercise 1" data-title="Gradient"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    // TODO: red from uv.x, green from uv.y, no blue
    var col = vec3f(0.0);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    // uv.x drives red, uv.y drives green
    var col = vec3f(uv, 0.0);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

Try removing the `pow` line. The gradient gets brighter and the midpoint looks washed out. That's the difference between linear light and the way screens encode brightness.

## [2]{.num} Hard split

On a CPU, you'd write "if x is past the middle, white; otherwise black." On a GPU, you usually write it as math instead.

The reason is how GPUs execute. Threads run in lockstep groups (often 32 at a time) that share one instruction stream. When threads in the same group take different sides of an `if`, the group runs *both* sides and masks out the results each thread shouldn't keep. This is called **divergence**. A small `if` costs almost nothing, but the habit of expressing choices as arithmetic keeps code fast and branch-free. WGSL gives you tools for it:

| Function | Result |
|---|---|
| `step(edge, x)` | `0.0` if `x < edge`, else `1.0` |
| `select(f, t, cond)` | `t` if `cond` is true, else `f`. Note the order: false value first. |
| `mix(a, b, t)` | Linear blend `a + (b - a) * t` |
| `clamp(x, lo, hi)` | `x` limited to the range |

::: task
Make the left half of the screen black and the right half white, using `step` instead of `if`. Then try moving the split with the mouse: `mouse.pos.x` is in pixels, so divide by the screen width to get the same 0-to-1 scale as `uv`.
:::

::: {.toy #ex02 data-label="Exercise 2" data-title="Hard split"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    // TODO: 0.0 on the left half, 1.0 on the right half, without `if`
    let v = 0.0;
    var col = vec3f(v);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    // Split at the middle, or at the mouse once it has moved over the canvas
    var edge = 0.5;
    if (mouse.pos.x > 0) { edge = f32(mouse.pos.x) / f32(screen_size.x); }

    let v = step(edge, uv.x);
    var col = vec3f(v);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

## [3]{.num} Checkerboard

Most patterns come from **tiling space**: split the plane into cells, then decide what happens in each one.

- `floor(x)` gives the index of the cell `x` falls in.
- `fract(x)` gives the position inside that cell, from 0 to 1.

Multiply `uv` by 8 and you get coordinates that run from 0 to 8. `floor` of that is a cell index from 0 to 7 on each axis. For a checkerboard, a cell is white when the sum of its x and y index is odd. The `%` operator works on floats in WGSL, so `(cell.x + cell.y) % 2.0` is either 0 or 1.

::: task
Draw an 8 × 8 checkerboard. Then try `fract(uv * 8.0)` as a color to see the coordinates *inside* each cell.
:::

::: {.toy #ex03 data-label="Exercise 3" data-title="Checkerboard"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    // TODO: cell index with floor(), then even/odd with %
    let cell = uv * 8.0;
    var col = vec3f(cell / 8.0, 0.0);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    let cell = floor(uv * 8.0);            // which cell, 0..7 on each axis
    let odd = (cell.x + cell.y) % 2.0;     // 0.0 or 1.0
    var col = vec3f(odd);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

The cells aren't square, because `uv` stretches to fit the screen's shape. The next part fixes that.

# [Part II]{.part} Shapes with distance

## [4]{.num} Circle

Neither `fragCoord` nor `uv` puts the origin in the middle of the screen. You may have met the range from −1 to 1 as **normalized device coordinates**, but that's the space a *vertex* shader outputs in the graphics pipeline, and a compute shader skips that pipeline entirely. So shader code builds its own centered space, almost always like this:

```wgsl
let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
```

This puts `(0, 0)` at the center with y running from −1 to 1. The key detail is **dividing by the height only**. That keeps x and y on the same scale, so a circle stays round instead of stretching into an ellipse. On a 16:9 screen, x runs from about −1.78 to 1.78.

With a centered space, a circle is one line: a point is inside a circle of radius `r` when `length(p) < r`. Better still, think of `length(p) - r` as a **signed distance**: negative inside the circle, zero on its edge, positive outside, and in each case equal to how far you are from the edge. Distance functions like this are the core of 2D shader art.

::: task
Draw a white circle of radius 0.5 in the center. Resize the page (or rotate your phone) and check that it stays round.
:::

::: {.toy #ex04 data-label="Exercise 4" data-title="Circle"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Centered, aspect-correct: (0,0) in the middle, y from -1 to 1
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    // TODO: white where length(p) < 0.5
    var col = vec3f(abs(p), 0.0);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let d = length(p) - 0.5;               // signed distance to the circle
    var col = vec3f(select(0.0, 1.0, d < 0.0));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

## [5]{.num} Soft circle

Zoom into the edge of that circle and you'll see stair steps. Each pixel is either fully in or fully out, a problem called **aliasing**. The fix is to fade across the edge over about one pixel.

`smoothstep(a, b, x)` returns 0 when `x ≤ a`, 1 when `x ≥ b`, and a smooth S-curve in between. Feed it the signed distance and fade across a band one pixel wide on each side of the edge.

How wide is one pixel in `p` units? `p.y` spans 2 units over the screen's height, so one pixel is `2.0 / f32(screen_size.y)`. (Fragment shaders can compute this with `fwidth`, but compute shaders have no neighboring pixels to compare against, so you work it out from the resolution.)

```wgsl
let px = 2.0 / f32(screen_size.y);
let inside = 1.0 - smoothstep(-px, px, d);   // 1 inside, 0 outside, soft edge
```

Once you have a mask from 0 to 1, `mix(background, foreground, mask)` paints with it. This mask-and-mix pattern is how nearly all 2D shader art is layered.

::: task
Give the circle an anti-aliased edge and paint it in a color of your choice on a dark background. Then make `px` ten times wider and see what happens.
:::

::: {.toy #ex05 data-label="Exercise 5" data-title="Soft circle"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let d = length(p) - 0.5;

    // TODO: replace the hard edge with smoothstep over about one pixel,
    //       then mix a background and a foreground color with the mask
    let inside = select(0.0, 1.0, d < 0.0);
    var col = vec3f(inside);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let d = length(p) - 0.5;
    let px = 2.0 / f32(screen_size.y);            // one pixel, in p units
    let inside = 1.0 - smoothstep(-px, px, d);

    let bg = vec3f(0.05, 0.06, 0.1);
    let fg = vec3f(1.0, 0.75, 0.3);
    var col = mix(bg, fg, inside);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

## [6]{.num} Rings and unions

Because a signed distance is a real distance, you can combine shapes with simple arithmetic:

| Operation | Code | Why it works |
|---|---|---|
| Union | `min(a, b)` | You're inside either shape if either distance is negative. |
| Intersection | `max(a, b)` | Inside both only if both are negative. |
| Subtraction | `max(a, -b)` | Inside `a` and outside `b`. |
| Outline (ring) | `abs(d) - w` | Distance to the edge itself, thickened by `w`. |
| Move | `sd(p - c)` | Evaluate the shape in coordinates centered at `c`. |

The last row is worth pausing on. To move a shape, you don't move the shape: you shift the coordinates you evaluate it in. That idea returns in Part III.

The starter code draws a single circle's distance field as contour lines, so you can see that `d` really is a distance everywhere on screen, not only near the edge.

::: task
Write a helper `fn sdCircle(p: vec2f, r: f32) -> f32`. Draw two overlapping circles as one merged shape using `min`, and a thin ring around both. Keep the edges anti-aliased.
:::

::: {.toy #ex06 data-label="Exercise 6" data-title="Rings and unions"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    // The distance field of one circle, drawn as contour lines.
    // Blue outside, orange inside, white on the edge.
    let d = length(p) - 0.5;
    var col = select(vec3f(0.3, 0.55, 0.95), vec3f(0.95, 0.6, 0.25), d < 0.0);
    col *= 0.75 + 0.25 * cos(d * 60.0);
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d)));

    // TODO: an sdCircle helper, two circles merged with min(), and a ring

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
    let px = 2.0 / f32(screen_size.y);

    // Two circles, each evaluated in shifted coordinates, merged with min
    let a = sdCircle(p - vec2f(-0.3, 0.0), 0.4);
    let b = sdCircle(p - vec2f( 0.3, 0.0), 0.4);
    let blobs = min(a, b);

    // A ring: distance to a circle's edge, thickened by 0.015
    let ring = abs(sdCircle(p, 0.85)) - 0.015;

    var col = vec3f(0.05, 0.06, 0.1);
    col = mix(col, vec3f(0.3, 0.55, 0.95), 1.0 - smoothstep(-px, px, blobs));
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(-px, px, ring));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

# [Part III]{.part} Time and motion

## [7]{.num} Moving circle

`time.elapsed` is the number of seconds since the shader started. Anything that depends on it animates, because every frame the whole image is recomputed with a slightly larger value.

To move a shape, shift the coordinates, as in the last table. A center that follows `(sin t, cos t)` traces a circle, since sine and cosine of the same angle are the coordinates of a point on the unit circle.

`time.frame` counts frames instead. Prefer `elapsed` for motion, since frame rates vary between machines, and use `frame` when you need something to happen exactly once or to seed randomness.

::: task
Make a circle of radius 0.2 orbit the center at radius 0.5. Then add a second circle that orbits twice as fast in the other direction.
:::

::: {.toy #ex07 data-label="Exercise 7" data-title="Moving circle"}
```wgsl
fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
    let px = 2.0 / f32(screen_size.y);

    // TODO: make the center move with time.elapsed
    let center = vec2f(0.5, 0.0);
    let d = sdCircle(p - center, 0.2);

    var col = vec3f(0.05, 0.06, 0.1);
    col = mix(col, vec3f(1.0, 0.75, 0.3), 1.0 - smoothstep(-px, px, d));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
    let px = 2.0 / f32(screen_size.y);
    let t = time.elapsed;

    let c1 = 0.5 * vec2f(sin(t), cos(t));                  // orbit clockwise
    let c2 = 0.5 * vec2f(sin(-2.0 * t), cos(-2.0 * t));    // twice as fast, reversed
    let d1 = sdCircle(p - c1, 0.2);
    let d2 = sdCircle(p - c2, 0.12);

    var col = vec3f(0.05, 0.06, 0.1);
    col = mix(col, vec3f(1.0, 0.75, 0.3), 1.0 - smoothstep(-px, px, d1));
    col = mix(col, vec3f(0.35, 0.6, 1.0), 1.0 - smoothstep(-px, px, d2));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

## [8]{.num} Ripples

A wave is a sine of position minus time. `sin(k * x - w * t)` is a pattern with `k` controlling how tightly the crests are packed and `w` controlling how fast they move. Because the crest sits where `k x - w t` is constant, it travels at speed `w / k`.

Use distance from the center as the position and you get concentric rings moving outward. `sin` returns −1 to 1, so remap it with `0.5 + 0.5 * sin(...)` before using it as a blend factor.

::: task
Color each pixel by `sin(length(p) * 20.0 - time.elapsed * 3.0)`, remapped to 0–1, and blend between two colors of your choice. Fade the waves out toward the edges. Then flip the sign of the time term.
:::

::: {.toy #ex08 data-label="Exercise 8" data-title="Ripples"}
```wgsl
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let d = length(p);

    // TODO: a wave that depends on d and time.elapsed, remapped to 0..1
    let wave = 0.0;
    var col = mix(vec3f(0.1, 0.2, 0.5), vec3f(0.9, 0.95, 1.0), wave);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let d = length(p);
    let wave = 0.5 + 0.5 * sin(d * 20.0 - time.elapsed * 3.0);
    var col = mix(vec3f(0.1, 0.2, 0.5), vec3f(0.9, 0.95, 1.0), wave);
    col *= smoothstep(1.5, 0.0, d);        // fade out toward the edges

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

## [9]{.num} Rotation

Rotating a point by angle `a` around the origin is a matrix multiply:

```wgsl
fn rot(a: f32) -> mat2x2f {
    let c = cos(a);
    let s = sin(a);
    return mat2x2f(c, s, -s, c);
}
```

WGSL matrices are **column-major**: `mat2x2f(c, s, -s, c)` has first column `(c, s)` and second column `(−s, c)`. Multiplying `rot(a) * p` turns `p` counter-clockwise by `a`.

As with moving, you rotate a shape by rotating the *coordinates* the other way. To spin a square by `+t`, evaluate the square at `rot(-t) * p`. This is the most common source of "why is it spinning backwards?"

For the square itself, the starter includes the standard box distance function. You don't need to derive it, but it works by measuring how far `abs(p)` sticks out past the box's corner.

::: task
Spin the square around the center. Then add a second, smaller square that orbits the first while spinning around its own center. You'll need to translate first, then rotate.
:::

::: {.toy #ex09 data-label="Exercise 9" data-title="Rotation"}
```wgsl
// Signed distance to an axis-aligned box with half-size b
fn sdBox(p: vec2f, b: vec2f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

// TODO: fn rot(a: f32) -> mat2x2f

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
    let px = 2.0 / f32(screen_size.y);

    // TODO: rotate the coordinates before measuring the box
    let q = p;
    let d = sdBox(q, vec2f(0.35));

    var col = vec3f(0.05, 0.06, 0.1);
    col = mix(col, vec3f(0.35, 0.6, 1.0), 1.0 - smoothstep(-px, px, d));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn sdBox(p: vec2f, b: vec2f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

// Counter-clockwise rotation. Columns are (c, s) and (-s, c).
fn rot(a: f32) -> mat2x2f {
    let c = cos(a);
    let s = sin(a);
    return mat2x2f(c, s, -s, c);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
    let px = 2.0 / f32(screen_size.y);
    let t = time.elapsed;

    // Big square: rotate the space by -t so the square turns by +t
    let d1 = sdBox(rot(-t) * p, vec2f(0.3));

    // Small square: move to its orbit position, then spin about its own center
    let center = rot(t * 0.5) * vec2f(0.75, 0.0);
    let d2 = sdBox(rot(-3.0 * t) * (p - center), vec2f(0.1));

    var col = vec3f(0.05, 0.06, 0.1);
    col = mix(col, vec3f(0.35, 0.6, 1.0), 1.0 - smoothstep(-px, px, d1));
    col = mix(col, vec3f(1.0, 0.75, 0.3), 1.0 - smoothstep(-px, px, d2));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

# [Part IV]{.part} Randomness

## [10]{.num} Static

WGSL has no `random()` function, and neither does any other shading language. A shader invocation has no hidden state that survives between calls, and thousands of threads can't share one global generator without stepping on each other.

Instead you use a **hash function**: a function that scrambles an integer so thoroughly that its outputs *look* random. Feed in something unique to this thread, such as the pixel coordinate, and you get a number that looks unrelated to its neighbors. Feed in the same input and you get the same output, which is useful: randomness in shaders is reproducible by default.

A good, fast choice is the **PCG hash**:

```wgsl
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A float in [0, 1). Keeping the top 24 bits fits exactly in an f32.
fn u2f(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}
```

The `u` suffix marks unsigned integer literals. Integer arithmetic on `u32` wraps around on overflow, which is exactly what a hash wants.

To combine several inputs, **nest** the hash: `pcg(x ^ pcg(y ^ pcg(frame)))`. Each layer mixes the previous result with a new input. To get several independent numbers for one pixel, keep hashing: `h`, `pcg(h)`, `pcg(pcg(h))`.

::: gotcha
If your seed is a float, convert it with `bitcast<u32>(x)`, which reinterprets the float's raw bits. Don't use `u32(x)`: that's a numeric conversion that truncates, so 0.2, 0.7 and 0.99 all become 0 and give identical "random" values.
:::

::: task
Fill the screen with colored TV static that changes every frame. Then make it hold still by removing the frame from the seed.
:::

::: {.toy #ex10 data-label="Exercise 10" data-title="Static"}
```wgsl
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn u2f(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // TODO: hash id.x, id.y and time.frame together, then take three
    //       numbers from the hash for red, green and blue
    var col = vec3f(u2f(pcg(id.x)));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn u2f(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Nest the hash to mix in each input. Drop time.frame for still noise.
    let h = pcg(id.x ^ pcg(id.y ^ pcg(time.frame)));
    let r = pcg(h);
    let g = pcg(r);
    let b = pcg(g);
    var col = vec3f(u2f(r), u2f(g), u2f(b));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

The solution chains `pcg` to get three numbers. Why not `pcg(h)`, `pcg(h + 1u)`, `pcg(h + 2u)`? Because the pixel to the right probably has a seed one larger, so its red would equal your green. Chaining avoids that kind of accidental correlation.

## [11]{.num} Random tiles

Here is the key trick behind procedural textures: **hash the cell, not the pixel**. Every pixel in a cell computes the same cell index, so every pixel gets the same random value, and the cell appears as one solid random color.

Cell indices from `floor` are floats. Convert them to integers first with `vec2i(...)`. Negative indices are fine, because `bitcast<u32>` of a negative `i32` is just a different bit pattern.

::: task
Split the screen into 8 × 8 cells like the checkerboard, and give each cell its own random color. Then make the pattern reshuffle once per second by mixing `u32(time.elapsed)` into the hash.
:::

::: {.toy #ex11 data-label="Exercise 11" data-title="Random tiles"}
```wgsl
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn u2f(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    let cell = vec2i(floor(uv * 8.0));

    // TODO: hash the cell (not the pixel) into a color
    var col = vec3f(vec2f(cell) / 8.0, 0.0);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn u2f(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

fn hash2(c: vec2i, seed: u32) -> u32 {
    return pcg(bitcast<u32>(c.x) ^ pcg(bitcast<u32>(c.y) ^ pcg(seed)));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    let cell = vec2i(floor(uv * 8.0));
    let h = hash2(cell, u32(time.elapsed));     // new pattern every second
    var col = vec3f(u2f(h), u2f(pcg(h)), u2f(pcg(pcg(h))));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

## [12]{.num} Value noise

Random tiles are blocky. For clouds, terrain or fire you want randomness that varies **smoothly**. **Value noise** gets it by putting a random value at every *corner* of the grid and blending between the four corners around each point:

1. `i = floor(x)` is the cell. Its corners are `i`, `i + (1,0)`, `i + (0,1)` and `i + (1,1)`. Hash each one.
2. `f = fract(x)` is the position inside the cell.
3. Blend the bottom two corners by `f.x`, the top two by `f.x`, then blend those two results by `f.y`.

Blending with `f` directly leaves visible creases at cell borders, because the slope jumps there. Using the curve `u = f * f * (3.0 - 2.0 * f)` instead (the same curve `smoothstep` uses) flattens the slope at each border and hides the grid.

A single layer of noise looks like soft blobs. For natural detail, add several layers, each at double the frequency and half the strength. This is called **fractal Brownian motion (fBm)**, and it's how the header of this page is drawn.

::: task
Write `noise(x: vec2f) -> f32` and display `noise(p * 4.0)`. Then write `fbm` with five octaves and animate it by adding time to the coordinates.
:::

::: {.toy #ex12 data-label="Exercise 12" data-title="Value noise"}
```wgsl
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A random value in [0, 1) for each integer grid point
fn hash(c: vec2i) -> f32 {
    return f32(pcg(bitcast<u32>(c.x) ^ pcg(bitcast<u32>(c.y))) >> 8u) / 16777216.0;
}

fn noise(x: vec2f) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    // TODO: hash the four corners of cell i and blend them smoothly
    return hash(i);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    var col = vec3f(noise(p * 4.0));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash(c: vec2i) -> f32 {
    return f32(pcg(bitcast<u32>(c.x) ^ pcg(bitcast<u32>(c.y))) >> 8u) / 16777216.0;
}

fn noise(x: vec2f) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);          // smooth blend weights

    let a = hash(i);
    let b = hash(i + vec2i(1, 0));
    let c = hash(i + vec2i(0, 1));
    let d = hash(i + vec2i(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Five octaves: each doubles the frequency and halves the amplitude
fn fbm(x0: vec2f) -> f32 {
    var x = x0;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 5; i++) {
        sum += amp * noise(x);
        x = x * 2.0 + vec2f(3.1, 1.7);         // offset so octaves don't align
        amp *= 0.5;
    }
    return sum;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let n = fbm(p * 2.5 + vec2f(time.elapsed * 0.3, 0.0));
    var col = mix(vec3f(0.1, 0.25, 0.6), vec3f(1.0), smoothstep(0.3, 0.8, n));

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

### Seeded generators

A hash gives one number per input. When one pixel needs *many* random numbers, it's more convenient to wrap the hash in a tiny generator: keep a `u32` state, and advance it each time you draw a number.

```wgsl
// A starting state from the pixel, the frame, and your own seed
fn rng_init(pixel: vec2u, frame: u32, seed: u32) -> u32 {
    return pcg(pixel.x ^ pcg(pixel.y ^ pcg(frame ^ pcg(seed))));
}

// The next float in [0, 1); updates the state through the pointer
fn rng_next(state: ptr<function, u32>) -> f32 {
    *state = pcg(*state);
    return f32(*state >> 8u) / 16777216.0;
}

var rng = rng_init(id.xy, time.frame, 1234u);
let a = rng_next(&rng);
let b = rng_next(&rng);    // a different number
```

The state must be a `var`, since it changes, and `&rng` passes a pointer so the function can update it.

Give each part of your shader its **own stream** with a different seed, for example `rng_init(id.xy, time.frame, 1u)` for color and `2u` for shape. With one shared generator, inserting one extra `rng_next` call shifts every number after it, which makes tweaking confusing. This is the same idea behind JAX's `split` and `fold_in`: random streams are derived from keys, not from a shared, mutating global.
