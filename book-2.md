
# [Part V]{.part} More than one pass

So far, every shader has had one entry point that computes each pixel from scratch. Fragment shaders on Shadertoy work the same way. Compute shaders can do more: run **several entry points per frame**, pass images from one to the next, and keep data from one frame to the following one.

## [13]{.num} Blur pass

A blur needs each pixel to read its *neighbors*. Within one pass, you can't read what other threads are computing: there's no order between them, so a neighbor may not have written yet. The fix is to split the work. One entry point draws the image; a second one reads the finished image and blurs it. The boundary between entry points guarantees that everything from the first is done before the second starts.

In this playground (as in compute.toys), entry points hand images to each other through **pass textures**: four scratch images, each the size of the screen, with four channels per pixel.

```wgsl
passStore(layer, coord, value)   // write a vec4f to pixel coord of scratch image `layer` (0–3)
passLoad(layer, coord, lod)      // read it back; `lod` is the mip level, always 0 here
```

A few details:

- `coord` is a `vec2i` in texture space, the same space as `id.xy`, with y pointing down.
- The third argument of `passLoad` is the **mip level**, not a channel. The pass textures have only one level, so it's always 0. You choose channels from the result with a swizzle: `passLoad(0, q, 0).x`.
- `passStore` writes all four channels at once. To update one channel, load the pixel, change it and store the whole `vec4f` back.
- That's 16 floats of scratch space per pixel: 4 layers × 4 channels.
- After each dispatch, the runtime copies what was stored into what `passLoad` reads. So a `passLoad` sees what the **previous** entry point stored, never what the current one is storing. This separation is what makes reading neighbors safe.

A **box blur** replaces each pixel with the average of the square of pixels around it. With radius `R`, that's `(2R + 1)²` samples. Pixels near the edge would read outside the image, so clamp the coordinate to the valid range.

::: task
The `scene` pass draws a sharp picture into layer 0. In `main_image`, replace each pixel with the average of the 13 × 13 square around it. Show the blurred version only to the right of the mouse (or of the middle, before the mouse moves), so you can compare.
:::

::: {.toy #ex13 data-label="Exercise 13" data-title="Blur pass"}
```wgsl
fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

// Pass 1: draw a sharp scene into scratch layer 0
@compute @workgroup_size(16, 16)
fn scene(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    // A checkerboard with orbiting circles: lots of hard edges
    let cell = floor(p * 6.0);
    let checker = abs((cell.x + cell.y) % 2.0);
    var col = mix(vec3f(0.06, 0.07, 0.12), vec3f(0.2, 0.22, 0.32), checker);
    for (var i = 0; i < 3; i++) {
        let a = time.elapsed * 0.7 + f32(i) * 2.094;
        let d = sdCircle(p - 0.55 * vec2f(cos(a), sin(a)), 0.22);
        let tint = 0.55 + 0.45 * cos(vec3f(0.0, 2.0, 4.0) + f32(i) * 2.0);
        col = select(col, tint, d < 0.0);
    }

    // Store in linear light so the blur averages real brightness
    passStore(0, vec2i(id.xy), vec4f(pow(col, vec3f(2.2)), 1.0));
}

// Pass 2: read layer 0 and write the screen
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let here = vec2i(id.xy);

    // TODO: average the (2R+1) x (2R+1) square around `here`,
    //       clamping coordinates to the screen
    let col = passLoad(0, here, 0).rgb;

    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

@compute @workgroup_size(16, 16)
fn scene(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let p = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.y);

    let cell = floor(p * 6.0);
    let checker = abs((cell.x + cell.y) % 2.0);
    var col = mix(vec3f(0.06, 0.07, 0.12), vec3f(0.2, 0.22, 0.32), checker);
    for (var i = 0; i < 3; i++) {
        let a = time.elapsed * 0.7 + f32(i) * 2.094;
        let d = sdCircle(p - 0.55 * vec2f(cos(a), sin(a)), 0.22);
        let tint = 0.55 + 0.45 * cos(vec3f(0.0, 2.0, 4.0) + f32(i) * 2.0);
        col = select(col, tint, d < 0.0);
    }
    passStore(0, vec2i(id.xy), vec4f(pow(col, vec3f(2.2)), 1.0));
}

const R = 6;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let here = vec2i(id.xy);
    let last = vec2i(screen_size) - 1;

    var sum = vec3f(0.0);
    for (var dy = -R; dy <= R; dy++) {
        for (var dx = -R; dx <= R; dx++) {
            let q = clamp(here + vec2i(dx, dy), vec2i(0), last);
            sum += passLoad(0, q, 0).rgb;
        }
    }
    let blurred = sum / f32((2 * R + 1) * (2 * R + 1));
    let sharp = passLoad(0, here, 0).rgb;

    // Compare: sharp on the left of the split, blurred on the right
    var split = i32(screen_size.x) / 2;
    if (mouse.pos.x > 0) { split = mouse.pos.x; }
    var col = select(sharp, blurred, here.x > split);
    if (abs(here.x - split) < 1) { col = vec3f(1.0); }

    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

A box blur of radius `R` costs `(2R + 1)²` reads per pixel: 169 here, and 1,681 at radius 20. The standard improvement is a **separable** blur. Blur horizontally into layer 1, then blur that vertically into the screen. The result is identical, and it costs `2 × (2R + 1)` reads instead. Try it: add a third entry point between `scene` and `main_image`.

## [14]{.num} Game of Life

Pass textures keep their contents **between frames**. That makes them memory: a simulation can read last frame's state and write this frame's state.

Conway's Game of Life is the classic first stateful compute shader. The grid is made of cells that are alive or dead. Every step, each cell counts its eight neighbors:

- A live cell with 2 or 3 live neighbors survives. Otherwise it dies.
- A dead cell with exactly 3 live neighbors comes alive.

You can't update the grid **in place**, because a cell's neighbors would see a mix of old and new values depending on which threads happened to run first. Every cell must read the old generation and write the new one. The usual pattern keeps two buffers and swaps their roles each frame, known as **ping-pong**. Here the runtime does it for you, since `passLoad` always reads the previous state and `passStore` writes the next one.

Three practical details:

- **Initialize on the first frame.** When `time.frame == 0u`, fill the grid randomly instead of stepping. Press **Restart** to get a new first frame.
- **Decide what happens at the edges.** Wrapping around, with `(c + grid) % grid`, turns the grid into a torus and keeps gliders flying forever.
- **Make cells bigger than pixels.** One cell per pixel is too small to watch. The starter stores cell `(i, j)` at pixel `(i, j)` of layer 0, but draws each cell as a 4 × 4 block.

::: task
Implement the rules in `life`. Right now it copies each cell unchanged, so the random soup just sits there. Hold the mouse button on the canvas to draw new live cells.
:::

::: {.toy #ex14 data-label="Exercise 14" data-title="Game of Life"}
```wgsl
const CELL = 4u;   // each cell is drawn as a CELL x CELL block of pixels

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// State of a cell (1.0 alive, 0.0 dead), wrapping around the edges
fn alive(c: vec2i, grid: vec2i) -> f32 {
    let w = (c + grid) % grid;
    return passLoad(0, w, 0).x;
}

// Step: one thread per cell. x = alive, y = a fading trail for display
@compute @workgroup_size(16, 16)
fn life(@builtin(global_invocation_id) id: vec3u) {
    let grid = vec2i(textureDimensions(screen) / CELL);
    let c = vec2i(id.xy);
    if (c.x >= grid.x || c.y >= grid.y) { return; }

    let prev = passLoad(0, c, 0);

    // First frame: random soup, about 30% alive
    if (time.frame == 0u) {
        let r = f32(pcg(id.x ^ pcg(id.y ^ 12345u)) >> 8u) / 16777216.0;
        let a = select(0.0, 1.0, r < 0.3);
        passStore(0, c, vec4f(a, a, 0.0, 1.0));
        return;
    }

    let me = alive(c, grid);

    // TODO: count the 8 neighbors with alive(), then apply the rules
    var next = me;

    // Hold the mouse to draw live cells
    let m = mouse.pos / i32(CELL);
    if (mouse.click == 1 && length(vec2f(c - m)) < 3.0) { next = 1.0; }

    passStore(0, c, vec4f(next, max(next, prev.y * 0.92), 0.0, 1.0));
}

// Display: read the cell under this pixel
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let s = passLoad(0, vec2i(id.xy / CELL), 0);
    var col = mix(vec3f(0.03, 0.035, 0.07), vec3f(0.2, 0.28, 0.75), s.y);
    col = mix(col, vec3f(1.0, 0.9, 0.7), s.x);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
```{.wgsl .solution}
const CELL = 4u;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn alive(c: vec2i, grid: vec2i) -> f32 {
    let w = (c + grid) % grid;
    return passLoad(0, w, 0).x;
}

@compute @workgroup_size(16, 16)
fn life(@builtin(global_invocation_id) id: vec3u) {
    let grid = vec2i(textureDimensions(screen) / CELL);
    let c = vec2i(id.xy);
    if (c.x >= grid.x || c.y >= grid.y) { return; }

    let prev = passLoad(0, c, 0);

    if (time.frame == 0u) {
        let r = f32(pcg(id.x ^ pcg(id.y ^ 12345u)) >> 8u) / 16777216.0;
        let a = select(0.0, 1.0, r < 0.3);
        passStore(0, c, vec4f(a, a, 0.0, 1.0));
        return;
    }

    let me = alive(c, grid);

    var n = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            n += alive(c + vec2i(dx, dy), grid);
        }
    }
    // Birth on exactly 3; survival on 2 or 3
    var next = select(0.0, 1.0, n == 3.0 || (me == 1.0 && n == 2.0));

    let m = mouse.pos / i32(CELL);
    if (mouse.click == 1 && length(vec2f(c - m)) < 3.0) { next = 1.0; }

    passStore(0, c, vec4f(next, max(next, prev.y * 0.92), 0.0, 1.0));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let s = passLoad(0, vec2i(id.xy / CELL), 0);
    var col = mix(vec3f(0.03, 0.035, 0.07), vec3f(0.2, 0.28, 0.75), s.y);
    col = mix(col, vec3f(1.0, 0.9, 0.7), s.x);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
```
:::

# [Part VI]{.part} Threads that cooperate

Everything so far has been **embarrassingly parallel**: each thread's output depends only on its inputs, never on other threads in the same pass. The last three exercises aren't like that. They need to combine thousands of values into one, which is the heart of GPU computing for machine learning.

## [15]{.num} Parallel sum

The task: add up 16,384 numbers. One thread doing it alone would take 16,384 steps while thousands of other threads sit idle. The parallel approach uses three new tools.

### Storage buffers

A **storage buffer** is a plain array in GPU memory that any thread can read and write. Declare one with a directive at the top of the file:

```wgsl
#storage data array<f32>
#storage partials array<f32>
```

The runtime turns each line into a buffer binding, so `data[i]` works like any array. Buffers here hold 2 million floats each. compute.toys allows **two** storage buffers, so the exercises stick to two, and pack several things into one buffer when they need more.

### Dispatch sizes

By default, an entry point gets enough workgroups to cover the screen. That's wrong for a 1D array, so set it explicitly:

```wgsl
#workgroup_count fill 64 1 1     // 64 workgroups × 256 threads = 16,384 threads
```

The directives only configure entry points; you still write `fill`, `reduce` and `finish` as ordinary `@compute` functions with exactly those names.

### Workgroup memory and barriers

Threads in the **same workgroup** can share a small, fast scratchpad:

```wgsl
var<workgroup> scratch: array<f32, 256>;
```

Every workgroup gets its own copy, visible to all 256 of its threads. To use it safely, threads must wait for each other. `workgroupBarrier()` makes each thread pause until every thread in its workgroup has reached the same point, and makes their writes to workgroup memory visible.

### The tree reduction

Inside one workgroup, the 256 values are summed in halving steps:

1. Each thread loads one value into `scratch[lid.x]`. Barrier.
2. With `stride = 128`, each thread with `lid.x < stride` adds `scratch[lid.x + stride]` into its own slot. Barrier.
3. Repeat with stride 64, 32, … down to 1. Barrier after every step.
4. `scratch[0]` now holds the workgroup's total. Thread 0 writes it out.

::: {.diagram .wide}
<figure class="diagram"><!--SVG:tree--><figcaption>Eight values summed in three steps instead of seven. With 256 values, it's 8 steps. Each dashed line is a barrier.</figcaption></figure>
:::

There's **no barrier across workgroups**: workgroups may run in any order, and not necessarily at the same time. So each workgroup writes its total to `partials[workgroup_id]`, and a second entry point, `finish`, runs a single workgroup that reduces those 64 partials the same way. The boundary between entry points is the global synchronization you don't otherwise have.

::: gotcha
**Every thread must reach every barrier.** Don't put `workgroupBarrier()` inside `if (lid.x < stride)`, and don't `return` early before one. WGSL's **uniformity analysis** rejects code where a barrier might be skipped by some threads. Keep the barrier inside the loop but outside the `if`.
:::

::: gotcha
**`lid` versus `gid`.** "Once per workgroup" is always `lid.x == 0u`. `gid.x == 0u` is true for exactly one thread in the entire dispatch, so only `partials[0]` would ever be written. This mix-up is probably the most common compute shader bug there is.
:::

When the data doesn't divide evenly into workgroups, don't skip threads, since that would break the barrier rule. Load the **identity element** instead: 0 for a sum, `-inf` for a max, 1 for a product. Here 64 × 256 is exactly 16,384, so it doesn't come up.

::: task
Fill in `reduce` and `finish`. The display shows the total as a bar, green when it equals 16,384 exactly and red otherwise. The strip underneath shows the 64 partial sums: each should be equally bright. If only the first one lights up, check `lid` versus `gid`. If the total flickers between frames, a barrier is missing.
:::

::: {.toy #ex15 data-label="Exercise 15" data-title="Parallel sum"}
```wgsl
#storage data array<f32>
#storage partials array<f32>
#workgroup_count fill 64 1 1
#workgroup_count reduce 64 1 1
#workgroup_count finish 1 1 1

const N = 16384u;      // 64 workgroups x 256 threads
const RESULT = 64u;    // partials[0..64) = per-workgroup sums, partials[64] = total

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    data[gid.x] = 1.0;   // all ones, so the correct total is 16384
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {
    // TODO: load data[gid.x] into scratch[lid.x], then barrier
    // TODO: tree reduction, stride 128, 64, ..., 1, barrier after each step
    // TODO: thread 0 of each workgroup writes scratch[0] to partials[wid.x]
}

@compute @workgroup_size(64)
fn finish(@builtin(local_invocation_id) lid: vec3u) {
    // TODO: the same reduction over the 64 partials, with one workgroup.
    //       Reuse scratch (first 64 slots). Thread 0 writes partials[RESULT].
}

// ---------- display: total as a bar, partials as a strip ----------
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);   // y points down here

    let total = partials[RESULT];
    var col = vec3f(0.06, 0.07, 0.12);
    if (uv.y > 0.3 && uv.y < 0.5) {
        col = vec3f(0.14, 0.16, 0.26);
        if (uv.x < total / f32(N)) {
            col = select(vec3f(0.95, 0.35, 0.3), vec3f(0.3, 0.85, 0.5), abs(total - f32(N)) < 0.5);
        }
    }
    let quarter = abs(fract(uv.x * 4.0 + 0.5) - 0.5) * f32(size.x) / 4.0;
    if (quarter < 1.0 && uv.y > 0.27 && uv.y < 0.53) { col = vec3f(0.55); }
    if (uv.y > 0.62 && uv.y < 0.72) {
        let i = min(u32(uv.x * 64.0), 63u);
        col = vec3f(0.35, 0.6, 1.0) * clamp(partials[i] / 256.0, 0.0, 1.0);
    }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
```{.wgsl .solution}
#storage data array<f32>
#storage partials array<f32>
#workgroup_count fill 64 1 1
#workgroup_count reduce 64 1 1
#workgroup_count finish 1 1 1

const N = 16384u;
const RESULT = 64u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    data[gid.x] = 1.0;
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {
    scratch[lid.x] = data[gid.x];   // gid picks the element, lid picks the slot
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();          // outside the if: every thread reaches it
    }

    if (lid.x == 0u) {               // once per workgroup: lid, not gid
        partials[wid.x] = scratch[0];
    }
}

@compute @workgroup_size(64)
fn finish(@builtin(local_invocation_id) lid: vec3u) {
    scratch[lid.x] = partials[lid.x];
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        partials[RESULT] = scratch[0];
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);

    let total = partials[RESULT];
    var col = vec3f(0.06, 0.07, 0.12);
    if (uv.y > 0.3 && uv.y < 0.5) {
        col = vec3f(0.14, 0.16, 0.26);
        if (uv.x < total / f32(N)) {
            col = select(vec3f(0.95, 0.35, 0.3), vec3f(0.3, 0.85, 0.5), abs(total - f32(N)) < 0.5);
        }
    }
    let quarter = abs(fract(uv.x * 4.0 + 0.5) - 0.5) * f32(size.x) / 4.0;
    if (quarter < 1.0 && uv.y > 0.27 && uv.y < 0.53) { col = vec3f(0.55); }
    if (uv.y > 0.62 && uv.y < 0.72) {
        let i = min(u32(uv.x * 64.0), 63u);
        col = vec3f(0.35, 0.6, 1.0) * clamp(partials[i] / 256.0, 0.0, 1.0);
    }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
:::

Once it's green, change `fill` to random values and verify the sum a second, dumb way: one entry point with `#workgroup_count check 1 1 1` in which a single thread loops over all 16,384 values. The two results won't match to the last bit, because floating-point addition isn't associative and the two methods add in different orders. That difference is normal, and worth seeing once.

This reduction is the single most important pattern on this page. Softmax, layer norm and attention all contain it.

## [16]{.num} Softmax

**Softmax** turns a list of scores into probabilities that are positive and sum to 1:

$$\operatorname{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

where $m$ is the largest of the $x_j$. Subtracting the maximum `m` doesn't change the answer, since it scales the top and bottom by the same factor. But it keeps `exp` from overflowing: `exp(89.0)` is already too large for an `f32`. Every real implementation does this.

On the GPU, softmax is **two reductions and an elementwise pass**:

1. **Max reduction**: the same tree as the sum, with `max` instead of `+`.
2. **Sum reduction** of `exp(x - m)`.
3. **Normalize**: every element becomes `exp(x - m) / sum`, written back in place.

Each reduction is two entry points (per-workgroup, then finish), so you'll write five new entry points. They run in the order they appear in the file.

Two things to plan yourself:

- **Where results live.** You still have two storage buffers. The starter reserves regions of `stats` for the partial maxima, the max, the partial sums and the sum.
- **Reusing the tree.** Listing an entry point twice in `#workgroup_count` does not run it twice. For repeated logic, write separate entry points that call a shared helper function. Helpers can contain `workgroupBarrier()`, as long as every thread in the workgroup calls them. The starter includes `tree_sum`; write `tree_max` alongside it.

::: task
Implement steps 1–3 between `fill` and `check_reduce`. The top bar is the sum of your outputs and turns green when it's 1. The bottom bar is the probability of element 5,000, which `fill` sets to 20.0, the largest value in the array. It should end up close to 0.96.
:::

::: {.toy #ex16 data-label="Exercise 16" data-title="Softmax"}
```wgsl
#storage data array<f32>
#storage stats array<f32>
#workgroup_count fill 64 1 1
#workgroup_count check_reduce 64 1 1
#workgroup_count check_finish 1 1 1

const N = 16384u;
const PLANT = 5000u;           // fill puts the largest value here

// Layout of the `stats` buffer
const MAX_PARTIALS = 0u;       // [0, 64)
const MAX = 64u;
const SUM_PARTIALS = 128u;     // [128, 192)
const SUM = 192u;
const CHECK_PARTIALS = 256u;   // [256, 320)
const CHECK = 320u;

var<workgroup> scratch: array<f32, 256>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Tree-sum scratch[0..width) into scratch[0]. Call from every thread.
fn tree_sum(lid: u32, width: u32) {
    for (var s = width / 2u; s > 0u; s >>= 1u) {
        if (lid < s) { scratch[lid] += scratch[lid + s]; }
        workgroupBarrier();
    }
}

// Random scores in [-10, 10), with one planted maximum of 20
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    let r = f32(pcg(gid.x) >> 8u) / 16777216.0;
    data[gid.x] = select(r * 20.0 - 10.0, 20.0, gid.x == PLANT);
}

// TODO: your entry points go here, in order, each with a #workgroup_count:
//   reduce_max, finish_max  ->  stats[MAX]
//   reduce_sum, finish_sum  ->  stats[SUM]   (sum of exp(x - max))
//   normalize               ->  data[i] = exp(data[i] - max) / sum

// ---------- check: sum the outputs, which should be 1 ----------
@compute @workgroup_size(256)
fn check_reduce(@builtin(global_invocation_id) gid: vec3u,
                @builtin(local_invocation_id) lid: vec3u,
                @builtin(workgroup_id) wid: vec3u) {
    scratch[lid.x] = data[gid.x];
    workgroupBarrier();
    tree_sum(lid.x, 256u);
    if (lid.x == 0u) { stats[CHECK_PARTIALS + wid.x] = scratch[0]; }
}

@compute @workgroup_size(64)
fn check_finish(@builtin(local_invocation_id) lid: vec3u) {
    scratch[lid.x] = stats[CHECK_PARTIALS + lid.x];
    workgroupBarrier();
    tree_sum(lid.x, 64u);
    if (lid.x == 0u) { stats[CHECK] = scratch[0]; }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);

    let total = stats[CHECK];
    let planted = data[PLANT];
    var col = vec3f(0.06, 0.07, 0.12);
    if (uv.y > 0.25 && uv.y < 0.42) {
        col = vec3f(0.14, 0.16, 0.26);
        if (uv.x < total) {
            col = select(vec3f(0.95, 0.35, 0.3), vec3f(0.3, 0.85, 0.5), abs(total - 1.0) < 1e-3);
        }
    }
    if (uv.y > 0.58 && uv.y < 0.75) {
        col = vec3f(0.14, 0.16, 0.26);
        if (uv.x < planted) { col = vec3f(0.35, 0.6, 1.0); }
    }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
```{.wgsl .solution}
#storage data array<f32>
#storage stats array<f32>
#workgroup_count fill 64 1 1
#workgroup_count reduce_max 64 1 1
#workgroup_count finish_max 1 1 1
#workgroup_count reduce_sum 64 1 1
#workgroup_count finish_sum 1 1 1
#workgroup_count normalize 64 1 1
#workgroup_count check_reduce 64 1 1
#workgroup_count check_finish 1 1 1

const N = 16384u;
const PLANT = 5000u;

const MAX_PARTIALS = 0u;
const MAX = 64u;
const SUM_PARTIALS = 128u;
const SUM = 192u;
const CHECK_PARTIALS = 256u;
const CHECK = 320u;

var<workgroup> scratch: array<f32, 256>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn tree_sum(lid: u32, width: u32) {
    for (var s = width / 2u; s > 0u; s >>= 1u) {
        if (lid < s) { scratch[lid] += scratch[lid + s]; }
        workgroupBarrier();
    }
}

// Same tree, different combining operation
fn tree_max(lid: u32, width: u32) {
    for (var s = width / 2u; s > 0u; s >>= 1u) {
        if (lid < s) { scratch[lid] = max(scratch[lid], scratch[lid + s]); }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    let r = f32(pcg(gid.x) >> 8u) / 16777216.0;
    data[gid.x] = select(r * 20.0 - 10.0, 20.0, gid.x == PLANT);
}

// ---- 1. max ----
@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3u,
              @builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
    scratch[lid.x] = data[gid.x];
    workgroupBarrier();
    tree_max(lid.x, 256u);
    if (lid.x == 0u) { stats[MAX_PARTIALS + wid.x] = scratch[0]; }
}

@compute @workgroup_size(64)
fn finish_max(@builtin(local_invocation_id) lid: vec3u) {
    scratch[lid.x] = stats[MAX_PARTIALS + lid.x];
    workgroupBarrier();
    tree_max(lid.x, 64u);
    if (lid.x == 0u) { stats[MAX] = scratch[0]; }
}

// ---- 2. sum of exp(x - max) ----
@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) gid: vec3u,
              @builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
    scratch[lid.x] = exp(data[gid.x] - stats[MAX]);
    workgroupBarrier();
    tree_sum(lid.x, 256u);
    if (lid.x == 0u) { stats[SUM_PARTIALS + wid.x] = scratch[0]; }
}

@compute @workgroup_size(64)
fn finish_sum(@builtin(local_invocation_id) lid: vec3u) {
    scratch[lid.x] = stats[SUM_PARTIALS + lid.x];
    workgroupBarrier();
    tree_sum(lid.x, 64u);
    if (lid.x == 0u) { stats[SUM] = scratch[0]; }
}

// ---- 3. normalize in place ----
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) gid: vec3u) {
    data[gid.x] = exp(data[gid.x] - stats[MAX]) / stats[SUM];
}

// ---------- check ----------
@compute @workgroup_size(256)
fn check_reduce(@builtin(global_invocation_id) gid: vec3u,
                @builtin(local_invocation_id) lid: vec3u,
                @builtin(workgroup_id) wid: vec3u) {
    scratch[lid.x] = data[gid.x];
    workgroupBarrier();
    tree_sum(lid.x, 256u);
    if (lid.x == 0u) { stats[CHECK_PARTIALS + wid.x] = scratch[0]; }
}

@compute @workgroup_size(64)
fn check_finish(@builtin(local_invocation_id) lid: vec3u) {
    scratch[lid.x] = stats[CHECK_PARTIALS + lid.x];
    workgroupBarrier();
    tree_sum(lid.x, 64u);
    if (lid.x == 0u) { stats[CHECK] = scratch[0]; }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);

    let total = stats[CHECK];
    let planted = data[PLANT];
    var col = vec3f(0.06, 0.07, 0.12);
    if (uv.y > 0.25 && uv.y < 0.42) {
        col = vec3f(0.14, 0.16, 0.26);
        if (uv.x < total) {
            col = select(vec3f(0.95, 0.35, 0.3), vec3f(0.3, 0.85, 0.5), abs(total - 1.0) < 1e-3);
        }
    }
    if (uv.y > 0.58 && uv.y < 0.75) {
        col = vec3f(0.14, 0.16, 0.26);
        if (uv.x < planted) { col = vec3f(0.35, 0.6, 1.0); }
    }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
:::

Why is the planted element's probability about 0.96? Its term is `e^0 = 1` after subtracting the max. The other 16,383 values are spread evenly over [−10, 10), so each contributes on average about $e^{-20} \times 1101 \approx 2.3 \times 10^{-6}$, for a combined total near 0.037. That gives 1 / 1.037 ≈ 0.964. Checking the GPU's answer against a number you can work out by hand is the habit that matters most for ML kernels, where a slightly wrong result produces no visible symptom.

# [Part VII]{.part} Matrix multiply

## [17]{.num} Matrix multiply

Matrix multiplication is the operation that most of machine learning runs on. For square `N × N` matrices, `C = A × B` means

$$C_{ij} = \sum_{k=0}^{N-1} A_{ik} \, B_{kj}.$$

The straightforward GPU version gives **one thread per output element**: thread `(j, i)` loops over `k` and computes `C[i][j]`. That's a 2D dispatch: `@workgroup_size(16, 16)` with `#workgroup_count matmul 16 16 1` covers a 256 × 256 output.

### 2D data in flat buffers

Storage buffers are one-dimensional, so a matrix is stored **row by row**. Element `(row, col)` of an `N`-wide matrix lives at index `row * N + col`. With only two buffers, pack `A` and `B` into one buffer at different offsets: `A` starts at 0 and `B` starts at `N * N`.

### Conversions, again

WGSL's conversion functions match the shape of what they produce. `f32(x)` converts a scalar; for vectors you use the vector constructor, which converts every component:

```wgsl
let ij = vec2u(uv * vec2f(DIMS));   // not u32(...): that only takes a scalar
```

Converting a float to `u32` truncates toward zero, so for non-negative values it's the same as `floor`.

### Knowing it's right

Fill `A` with the **identity matrix** (1 on the diagonal, 0 elsewhere) and `B` with random values. Then `C` must equal `B` exactly, since each output is one `1 × b` plus zeros. The display shows `C` on the left, and on the right a map of where `C` matches `B`: green where it does, red where it doesn't.

::: task
Write `matmul`. With the starter's placeholder, `C` is all zeros, so the left panel is black and the right one is red. When it's correct, the left panel looks like static and the right one is solid green.
:::

::: {.toy #ex17 data-label="Exercise 17" data-title="Matrix multiply"}
```wgsl
#storage inputs array<f32>
#storage outputs array<f32>
#workgroup_count fill 16 16 1
#workgroup_count matmul 16 16 1

const N = 256u;          // matrices are N x N; 16 x 16 workgroups of 16 x 16
const A = 0u;            // A starts here in `inputs`
const B = N * N;         // B starts here in `inputs`

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A = identity, B = random. gid.y is the row, gid.x the column.
@compute @workgroup_size(16, 16)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * N + gid.x;
    inputs[A + index] = select(0.0, 1.0, gid.x == gid.y);
    inputs[B + index] = f32(pcg(index) >> 8u) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.y;
    let col = gid.x;
    // TODO: outputs[row * N + col] = sum over k of A[row][k] * B[k][col]
    outputs[row * N + col] = 0.0;
}

// ---------- display: C on the left, C == B check on the right ----------
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let side = min(size.x / 2u, size.y);
    let panel = id.x / side;
    let local = vec2u(id.x % side, id.y);
    if (panel > 1u || local.y >= side) {
        textureStore(screen, id.xy, vec4f(0.004, 0.004, 0.008, 1.));
        return;
    }
    let ij = min(local * N / side, vec2u(N - 1u));    // (column, row)
    let index = ij.y * N + ij.x;
    let c = outputs[index];
    let ok = abs(c - inputs[B + index]) < 1e-5;

    var col = vec3f(c);
    if (panel == 1u) { col = select(vec3f(0.9, 0.2, 0.15), vec3f(0.2, 0.75, 0.4), ok); }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
```{.wgsl .solution}
#storage inputs array<f32>
#storage outputs array<f32>
#workgroup_count fill 16 16 1
#workgroup_count matmul 16 16 1

const N = 256u;
const A = 0u;
const B = N * N;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(16, 16)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * N + gid.x;
    inputs[A + index] = select(0.0, 1.0, gid.x == gid.y);
    inputs[B + index] = f32(pcg(index) >> 8u) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;                               // accumulate in a register
    for (var k = 0u; k < N; k++) {
        acc += inputs[A + row * N + k] * inputs[B + k * N + col];
    }
    outputs[row * N + col] = acc;                // one write at the end
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let side = min(size.x / 2u, size.y);
    let panel = id.x / side;
    let local = vec2u(id.x % side, id.y);
    if (panel > 1u || local.y >= side) {
        textureStore(screen, id.xy, vec4f(0.004, 0.004, 0.008, 1.));
        return;
    }
    let ij = min(local * N / side, vec2u(N - 1u));
    let index = ij.y * N + ij.x;
    let c = outputs[index];
    let ok = abs(c - inputs[B + index]) < 1e-5;

    var col = vec3f(c);
    if (panel == 1u) { col = select(vec3f(0.9, 0.2, 0.15), vec3f(0.2, 0.75, 0.4), ok); }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
:::

The solution accumulates into a local variable and writes the output once. Writing `outputs[index] += ...` inside the loop also gives the right answer, but it reads and writes global memory 256 times instead of once. On a GPU, that difference is usually what separates fast code from slow code.

### Stretch: tiling

The naive kernel does 2 reads for every multiply-add. Every thread in a row of the output reads the *same* row of `A` from global memory, and every thread in a column reads the same column of `B`: each value is fetched 256 times. GPUs can do arithmetic far faster than they can fetch from memory, so the naive kernel spends most of its time waiting.

**Tiling** fixes this with workgroup memory. Each 16 × 16 workgroup walks along `k` in steps of 16. At each step:

1. Each thread loads **one** element of a 16 × 16 tile of `A` and one of `B` into shared arrays. Barrier.
2. Each thread does 16 multiply-adds reading only from the shared tiles. Barrier, so no one overwrites a tile still in use.

Same math, same answer, but each value now comes from global memory 16 times less often.

To **measure** it, the stretch version runs `matmul` 16 times per frame with `#dispatch_count`. Watch the fps counter in the corner, then flip `TILED` to `true` once your tiled version works. On a fast GPU you may need to raise the dispatch count before the naive version slows down at all.

::: {.toy #ex17b data-label="Stretch" data-title="Tiled matrix multiply"}
```wgsl
#storage inputs array<f32>
#storage outputs array<f32>
#workgroup_count fill 16 16 1
#workgroup_count matmul 16 16 1
#dispatch_count matmul 16

const N = 256u;
const TILE = 16u;
const A = 0u;
const B = N * N;
const TILED = false;     // flip once matmul_tiled works, and compare fps

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(16, 16)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * N + gid.x;
    inputs[A + index] = select(0.0, 1.0, gid.x == gid.y);
    inputs[B + index] = f32(pcg(index) >> 8u) / 16777216.0;
}

fn matmul_naive(row: u32, col: u32) -> f32 {
    var acc = 0.0;
    for (var k = 0u; k < N; k++) {
        acc += inputs[A + row * N + k] * inputs[B + k * N + col];
    }
    return acc;
}

// Every thread in the workgroup must call this (it contains barriers)
fn matmul_tiled(row: u32, col: u32, lid: vec2u) -> f32 {
    var acc = 0.0;
    for (var t = 0u; t < N; t += TILE) {
        // TODO: load tileA[lid.y][lid.x] = A[row][t + lid.x]
        //       and   tileB[lid.y][lid.x] = B[t + lid.y][col], then barrier
        // TODO: acc += tileA[lid.y][k] * tileB[k][lid.x] for k in 0..TILE
        // TODO: barrier before the next tile overwrites these
    }
    return acc;
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u) {
    var acc = 0.0;
    if (TILED) { acc = matmul_tiled(gid.y, gid.x, lid.xy); }
    else { acc = matmul_naive(gid.y, gid.x); }
    outputs[gid.y * N + gid.x] = acc;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let side = min(size.x / 2u, size.y);
    let panel = id.x / side;
    let local = vec2u(id.x % side, id.y);
    if (panel > 1u || local.y >= side) {
        textureStore(screen, id.xy, vec4f(0.004, 0.004, 0.008, 1.));
        return;
    }
    let ij = min(local * N / side, vec2u(N - 1u));
    let index = ij.y * N + ij.x;
    let c = outputs[index];
    let ok = abs(c - inputs[B + index]) < 1e-5;

    var col = vec3f(c);
    if (panel == 1u) { col = select(vec3f(0.9, 0.2, 0.15), vec3f(0.2, 0.75, 0.4), ok); }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
```{.wgsl .solution}
#storage inputs array<f32>
#storage outputs array<f32>
#workgroup_count fill 16 16 1
#workgroup_count matmul 16 16 1
#dispatch_count matmul 16

const N = 256u;
const TILE = 16u;
const A = 0u;
const B = N * N;
const TILED = true;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(16, 16)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * N + gid.x;
    inputs[A + index] = select(0.0, 1.0, gid.x == gid.y);
    inputs[B + index] = f32(pcg(index) >> 8u) / 16777216.0;
}

fn matmul_naive(row: u32, col: u32) -> f32 {
    var acc = 0.0;
    for (var k = 0u; k < N; k++) {
        acc += inputs[A + row * N + k] * inputs[B + k * N + col];
    }
    return acc;
}

fn matmul_tiled(row: u32, col: u32, lid: vec2u) -> f32 {
    var acc = 0.0;
    for (var t = 0u; t < N; t += TILE) {
        // Each thread fetches one element of each tile
        tileA[lid.y][lid.x] = inputs[A + row * N + (t + lid.x)];
        tileB[lid.y][lid.x] = inputs[B + (t + lid.y) * N + col];
        workgroupBarrier();

        // 16 multiply-adds from fast shared memory
        for (var k = 0u; k < TILE; k++) {
            acc += tileA[lid.y][k] * tileB[k][lid.x];
        }
        workgroupBarrier();
    }
    return acc;
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u) {
    var acc = 0.0;
    if (TILED) { acc = matmul_tiled(gid.y, gid.x, lid.xy); }
    else { acc = matmul_naive(gid.y, gid.x); }
    outputs[gid.y * N + gid.x] = acc;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let side = min(size.x / 2u, size.y);
    let panel = id.x / side;
    let local = vec2u(id.x % side, id.y);
    if (panel > 1u || local.y >= side) {
        textureStore(screen, id.xy, vec4f(0.004, 0.004, 0.008, 1.));
        return;
    }
    let ij = min(local * N / side, vec2u(N - 1u));
    let index = ij.y * N + ij.x;
    let c = outputs[index];
    let ok = abs(c - inputs[B + index]) < 1e-5;

    var col = vec3f(c);
    if (panel == 1u) { col = select(vec3f(0.9, 0.2, 0.15), vec3f(0.2, 0.75, 0.4), ok); }
    textureStore(screen, id.xy, vec4f(pow(col, vec3f(2.2)), 1.));
}
```
:::

With matmul, softmax and one more reduction (layer norm), you have the kernels for the forward pass of a small transformer.

# [After this]{.part} Where to go next

## Debugging without print

- **Turn values into colors.** `col = vec3f(length(p))`, `col = vec3f(fract(x))`, `col = vec3f(f32(lid.x) / 256.0)`. If you can see it, you can reason about it.
- **Draw numbers as bars.** Every reduction exercise above does this. A bar that's exactly 1/64 full, or 1/256 full, usually points straight at the bug.
- **Use inputs with known answers.** All ones, the identity matrix, a planted maximum. Random data hides bugs; designed data exposes them.
- **Flicker means a race.** If a result changes from frame to frame when the inputs don't, some thread is reading memory another thread hasn't finished writing. Look for a missing barrier.

## WGSL directives

Real WGSL has a few **directives** at the top of a file. They're different from this playground's `#` lines, which are rewritten before WGSL sees the code.

```wgsl
enable f16;                                      // optional hardware feature
requires readonly_and_readwrite_storage_textures; // newer language feature
diagnostic(off, derivative_uniformity);          // tune a compiler check
```

- **`enable`** opts in to optional features. The two most relevant to compute are `f16` (half-precision floats, half the memory, often faster, and standard for ML inference) and `subgroups`, which gives access to the threads that run in lockstep. `subgroupAdd` sums a value across a subgroup with no shared memory and no barriers, which can speed up reductions a lot. Both must also be requested when creating the device in JavaScript.
- **`requires`** declares language features that some browsers may not have yet, so older implementations fail with a clear error.
- **`diagnostic`** changes how the compiler reports a class of issue, most often uniformity analysis. Use it sparingly: the check usually exists because the code really could misbehave, and it's the same class of bug as a barrier inside an `if`.

Attributes such as `@compute`, `@workgroup_size` and `@builtin` are not directives. They attach to one function or variable.

## Your own WebGPU program

This playground hides the host-side setup: requesting a device, creating buffers and bind groups, building pipelines, and reading results back. That setup is where much of the real-world difficulty lives, and it's the natural next step. A good first program is deliberately boring: add two arrays on the GPU, read the result back, and compare it with the same sum computed in JavaScript. It touches every piece of plumbing with nothing visual to distract you. Then port your softmax and matmul over, and check them against CPU results.

For reference as you go: [WebGPU Fundamentals](https://webgpufundamentals.org) explains the host API step by step, the [WGSL specification](https://www.w3.org/TR/WGSL/) is the final word on the language, and [compute.toys](https://compute.toys) has hundreds of shaders written in the same style as this page.
