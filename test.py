import json, sys, subprocess, time, pathlib
from playwright.sync_api import sync_playwright

root = pathlib.Path(__file__).parent
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shots")
out.mkdir(parents=True, exist_ok=True)
srv = subprocess.Popen(["python3", "-m", "http.server", "8766"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

JS_RUN = """async ([i, which]) => {
  const t = gpuBook.toys[i];
  const code = which === 'solution' ? t.solution : t.starter;
  if (code == null) return null;
  t.editor.value = code;
  t.checkFrames = 0;
  await t.compile();
  const msgs = [...t.el.querySelectorAll('.msg')].map(m => m.textContent);
  if (t.state === 'error') return {id: t.id, which, error: true, msgs};
  t.restart();
  const d = gpuBook.G.device;
  d.pushErrorScope('validation');
  for (let f = 0; f < 3; f++) t.frame(1/60);
  await d.queue.onSubmittedWorkDone();
  const verr = await d.popErrorScope();
  let storage = null;
  if (t.pp.storage.length) storage = [await t.readStorage(0, 8), await t.readStorage(1, 330)];
  return {id: t.id, which, error: !!verr, msgs: verr ? [verr.message] : msgs, dispatch: t.dispatchEl.textContent, storage};
}"""

with sync_playwright() as p:
    b = p.chromium.launch(args=["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
    pg.goto("http://localhost:8766/thread-by-thread.html?nopresent")
    pg.wait_for_function("window.gpuBook && gpuBook.G.device", timeout=30000)
    pg.evaluate("gpuBook.setAutoRun(false)")
    n = pg.evaluate("gpuBook.toys.length")
    results = []
    for i in range(1, n):   # 0 is the hero
        for which in ("starter", "solution"):
            r = pg.evaluate(JS_RUN, [i, which])
            if r is None: continue
            results.append(r)
            flag = "FAIL" if r["error"] else "ok  "
            extra = ""
            if r.get("storage"):
                s1 = r["storage"][1]
                extra = f" s0[:3]={[round(x,4) for x in r['storage'][0][:3]]} s1[64]={s1[64]:.4f} s1[192]={s1[192]:.4f} s1[320]={s1[320]:.6f}"
            print(flag, r["id"], which, "|", (r.get("dispatch") or "")[:90], extra)
            if r["error"]:
                for m in r["msgs"]: print("      ", m[:300])
            if which == "solution" or r["error"]:
                import base64
                url = pg.evaluate(f"(async()=>{{const t=gpuBook.toys[{i}]; t.frame(1/60); return await t.readScreen()}})()")
                (out / f"{r['id']}-{which}.png").write_bytes(base64.b64decode(url.split(",")[1]))
    print("\n".join(l for l in logs if "error" in l.lower() or "PAGEERROR" in l)[:3000])
    b.close()
srv.terminate()
