// Visual QA for the static site in a real (headless) Chrome, over the DevTools protocol.
//
//   python -m http.server 8765 --bind 127.0.0.1 --directory site/dist      (in another terminal)
//   node site/visual_qa.mjs [--base http://127.0.0.1:8765] [--out site/qa]
//
// For each language x viewport it: loads the page; waits for fonts and for every <img> to finish loading
// AND decoding; scrolls the full page; counts expected vs loaded images; records failed requests
// (HTTP >= 400 or network errors), console errors and horizontal overflow; then captures the full page
// in segments of at most SEG px (single very tall captures exceed Chrome's texture limit and come back
// with blank regions). Writes <out>/visual_qa.json and PNGs. Exit code 1 on any failure.
// Blank captures are checked separately by tests/test_site.py (pixel variance of every PNG listed here).
// Node >= 22 (global WebSocket). Uses a throwaway browser profile, never the user's own.
import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const BASE = arg("--base", "http://127.0.0.1:8765").replace(/\/$/, "");
const OUT = arg("--out", "site/qa");
const SEG = 4000;
const PAGES = [["en", "/"], ["zh", "/zh-tw/"], ["print1", "/print/one-page/"]];
const VIEWPORTS = { desktop: [1440, 900, 1, false], tablet: [820, 1180, 1, false], mobile: [375, 812, 2, true] };
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium"]
  .find((p) => existsSync(p));
if (!CHROME) { console.error("no Chrome/Edge found"); process.exit(2); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(OUT, { recursive: true });
const prof = mkdtempSync(join(tmpdir(), "ivad-qa-"));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--hide-scrollbars", `--user-data-dir=${prof}`, `--remote-debugging-port=${port}`, "about:blank"], { stdio: "ignore" });

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const t = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const p = t.find((x) => x.type === "page");
      if (p) return p.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error("Chrome DevTools endpoint did not come up");
}

const ws = new WebSocket(await target());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let seq = 0;
const pending = new Map();
const listeners = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); }
  else if (m.method) listeners.forEach((f) => f(m));
});
const send = (method, params = {}) => new Promise((res, rej) => { const id = ++seq; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result.value;

await send("Page.enable"); await send("Network.enable"); await send("Runtime.enable"); await send("Log.enable");
await send("Network.setCacheDisabled", { cacheDisabled: true });
// Authored smooth scrolling can still be in flight when mobile screenshots are
// captured. Disable animation in QA and use instant scrolling for deterministic captures.
await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
let failures = [], consoleErrors = [];
const reqUrl = new Map();
listeners.push((m) => {
  if (m.method === "Network.requestWillBeSent") reqUrl.set(m.params.requestId, m.params.request.url);
  if (m.method === "Network.responseReceived" && m.params.response.status >= 400) failures.push(`${m.params.response.status} ${m.params.response.url}`);
  if (m.method === "Network.loadingFailed" && !m.params.canceled) failures.push(`${m.params.errorText} ${reqUrl.get(m.params.requestId) || "?"}`);
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") consoleErrors.push(m.params.args.map((a) => a.value).join(" "));
  if (m.method === "Runtime.exceptionThrown") consoleErrors.push(m.params.exceptionDetails.text);
  if (m.method === "Log.entryAdded" && m.params.entry.level === "error") consoleErrors.push(m.params.entry.text + " " + (m.params.entry.url || ""));
});

const report = [];
let bad = 0;
for (const [lang, path] of PAGES) {
  for (const [vp, [w, h, dpr, mobile]] of Object.entries(VIEWPORTS)) {
    if (lang.startsWith("print") && vp !== "desktop") continue;
    failures = []; consoleErrors = [];
    await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: dpr, mobile });
    const loaded = new Promise((r) => { const f = (m) => { if (m.method === "Page.loadEventFired") { listeners.splice(listeners.indexOf(f), 1); r(); } }; listeners.push(f); });
    await send("Page.navigate", { url: BASE + path });
    await loaded;
    // scroll the whole page (would trigger any lazy loading), then wait for every image to load and decode
    await evaluate(`(async()=>{for(let y=0;y<document.documentElement.scrollHeight;y+=innerHeight){scrollTo({top:y,left:0,behavior:'instant'});await new Promise(r=>setTimeout(r,30));}scrollTo({top:0,left:0,behavior:'instant'});await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));})()`);
    const info = await evaluate(`(async()=>{
      await document.fonts.ready;
      const imgs=[...document.images];
      await Promise.all(imgs.map(i=>i.complete?i.decode().catch(()=>null):new Promise(r=>{i.onload=i.onerror=r}).then(()=>i.decode().catch(()=>null))));
      const st=i=>getComputedStyle(i);
      return {title:document.title, lang:document.documentElement.lang,
        scrollY, visualViewportOffsetTop:visualViewport?.offsetTop ?? 0,
        expected:imgs.length,
        loaded:imgs.filter(i=>i.complete&&i.naturalWidth>0).length,
        visible:imgs.filter(i=>st(i).display!=='none'&&st(i).visibility!=='hidden'&&i.getBoundingClientRect().width>0&&i.getBoundingClientRect().height>0).length,
        images:imgs.map(i=>({src:i.getAttribute('src'),resolved:i.currentSrc,natural:[i.naturalWidth,i.naturalHeight],box:[Math.round(i.getBoundingClientRect().width),Math.round(i.getBoundingClientRect().height)],alt:!!i.alt})),
        svg:document.querySelectorAll('svg').length,
        overflowX:document.documentElement.scrollWidth>document.documentElement.clientWidth,
        scrollWidth:document.documentElement.scrollWidth, clientWidth:document.documentElement.clientWidth,
        height:document.documentElement.scrollHeight,
        brokenLinks:[...document.querySelectorAll('a[href^="#"]')].map(a=>a.getAttribute('href')).filter(h=>h.length>1&&!document.getElementById(h.slice(1))),
        placeholders:(document.body.innerText.match(/\\{\\{|\\}\\}|\\{%|TODO|TBD|lorem ipsum|undefined|NaN|\\[object/g)||[])};
    })()`);
    // same-site links to other files (PDF, language switch) must resolve
    const hrefs = await evaluate(`[...document.querySelectorAll('a[href]')].map(a=>a.href).filter(h=>h.startsWith(location.origin)&&!h.includes('#'))`);
    const linkStatus = {};
    for (const u of [...new Set(hrefs)]) linkStatus[u] = (await fetch(u, { method: "HEAD" })).status;
    const badLinks = Object.entries(linkStatus).filter(([, s]) => s >= 400).map(([u, s]) => `${s} ${u}`);
    // first screen BEFORE any beyond-viewport capture, after two animation frames (a capture taken right
    // after a beyond-viewport capture under mobile emulation came back blank)
    const frames = () => evaluate("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>r(1))))");
    await frames();
    const first = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: w, height: h, scale: 1 } });
    const ff = join(OUT, `${lang}_${vp}_first.png`);
    writeFileSync(ff, Buffer.from(first.data, "base64"));
    const metrics = await send("Page.getLayoutMetrics");
    const H = Math.ceil(metrics.cssContentSize.height), W = Math.ceil(metrics.cssContentSize.width);
    const shots = [];
    for (let y = 0, k = 0; y < H; y += SEG, k++) {
      const hh = Math.min(SEG, H - y);
      // A tiny final slice can contain only legitimate footer padding. Overlap the
      // previous segment so pixel-variance QA can distinguish it from a failed capture.
      const clipY = hh < 120 ? Math.max(0, H - SEG) : y;
      const clipH = hh < 120 ? Math.min(SEG, H) : hh;
      await frames();
      const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
        clip: { x: 0, y: clipY, width: W, height: clipH, scale: 1 } });
      const f = join(OUT, `${lang}_${vp}_full_${String(k).padStart(2, "0")}.png`);
      writeFileSync(f, Buffer.from(r.data, "base64"));
      shots.push(f.replaceAll("\\", "/"));
    }
    const ok = info.scrollY === 0 && info.visualViewportOffsetTop === 0 &&
      info.expected === info.loaded && info.loaded === info.visible && !info.overflowX && !failures.length &&
      !consoleErrors.length && !info.brokenLinks.length && !info.placeholders.length && !badLinks.length;
    if (!ok) bad++;
    report.push({ page: lang, path, viewport: vp, size: [w, h, dpr], ok, ...info, failedRequests: failures,
      consoleErrors, badLinks, linksChecked: Object.keys(linkStatus).length, fullPageSegments: shots, firstScreen: ff.replaceAll("\\", "/") });
    console.log(`${ok ? "PASS" : "FAIL"} ${lang} ${vp}: images ${info.loaded}/${info.expected} loaded, ${info.visible} visible; ` +
      `height ${H}px in ${shots.length} segment(s); overflowX ${info.overflowX}; failed requests ${failures.length}; ` +
      `console errors ${consoleErrors.length}; broken anchors ${info.brokenLinks.length}; bad links ${badLinks.length}/${Object.keys(linkStatus).length}; placeholders ${info.placeholders.length}`);
  }
}
writeFileSync(join(OUT, "visual_qa.json"), JSON.stringify({ base: BASE, chrome: CHROME.split("/").pop(), generated: new Date().toISOString(), pages: report }, null, 1));
ws.close(); chrome.kill();
await sleep(500);
try { rmSync(prof, { recursive: true, force: true }); } catch { /* profile still locked: harmless temp dir */ }
process.exit(bad ? 1 : 0);
