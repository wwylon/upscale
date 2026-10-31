/**
 * Bridges the design-doc UI (DCLogic) to the FastAPI job API.
 * Loaded after the design page; call sites invoke window.UpscaleBackend.*.
 */
(function () {
  const API = "";

  /** Design sliders are often 0–100; API expects 0–1. */
  function _unit(v, fallback) {
    if (v == null || v === "") return fallback;
    const n = +v;
    if (Number.isNaN(n)) return fallback;
    return n > 1 ? n / 100 : n;
  }

  async function postJob(file, settings) {
    const fd = new FormData();
    fd.append("file", file, file.name || "image.png");
    fd.append("settings", JSON.stringify(settings));
    const res = await fetch(API + "/api/jobs", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  function listenEvents(jobId, onEvent) {
    const es = new EventSource(API + "/api/jobs/" + jobId + "/events");
    es.onmessage = (ev) => {
      try {
        onEvent(JSON.parse(ev.data));
      } catch (_) {}
    };
    es.onerror = () => es.close();
    return es;
  }

  function trackEs(comp, es) {
    if (!comp) return es;
    if (!comp._extraEs) comp._extraEs = [];
    comp._extraEs.push(es);
    return es;
  }

  function closeAllEs(comp) {
    if (!comp) return;
    if (comp._jobEs) {
      try {
        comp._jobEs.close();
      } catch (_) {}
      comp._jobEs = null;
    }
    const extras = comp._extraEs || [];
    comp._extraEs = [];
    for (const es of extras) {
      try {
        es.close();
      } catch (_) {}
    }
  }

  function revokeUrl(u) {
    if (u && typeof u === "string" && u.indexOf("blob:") === 0) {
      try {
        URL.revokeObjectURL(u);
      } catch (_) {}
    }
  }

  function revokeStateBlobs(comp, keys) {
    if (!comp || !comp.state) return;
    const s = comp.state;
    (keys || ["displayUrl", "urlAfter", "resultUrl", "url", "urlBefore"]).forEach((k) => {
      revokeUrl(s[k]);
    });
  }

  function settingsFromAdvanced(comp) {
    const s = comp.state;
    const k = typeof comp.curScale === "function" ? comp.curScale() : +s.scale || 4;
    const filt = s.filt || {};
    const f = (key) => (filt[key] && filt[key].on ? (filt[key].v || 0) / 100 : 0);
    const settings = {
      scale: s.mode === "scale" ? k : null,
      target_size: s.mode === "target" ? +s.target || null : null,
      width: s.mode === "dims" ? +s.dw || null : null,
      height: s.mode === "dims" ? +s.dh || null : null,
      model: s.model,
      format: s.fmt || "auto",
      quality: s.q || 90,
      tiff_compression: s.tiff || "lzw",
      // Advanced UI no longer exposes strength/blend; keep neutral defaults.
      strength: 1,
      denoise: _unit(s.denoise, 0.5),
      blend_model: null,
      blend: 0.5,
      device: s.device || "auto",
      half: !!s.half,
      tile: s.tile === "auto" ? "auto" : +s.tile || "auto",
      overlap: +s.overlap || 32,
      filters: {
        pre_denoise: f("denoise"),
        jpeg_clean: f("jpeg"),
        deblur: f("deblur"),
        edge: f("edge"),
        grain: f("grain"),
        color_contrast: f("color"),
        halo_reduce: f("halo"),
        seamless: !!s.seamless,
        face_restore: 0,
      },
      mid_pass_previews: true,
      fast: !!s.fast,
      sr_alpha: !!s.srAlpha,
    };
    if (s.namePattern) {
      settings.name_pattern = {
        pattern: s.namePattern,
        on_exists: s.onExists || "number",
      };
    }
    if (s.method === "classical" && s.algorithm) {
      settings.algorithm = s.algorithm;
      if (s.algorithm === "vectorize_scale") {
        const n = (v, d) => {
          const x = Number(v);
          return Number.isFinite(x) ? x : d;
        };
        settings.vectorize = {
          engine: s.vecEngine === "autotrace" ? "autotrace" : "vtracer",
          color_precision: Math.max(1, Math.min(8, n(s.vecColors, 6))),
          filter_speckle: Math.max(0, Math.min(64, n(s.vecSpeckle, 4))),
          corner_threshold: Math.max(0, Math.min(180, n(s.vecCorners, 60))),
          mode: s.vecMode === "polygon" ? "polygon" : "spline",
          max_trace_edge: Math.max(256, Math.min(4096, n(s.vecTrace, 1024))),
        };
      }
    }
    if (s.test && s.w) {
      const size = +s.cropSize || 128;
      const cx = s.cx != null ? s.cx : 0.5;
      const cy = s.cy != null ? s.cy : 0.5;
      const x = Math.max(0, Math.round(cx * s.w - size / 2));
      const y = Math.max(0, Math.round(cy * s.h - size / 2));
      settings.crop = { x, y, w: Math.min(size, s.w - x), h: Math.min(size, s.h - y) };
    }
    return settings;
  }

  function settingsFromSimple(comp) {
    const s = comp.state;
    const map = {
      photo: "RealESRGAN_x4plus",
      art: "RealESRGAN_x4plus_anime_6B",
      old: "realesr-general-wdn-x4v3",
    };
    const kind = s.type || s.kind || "photo";
    const scale = +(s.scale || 4);
    const settings = {
      scale,
      model: map[kind] || map.photo,
      format: "auto",
      device: "auto",
      mid_pass_previews: true,
    };
    if (s.preview && s.w) {
      const size = 512;
      const x = Math.max(0, Math.round(s.w / 2 - size / 2));
      const y = Math.max(0, Math.round(s.h / 2 - size / 2));
      settings.crop = {
        x,
        y,
        w: Math.min(size, s.w - x),
        h: Math.min(size, s.h - y),
      };
    }
    return settings;
  }

  function applyModelStatus(models) {
    const list = Array.isArray(models) ? models : [];
    window.__UPSCALE_MODELS__ = list;
    const installed = list.filter((m) => m.status === "installed").map((m) => m.name);
    const failed = {};
    list
      .filter((m) => m.status === "failed")
      .forEach((m) => {
        failed[m.name] = "failed to load";
      });
    window.__UPSCALE_DOWNLOADED__ = installed;
    window.__UPSCALE_MODEL_ERR__ = failed;

    // Patch any live DCLogic instance that tracks DOWNLOADED / ERR / models table
    const roots = document.querySelectorAll("x-dc");
    roots.forEach((el) => {
      const inst = el._dc || el.__dc || el._component || el.component;
      if (!inst) return;
      if (Array.isArray(inst.DOWNLOADED)) {
        inst.DOWNLOADED = installed.slice();
        inst.ERR = Object.assign({}, failed);
        if (typeof inst.forceUpdate === "function") inst.forceUpdate();
        else if (typeof inst.setState === "function") inst.setState({});
      }
      if (Array.isArray(inst.state && inst.state.models) && inst.state.models[0] && Array.isArray(inst.state.models[0])) {
        const byName = Object.fromEntries(list.map((m) => [m.name, m]));
        const next = inst.state.models.map((row) => {
          const info = byName[row[0]];
          if (!info) return row;
          const st =
            info.status === "installed" ? "ok" : info.status === "failed" ? "err" : "dl";
          const copy = row.slice();
          copy[6] = st;
          if (info.size_bytes) copy[4] = Math.max(1, Math.round(info.size_bytes / 1e6));
          return copy;
        });
        // Append API-only models not in the design table
        list.forEach((m) => {
          if (!next.some((r) => r[0] === m.name)) {
            next.push([
              m.name,
              m.description || "",
              m.tags && m.tags[0] ? m.tags[0] : "Custom",
              m.scale || 4,
              m.size_bytes ? Math.max(1, Math.round(m.size_bytes / 1e6)) : 0,
              m.source === "built-in" ? "Built-in" : "External",
              m.status === "installed" ? "ok" : m.status === "failed" ? "err" : "dl",
            ]);
          }
        });
        inst.setState({ models: next });
      }
    });
  }

  async function hydrateSystem(comp) {
    try {
      const s = (comp && comp.state) || {};
      const qs = new URLSearchParams();
      if (s.model && s.model !== "__custom") qs.set("model", s.model);
      if (s.tile && s.tile !== "auto") qs.set("tile", String(s.tile));
      if (s.half) qs.set("half", "true");
      if (s.device && s.device !== "auto") qs.set("device", s.device);
      const sysUrl = API + "/api/system" + (qs.toString() ? "?" + qs.toString() : "");
      const sys = await (await fetch(sysUrl)).json();
      const models = await (await fetch(API + "/api/models")).json();
      window.__UPSCALE_SYS__ = sys;
      try {
        localStorage.setItem(
          "upscale.sys",
          JSON.stringify({ at: Date.now(), sys })
        );
      } catch (_) {}
      applyModelStatus(models);
      let algorithms = null;
      try {
        const al = await (await fetch(API + "/api/algorithms")).json();
        if (al.items && al.items.length) {
          algorithms = al.items.map((it) => ({
            name: it.name,
            label: it.label || it.name.replace(/_/g, " "),
            group: it.group || "plugin",
            description: it.description || "",
          }));
        } else {
          algorithms = (al.algorithms || []).map((n) => ({
            name: n,
            label: n.replace(/_/g, " "),
            group: "builtin",
            description: "",
          }));
        }
      } catch (_) {}
      let presets = null;
      try {
        presets = await (await fetch(API + "/api/presets")).json();
      } catch (_) {}
      const label =
        (sys.gpu_name || sys.device || "CPU") +
        (sys.vram_free_mb != null ? " · " + Math.round(sys.vram_free_mb) + " MB free VRAM" : "");
      const patch = { deviceLabel: label };
      if (algorithms && algorithms.length) patch.apiAlgorithms = algorithms;
      if (presets) patch.apiPresets = presets;
      if (sys.backend) patch.backend = sys.backend;
      if (sys.backend === "onnx") {
        const missing = (models || []).filter(
          (m) => m.status === "not_downloaded" && /span|RealESRGAN_x4plus$/.test(m.name)
        );
        if (missing.length) {
          patch.onnxHint =
            "ONNX models missing — export with scripts/export_onnx.py or place .onnx under " +
            (sys.models_dir || "models/");
        }
      }
      const roots = document.querySelectorAll("x-dc");
      roots.forEach((el) => {
        const inst = el._dc || el.__dc || el._component || el.component;
        if (inst && typeof inst.setState === "function") {
          inst.setState(patch);
          if (typeof inst.forceUpdate === "function") inst.forceUpdate();
        }
      });
      if (comp && typeof comp.setState === "function") {
        comp.setState(patch);
      }
    } catch (e) {
      console.warn("system hydrate failed", e);
    }
  }

  async function addModel(body) {
    const res = await fetch(API + "/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    await hydrateSystem(null);
    return out;
  }

  async function uploadModel(file, { name, scale, description } = {}) {
    const fd = new FormData();
    fd.append("file", file, file.name || "model.pth");
    if (name) fd.append("name", name);
    if (scale != null) fd.append("scale", String(scale));
    if (description) fd.append("description", description);
    const res = await fetch(API + "/api/models/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    await hydrateSystem(null);
    return out;
  }

  const WS_KEY = "upscale.workspace";
  const IDB_NAME = "upscale-db";
  const IDB_STORE = "session";
  const IDB_IMAGE = "image";
  const IDB_BATCH = "images";
  const MAX_PERSIST_BYTES = 80 * 1024 * 1024;

  const WS_SETTINGS_KEYS = [
    "mode",
    "scale",
    "target",
    "model",
    "custom",
    "device",
    "tile",
    "overlap",
    "half",
    "tiff",
    "fmt",
    "q",
    "denoise",
    "strength",
    "blendModel",
    "blend",
    "method",
    "algorithm",
    "vecEngine",
    "vecMode",
    "vecColors",
    "vecSpeckle",
    "vecCorners",
    "vecTrace",
    "test",
    "cropSize",
    "cx",
    "cy",
    "dpi",
    "dw",
    "dh",
    "filt",
    "cmpSel",
    "cmpAlgoSel",
    "cmpBig",
    "cmpZ",
    "split",
    "zoom",
    "px",
    "py",
    "tool",
    "type",
    "preview",
    "name",
    "adv",
    "history",
  ];

  function fmtDims(w, h) {
    if (!w) return "—";
    return Number(w).toLocaleString("en") + "×" + Number(h).toLocaleString("en");
  }

  function historyRow(entry) {
    return {
      name: entry.name || "image",
      info: entry.info || "",
      time: entry.time || "",
      jobId: entry.jobId || null,
      resultUrl: entry.resultUrl || null,
      displayUrl: entry.displayUrl || null,
      at: Date.now(),
    };
  }

  function withHistory(prevHistory, entry) {
    const row = historyRow(entry);
    return [row, ...(prevHistory || []).filter((h) => h.jobId !== row.jobId)].slice(
      0,
      12
    );
  }

  function pushHistory(comp, entry) {
    if (!comp || typeof comp.setState !== "function") return;
    comp.setState((st) => ({ history: withHistory(st.history, entry) }));
  }

  function idbOpen() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(IDB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(IDB_STORE)) db.createObjectStore(IDB_STORE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error || new Error("idb open failed"));
    });
  }

  async function idbPut(key, value) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function idbGet(key) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readonly");
      const req = tx.objectStore(IDB_STORE).get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async function idbDel(key) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  function workspaceFromState(s, comp) {
    const out = {};
    for (const k of WS_SETTINGS_KEYS) {
      if (s[k] !== undefined) out[k] = s[k];
    }
    if (s.file && s.file.name) out.fileName = s.file.name;
    if (s.activeId) out.activeId = s.activeId;
    // Prefer explicit state.jobId. Never fall back to comp._jobId after Clear
    // (state.jobId is null but _jobId may still hold the last run).
    const jobId = s.jobId || null;
    if (jobId) out.jobId = jobId;
    if (s.status === "done" || s.status === "idle" || s.status === "running") {
      out.status = s.status === "running" ? "idle" : s.status;
    }
    const files = s.files || [];
    if (files.length) {
      out.batch = files.map((f) => ({
        id: f.id,
        name: f.name,
        w: f.w || 0,
        h: f.h || 0,
        st: f.st || "queued",
        pct: f.pct || 0,
        jobId: f.jobId || null,
        hasAlpha: !!f.hasAlpha,
      }));
      if (s.hasAlpha != null) out.hasAlpha = !!s.hasAlpha;
    } else {
      // Cleared workspace — do not keep a lone global jobId for ghost restore.
      delete out.jobId;
      out.status = "idle";
    }
    out.savedAt = Date.now();
    return out;
  }

  async function persistWorkspace(comp) {
    if (!comp || comp._wsRestoring) return;
    try {
      const s = comp.state || {};
      const files = s.files || [];
      const empty = !s.file && !files.length;
      if (empty) {
        // Cleared UI — wipe storage instead of writing a settings-only ghost.
        localStorage.removeItem(WS_KEY);
        try {
          localStorage.removeItem("upscale.lastJobId");
        } catch (_) {}
        await idbDel(IDB_IMAGE);
        await idbDel(IDB_BATCH);
        return;
      }
      localStorage.setItem(WS_KEY, JSON.stringify(workspaceFromState(s, comp)));
      const file = s.file;
      if (file instanceof Blob) {
        if (file.size > MAX_PERSIST_BYTES) {
          console.warn("Image too large to persist across refresh");
          await idbDel(IDB_IMAGE);
        } else {
          await idbPut(IDB_IMAGE, {
            name: file.name || s.name || "image.png",
            type: file.type || "image/png",
            blob: file,
            at: Date.now(),
          });
        }
      } else if (!file) {
        await idbDel(IDB_IMAGE);
      }

      const batchFiles = files.filter((f) => f && f.file instanceof Blob);
      let total = 0;
      for (const f of batchFiles) total += f.file.size || 0;
      if (!batchFiles.length) {
        await idbDel(IDB_BATCH);
      } else if (total > MAX_PERSIST_BYTES) {
        console.warn("Batch images too large to persist across refresh");
        await idbDel(IDB_BATCH);
      } else {
        await idbPut(
          IDB_BATCH,
          batchFiles.map((f) => ({
            id: f.id,
            name: f.name || "image.png",
            type: (f.file && f.file.type) || "image/png",
            blob: f.file,
          }))
        );
      }
    } catch (e) {
      console.warn("persistWorkspace failed", e);
    }
  }

  async function clearWorkspace(comp) {
    try {
      if (comp) {
        comp._wsRestoring = true;
        comp._jobId = null;
        comp._batchJobIds = null;
        try {
          if (comp._jobEs) {
            comp._jobEs.close();
            comp._jobEs = null;
          }
        } catch (_) {}
      }
      localStorage.removeItem(WS_KEY);
      try {
        localStorage.removeItem("upscale.lastJobId");
      } catch (_) {}
      await idbDel(IDB_IMAGE);
      await idbDel(IDB_BATCH);
    } catch (e) {
      console.warn("clearWorkspace failed", e);
    } finally {
      if (comp) {
        // Allow a beat so Clear's setState + didUpdate persist do not race-rewrite.
        setTimeout(() => {
          if (comp) comp._wsRestoring = false;
        }, 500);
      }
    }
  }

  function loadImageSize(url) {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight });
      im.onerror = () => resolve({ w: 0, h: 0 });
      im.src = url;
    });
  }

  function detectAlphaUrl(url) {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => {
        try {
          const max = 96;
          const sc = Math.min(1, max / Math.max(im.naturalWidth, im.naturalHeight, 1));
          const w = Math.max(1, Math.round(im.naturalWidth * sc));
          const h = Math.max(1, Math.round(im.naturalHeight * sc));
          const cv = document.createElement("canvas");
          cv.width = w;
          cv.height = h;
          const g = cv.getContext("2d", { willReadFrequently: true });
          g.drawImage(im, 0, 0, w, h);
          const d = g.getImageData(0, 0, w, h).data;
          for (let i = 3; i < d.length; i += 4) {
            if (d[i] < 254) {
              resolve(true);
              return;
            }
          }
        } catch (_) {}
        resolve(false);
      };
      im.onerror = () => resolve(false);
      im.src = url;
    });
  }

  async function hydrateJobResult(jobId) {
    if (!jobId) return null;
    try {
      const st = await (await fetch(API + "/api/jobs/" + jobId)).json();
      if (!st || st.state !== "done") return null;
      const resultUrl = API + "/api/jobs/" + jobId + "/result";
      const displayApi = API + "/api/jobs/" + jobId + "/display";
      const displayUrl = await waitForJobImage(displayApi, 8, 150);
      return { resultUrl, displayUrl, urlAfter: displayUrl, jobId, st: "done", pct: 100 };
    } catch (_) {
      return null;
    }
  }

  function rememberView(view) {
    const v = view === "advanced" ? "advanced" : "simple";
    try {
      localStorage.setItem("upscale.view", v);
    } catch (_) {}
    try {
      document.cookie =
        "upscale_view=" + encodeURIComponent(v) + ";path=/;max-age=31536000;SameSite=Lax";
    } catch (_) {}
  }

  async function stashSession(comp, view) {
    try {
      rememberView(view || "simple");
      await persistWorkspace(comp);
    } catch (e) {
      console.warn("stashSession failed", e);
    }
  }

  function bindWorkspace(comp) {
    if (!comp || comp._wsBound) return;
    comp._wsBound = true;
    let timer = null;
    const schedule = () => {
      if (comp._wsRestoring) return;
      clearTimeout(timer);
      timer = setTimeout(() => persistWorkspace(comp), 350);
    };
    const prev = comp.componentDidUpdate;
    comp.componentDidUpdate = function (...args) {
      if (typeof prev === "function") prev.apply(this, args);
      schedule();
    };
    const flush = () => {
      clearTimeout(timer);
      persistWorkspace(comp);
    };
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") flush();
    });
  }

  async function restoreSession(comp) {
    if (!comp) return;
    comp._wsRestoring = true;
    try {
      let data = {};
      try {
        data = JSON.parse(localStorage.getItem(WS_KEY) || "{}") || {};
      } catch (_) {
        data = {};
      }

      const patch = {};
      for (const k of WS_SETTINGS_KEYS) {
        if (data[k] !== undefined) patch[k] = data[k];
      }
      if (data.typeKind && patch.type == null) patch.type = data.typeKind;

      let batchStored = null;
      try {
        batchStored = await idbGet(IDB_BATCH);
      } catch (e) {
        console.warn("idb batch restore failed", e);
      }

      const batchMeta = Array.isArray(data.batch) ? data.batch : [];
      if (batchMeta.length && Array.isArray(batchStored) && batchStored.length) {
        const files = [];
        for (const meta of batchMeta) {
          const stored =
            batchStored.find((x) => x && x.id === meta.id) ||
            batchStored.find((x) => x && x.name === meta.name);
          if (!stored || !stored.blob) continue;
          const file = new File(
            [stored.blob],
            meta.name || stored.name || "image.png",
            { type: stored.type || "image/png" }
          );
          const url = URL.createObjectURL(file);
          let w = meta.w || 0;
          let h = meta.h || 0;
          if (!w || !h) {
            const sz = await loadImageSize(url);
            w = sz.w;
            h = sz.h;
          }
          const item = {
            id: meta.id || Math.random().toString(36).slice(2),
            file,
            url,
            name: file.name,
            w,
            h,
            st: meta.st || "queued",
            pct: meta.pct || 0,
            jobId: meta.jobId || null,
            hasAlpha: !!meta.hasAlpha,
            resultUrl: null,
            displayUrl: null,
            urlAfter: null,
          };
          if (!item.hasAlpha) item.hasAlpha = await detectAlphaUrl(url);
          if (item.jobId) {
            const hydrated = await hydrateJobResult(item.jobId);
            if (hydrated) {
              item.st = "done";
              item.pct = 100;
              item.resultUrl = hydrated.resultUrl;
              item.displayUrl = hydrated.displayUrl;
              item.urlAfter = hydrated.urlAfter;
            } else {
              item.st = "queued";
              item.pct = 0;
              item.jobId = null;
            }
          }
          files.push(item);
        }
        if (files.length) {
          const active =
            files.find((f) => f.id === data.activeId) ||
            files.find((f) => f.st === "done") ||
            files[0];
          const base = {
            ...patch,
            files,
            file: active.file,
            url: active.url,
            name: active.name,
            w: active.w,
            h: active.h,
            activeId: active.id,
            hasAlpha: !!active.hasAlpha,
            jobId: active.jobId || null,
            resultUrl: active.resultUrl || null,
            displayUrl: active.displayUrl || null,
            urlAfter: active.urlAfter || active.displayUrl || null,
            status: files.some((f) => f.st === "done") ? "done" : data.status || "idle",
            pct: active.st === "done" ? 100 : 0,
          };
          if (base.dw == null) base.dw = Math.round(active.w * (base.scale || 4));
          if (base.dh == null) base.dh = Math.round(active.h * (base.scale || 4));
          if (active.jobId) comp._jobId = active.jobId;
          if (base.status === "running") base.status = "idle";
          comp.setState(base);
          return;
        }
      }

      let file = null;
      try {
        const stored = await idbGet(IDB_IMAGE);
        if (stored && stored.blob) {
          file = new File([stored.blob], stored.name || data.fileName || "image.png", {
            type: stored.type || "image/png",
          });
        }
      } catch (e) {
        console.warn("idb image restore failed", e);
      }

      const finish = async (base) => {
        const hasImage =
          !!(base.file || base.url || (base.files && base.files.length));
        const jobId = hasImage ? data.jobId : null;
        if (jobId) {
          const hydrated = await hydrateJobResult(jobId);
          if (hydrated) {
            base.status = "done";
            base.resultUrl = hydrated.resultUrl;
            base.displayUrl = hydrated.displayUrl;
            base.urlAfter = hydrated.urlAfter;
            base.pct = 100;
            base.jobId = jobId;
            if (comp) comp._jobId = jobId;
            if (base.files && base.files[0]) {
              base.files = base.files.map((f, i) =>
                i === 0
                  ? {
                      ...f,
                      st: "done",
                      pct: 100,
                      jobId,
                      resultUrl: hydrated.resultUrl,
                      displayUrl: hydrated.displayUrl,
                      urlAfter: hydrated.urlAfter,
                    }
                  : f
              );
            }
          }
        } else {
          base.resultUrl = null;
          base.displayUrl = null;
          base.urlAfter = null;
          base.jobId = null;
          if (!hasImage) base.status = "idle";
        }
        if (base.status === "running") base.status = "idle";
        comp.setState(base);
      };

      if (file) {
        const url = URL.createObjectURL(file);
        await new Promise((resolve) => {
          const im = new Image();
          im.onload = async () => {
            const hasAlpha = await detectAlphaUrl(url);
            const it = {
              id: Math.random().toString(36).slice(2),
              file,
              url,
              name: file.name,
              w: im.naturalWidth,
              h: im.naturalHeight,
              st: "queued",
              pct: 0,
              hasAlpha,
            };
            const base = {
              ...patch,
              file,
              url,
              name: file.name,
              w: it.w,
              h: it.h,
              files: [it],
              activeId: it.id,
              hasAlpha,
              pct: patch.pct || 0,
            };
            if (base.dw == null) base.dw = Math.round(it.w * (base.scale || 4));
            if (base.dh == null) base.dh = Math.round(it.h * (base.scale || 4));
            await finish(base);
            resolve();
          };
          im.onerror = () => {
            comp.setState(patch);
            resolve();
          };
          im.src = url;
        });
      } else if (Object.keys(patch).length) {
        await finish(patch);
      }
    } catch (e) {
      console.warn("restoreSession failed", e);
    } finally {
      comp._wsRestoring = false;
      bindWorkspace(comp);
      persistWorkspace(comp);
    }
  }

  try {
    const cached = JSON.parse(localStorage.getItem("upscale.sys") || "null");
    if (cached && cached.sys && Date.now() - (cached.at || 0) < 86400000) {
      window.__UPSCALE_SYS__ = cached.sys;
    }
  } catch (_) {}

  async function downloadModel(name) {
    const res = await fetch(API + "/api/models/" + encodeURIComponent(name) + "/download", {
      method: "POST",
    });
    if (!res.ok) throw new Error(await res.text());
    const body = await res.json();
    await hydrateSystem(null);
    return body;
  }

  function stageMessage(ev) {
    const t = ev.type || "progress";
    if (t === "loading_model") return "Loading model…";
    if (t === "pre_filters") return "Applying filters…";
    if (t === "filters") return "Applying filters…";
    if (t === "writing") return "Writing output…";
    if (t === "paused") return "Paused";
    if (t === "downloading") return "Downloading model…";
    if (t === "tile" && ev.pass != null && ev.tile != null) {
      return (
        "Pass " +
        ev.pass +
        (ev.passes != null ? "/" + ev.passes : "") +
        " · tile " +
        ev.tile +
        (ev.tiles != null ? "/" + ev.tiles : "")
      );
    }
    if (ev.message) return ev.message;
    if (ev.pass) return "Pass " + ev.pass + (ev.passes != null ? "/" + ev.passes : "");
    return "Working…";
  }

  async function waitForJobImage(url, attempts, delayMs, prevBlob) {
    const n = attempts == null ? 40 : attempts;
    const delay = delayMs == null ? 120 : delayMs;
    const urls = Array.isArray(url) ? url : [url];
    for (let i = 0; i < n; i++) {
      for (const u of urls) {
        if (!u) continue;
        try {
          const res = await fetch(u + (u.includes("?") ? "&" : "?") + "t=" + Date.now(), {
            cache: "no-store",
          });
          if (res.ok) {
            const blob = await res.blob();
            if (blob && blob.size > 32) {
              const obj = URL.createObjectURL(blob);
              if (prevBlob && prevBlob !== obj) revokeUrl(prevBlob);
              return obj;
            }
          }
        } catch (_) {}
      }
      await new Promise((r) => setTimeout(r, delay));
    }
    return urls.find(Boolean) || url;
  }

  async function runOne(comp, file, settings) {
    closeAllEs(comp);
    revokeStateBlobs(comp, ["displayUrl", "urlAfter", "resultUrl"]);
    comp.setState({ status: "running", paused: false, blocked: false, pct: 0, stage: "Starting…" });
    const { id } = await postJob(file, settings);
    comp._jobId = id;
    return new Promise((resolve, reject) => {
      comp._jobEs = listenEvents(id, (ev) => {
        const t = ev.type || "progress";
        if (t === "paused") {
          comp.setState({ paused: true, status: "running", stage: stageMessage(ev) });
          return;
        }
        if (t === "resumed") {
          comp.setState({ paused: false, status: "running", stage: stageMessage(ev) || "Resumed…" });
          return;
        }
        if (t === "progress" && ev.message === "resumed") {
          comp.setState({ paused: false, status: "running", stage: "Resumed…" });
          return;
        }
        if (
          t === "progress" ||
          t === "tile" ||
          t === "pass_complete" ||
          t === "loading_model" ||
          t === "filters" ||
          t === "pre_filters" ||
          t === "writing" ||
          t === "queued"
        ) {
          const pct = Math.floor((ev.pct != null ? ev.pct : 0) * 100);
          const msg = stageMessage(ev);
          const patch = { pct, stage: msg, status: "running", paused: false };
          if (t === "loading_model" && ev.pct == null) patch.pct = Math.max(comp.state.pct || 0, 2);
          comp.setState(patch);
          if (ev.artifact) {
            const url = API + "/api/jobs/" + id + "/passes/" + (ev.pass || 1);
            comp.setState((st) => ({
              film: [...(st.film || []), url],
            }));
          }
        } else if (t === "done") {
          comp._jobEs.close();
          const resultUrl = API + "/api/jobs/" + id + "/result";
          const displayApi = API + "/api/jobs/" + id + "/display";
          // Show result URL immediately (quoted CSS); refine to blob when ready.
          comp.setState({
            status: "done",
            pct: 100,
            stage: "Loading preview…",
            jobId: id,
            resultUrl,
            displayUrl: resultUrl,
            urlAfter: resultUrl,
            split: 50,
            zoom: 1,
            px: 0,
            py: 0,
          });
          waitForJobImage([displayApi, resultUrl])
            .then((displayUrl) => {
              const patch = {
                status: "done",
                pct: 100,
                split: 50,
                zoom: 1,
                px: 0,
                py: 0,
                paused: false,
                stage: "",
                resultUrl,
                displayUrl,
                urlAfter: displayUrl,
                jobId: id,
              };
              if (ev.cli) patch.cliHint = ev.cli;
              const s = comp.state || {};
              const srcW = s.test
                ? Math.min(+s.cropSize || s.w || 0, s.w || 0)
                : s.w || 0;
              const srcH = s.test
                ? Math.min(+s.cropSize || s.h || 0, s.h || 0)
                : s.h || 0;
              let scaleLabel = "—";
              try {
                scaleLabel =
                  typeof comp.curScale === "function"
                    ? (+comp.curScale()).toFixed(1) + "×"
                    : (+s.scale || 1).toFixed(1) + "×";
              } catch (_) {}
              let outLabel = "—";
              try {
                outLabel =
                  typeof comp.outDims === "function" ? comp.outDims(s) : "—";
              } catch (_) {}
              patch.history = withHistory(s.history, {
                name: s.name || "image",
                info: fmtDims(srcW, srcH) + " → " + outLabel,
                time: scaleLabel,
                jobId: id,
                resultUrl,
                displayUrl,
              });
              comp.setState(patch);
              persistWorkspace(comp);
              if (window.UpscaleBackend && window.UpscaleBackend.invalidateEstimate) {
                window.UpscaleBackend.invalidateEstimate(comp);
              }
              try {
                if (window.UpscaleDesktop && window.UpscaleDesktop.notifyDone) {
                  window.UpscaleDesktop.notifyDone(comp.state.name || "image");
                } else if (
                  typeof Notification !== "undefined" &&
                  Notification.permission === "granted"
                ) {
                  new Notification("Upscale finished", {
                    body: comp.state.name || "Image ready",
                  });
                }
              } catch (_) {}
              resolve(resultUrl);
            })
            .catch((err) => reject(err));
        } else if (t === "cancelled") {
          comp._jobEs.close();
          const err = new Error("cancelled");
          err.cancelled = true;
          comp.setState({ status: "idle", pct: 0, paused: false, stage: "Cancelled" });
          reject(err);
        } else if (t === "error") {
          comp._jobEs.close();
          comp.setState({ status: "idle", pct: 0, paused: false, stage: ev.message || t });
          reject(new Error(ev.message || t));
        }
      });
    });
  }

  function isCancelledError(e) {
    return !!(e && (e.cancelled || /cancel/i.test(String(e.message || e))));
  }

  window.UpscaleBackend = {
    async runJob(comp, opts) {
      try {
        const force = !!(opts && opts.force);
        if (force && comp && typeof comp.setState === "function") {
          comp.setState({ blocked: false });
        }
        const settings = settingsFromAdvanced(comp);
        await runOne(comp, comp.state.file, settings);
        this.refreshInterrupted(comp);
      } catch (e) {
        console.error(e);
        if (!isCancelledError(e)) {
          alert("Upscale failed: " + (e.message || e));
        }
        comp.setState({ status: "idle", paused: false, stage: isCancelledError(e) ? "Cancelled" : (comp.state.stage || "") });
        this.refreshInterrupted(comp);
      }
    },

    /** Upscale only the current test crop with the selected model/settings. */
    async runCropJob(comp) {
      try {
        if (!comp.state.file) {
          alert("Choose an image first.");
          return;
        }
        const settings = settingsFromAdvanced(comp);
        // Force crop even if Scale tab somehow called us.
        const s = comp.state;
        const size = +s.cropSize || 128;
        const cx = s.cx != null ? s.cx : 0.5;
        const cy = s.cy != null ? s.cy : 0.5;
        const x = Math.max(0, Math.round(cx * s.w - size / 2));
        const y = Math.max(0, Math.round(cy * s.h - size / 2));
        settings.crop = {
          x,
          y,
          w: Math.min(size, s.w - x),
          h: Math.min(size, s.h - y),
        };
        await runOne(comp, comp.state.file, settings);
        this.refreshInterrupted(comp);
      } catch (e) {
        console.error(e);
        if (!isCancelledError(e)) {
          alert("Crop upscale failed: " + (e.message || e));
        }
        comp.setState({
          status: "idle",
          paused: false,
          stage: isCancelledError(e) ? "Cancelled" : comp.state.stage || "",
        });
      }
    },

    async runBatch(comp) {
      const files = (comp.state.files || []).slice();
      comp._batchJobIds = [];
      comp._batchAbort = false;
      comp.setState({ status: "running", paused: false, pct: 0 });
      let aborted = false;
      for (let i = 0; i < files.length; i++) {
        if (comp._batchAbort) {
          aborted = true;
          break;
        }
        const it = files[i];
        const itemId = it.id;
        comp.setState((st) => ({
          stage: "Image " + (i + 1) + " of " + files.length + " · " + it.name,
          activeId: itemId,
          file: it.file,
          url: it.url,
          name: it.name,
          w: it.w,
          h: it.h,
          hasAlpha: !!it.hasAlpha,
          resultUrl: null,
          displayUrl: null,
          urlAfter: null,
          jobId: null,
          files: (st.files || []).map((x) =>
            x.id === itemId ? { ...x, st: "run", pct: 0 } : x
          ),
        }));
        try {
          const settings = settingsFromAdvanced(comp);
          delete settings.crop;
          await runOne(comp, it.file, settings);
          if (comp._batchAbort) {
            aborted = true;
            break;
          }
          const jobId = comp._jobId || null;
          if (jobId) comp._batchJobIds.push(jobId);
          // Stable per-job API URLs so strip switching never reuses another
          // image's blob preview from the last completed runOne().
          const stableResult = jobId
            ? API + "/api/jobs/" + jobId + "/result"
            : null;
          const stableDisplay = jobId
            ? API + "/api/jobs/" + jobId + "/display"
            : stableResult;
          comp.setState((st) => ({
            files: st.files.map((x) =>
              x.id === itemId
                ? {
                    ...x,
                    st: "done",
                    pct: 100,
                    jobId,
                    resultUrl: stableResult,
                    displayUrl: stableDisplay,
                    urlAfter: stableDisplay,
                  }
                : x
            ),
            // Keep the just-finished image selected with matching result URLs.
            resultUrl: stableResult,
            displayUrl: stableDisplay,
            urlAfter: stableDisplay,
            jobId,
          }));
        } catch (e) {
          const cancelled = isCancelledError(e) || comp._batchAbort;
          comp.setState((st) => ({
            files: st.files.map((x) =>
              x.id === itemId
                ? { ...x, st: cancelled ? "cancelled" : "error" }
                : x
            ),
          }));
          if (cancelled) {
            aborted = true;
            break;
          }
        }
      }
      if (aborted) {
        comp.setState({
          status: "idle",
          pct: 0,
          paused: false,
          stage: "Cancelled",
          resultUrl: null,
          displayUrl: null,
          urlAfter: null,
          jobId: null,
        });
        comp._jobId = null;
        await persistWorkspace(comp);
        this.refreshInterrupted(comp);
        return;
      }
      // Reselect first completed item so the canvas matches a strip pick.
      const doneFiles = (comp.state.files || []).filter((f) => f.st === "done");
      const active =
        doneFiles.find((f) => f.id === (comp.state && comp.state.activeId)) ||
        doneFiles[doneFiles.length - 1] ||
        null;
      if (active) {
        const resultUrl =
          active.resultUrl ||
          (active.jobId ? API + "/api/jobs/" + active.jobId + "/result" : null);
        const displayUrl =
          active.displayUrl ||
          active.urlAfter ||
          (active.jobId ? API + "/api/jobs/" + active.jobId + "/display" : null) ||
          resultUrl;
        comp._jobId = active.jobId || null;
        comp.setState({
          status: "done",
          pct: 100,
          activeId: active.id,
          file: active.file,
          url: active.url,
          name: active.name,
          w: active.w,
          h: active.h,
          hasAlpha: !!active.hasAlpha,
          jobId: active.jobId || null,
          resultUrl,
          displayUrl,
          urlAfter: displayUrl,
        });
      } else {
        comp.setState({ status: "done", pct: 100 });
      }
      await persistWorkspace(comp);
      if (comp._batchJobIds.length > 1) {
        try {
          await this.zipJobs(comp._batchJobIds);
        } catch (e) {
          console.warn("batch zip failed", e);
        }
      }
    },

    async runSimple(comp) {
      try {
        const file = comp.state.file;
        if (!file) throw new Error("No image loaded");
        const settings = settingsFromSimple(comp);
        const url = await runOne(comp, file, settings);
        // Do not overwrite source url — before/after slider needs both
        if (url) {
          const displayUrl = comp.state.displayUrl || url;
          comp.setState({
            status: "done",
            pct: 100,
            split: 50,
            resultUrl: url,
            displayUrl,
            urlAfter: displayUrl,
          });
        }
      } catch (e) {
        if (!isCancelledError(e)) {
          alert("Upscale failed: " + (e.message || e));
        }
        comp.setState({
          status: "idle",
          stage: isCancelledError(e) ? "Cancelled" : "",
        });
      }
    },

    async compareModels(comp) {
      try {
        await this.compareViaApi(comp);
      } catch (e) {
        if (isCancelledError(e) || (e && e.name === "AbortError") || (comp && comp._batchAbort)) {
          comp.setState({ status: "idle", paused: false, stage: "Cancelled" });
          return;
        }
        console.warn("compare API failed, falling back to sequential jobs", e);
        const s = comp.state;
        const sel = this._compareSelection(comp);
        if (!s.file || !sel.total) return;
        const size = +s.cropSize || 128;
        const cx = s.cx != null ? s.cx : 0.5;
        const cy = s.cy != null ? s.cy : 0.5;
        const x = Math.max(0, Math.round(cx * s.w - size / 2));
        const y = Math.max(0, Math.round(cy * s.h - size / 2));
        const crop = {
          x,
          y,
          w: Math.min(size, s.w - x),
          h: Math.min(size, s.h - y),
        };
        const variants = [
          ...sel.models.map((n) => ({
            id: n,
            name: n,
            kind: "model",
            settings: { model: n, algorithm: null, filters: {} },
          })),
          ...sel.algorithms.map((n) => ({
            id: "algo:" + n,
            name: "Classical · " + n,
            kind: "algorithm",
            settings: { model: sel.base_model, algorithm: n, filters: {} },
          })),
          ...sel.filters.map((n) => {
            const filters = {
              pre_denoise: 0,
              jpeg_clean: 0,
              deblur: 0,
              edge: 0,
              grain: 0,
              color_contrast: 0,
              halo_reduce: 0,
            };
            const map = {
              denoise: "pre_denoise",
              jpeg: "jpeg_clean",
              deblur: "deblur",
              edge: "edge",
              grain: "grain",
              color: "color_contrast",
              halo: "halo_reduce",
            };
            if (n !== "none" && map[n]) {
              filters[map[n]] = sel.filter_amounts[n] != null ? sel.filter_amounts[n] : 0.35;
            }
            return {
              id: "filt:" + n,
              name: n === "none" ? "No filters" : "Filter · " + n,
              kind: "filter",
              settings: {
                model: sel.base_model,
                algorithm: sel.base_algorithm,
                filters,
              },
            };
          }),
        ];
        const items = variants.map((v) => ({
          id: v.id,
          name: v.name,
          kind: v.kind,
          done: false,
          secs: 0,
          url: null,
        }));
        comp.setState({ cmp: items, cpx: 0, cpy: 0, status: "running" });
        for (let i = 0; i < variants.length; i++) {
          if (comp._batchAbort) break;
          const v = variants[i];
          const t0 = performance.now();
          try {
            const base = settingsFromAdvanced(comp);
            const settings = {
              ...base,
              scale:
                typeof comp.curScale === "function" ? comp.curScale() : base.scale || 4,
              target_size: null,
              width: null,
              height: null,
              model: v.settings.model,
              algorithm: v.settings.algorithm || null,
              filters: { ...(base.filters || {}), ...(v.settings.filters || {}) },
              crop,
              format: "png",
              mid_pass_previews: false,
            };
            if (!settings.algorithm) delete settings.algorithm;
            const { id } = await postJob(s.file, settings);
            comp._jobId = id;
            await new Promise((resolve, reject) => {
              const es = trackEs(
                comp,
                listenEvents(id, (ev) => {
                  if (ev.type === "done") {
                    es.close();
                    resolve();
                  } else if (ev.type === "error" || ev.type === "cancelled") {
                    es.close();
                    reject(
                      Object.assign(new Error(ev.message || ev.type), {
                        cancelled: ev.type === "cancelled",
                      })
                    );
                  }
                })
              );
            });
            if (comp._batchAbort) break;
            const secs = (performance.now() - t0) / 1000;
            const url = API + "/api/jobs/" + id + "/result";
            comp.setState((st) => ({
              cmp: st.cmp.map((c, xi) =>
                xi === i ? { ...c, done: true, secs, url } : c
              ),
            }));
          } catch (err) {
            if (comp._batchAbort || isCancelledError(err)) break;
            comp.setState((st) => ({
              cmp: st.cmp.map((c, xi) =>
                xi === i ? { ...c, done: true, secs: 0, error: String(err) } : c
              ),
            }));
          }
        }
        comp._jobId = null;
        comp.setState({
          status: "idle",
          stage: comp._batchAbort ? "Cancelled" : "",
        });
      }
    },

    async downloadBatchZip(comp) {
      const ids = (comp._batchJobIds || []).filter(Boolean);
      if (!ids.length) {
        alert("No completed batch jobs to zip yet.");
        return;
      }
      await this.zipJobs(ids);
    },

    async refreshInterrupted(comp) {
      try {
        const data = await (await fetch(API + "/api/jobs/interrupted")).json();
        const jobs = data.jobs || [];
        const first = jobs[0] || null;
        if (comp && typeof comp.setState === "function") {
          comp.setState({ interrupted: first, interruptedAll: jobs });
        }
        return jobs;
      } catch (e) {
        console.warn("interrupted poll failed", e);
        return [];
      }
    },

    async resumeInterrupted(comp) {
      const interrupted = (comp.state && comp.state.interrupted) || null;
      const id = (interrupted && interrupted.id) || comp._jobId;
      if (!id) {
        alert("No interrupted job to resume.");
        return;
      }
      const ck = (interrupted && interrupted.checkpoint) || {};
      const passIdx = ck.pass_index;
      const passLabel =
        passIdx == null || passIdx === "" ? "?" : String(passIdx);
      const pct0 =
        ck.pct_approx != null
          ? Math.floor(Number(ck.pct_approx) * 100)
          : ck.flush_row && ck.out_h
            ? Math.floor((Number(ck.flush_row) / Number(ck.out_h)) * 100)
            : 0;
      const previewUrl = API + "/api/jobs/" + id + "/preview?t=" + Date.now();
      // Keep original source blob for before/after; show checkpoint as separate preview.
      const patch = {
        status: "running",
        paused: false,
        pct: pct0,
        stage: "Resuming from checkpoint · pass " + passLabel + "…",
        interrupted: null,
        jobId: id,
        name: (interrupted && interrupted.input) || (comp.state && comp.state.name) || "image",
        checkpointPreview: previewUrl,
      };
      if (!(comp.state && comp.state.url)) {
        patch.url = previewUrl;
        patch.urlBefore = previewUrl;
      }
      const settings = interrupted && interrupted.settings;
      if (settings && typeof settings === "object") {
        if (settings.model) patch.model = settings.model;
        if (settings.scale != null) {
          patch.mode = "scale";
          patch.scale = settings.scale;
        }
        if (settings.algorithm) {
          patch.method = "classical";
          patch.algorithm = settings.algorithm;
          const vz = settings.vectorize || {};
          if (vz.engine) patch.vecEngine = vz.engine === "autotrace" ? "autotrace" : "vtracer";
          if (vz.color_precision != null) patch.vecColors = +vz.color_precision;
          if (vz.filter_speckle != null) patch.vecSpeckle = +vz.filter_speckle;
          if (vz.corner_threshold != null) patch.vecCorners = +vz.corner_threshold;
          if (vz.mode) patch.vecMode = vz.mode === "polygon" ? "polygon" : "spline";
          if (vz.max_trace_edge != null) patch.vecTrace = +vz.max_trace_edge;
        } else if (settings.model) {
          patch.method = "neural";
        }
        if (settings.format) patch.fmt = settings.format;
        if (settings.device) patch.device = settings.device;
        if (settings.tile != null) patch.tile = settings.tile;
        if (settings.overlap != null) patch.overlap = settings.overlap;
      }
      try {
        const res = await fetch(API + "/api/jobs/" + id + "/resume-checkpoint", {
          method: "POST",
        });
        if (!res.ok) throw new Error(await res.text());
        comp._jobId = id;
        comp._jobEs && comp._jobEs.close();
        comp.setState(patch);
        return await new Promise((resolve, reject) => {
          comp._jobEs = listenEvents(id, (ev) => {
            const t = ev.type || "progress";
            if (t === "done") {
              comp._jobEs.close();
              const resultUrl = API + "/api/jobs/" + id + "/result";
              const displayUrl = API + "/api/jobs/" + id + "/display";
              const hist = withHistory(comp.state && comp.state.history, {
                name: (comp.state && comp.state.name) || "image",
                info: "Resumed job",
                time: "resume",
                jobId: id,
                resultUrl,
                displayUrl,
              });
              waitForJobImage(displayUrl, 8, 100).then((url) => {
                comp.setState({
                  status: "done",
                  pct: 100,
                  resultUrl,
                  displayUrl: url,
                  urlAfter: url,
                  jobId: id,
                  history: hist,
                  stage: "",
                });
                persistWorkspace(comp);
                this.refreshInterrupted(comp);
                resolve(resultUrl);
              });
            } else if (t === "error" || t === "cancelled") {
              comp._jobEs.close();
              comp.setState({ status: "idle", stage: ev.message || t, paused: false });
              this.refreshInterrupted(comp);
              reject(new Error(ev.message || t));
            } else if (ev.pct != null || ev.message) {
              comp.setState({
                pct: ev.pct != null ? Math.floor(ev.pct * 100) : pct0,
                stage: ev.message || "Resuming…",
                status: "running",
              });
            }
          });
        });
      } catch (e) {
        console.error(e);
        alert("Resume failed: " + (e.message || e));
        comp.setState({ status: "idle", stage: "" });
        this.refreshInterrupted(comp);
      }
    },

    downloadModel,
    hydrateSystem,
    addModel,
    uploadModel,
    rememberView,
    stashSession,
    restoreSession,
    persistWorkspace,
    clearWorkspace,
    bindWorkspace,
    pushHistory,

    exportSettings(comp) {
      const data = workspaceFromState((comp && comp.state) || {}, comp);
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "upscale-settings.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    },

    importSettings(comp, data) {
      if (!comp || !data || typeof data !== "object") return;
      const patch = {};
      for (const k of WS_SETTINGS_KEYS) {
        if (data[k] !== undefined) patch[k] = data[k];
      }
      comp.setState(patch, () => persistWorkspace(comp));
    },

    async runBenchmark(comp) {
      const s = (comp && comp.state) || {};
      let path = null;
      const stageBlob = async (blob, name) => {
        const f = new FormData();
        f.append("file", blob, name || "bench.png");
        f.append(
          "settings",
          JSON.stringify({
            scale: 1,
            algorithm: "nearest",
            device: "cpu",
          })
        );
        const jobRes = await fetch(API + "/api/jobs", { method: "POST", body: f });
        if (!jobRes.ok) throw new Error(await jobRes.text());
        const job = await jobRes.json();
        fetch(API + "/api/jobs/" + job.id + "/cancel", { method: "POST" }).catch(
          () => {}
        );
        return job.input || null;
      };
      if (s.file) {
        let blob = s.file;
        if (s.test && s.w) {
          const size = +s.cropSize || 128;
          const cx = s.cx != null ? s.cx : 0.5;
          const cy = s.cy != null ? s.cy : 0.5;
          const x = Math.max(0, Math.round(cx * s.w - size / 2));
          const y = Math.max(0, Math.round(cy * s.h - size / 2));
          const w = Math.min(size, s.w - x);
          const h = Math.min(size, s.h - y);
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const img = await createImageBitmap(s.file);
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, x, y, w, h, 0, 0, w, h);
          blob = await new Promise((resolve) =>
            canvas.toBlob(resolve, "image/png")
          );
        }
        path = await stageBlob(blob, "bench.png");
      } else if (comp && comp._jobId) {
        const meta = await (await fetch(API + "/api/jobs/" + comp._jobId)).json();
        path = meta.input || null;
      }
      if (!path) {
        throw new Error(
          "Load an image before benchmarking — needs a file under the jobs folder"
        );
      }
      const res = await fetch(API + "/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path,
          model: s.model || "RealESRGAN_x4plus",
          device: s.device === "cuda" ? "cuda" : "cpu",
          crop_size: 64,
          scale: 2,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    openHistory(comp, entry) {
      if (!comp || !entry) return;
      revokeStateBlobs(comp, ["displayUrl", "urlAfter", "resultUrl"]);
      const patch = { status: "done", pct: 100, paused: false };
      if (entry.jobId) {
        comp._jobId = entry.jobId;
        patch.jobId = entry.jobId;
        patch.resultUrl = API + "/api/jobs/" + entry.jobId + "/result";
        patch.displayUrl = API + "/api/jobs/" + entry.jobId + "/display";
        patch.urlAfter = patch.displayUrl;
        // Prefer stored source preview when available.
        if (entry.sourceUrl) patch.url = entry.sourceUrl;
        else if (entry.url) patch.url = entry.url;
        else patch.url = API + "/api/jobs/" + entry.jobId + "/preview";
        waitForJobImage(patch.displayUrl, 8, 100).then((url) => {
          if (comp._jobId !== entry.jobId) return;
          comp.setState({ displayUrl: url, urlAfter: url, status: "done", pct: 100 });
        });
      } else if (entry.displayUrl || entry.resultUrl) {
        patch.displayUrl = entry.displayUrl || entry.resultUrl;
        patch.resultUrl = entry.resultUrl || entry.displayUrl;
        patch.urlAfter = patch.displayUrl;
        if (entry.sourceUrl || entry.url) patch.url = entry.sourceUrl || entry.url;
      }
      if (entry.name) patch.name = entry.name;
      if (entry.w) patch.w = entry.w;
      if (entry.h) patch.h = entry.h;
      comp.setState(patch);
    },

    cancel(comp) {
      // Stop batch from starting the next image after this job dies.
      if (comp) comp._batchAbort = true;
      closeAllEs(comp);
      const resetBatchRows = () => {
        if (!comp || !comp.setState) return;
        comp.setState((st) => ({
          files: (st.files || []).map((f) =>
            f && f.st === "run" ? { ...f, st: "queued", pct: 0 } : f
          ),
        }));
      };
      if (comp && comp._compareAbort) {
        try {
          comp._compareAbort.abort();
        } catch (_) {}
      }
      const compareId = comp && comp._compareId;
      if (compareId) {
        fetch(API + "/api/compare/" + compareId + "/cancel", { method: "POST" }).catch(
          () => {}
        );
        comp._compareId = null;
        if (comp) {
          resetBatchRows();
          comp.setState({
            status: "idle",
            pct: 0,
            paused: false,
            stage: "Cancelled",
          });
        }
        return;
      }
      const id =
        (comp && comp._jobId) ||
        (comp && comp.state && comp.state.jobId) ||
        null;
      if (!id) {
        if (comp) {
          resetBatchRows();
          comp.setState({
            status: "idle",
            pct: 0,
            paused: false,
            stage: "Cancelled",
          });
        }
        return;
      }
      resetBatchRows();
      comp._jobId = id;
      try {
        localStorage.setItem("upscale.lastJobId", id);
      } catch (_) {}
      comp.setState({ stage: "Cancelling…", status: "running", paused: false });
      // Also clear pause so the worker can notice cancel while waiting.
      fetch(API + "/api/jobs/" + id + "/resume", { method: "POST" }).catch(() => {});
      fetch(API + "/api/jobs/" + id + "/cancel", { method: "POST" })
        .then((res) => {
          if (!res.ok) throw new Error("cancel HTTP " + res.status);
        })
        .catch((e) => {
          console.warn("cancel request failed", e);
          comp.setState({
            status: "idle",
            pct: 0,
            paused: false,
            stage: "Cancel failed — try again",
          });
        });
      // Keep EventSource open until cancelled/done/error arrives (runOne handler).
      // Reminder if the worker is mid-tile (cancel applies between tiles).
      clearTimeout(comp._cancelFallback);
      comp._cancelFallback = setTimeout(() => {
        if (comp.state && comp.state.status === "running" && comp._batchAbort) {
          comp.setState({
            stage: "Cancelling… (finishing current tile)",
            paused: false,
          });
        }
        this.refreshInterrupted(comp);
      }, 1500);
    },
    pause(comp) {
      if (!comp._jobId) return;
      fetch(API + "/api/jobs/" + comp._jobId + "/pause", { method: "POST" });
    },
    resume(comp) {
      if (!comp._jobId) return;
      fetch(API + "/api/jobs/" + comp._jobId + "/resume", { method: "POST" });
    },

    async fetchEstimate(body) {
      const res = await fetch(API + "/api/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    /** Decode TGA/PSD/PSB/JP2/etc. to a PNG preview blob + dimensions. */
    async decodePreview(file) {
      const fd = new FormData();
      fd.append("file", file, file.name || "image.bin");
      const res = await fetch(API + "/api/decode-preview", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      return {
        blob,
        w: +(res.headers.get("X-Image-Width") || 0),
        h: +(res.headers.get("X-Image-Height") || 0),
        hasAlpha: res.headers.get("X-Has-Alpha") === "1",
      };
    },

    async fetchAlgorithms() {
      const res = await fetch(API + "/api/algorithms");
      if (!res.ok) return null;
      return res.json();
    },

    async fetchPresets() {
      const res = await fetch(API + "/api/presets");
      if (!res.ok) return null;
      return res.json();
    },

    async applyPreset(id, kind) {
      const res = await fetch(API + "/api/presets/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, kind: kind || "option" }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async suggest(file, fast) {
      const fd = new FormData();
      fd.append("file", file, file.name || "image.png");
      fd.append("fast", fast ? "true" : "false");
      const res = await fetch(API + "/api/suggest", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    /**
     * Debounced server estimate → comp.state.apiEtaLow/High + apiVramFit.
     * Call from renderVals / after settings change.
     */
    maybeEstimate(comp, body) {
      if (!comp || !body || !body.src_w || !body.src_h || !body.scale) return;
      const key = JSON.stringify(body);
      if (comp._estKey === key) return;
      comp._estKey = key;
      clearTimeout(comp._estTimer);
      comp._estTimer = setTimeout(async () => {
        try {
          // Server looks up learned MP/s from past jobs; do not override with a
          // single global localStorage rate (that poisoned every model the same).
          const est = await this.fetchEstimate(body);
          if (comp._estKey !== key) return;
          comp.setState({
            apiEtaLow: est.seconds_low,
            apiEtaHigh: est.seconds_high,
            apiVramFit: est.vram_fit,
            apiEtaCalibrated: !!est.calibrated,
            apiEtaSamples: est.samples || 0,
            apiEstimateAt: Date.now(),
          });
        } catch (e) {
          console.warn("estimate failed", e);
        }
      }, 280);
    },

    /** Force a fresh estimate after a completed run (stats file just updated). */
    invalidateEstimate(comp) {
      if (!comp) return;
      comp._estKey = null;
      clearTimeout(comp._estTimer);
    },

    async zipJobs(jobIds) {
      const res = await fetch(API + "/api/zip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_ids: jobIds }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "upscale-batch.zip";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    },

    _compareSelection(comp) {
      const s = comp.state || {};
      const models = s.cmpSel || [];
      const algorithms = s.cmpAlgoSel || [];
      const filters = [];
      const filter_amounts = {};
      if (s.cmpIncludeFilters && s.filt) {
        Object.entries(s.filt).forEach(([k, x]) => {
          if (x && x.on) {
            filters.push(k);
            filter_amounts[k] = (x.v != null ? +x.v : 0) / 100;
          }
        });
      }
      return {
        models,
        algorithms,
        filters,
        filter_amounts,
        base_model: s.model || "RealESRGAN_x4plus",
        base_algorithm:
          s.method === "classical" && s.algorithm ? s.algorithm : null,
        total: models.length + algorithms.length + filters.length,
      };
    },

    async compareViaApi(comp) {
      const s = comp.state;
      const sel = this._compareSelection(comp);
      if (!s.file || !sel.total) return;
      const size = +s.cropSize || 128;
      const cx = s.cx != null ? s.cx : 0.5;
      const cy = s.cy != null ? s.cy : 0.5;
      const x = Math.max(0, Math.round(cx * s.w - size / 2));
      const y = Math.max(0, Math.round(cy * s.h - size / 2));
      const fd = new FormData();
      fd.append("file", s.file, s.file.name || "image.png");
      fd.append(
        "body",
        JSON.stringify({
          models: sel.models,
          algorithms: sel.algorithms,
          filters: sel.filters,
          filter_amounts: sel.filter_amounts,
          base_model: sel.base_model,
          base_algorithm: sel.base_algorithm,
          crop: { x, y, w: Math.min(size, s.w - x), h: Math.min(size, s.h - y) },
          scale:
            typeof comp.curScale === "function" ? comp.curScale() : s.scale || 4,
          device: s.device || "auto",
          half: !!s.half,
        })
      );
      const placeholder = [
        ...sel.models.map((n) => ({ id: n, name: n, kind: "model" })),
        ...sel.algorithms.map((n) => ({
          id: "algo:" + n,
          name: "Classical · " + n,
          kind: "algorithm",
        })),
        ...(sel.filters || []).map((n) => ({
          id: "filt:" + n,
          name: "Filter · " + n,
          kind: "filter",
        })),
      ].map((x) => ({ ...x, done: false, secs: 0, url: null }));
      const ac = new AbortController();
      comp._compareAbort = ac;
      comp._compareId = null;
      comp._batchAbort = false;
      comp.setState({
        cmp: placeholder,
        cpx: 0,
        cpy: 0,
        status: "running",
        stage: "Comparing variants…",
        pct: 0,
        paused: false,
      });
      const mapResults = (runId, rows) =>
        (rows || []).map((r) => {
          const id = r.id || r.model;
          return {
            id,
            name: r.label || id,
            kind: r.kind || "model",
            done: true,
            secs: r.seconds || 0,
            url: r.path
              ? API + "/api/compare/" + runId + "/" + encodeURIComponent(id)
              : null,
            error: r.error || null,
          };
        });
      const mergePartial = (runId, rows, stMsg, doneCount, totalCount) => {
        const doneMap = {};
        mapResults(runId, rows).forEach((r) => {
          doneMap[r.id] = r;
        });
        const total = totalCount || placeholder.length || 1;
        const done = doneCount != null ? doneCount : Object.keys(doneMap).length;
        const pct = Math.max(0, Math.min(99, Math.round((done / total) * 100)));
        comp.setState((st) => ({
          cmp: (st.cmp || placeholder).map((c) => doneMap[c.id] || c),
          stage: stMsg || "Comparing variants…",
          pct,
        }));
      };
      try {
        const res = await fetch(API + "/api/compare", {
          method: "POST",
          body: fd,
          signal: ac.signal,
        });
        if (!res.ok) throw new Error(await res.text());
        const started = await res.json();
        const runId = started.id;
        if (!runId) throw new Error("compare returned no id");
        comp._compareId = runId;
        const sleep = (ms) =>
          new Promise((resolve, reject) => {
            const t = setTimeout(resolve, ms);
            ac.signal.addEventListener(
              "abort",
              () => {
                clearTimeout(t);
                reject(Object.assign(new Error("cancelled"), { cancelled: true }));
              },
              { once: true }
            );
          });
        while (true) {
          if (comp._batchAbort || ac.signal.aborted) {
            fetch(API + "/api/compare/" + runId + "/cancel", { method: "POST" }).catch(
              () => {}
            );
            throw Object.assign(new Error("cancelled"), { cancelled: true });
          }
          const stRes = await fetch(API + "/api/compare/" + runId + "/status", {
            signal: ac.signal,
          });
          if (!stRes.ok) throw new Error(await stRes.text());
          const st = await stRes.json();
          const total =
            st.total != null ? st.total : placeholder.length || sel.total || 1;
          const doneCount =
            st.done_count != null
              ? st.done_count
              : (st.results && st.results.length) || 0;
          if (st.results && st.results.length)
            mergePartial(runId, st.results, st.message, doneCount, total);
          else if (st.message || doneCount) {
            comp.setState({
              stage: st.message || "Comparing variants…",
              pct: Math.max(
                0,
                Math.min(99, Math.round((doneCount / total) * 100))
              ),
            });
          }
          if (st.state === "done") {
            const results = mapResults(runId, st.results);
            // Keep placeholder slots for variants that never ran.
            const byId = {};
            results.forEach((r) => {
              byId[r.id] = r;
            });
            comp.setState({
              cmp: placeholder.map((c) => byId[c.id] || { ...c, done: true }),
              status: "idle",
              stage: "",
              pct: 100,
            });
            return st;
          }
          if (st.state === "cancelled") {
            const results = mapResults(runId, st.results);
            const byId = {};
            results.forEach((r) => {
              byId[r.id] = r;
            });
            comp.setState({
              cmp: placeholder.map((c) => byId[c.id] || c),
              status: "idle",
              stage: "Cancelled",
            });
            throw Object.assign(new Error("cancelled"), { cancelled: true });
          }
          if (st.state === "error") {
            throw new Error(st.error || "compare failed");
          }
          await sleep(400);
        }
      } catch (e) {
        if (isCancelledError(e) || (e && e.name === "AbortError") || comp._batchAbort) {
          comp.setState({ status: "idle", paused: false, stage: "Cancelled" });
          throw Object.assign(e instanceof Error ? e : new Error("cancelled"), {
            cancelled: true,
          });
        }
        throw e;
      } finally {
        comp._compareId = null;
        comp._compareAbort = null;
      }
    },
  };

  function _fmtBytes(n) {
    if (n == null) return "?";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
    return (n / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }

  async function showAbout() {
    try {
      const a = await (await fetch(API + "/api/about")).json();
      const m = a.models || {};
      alert(
        "Upscale " +
          (a.version || "") +
          "\n\n" +
          (a.description || "") +
          "\n\nLicense: " +
          (a.license || "MIT") +
          "\nModels folder: " +
          (a.models_dir || "") +
          "\nInstalled: " +
          (m.installed_count || 0) +
          " (" +
          _fmtBytes(m.total_bytes) +
          ")\n\n" +
          (a.models_policy || "")
      );
    } catch (e) {
      alert("Could not load About: " + e);
    }
  }

  async function showLicense() {
    try {
      const L = await (await fetch(API + "/api/license")).json();
      const text = L.text || "MIT";
      alert(text.length > 3500 ? text.slice(0, 3500) + "\n…" : text);
    } catch (e) {
      alert("Could not load License: " + e);
    }
  }

  async function checkUpdates() {
    try {
      const u = await (await fetch(API + "/api/updates")).json();
      let msg = u.message || ("Current version: " + (u.current || "?"));
      if (u.update_available && u.url) msg += "\n\n" + u.url;
      alert(msg);
    } catch (e) {
      alert("Update check failed: " + e);
    }
  }

  // Expose helpers the synced design HTML can call for after-image URLs
  window.UpscaleUI = {
    afterUrl(s) {
      return s && s.status === "done" && s.resultUrl ? s.resultUrl : s && s.url;
    },
    beforeUrl(s) {
      return s && s.url;
    },
    modelStatus(name) {
      const list = window.__UPSCALE_MODELS__;
      if (Array.isArray(list)) {
        const m = list.find((x) => x.name === name);
        if (m) {
          if (m.status === "failed") return "err";
          return m.status === "installed" ? "ok" : "dl";
        }
      }
      if (window.__UPSCALE_MODEL_ERR__ && window.__UPSCALE_MODEL_ERR__[name]) return "err";
      if (window.__UPSCALE_DOWNLOADED__ && window.__UPSCALE_DOWNLOADED__.includes(name)) return "ok";
      return "dl";
    },
    showAbout,
    showLicense,
    checkUpdates,
  };

  function boot() {
    hydrateSystem(null);
    // Design components mount slightly later — re-apply once
    setTimeout(() => hydrateSystem(null), 800);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
