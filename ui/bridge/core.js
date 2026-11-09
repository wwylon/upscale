/** Upscale bridge — core. Attaches to window.__UpscaleBridge. */

(function (B) {
B.API = B.API || "";
  const API = B.API;

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

  /**
   * onError(forced) should return true to tear the stream down and settle.
   * Return false on a transient error so EventSource can reconnect.
   * closeEs() calls onError(true) so a waiting runOne/compare promise cannot hang.
   */
  function listenEvents(jobId, onEvent, onError) {
    const es = new EventSource(API + "/api/jobs/" + jobId + "/events");
    es._onError = typeof onError === "function" ? onError : null;
    es.onmessage = (ev) => {
      try {
        onEvent(JSON.parse(ev.data));
      } catch (_) {}
    };
    es.onerror = () => {
      if (typeof es._onError !== "function") return;
      let settle = false;
      try {
        settle = es._onError(false) === true;
      } catch (_) {
        settle = true;
      }
      if (!settle) return;
      es._onError = null;
      try {
        es.close();
      } catch (_) {}
    };
    return es;
  }

  function closeEs(es) {
    if (!es) return;
    const hook = es._onError;
    es._onError = null;
    try {
      es.close();
    } catch (_) {}
    if (typeof hook === "function") {
      try {
        hook(true);
      } catch (_) {}
    }
  }

  function trackEs(comp, es) {
    if (!comp) return es;
    if (!comp._extraEs) comp._extraEs = [];
    comp._extraEs.push(es);
    return es;
  }

  function closeExtraEs(comp) {
    if (!comp) return;
    const extras = comp._extraEs || [];
    comp._extraEs = [];
    for (const es of extras) closeEs(es);
  }

  function closeAllEs(comp) {
    if (!comp) return;
    if (comp._jobEs) {
      const es = comp._jobEs;
      comp._jobEs = null;
      closeEs(es);
    }
    closeExtraEs(comp);
  }

  /** True when this listener must not write results (cleared, replaced, or cancelled). */
  function runResultStale(comp, id, gen) {
    if (!comp) return true;
    if (gen != null && comp._runGen !== gen) return true;
    if (comp._jobId !== id) return true;
    if (comp._cancelGen === gen) return true;
    return false;
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
        failed[m.name] = i18n.t("models.failed_to_load");
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
              m.tags && m.tags[0] ? m.tags[0] : i18n.t("models.kind.custom"),
              m.scale || 4,
              m.size_bytes ? Math.max(1, Math.round(m.size_bytes / 1e6)) : 0,
              m.source === "built-in" ? i18n.t("models.source.builtin") : i18n.t("models.source.external"),
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
        (sys.vram_free_mb != null ? " · " + i18n.t("sys.vram_free", { mb: Math.round(sys.vram_free_mb) }) : "");
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
            i18n.t("sys.onnx_missing", { dir: sys.models_dir || "models/" });
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

  B._unit = typeof _unit === "function" ? _unit : B._unit;
  B.postJob = typeof postJob === "function" ? postJob : B.postJob;
  B.listenEvents = typeof listenEvents === "function" ? listenEvents : B.listenEvents;
  B.trackEs = typeof trackEs === "function" ? trackEs : B.trackEs;
  B.closeAllEs = typeof closeAllEs === "function" ? closeAllEs : B.closeAllEs;
  B.hydrateSystem = typeof hydrateSystem === "function" ? hydrateSystem : B.hydrateSystem;
  B.addModel = typeof addModel === "function" ? addModel : B.addModel;
  B.uploadModel = typeof uploadModel === "function" ? uploadModel : B.uploadModel;
  B.downloadModel = typeof downloadModel === "function" ? downloadModel : B.downloadModel;
  B.applyModelStatus = typeof applyModelStatus === "function" ? applyModelStatus : B.applyModelStatus;
  B.revokeStateBlobs = typeof revokeStateBlobs === "function" ? revokeStateBlobs : B.revokeStateBlobs;
  B.isCancelledError = typeof isCancelledError === "function" ? isCancelledError : B.isCancelledError;

})(window.__UpscaleBridge = window.__UpscaleBridge || {});
