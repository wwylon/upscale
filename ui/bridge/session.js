/** Upscale bridge — session. Attaches to window.__UpscaleBridge. */

(function (B) {
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
    } else if (!s.file && !s.jobId) {
      // Cleared workspace — do not keep a lone global jobId for ghost restore.
      // A simple-view image has state.file / state.jobId and an empty files list.
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
    if (t === "loading_model") return i18n.t("stage.loading_model");
    if (t === "pre_filters") return i18n.t("stage.applying_filters");
    if (t === "filters") return i18n.t("stage.applying_filters");
    if (t === "writing") return i18n.t("stage.writing");
    if (t === "paused") return i18n.t("stage.paused");
    if (t === "downloading") return i18n.t("stage.downloading");
    if (t === "tile" && ev.pass != null && ev.tile != null) {
      return i18n.t("stage.pass_tile", {
        pass: ev.pass + (ev.passes != null ? "/" + ev.passes : ""),
        tile: ev.tile + (ev.tiles != null ? "/" + ev.tiles : ""),
      });
    }
    if (ev.message) return ev.message;
    if (ev.pass) return i18n.t("stage.pass", { pass: ev.pass + (ev.passes != null ? "/" + ev.passes : "") });
    return i18n.t("stage.working");
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
    const gen = (comp._runGen = (comp._runGen || 0) + 1);
    closeAllEs(comp);
    revokeStateBlobs(comp, ["displayUrl", "urlAfter", "resultUrl"]);
    comp.setState({ status: "running", paused: false, blocked: false, pct: 0, stage: i18n.t("stage.starting") });
    const { id } = await postJob(file, settings);
    if (comp._runGen !== gen) {
      const err = new Error("cancelled");
      err.cancelled = true;
      throw err;
    }
    comp._jobId = id;
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (fn, val) => {
        if (settled) return;
        settled = true;
        if (comp._jobEs) comp._jobEs._onError = null;
        fn(val);
      };
      // forced===true means the stream was closed (new run / clear extras).
      // Otherwise settle only after cancel or when this generation was replaced,
      // so a transient EventSource error can reconnect instead of hanging.
      const onStreamError = (forced) => {
        const cancelThis = comp._cancelGen === gen;
        const replaced = comp._runGen !== gen;
        if (forced !== true && !cancelThis && !replaced) return false;
        if (settled) return true;
        const err = new Error("cancelled");
        err.cancelled = true;
        if (cancelThis && comp._runGen === gen && comp._jobId === id) {
          comp.setState({ status: "idle", pct: 0, paused: false, stage: i18n.t("stage.cancelled") });
        }
        finish(reject, err);
        return true;
      };
      comp._jobEs = listenEvents(
        id,
        (ev) => {
        const t = ev.type || "progress";
        const stale = () => comp._runGen !== gen || comp._jobId !== id;
        const quiet = () => stale() || comp._cancelGen === gen;
        if (t === "paused") {
          if (quiet()) return;
          comp.setState({ paused: true, status: "running", stage: stageMessage(ev) });
          return;
        }
        if (t === "resumed" || (t === "progress" && ev.message === "resumed")) {
          if (quiet()) return;
          comp.setState({ paused: false, status: "running", stage: stageMessage(ev) || i18n.t("stage.resumed") });
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
          if (quiet()) return;
          const msg = stageMessage(ev);
          comp.setState((st) => {
            let pct = Math.floor((ev.pct != null ? ev.pct : 0) * 100);
            if (t === "loading_model" && ev.pct == null) pct = Math.max(st.pct || 0, 2);
            const patch = { pct, stage: msg, status: "running" };
            if (st.paused) patch.paused = true;
            if (st.activeId && Array.isArray(st.files) && st.files.length) {
              patch.files = st.files.map((x) =>
                x.id === st.activeId && x.st !== "done" && x.st !== "error" && x.st !== "cancelled"
                  ? { ...x, pct, st: "run" }
                  : x
              );
            }
            return patch;
          });
          if (ev.artifact) {
            const url = API + "/api/jobs/" + id + "/passes/" + (ev.pass || 1);
            comp.setState((st) => ({
              film: [...(st.film || []), url],
            }));
          }
        } else if (t === "done") {
          if (comp._jobEs) {
            comp._jobEs._onError = null;
            try { comp._jobEs.close(); } catch (_) {}
          }
          if (runResultStale(comp, id, gen)) {
            if (comp._cancelGen === gen || comp._runGen !== gen) {
              const err = new Error("cancelled");
              err.cancelled = true;
              finish(reject, err);
            } else {
              finish(resolve, null);
            }
            return;
          }
          const resultUrl = API + "/api/jobs/" + id + "/result";
          const displayApi = API + "/api/jobs/" + id + "/display";
          // Show result URL immediately (quoted CSS); refine to blob when ready.
          comp.setState({
            status: "done",
            pct: 100,
            paused: false,
            stage: i18n.t("stage.loading_preview"),
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
              if (runResultStale(comp, id, gen)) {
                revokeUrl(displayUrl);
                if (comp._cancelGen === gen || comp._runGen !== gen) {
                  const err = new Error("cancelled");
                  err.cancelled = true;
                  finish(reject, err);
                } else {
                  finish(resolve, null);
                }
                return;
              }
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
                  new Notification(i18n.t("notify.finished"), {
                    body: comp.state.name || i18n.t("notify.image_ready"),
                  });
                }
              } catch (_) {}
              finish(resolve, resultUrl);
            })
            .catch((err) => finish(reject, err));
        } else if (t === "cancelled") {
          if (comp._jobEs) {
            comp._jobEs._onError = null;
            try { comp._jobEs.close(); } catch (_) {}
          }
          const err = new Error("cancelled");
          err.cancelled = true;
          if (!stale()) {
            comp.setState({ status: "idle", pct: 0, paused: false, stage: i18n.t("stage.cancelled") });
          }
          finish(reject, err);
        } else if (t === "error") {
          if (comp._jobEs) {
            comp._jobEs._onError = null;
            try { comp._jobEs.close(); } catch (_) {}
          }
          if (stale() || comp._cancelGen === gen) {
            const err = new Error("cancelled");
            err.cancelled = true;
            finish(reject, err);
            return;
          }
          comp.setState({ status: "idle", pct: 0, paused: false, stage: ev.message || t });
          finish(reject, new Error(ev.message || t));
        }
      },
        onStreamError
      );
    });
  }

  function isCancelledError(e) {
    return !!(e && (e.cancelled || /cancel/i.test(String(e.message || e))));
  }

  B.persistWorkspace = typeof persistWorkspace === "function" ? persistWorkspace : B.persistWorkspace;
  B.clearWorkspace = typeof clearWorkspace === "function" ? clearWorkspace : B.clearWorkspace;
  B.restoreSession = typeof restoreSession === "function" ? restoreSession : B.restoreSession;
  B.stashSession = typeof stashSession === "function" ? stashSession : B.stashSession;
  B.bindWorkspace = typeof bindWorkspace === "function" ? bindWorkspace : B.bindWorkspace;
  B.workspaceFromState = typeof workspaceFromState === "function" ? workspaceFromState : B.workspaceFromState;
  B.WS_SETTINGS_KEYS = typeof WS_SETTINGS_KEYS !== "undefined" ? WS_SETTINGS_KEYS : B.WS_SETTINGS_KEYS;

})(window.__UpscaleBridge = window.__UpscaleBridge || {});
