const STORAGE = {
  theme: "upscale.theme",
  fs: "upscale.fs",
  lang: "upscale.lang",
};

export function applyTheme() {
  const t = localStorage.getItem(STORAGE.theme) || "system";
  let dark = t === "dark";
  if (t === "system") {
    dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

export function applyTextSize() {
  const fs = localStorage.getItem(STORAGE.fs) || "1";
  document.documentElement.style.setProperty("--fs", fs);
}

export async function loadLocale(code) {
  const lang = code || localStorage.getItem(STORAGE.lang) || "en";
  try {
    const r = await fetch(`/api/locales/${lang}`);
    if (!r.ok) return {};
    return r.json();
  } catch {
    return {};
  }
}

export function t(dict, key, fallback) {
  return dict[key] || fallback || key;
}

export function bindSettingsPanel() {
  const themeSel = document.getElementById("set-theme");
  const fsSel = document.getElementById("set-fs");
  const langSel = document.getElementById("set-lang");
  if (themeSel) {
    themeSel.value = localStorage.getItem(STORAGE.theme) || "system";
    themeSel.addEventListener("change", () => {
      localStorage.setItem(STORAGE.theme, themeSel.value);
      applyTheme();
    });
  }
  if (fsSel) {
    fsSel.value = localStorage.getItem(STORAGE.fs) || "1";
    fsSel.addEventListener("change", () => {
      localStorage.setItem(STORAGE.fs, fsSel.value);
      applyTextSize();
    });
  }
  if (langSel) {
    langSel.value = localStorage.getItem(STORAGE.lang) || "en";
    langSel.addEventListener("change", () => {
      localStorage.setItem(STORAGE.lang, langSel.value);
      document.dispatchEvent(new CustomEvent("upscale:lang"));
    });
  }
  applyTheme();
  applyTextSize();
}

export async function openHelpDrawer(topic) {
  const backdrop = document.getElementById("help-backdrop");
  const drawer = document.getElementById("help-drawer");
  const body = document.getElementById("help-body");
  if (!backdrop || !drawer || !body) return;
  const lang = localStorage.getItem(STORAGE.lang) || "en";
  body.textContent = "Loading…";
  backdrop.classList.add("open");
  drawer.classList.add("open");
  try {
    const r = await fetch(`/api/help/${topic}?lang=${encodeURIComponent(lang)}`);
    const data = await r.json();
    body.textContent = data.markdown || "No content.";
  } catch (e) {
    body.textContent = String(e);
  }
  backdrop.onclick = () => {
    backdrop.classList.remove("open");
    drawer.classList.remove("open");
  };
}
