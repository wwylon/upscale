// Localization settings for upscale, loaded before i18n.js.
// Catalogues: locales/*.json (served at /api/locales/{code}) (en = source; pl translated by llm-localize; zh/ar are
// frozen layout checks; en-XX is the URL-only QA marker locale).
window.I18N_CONFIG = {
  url: '/api/locales/{locale}',
  storageKey: 'upscale_lang',
  locales: ['en', 'pl', 'zh', 'ar'],
  qaLocales: ['en-XX'],
  textScale: { zh: 1.15 },
  // The DC runtime (support.js) evaluates i18n.t() while rendering, before any
  // promise could resolve: load catalogues synchronously and patch keyed nodes
  // as React inserts them. English markup is already correct, so it is left as is.
  sync: true,
  observe: true,
  applySource: false,
  // <x-dc> holds raw template source that support.js re-reads: never touch it.
  skipSelector: 'x-dc',
  // The server redirects bare URLs to /<code>/ and reads the cookie.
  pathPrefix: true,
  cookie: true,
  prefsKey: 'upscale.prefs',
  navigator: false,
  // llm-localize translator overlay: CTRL to inspect, CTRL+click to edit in place.
  // Needs `llm-localize web` running; open the app with ?translate=1 to load it.
  // Saving from here also needs [web] overlay_edit = true in .llm-localize.toml.
  overlay: { enabled: true, api: 'http://127.0.0.1:8770', project: 'upscale', flatNs: '' },
};
