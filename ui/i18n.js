/*! llm-localize web runtime v1.0.0 — shipped by `llm-localize runtime install`.
 * Do not edit this copy: configure it with window.I18N_CONFIG (i18n.config.js) and
 * refresh it with `llm-localize runtime update <project>`. Spec: docs/web-runtime.md.
 *
 * Catalogues: flat JSON, key → ICU message, one file per locale. The source locale
 * (default 'en') is also the fallback for keys a locale lacks.
 *
 *   i18n.t('nav.search')                         → "Search" / "Szukaj"
 *   i18n.t('pager.showing', {n: 3, total: 9})    → "Showing 3 of 9"
 *   i18n.t('docs.count', {n})  with "{n, plural, one {# doc} other {# docs}}"
 *
 * Markup:  <button data-i18n="nav.search">Search</button>
 *          <input data-i18n-placeholder="search.ph" placeholder="Search…">
 *          <a data-i18n-title="x.tip" data-i18n-aria-label="x.aria">…</a>
 * Keep the source text in the markup so the page reads fine before load. An element
 * with child elements is never rewritten (put the key on the innermost element).
 *
 * I18N_CONFIG options (all optional):
 *   base        '/static/locales'   catalogue directory; files are <base>/<code>.json
 *   url         —                   URL template instead of base, e.g. '/api/locales/{locale}'
 *   source      'en'                source / fallback locale
 *   locales     ['en']              real locales, offered by pickers (i18n.locales)
 *   qaLocales   ['en-XX']           URL-only QA marker locales (ADR-110): never persisted
 *   storageKey  'app_lang'          localStorage key (and cookie name) for the choice
 *   cookie      false               also read/write the choice as a cookie (server-side picks)
 *   prefsKey    —                   localStorage JSON object whose .lang holds the choice
 *   navigator   true                fall back to navigator.language
 *   pathPrefix  false               /<code>/page.html selects the locale; setLocale keeps it
 *   sync        false               blocking XHR load, for runtimes that render before a promise
 *   observe     = sync              patch data-i18n* nodes as they are inserted (MutationObserver)
 *   applySource true                also apply the DOM in the source locale
 *   skipSelector —                  never touch nodes inside this selector (raw templates)
 *   rtl         ['ar','he',…]       base codes that get dir="rtl"
 *   textScale   {zh:1.15,ja:1.1,ko:1.1}  body zoom per base code (px-based CSS)
 *   overlay     —                   {enabled, api, project, flatNs, localOnly}: the
 *                                   llm-localize translator overlay, localhost + ?translate=1
 */
(function (global) {
  'use strict';

  var VERSION = '1.0.0';
  var cfg = global.I18N_CONFIG || {};
  var SOURCE = cfg.source || 'en';
  var BASE = cfg.base || '/static/locales';
  var URL_TEMPLATE = cfg.url || null;
  var STORAGE_KEY = cfg.storageKey || 'app_lang';
  var SUPPORTED = (cfg.locales || [SOURCE]).slice();
  if (SUPPORTED.indexOf(SOURCE) < 0) SUPPORTED.unshift(SOURCE);
  var QA_LOCALES = cfg.qaLocales || ['en-XX'];
  var QA_KEY = STORAGE_KEY + '_qa';
  var SYNC = !!cfg.sync;
  var OBSERVE = cfg.observe != null ? !!cfg.observe : SYNC;
  var APPLY_SOURCE = cfg.applySource !== false;
  var SKIP = cfg.skipSelector || null;
  var RTL = cfg.rtl || ['ar', 'he', 'fa', 'ur', 'ckb', 'ps', 'dv', 'yi', 'sd', 'ug'];
  var TEXT_SCALE = cfg.textScale || { zh: 1.15, ja: 1.1, ko: 1.1 };
  var doc = global.document || null;

  // ---- storage helpers (private mode / blocked storage never breaks the page) ----
  function lsGet(k) { try { return global.localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { global.localStorage.setItem(k, v); } catch (e) { /* blocked */ } }
  function lsDel(k) { try { global.localStorage.removeItem(k); } catch (e) { /* blocked */ } }
  function ssGet(k) { try { return global.sessionStorage.getItem(k); } catch (e) { return null; } }
  function ssSet(k, v) { try { global.sessionStorage.setItem(k, v); } catch (e) { /* blocked */ } }
  function ssDel(k) { try { global.sessionStorage.removeItem(k); } catch (e) { /* blocked */ } }
  function query(name) {
    try { return new URLSearchParams(global.location.search).get(name); } catch (e) { return null; }
  }
  function readCookie(name) {
    try {
      var m = doc.cookie.match(new RegExp('(?:^|;\\s*)' + name + '=([^;]*)'));
      return m ? decodeURIComponent(m[1]) : null;
    } catch (e) { return null; }
  }
  function readPrefs() {
    if (!cfg.prefsKey) return null;
    try { return (JSON.parse(lsGet(cfg.prefsKey) || '{}') || {}).lang || null; } catch (e) { return null; }
  }

  // ---- locale choice ----
  function qaLocale(code) {
    if (!code) return null;
    for (var i = 0; i < QA_LOCALES.length; i++) {
      if (QA_LOCALES[i].toLowerCase() === String(code).toLowerCase()) return QA_LOCALES[i];
    }
    return null;
  }
  function supported(code) {
    if (!code) return null;
    if (SUPPORTED.indexOf(code) >= 0) return code;
    var base = String(code).split(/[-_]/)[0].toLowerCase();
    return SUPPORTED.indexOf(base) >= 0 ? base : null;
  }
  function pathLocale() {
    if (!cfg.pathPrefix) return null;
    try {
      var m = global.location.pathname.match(/^\/([A-Za-z0-9_-]+)\//);
      if (!m) return null;
      return SUPPORTED.indexOf(m[1]) >= 0 ? m[1] : qaLocale(m[1]);
    } catch (e) { return null; }
  }

  // Order: path prefix → ?lang= → cookie → localStorage → prefs → navigator → source.
  // QA locales come only from the URL and are held for the tab (sessionStorage).
  function pickLocale() {
    var fromPath = pathLocale();
    if (qaLocale(fromPath)) return qaLocale(fromPath);
    var q = query('lang');
    var qa = qaLocale(q);
    if (qa) { ssSet(QA_KEY, qa); return qa; }
    if (fromPath || supported(q)) {
      ssDel(QA_KEY);
    } else {
      var held = qaLocale(ssGet(QA_KEY));
      if (held) return held;
    }
    var nav = cfg.navigator === false ? null : (global.navigator && global.navigator.language) || null;
    var want = [fromPath, q, cfg.cookie ? readCookie(STORAGE_KEY) : null, lsGet(STORAGE_KEY), readPrefs(), nav];
    for (var i = 0; i < want.length; i++) {
      var hit = supported(want[i]);
      if (hit) return hit;
    }
    return SOURCE;
  }

  var locale = pickLocale();
  var baseCode = String(locale).split('-')[0].toLowerCase();
  var dir = RTL.indexOf(baseCode) >= 0 ? 'rtl' : 'ltr';

  // ---- document layout: lang, dir, CJK zoom (before the first paint) ----
  function applyDocumentLayout() {
    var html = doc && doc.documentElement;
    if (!html) return;
    html.lang = locale;
    html.dir = dir;
    var scale = TEXT_SCALE[baseCode];
    if (scale && scale !== 1 && !doc.getElementById('i18n-text-scale')) {
      var style = doc.createElement('style');
      style.id = 'i18n-text-scale';
      style.textContent = 'html:lang(' + baseCode + ') body{zoom:' + scale + '}';
      (doc.head || html).appendChild(style);
    }
  }
  applyDocumentLayout();

  // ---- catalogue loading ----
  function urlFor(code) {
    if (URL_TEMPLATE) return URL_TEMPLATE.replace('{locale}', encodeURIComponent(code));
    return BASE.replace(/\/+$/, '') + '/' + code + '.json';
  }
  function asDict(j) { return j && typeof j === 'object' && !Array.isArray(j) ? j : {}; }
  function loadSync(code) {
    try {
      var xhr = new global.XMLHttpRequest();
      xhr.open('GET', urlFor(code), false);
      xhr.send(null);
      if (xhr.status >= 200 && xhr.status < 300) return asDict(JSON.parse(xhr.responseText));
    } catch (e) { /* offline / file:// — the markup's source text stays */ }
    return {};
  }
  function loadAsync(code) {
    try {
      return global.fetch(urlFor(code), { cache: 'no-cache' })
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(asDict)
        .catch(function () { return {}; });
    } catch (e) { return Promise.resolve({}); }
  }

  var dicts = { src: {}, cur: {} };

  // ---- ICU subset: {name}, plural (=n, categories, #), selectordinal, select ----
  var rules = {};
  function pluralCategory(n, ordinal) {
    var k = locale + (ordinal ? ':o' : '');
    if (!(k in rules)) {
      try { rules[k] = new Intl.PluralRules(locale, { type: ordinal ? 'ordinal' : 'cardinal' }); }
      catch (e) { rules[k] = null; }
    }
    if (rules[k]) return rules[k].select(n);
    return n === 1 ? 'one' : 'other';
  }

  function matchBrace(s, open) {
    var depth = 0;
    for (var j = open; j < s.length; j++) {
      if (s[j] === '{') depth++;
      else if (s[j] === '}' && --depth === 0) return j;
    }
    return -1;
  }

  // '#' stands for the number only at the branch's own level, not inside nested messages.
  function replaceHash(branch, value) {
    var out = '';
    var depth = 0;
    for (var i = 0; i < branch.length; i++) {
      var ch = branch[i];
      if (ch === '{') depth++;
      else if (ch === '}') depth--;
      out += ch === '#' && depth === 0 ? value : ch;
    }
    return out;
  }

  function format(msg, vars) {
    msg = msg == null ? '' : String(msg);
    if (!vars || msg.indexOf('{') < 0) return msg;
    var out = '';
    var i = 0;
    while (i < msg.length) {
      var ch = msg[i];
      if (ch !== '{') { out += ch; i++; continue; }
      var end = matchBrace(msg, i);
      if (end < 0) { out += msg.slice(i); break; }
      out += expand(msg.slice(i + 1, end), vars);
      i = end + 1;
    }
    return out;
  }

  function expand(body, vars) {
    var first = body.indexOf(',');
    var name = (first < 0 ? body : body.slice(0, first)).trim();
    var second = first < 0 ? -1 : body.indexOf(',', first + 1);
    if (second < 0) return vars[name] == null ? '' : String(vars[name]);
    var kind = body.slice(first + 1, second).trim();
    var rest = body.slice(second + 1);
    var branches = {};
    var re = /\s*(=?[\w-]+)\s*\{/g;
    var m;
    while ((m = re.exec(rest))) {
      var start = m.index + m[0].length - 1;
      var close = matchBrace(rest, start);
      if (close < 0) break;
      branches[m[1]] = rest.slice(start + 1, close);
      re.lastIndex = close + 1;
    }
    var v = vars[name];
    var pick;
    if (kind === 'plural' || kind === 'selectordinal') {
      var n = Number(v);
      pick = branches['=' + v];
      if (pick == null) pick = branches[pluralCategory(n, kind === 'selectordinal')];
      if (pick == null) pick = branches.other || '';
      return format(replaceHash(pick, String(v)), vars);
    }
    pick = branches[String(v)];
    if (pick == null) pick = branches.other || '';
    return format(pick, vars);
  }

  function has(key) {
    var msg = dicts.cur[key];
    if (msg != null && msg !== '') return true;
    return dicts.src[key] != null;
  }

  function t(key, vars) {
    var msg = dicts.cur[key];
    if (msg == null || msg === '') msg = dicts.src[key];
    if (msg == null) return key;
    return format(msg, vars);
  }

  // ---- DOM ----
  var ATTRS = ['title', 'placeholder', 'aria-label', 'alt'];
  var SELECTOR = '[data-i18n],' + ATTRS.map(function (a) { return '[data-i18n-' + a + ']'; }).join(',');

  function setText(el, value) {
    var kids = el.childNodes;
    for (var i = 0; i < kids.length; i++) {
      if (kids[i].nodeType === 1) return; // has child elements: never clobber them
    }
    if (kids.length === 1) {
      if (kids[0].nodeValue !== value) kids[0].nodeValue = value;
    } else if (el.textContent !== value) {
      el.textContent = value;
    }
  }

  function applyOne(el) {
    var k = el.getAttribute('data-i18n');
    if (k) setText(el, t(k));
    for (var i = 0; i < ATTRS.length; i++) {
      var ak = el.getAttribute('data-i18n-' + ATTRS[i]);
      if (ak) {
        var v = t(ak);
        if (el.getAttribute(ATTRS[i]) !== v) el.setAttribute(ATTRS[i], v);
      }
    }
  }

  function skipped(el) { return !!(SKIP && el.closest && el.closest(SKIP)); }

  function applyDOM(root) {
    root = root || doc;
    if (!root || !root.querySelectorAll) return;
    if (locale !== SOURCE || APPLY_SOURCE) {
      if (root.nodeType === 1 && root.matches && root.matches(SELECTOR) && !skipped(root)) applyOne(root);
      var list = root.querySelectorAll(SELECTOR);
      for (var i = 0; i < list.length; i++) if (!skipped(list[i])) applyOne(list[i]);
    }
    if (root === doc) applyDocumentLayout();
  }

  function setLocale(code) {
    var want = supported(code);
    if (!want) return;
    lsSet(STORAGE_KEY, want);
    ssDel(QA_KEY);
    if (cfg.prefsKey) {
      try {
        var p = JSON.parse(lsGet(cfg.prefsKey) || '{}') || {};
        p.lang = want;
        lsSet(cfg.prefsKey, JSON.stringify(p));
      } catch (e) { /* bad json */ }
    }
    if (cfg.cookie) {
      try { doc.cookie = STORAGE_KEY + '=' + encodeURIComponent(want) + ';path=/;max-age=31536000;SameSite=Lax'; }
      catch (e) { /* blocked */ }
    }
    if (want === locale) return;
    var url = new URL(global.location.href);
    url.searchParams.delete('lang');
    if (pathLocale()) url.pathname = url.pathname.replace(/^\/[^/]+/, '/' + want);
    global.location.assign(url.toString());
  }

  function whenDomReady(fn) {
    if (!doc) return;
    if (doc.readyState !== 'loading') fn();
    else doc.addEventListener('DOMContentLoaded', function () { fn(); });
  }

  function startObserver() {
    if (!OBSERVE || !doc || typeof global.MutationObserver === 'undefined') return;
    if (locale === SOURCE && !APPLY_SOURCE) return;
    // Callbacks run as microtasks, so inserted nodes are patched before paint.
    new global.MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        var added = records[i].addedNodes;
        for (var j = 0; j < added.length; j++) if (added[j].nodeType === 1) applyDOM(added[j]);
      }
    }).observe(doc.documentElement, { childList: true, subtree: true });
  }

  var ready;
  if (SYNC) {
    dicts.src = loadSync(SOURCE);
    dicts.cur = locale === SOURCE ? dicts.src : loadSync(locale);
    startObserver();
    whenDomReady(function () { applyDOM(); });
    ready = Promise.resolve();
  } else {
    ready = Promise.all([loadAsync(SOURCE), locale === SOURCE ? Promise.resolve(null) : loadAsync(locale)])
      .then(function (res) {
        dicts.src = res[0] || {};
        dicts.cur = res[1] || dicts.src;
        startObserver();
        whenDomReady(function () { applyDOM(); });
      });
  }

  // ---- llm-localize translator overlay (CTRL to inspect, CTRL+click to edit) ----
  // Opt-in via I18N_CONFIG.overlay; loaded only on localhost (unless localOnly:false),
  // only after ?translate=1 (remembered; ?translate=0 turns it off), and it needs
  // `llm-localize web` answering at overlay.api.
  var OVERLAY_FLAG = 'llm_overlay';
  var LOOPBACK = ['localhost', '127.0.0.1', '[::1]', '::1'];
  function overlayWanted(o) {
    if (!o || !o.enabled || !doc) return false;
    var host = '';
    try { host = global.location.hostname; } catch (e) { return false; }
    if (o.localOnly !== false && LOOPBACK.indexOf(host) < 0) return false;
    var q = query('translate');
    if (q === '1') lsSet(OVERLAY_FLAG, '1');
    if (q === '0') lsDel(OVERLAY_FLAG);
    var stored = lsGet(OVERLAY_FLAG);
    return stored === '1' || (stored == null && q === '1');
  }
  function loadOverlay() {
    var o = cfg.overlay;
    if (!overlayWanted(o)) return false;
    var api = String(o.api || 'http://127.0.0.1:8770').replace(/\/+$/, '');
    global.__LLM_TRANSLATOR__ = {
      locale: function () { return locale; },
      catalogLocale: function () { return locale; },
      project: function () { return o.project || ''; },
      applyTranslation: function (e) {
        if (!e || !e.key) return;
        if (dicts.cur === dicts.src) dicts.cur = Object.assign({}, dicts.src);
        dicts.cur[e.key] = e.value;
        applyDOM();
      },
    };
    var s = doc.createElement('script');
    s.src = api + '/overlay.js';
    s.async = true;
    s.setAttribute('data-project', o.project || '');
    s.setAttribute('data-locale', locale);
    s.setAttribute('data-api', api);
    s.setAttribute('data-flat-ns', o.flatNs || '');
    (doc.head || doc.documentElement).appendChild(s);
    return true;
  }
  var overlayLoaded = loadOverlay();

  global.i18n = {
    version: VERSION,
    t: t,
    has: has,
    format: format,
    applyDOM: applyDOM,
    setLocale: setLocale,
    pluralCategory: pluralCategory,
    get locale() { return locale; },
    get dir() { return dir; },
    get isQa() { return QA_LOCALES.indexOf(locale) >= 0; },
    get overlay() { return overlayLoaded; },
    source: SOURCE,
    locales: SUPPORTED.slice(),
    whenReady: function () { return ready; },
  };
})(typeof window !== 'undefined' ? window : globalThis);
