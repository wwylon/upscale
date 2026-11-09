# Public marketing site (`site/`)

Static landing for GitHub Pages. **Not** the operator/help tree under `docs/`.

## Enable Pages (maintainer)

GitHub's branch Source only allows `/` or `/docs`, not `/site`. Deploy via
`.github/workflows/pages.yml` (Actions).

1. Push `site/` + the workflow to public `main` (house publish).
2. Repo **Settings -> Pages**: Source = **GitHub Actions** (or
   `gh api repos/wwylon/upscale/pages -X POST -f build_type=workflow`).
3. Confirm https://wwylon.github.io/upscale/
4. `gh repo edit wwylon/upscale --homepage https://wwylon.github.io/upscale/`
5. **Settings -> Social preview**: upload `../docs/assets/social.jpg`.
6. [Google Search Console](https://search.google.com/search-console): URL-prefix
   `https://wwylon.github.io/upscale/`, verify (`google....html` in this folder),
   submit `https://wwylon.github.io/upscale/sitemap.xml`, URL Inspection on `/`.
