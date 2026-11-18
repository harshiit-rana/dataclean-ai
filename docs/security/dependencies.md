# Dependency vulnerabilities: what was fixed, and what is accepted for now

Dependabot alerts were switched on for the repository on 16 October 2026 (#109). On the first scan GitHub reported 17
alerts, all in the frontend's npm dependencies.

## Fixed (16 October 2026)

| Package | From | To | What it fixed |
| --- | --- | --- | --- |
| react-router-dom / react-router | 6.30.4 | 6.30.6 | Open redirect leading to XSS (the 6.30.x advisory) |
| vite | 5.4.21 | 6.4.4 | `server.fs.deny` bypass on Windows, path traversal in `.map` handling, NTLM hash disclosure via launch-editor (all in the development server) |
| esbuild (via vite) | 0.21.5 | 0.25.12 | Any website could send requests to the development server and read the response |
| vitest / @vitest/mocker | 4.1.10 | 4.1.12 | Path traversal through a redirect mock (test runner only) |
| undici (via jsdom) | 7.29.0 | 7.30.1 | TLS validation bypass in BalancedPool and five lower-severity issues (test environment only) |
| source-map-js | 1.2.1 | 1.2.2 | Event-loop denial of service from crafted source maps (build only) |
| nanoid | below 3.3.18 | 3.3.20 | A custom generator looping when size is zero (build only) |

The frontend suite (520 tests) and the production build pass on the new versions.

## Fixed (18 November 2026)

Dependabot's first version-update PRs arrived once security updates were switched on (#109).

| Package | From | To | What it fixed |
| --- | --- | --- | --- |
| zod | 3.25.76 | removed | Memory exhaustion validating oversized collections (high). zod was in `package.json` from the first scaffold but nothing imports it, so it was removed rather than moved to version 4 |
| vite | 6.4.4 | 6.4.6 | `server.fs.deny` bypassed with queries (development server) |
| postcss-selector-parser (via tailwindcss) | 6.1.4 | 6.1.6 | Quadratic parsing of flat selectors (build only) |

The frontend suite (541 tests), the type check and the production build pass on the new versions.

## Accepted for now, each needing a major upgrade

| Package | Advisory | Why it is accepted for now | What would fix it |
| --- | --- | --- | --- |
| react-router 6 | Open redirect via a backslash in `<Link>` / `useNavigate`; constructor injection in SSR hydration | The app has no server-side rendering. Every navigation target is a path the app builds itself, starting with a fixed prefix (`/model/`, `/review/`, ...) and a server-made UUID. The one id a user types ("Open a project with its key") is checked against the UUID format before it is stored or navigated to. | React Router 7 (a major upgrade, with route API changes) |
| tailwindcss 3 (braces, micromatch, chokidar, fast-glob, postcss-selector-parser, postcss-nested) | Denial of service from deeply nested glob patterns or crafted selectors | Build-time only, and the only globs and CSS they process are the project's own. Nothing from a user reaches them, and none of it ships to the browser. | Tailwind CSS 4 (a major upgrade with a new configuration format) |
| picomatch (via vite, vitest, tailwindcss and their file watchers) | ReDoS through repeated extglobs (high; npm audit lists it under fdir, tinyglobby, anymatch, readdirp, chokidar, sucrase, micromatch and fast-glob too) | Build- and test-time only: it matches the project's own file globs, never anything a user sends, and none of it ships to the browser. No fixed release exists yet: 4.0.7 and 2.3.2, the newest of each line, are both in the affected range (18 November 2026). | A patched picomatch release, which Dependabot will propose |

These are worth doing as their own pieces of work, each with the frontend suite and a look at every page. They are
not done here because a major upgrade of the router or the styling system is not a security patch, and doing it in
the same change would hide whatever it breaks.

## How updates arrive from here

Dependabot alerts are on. Dependabot's weekly version-update PRs (`.github/dependabot.yml`) started on 18 November 2026, once security
updates were switched on in the repository settings (#109). Each update PR is reviewed by hand for compatibility, security effect and regression risk
before it is merged. A green CI run is necessary, not sufficient, and nothing is merged automatically.
