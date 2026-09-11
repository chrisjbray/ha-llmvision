# Local fork: thinking controls

- [x] Add a per-call reasoning effort override without changing provider defaults or existing budget settings.
- [x] Restore OpenRouter reasoning support with its canonical `reasoning` object, including Anthropic models. Add native Gemini 3 thinking levels while retaining existing budgets.
- [x] Keep thought text out of answers and authorization headers out of debug output.
- [x] Add regression tests and service documentation. Pinned unit suite: 549 passed on three consecutive runs, 4 integration tests deselected. Correct stale stream-test mocks and join already-closed SQLite workers before HA's leak check; keep checks for open-connection leaks.
- [x] Stack the fix on the existing local fork with verified transcript attribution. Local only: do not push or deploy to Home Assistant.
