# enhance_prompt — bugs found this session + production checklist

## Bugs found and fixed

| # | Bug | Fix |
|---|-----|-----|
| 1 | `tests/test_config.py` had a real-looking Supabase service role key (`sb_secret_...`) hardcoded as a literal string. It had already been committed to git history. | Replaced with an obvious fake placeholder. **The key itself must still be rotated in Supabase** — removing it from the working tree does not undo the earlier commit. |
| 2 | `worker.py`'s fal submission never stripped `source_feature_type` from `input_params`, and never forwarded the tool's own `ai_steps.model` — so the payload sent to `openrouter/router` carried a stray unknown field and no explicit `model`. | Added `source_feature_type` to `FAL_NON_SCHEMA_FIELDS`; inject `tool.ai_steps["model"]` into the outgoing params when present. |
| 3 | `recolor.py`'s `build_instruction` signature changed to `(job, tool_definition, db)`, but `tests/test_recolor.py` (5 tests) and `tests/test_fal_integration.py` still called it with 2 args — all silently broken until run. | Updated every call site to pass a `db` argument. |
| 4 | `/generate` hardcoded `color/quality/size/target_area` as the only accepted form fields and required at least one image (`File(...)`) — `enhance_prompt` needs `prompt`/`source_feature_type` and zero images, so **the tool could not be invoked through the real API at all**. | Added `prompt`/`source_feature_type` as accepted `Form` fields; made `images` optional (`File(default=[])`). |
| 5 | `/jobs/{id}` and `/batches/{id}` never returned `output_text` — even a successfully completed enhance_prompt job's result was invisible to any client. | Added `outputText` to both response payloads. |
| 6 | `app/worker.py` never imported the `User` model. SQLAlchemy never registered the `users` table in that process, so `generation_jobs.user_id`'s `ForeignKey("users.id")` failed to resolve the first time the local timeout-checker cron actually touched a job row — poisoning the DB session (`PendingRollbackError`) and leaving jobs stuck `queued` forever, holding the per-user generation lock. This had been silently masked in every earlier test because the real fal webhook (delivered to the *deployed* server) always resolved jobs before the local cron got a chance to touch them. | Added a central `app/models/__init__.py` that imports every model (so table registration can never again depend on import order), and made `worker.py` import it. **Not yet committed** — see below. |
| 7 | Two zombie `generation_jobs` rows (created before fix #6) were stuck `queued` forever, permanently holding the generation lock for the test account. | Manually failed + refunded + released via `fail_and_release` (one-off DB fix, not a code change — this won't recur now that #6 is fixed, but any future crash mid-processing could still leave a job stuck until the 5-min lock TTL or the 10-min sweep catches it). |
| 8 | Local `.env` had `PUBLIC_BACKEND_URL` pointing at the deployed Render URL, so local testing sent fal's webhook to a completely different (stale) running process instead of the local one — the actual cause of the "fal succeeded but job failed" symptom reported mid-session. | Changed local `.env` to `http://127.0.0.1:8000` so local testing resolves entirely through the local timeout-checker cron instead of a real inbound webhook. **This is a local-only dev convenience — see checklist item 1 below, it must not ship this way.** |

Full test suite: **271 passed** as of this audit.

## Before this goes to production

1. **Revert the local webhook workaround for any deployed environment.** Item 8 above only makes sense for a laptop that fal's cloud can't reach. Render's own `PUBLIC_BACKEND_URL` env var (set in Render's dashboard, not `.env`) must stay pointed at the real public backend URL — confirm it's still correct there; don't copy the local `.env` value over it.

2. **Deploy this branch.** `git status` shows this branch (`feature/enhance-prompt-tool`) is already pushed to `origin` up to commit `9799c4b`, which contains the enhance_prompt feature itself (tool file, registry, model column, routes, generation.py webhook branch). **Two fixes from this session are still uncommitted**: `app/models/__init__.py` and `app/worker.py` (bug #6, the FK-registration fix). Commit and push those before deploying, or the same worker crash will reproduce in production the first time the timeout-checker beats a webhook to a job.

3. **Apply the DB migration to the production database.** This repo has no Alembic/migration files — schema changes are applied by hand. Confirm, on the actual production Supabase project (not just the dev one this session used):
   - `generation_jobs.output_text` column exists (`ALTER TABLE generation_jobs ADD COLUMN output_text TEXT;` if not).
   - The `tool_definitions` row for `enhance_prompt` exists and matches (see the INSERT already run in dev — same statement needs to run against prod).
   - Confirm whether dev and prod actually share one Supabase project or are separate — if separate, every DB step in this checklist needs to be repeated against prod specifically.

4. **Rotate the leaked Supabase service role key** (bug #1) if it hasn't been already — this is unrelated to enhance_prompt but was discovered during this work and is still outstanding.

5. **Decide `enhance_prompt`'s real pricing.** `credit_cost_per_output` is currently `0` (free) in the row used for testing — confirm that's intentional for launch, not a placeholder left over from testing.

6. **Add `ai_steps.enhance_hint` to every tool that should support "enhance my prompt" before launch.** Right now only `recolor` and the not-yet-built `creative_photoshoot` placeholder have it. Any other `source_feature_type` a user could send will fall back to the generic hint (`enhance_prompt.py`'s fallback string) with a warning logged — check that's the desired behavior for tools without a specific hint yet, or add hints for them first.

7. **Deactivate/remove the `test_tool` row** (`is_active=true`, `fal_model_id='fake-model-for-testing'`) from whichever database is actually production, if it isn't already excluded — it's callable through `/generate` by anyone who knows the `feature_type` string, even though it's not linked from the real frontend.

8. **Frontend work, not yet started:** the `outputText` field is now returned by `/jobs/{id}` and `/batches/{id}`, but nothing on the actual product frontend (as opposed to the throwaway `test-console/testconsole.html`) has been built to display it. This backend work is not useful to end users until that UI exists.

9. **`creative_photoshoot` stays `is_active=false`** until that tool is actually built — don't flip it to `true` as a side effect of any other deploy.

10. **General hardening not specific to this feature, but relevant to relying on it in production:**
    - A job that crashes mid-processing (like bug #6 did) still needs its lock/DB row cleaned up by hand today — the 5-minute lock TTL and 10-minute sweep are the only automatic backstops. Consider whether that's an acceptable recovery time for production traffic.
    - No error monitoring/alerting (e.g. Sentry) is wired up — a worker crash like bug #6 would previously have been silent until someone noticed stuck jobs.
