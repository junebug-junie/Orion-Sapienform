# Concept-profile config adapter in Orch

## Why this exists

`concept_induction_pass` resolves concept profiles through the repository seam (`build_concept_profile_repository`).
That seam expects concept-profile settings (store path, backend controls, graph endpoint/auth, cutover policy,
and parity/readiness thresholds). The generic Orch service settings model is intentionally narrower and does
not own these fields.

Passing generic Orch settings directly to the repository seam can fail at runtime (for example, missing
`store_path`) even when the container environment already exports the required concept-profile variables.

## Boundary decision

Orch now uses a dedicated adapter model (`OrchConceptProfileSettings`) in `app/concept_profile_config.py`.

- The adapter reads concept-profile env vars directly.
- Orch may pass its generic settings object only as an optional overlay source for seam-owned fields.
- The repository seam always receives a concept-profile-shaped config object.

## Seam-owned fields

The adapter owns the fields used by repository construction and cutover behavior:

- local store path + subjects
- backend mode (`local|graph|shadow`)
- `concept_induction_pass` backend override
- graph cutover fallback policy (`fail_open_local|fail_closed`)
- graph endpoint/url/repo/auth/timeout/graph URI
- parity/readiness threshold fields consumed by repository parity evidence configuration

## Why this keeps the seam clean

This keeps Orch service settings focused on Orch service concerns while keeping concept-profile repository
requirements explicit and typed at the seam. Workflow code remains backend-agnostic and does not perform
raw graph reads.
