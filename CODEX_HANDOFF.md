# Codex Handoff

## Canonical workspace
Use this folder as the project root:
- `D:\NDVI-dataset\MVP - ndvi progression`

## Why this handoff exists
The previous Codex workspace path was:
- `D:\Strawberry baseline`

That folder no longer exists, so the old session could not be safely continued in place.

## What is present in this folder now
Current recovered artifacts observed in this folder:
- `audit_progression_dataset.py`
- `progression_feasibility_audit.md`
- `progression_feasibility_summary.json`

## Repo status
This folder is **not** a git repository.
- `.git`: not present
- branch: N/A
- remote origin: N/A

## Recovered progress from the prior Codex session
### Workspace diagnosis
- Codex had been pointing to `D:\Strawberry baseline`.
- That path is gone.
- The correct surviving folder to use is `D:\NDVI-dataset\MVP - ndvi progression`.
- This did not appear to be a simple case/space/WSL normalization issue; the old path itself was missing.

### NDVI progression audit status
Prior audit conclusion captured in the session:
- Current readiness: **Level 2**
- Data basis: 2 field units, 6 dates total
- Matching status: raw-NDVI-to-annotation matching was reported as complete for the currently inspected set
- Gaps: no RGB timeline, no machine-readable labels, very limited repeated georeferenced comparisons
- Operational interpretation: enough for a farmer demo and manual progression review, not enough yet for reliable model training

### Important limitation
Any files that only existed inside `D:\Strawberry baseline` are **not recoverable from that missing path** through this handoff alone. Only surviving on-disk artifacts and chat context were carried forward here.

## Recommended next steps in the new Codex session
1. Open Codex directly in `D:\NDVI-dataset\MVP - ndvi progression`.
2. Start by inventorying the folder contents.
3. Confirm which missing assets must be reconstructed versus which already exist elsewhere on disk.
4. If needed, create a manual annotation template and progression dataset structure here.
5. Rebuild any missing MVP code in this workspace only after confirming scope.

## Suggested first prompt for the new Codex session
Use this exact folder as the workspace: `D:\NDVI-dataset\MVP - ndvi progression`.
The old workspace `D:\Strawberry baseline` is gone.
Start by inspecting the files in this folder, summarize what survives, and continue the NDVI progression audit from the existing artifacts:
- `audit_progression_dataset.py`
- `progression_feasibility_audit.md`
- `progression_feasibility_summary.json`
Do not assume git is configured. First produce an inventory and recovery plan before modifying code.
