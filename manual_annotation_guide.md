# Manual Annotation Guide

## 1. Purpose

Convert Level 2 whole-field NDVI progression into Level 3 annotated disease progression.

## 2. Why This Is Needed

The audit found repeated NDVI dates and matching human annotated overlays, but the annotations are baked into PNG files and are not machine-readable. This template captures reviewer labels, scores, progression states, and notes in a structured format that can be audited and expanded into repeatable zone labels.

## 3. Scoring

### Disease score 0-5

- 0: no visible disease signal
- 1: trace or ambiguous visible symptoms
- 2: mild localized symptoms
- 3: moderate clear symptoms
- 4: severe broad symptoms
- 5: extreme symptoms or likely crop-loss area

### NDVI stress score 0-5

- 0: no NDVI stress signal
- 1: trace or ambiguous NDVI stress
- 2: mild localized NDVI stress
- 3: moderate clear NDVI stress
- 4: severe broad NDVI stress
- 5: extreme NDVI stress signal

### Confidence 0-5

- 0: unusable or no confidence
- 1: very low confidence
- 2: low confidence
- 3: moderate confidence
- 4: high confidence
- 5: very high confidence

## 4. Important Chimera Rule

Track visual disease and NDVI stress separately.

If symptoms visually worsen but NDVI does not, mark:

`overall_progression_state = worsening_visual_only`

## 5. Minimum Work To Reach Level 3

- Complete all 6 current field/date rows.
- Then add at least 15 repeatable zones for Mallard-Avenue across 3 dates.
- Add at least 10 repeatable zones for McIntyre-Road across 2 dates.
- Assign stable `zone_id` values.
- Add 0-5 scores and notes.

## 6. Current Six Rows

- Mallard-Avenue 2026-05-12
- Mallard-Avenue 2026-05-18
- Mallard-Avenue 2026-05-22
- Mallard-Avenue 2026-05-29
- McIntyre-Road 2026-05-08
- McIntyre-Road 2026-05-27
