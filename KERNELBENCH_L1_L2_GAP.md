# KernelBench L1/L2 Port Plan and Gap List

This document tracks progress toward:

> Port all KernelBench Level 1 and Level 2 challenges.

Source goal reference:
- `/Users/rotemisrael/Documents/python/problems/README.md`

## How this gap list was computed
- Parsed every `problem.md` in `/Users/rotemisrael/Documents/python/problems/problems` and `/Users/rotemisrael/Documents/python/problems/staging`.
- Counted a challenge as "ported" when the markdown includes a KernelBench source link matching:
  - `KernelBench/level1/<id>_...`
  - `KernelBench/level2/<id>_...`

## Current coverage snapshot
- Level 1: 38/100 linked as ported
- Level 2: 2/100 linked as ported

## Ported IDs (detected)
- Level 1: 1, 4, 5, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 25, 33, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 67, 89, 90, 96, 97, 100
- Level 2: 59, 76

## Missing IDs

### Level 1 missing (62)
2, 3, 6, 7, 8, 9, 16, 17, 18, 24, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 98, 99

### Level 2 missing (98)
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100

## Execution plan
1. Baseline validation
- Confirm whether "ported" means:
  - strict one-to-one KernelBench parity (same behavior + test shape range), or
  - Tensara-native variants are acceptable.
- If strict parity is required, add source URL in every ported `problem.md` to avoid false negatives.

2. Batch the migration
- Batch A (Level 1): IDs 2-32 not already done.
- Batch B (Level 1): IDs 34-66 not already done.
- Batch C (Level 1): IDs 68-99 not already done.
- Batch D (Level 2): IDs 1-50 not already done.
- Batch E (Level 2): IDs 51-100 not already done.

3. Per-problem workflow (repeatable)
- Create slug folder under `problems/<slug>/`.
- Implement `def.py` with `reference_solution`, `generate_test_cases`, `verify_result`, `get_function_signature`, `get_flops`.
- Write `problem.md` with required front matter and source attribution link.
- Add optional `torch.py` and `tinygrad.py` baselines when relevant.
- Run `pnpm sync-problems` and validate in local Tensara.

4. Quality gates before marking complete
- Correctness: correct submissions pass, incorrect fail.
- Numerical stability: tolerance and dtype handling are explicit.
- FLOPs formula sanity-checked on min/max testcase sizes.
- Metadata quality: tags, parameters, and difficulty are consistent.

5. Tracking discipline
- Keep this file updated after each merged batch.
- Maintain a short changelog section per batch with new IDs closed.

## Notes
- This list is link-based. If a problem is already ported but missing KernelBench attribution in `problem.md`, it will currently appear as missing.
