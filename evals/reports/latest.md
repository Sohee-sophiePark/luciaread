# Eval report

Tier 1 unit tests: 78 passed in 2.30s

| Golden | Case | Criteria | Result | Status | LLM calls | Notes |
|---|---|---|---|---|---|---|
| G1 | S1 | 3,7 | pass | AWAITING_SIGNOFF | 7 |  |
| G2 | S2 | 3 | pass | AWAITING_SIGNOFF | 6 |  |
| G3 | S3 | 3 | pass | AWAITING_SIGNOFF | 6 |  |
| G4 | S4 | 4 | pass | AWAITING_SIGNOFF | 7 |  |
| G5 | S5 | 4 | pass | AWAITING_SIGNOFF | 6 |  |
| G6 | S6 | 5 | pass | REJECTED | 1 |  |
| G7 | S7 | 4 | pass | AWAITING_SIGNOFF | 6 |  |
| G8 | pytest | 3,6 | pass |  |  | test_gate_failure_triggers_revision_with_feedback |
| G9 | pytest | 6 | pass |  |  | test_evaluator_loop_is_bounded |
| G10 | pytest | 7 | pass |  |  | test_signoff_writes_report_and_return_does_not |
| G11 | pytest | 5 | pass |  |  | test_colour_photo_rejected_without_llm_call |
| G12 | pytest | 2 | pass |  |  | test_committed_manifest_is_patient_disjoint |
