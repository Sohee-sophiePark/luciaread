/** Mirrors of the Python models the UI reads (src/luciaread/models.py). */
export type Status = "CREATED" | "INPUT_GATED" | "REJECTED" | "ROUTED" | "ANALYZING" | "WRITING" | "AWAITING_SIGNOFF"
  | "NEEDS_CLINICIAN_REVIEW" | "SIGNED_OFF" | "RETURNED" | "FAILED" | "DEGRADED";
export interface Metric { key: string; value: number; unit: "pct" | "px" | "score"; label: string; source_tool: string }
export interface Flag { flag_id: string; severity: "info" | "review"; metric_refs: string[]; message: string }
export interface Finding { id: string; agent: string; text: string; metric_refs: string[]; flag_refs: string[] }
export interface GateResult { gate: string; passed: boolean; violations: string[] }
export interface EvalVerdict { verdict: string; passed: boolean; checks: Record<string, boolean>; scores: Record<string, number>; issues: string[] }
export interface Rendered {
  modality: "cxr" | "oct"; label: string; label_display: string; probability_pct: number; triage: "routine" | "needs_review";
  headline: string; summary: string; key_points: string[]; review_note: string; disclaimer: string;
}
export interface SignOff { decision: "sign_off" | "return"; note: string; at: string | null }
export interface RunState {
  run_id: string; case_id: string; status: Status; modality: "cxr" | "oct" | null; label: string | null; triage: "routine" | "needs_review" | null;
  metrics: Record<string, Metric>; flags: Record<string, Flag>; findings: Finding[]; gate_results: GateResult[];
  eval_verdicts: EvalVerdict[]; revision_count: number; final: Rendered | null; signoff: SignOff | null;
  message: string | null; error: string | null; budget: Record<string, number>; route: { modality: string; reason: string } | null;
}
export interface TraceEvent { seq: number; run_id: string; t_ms: number; type: string; agent: string; level: string; payload: Record<string, any> }
export interface Case { id: string; kind?: "test" | "edge"; title: string; file: string; source: string | null; changes: string | null; sha256: string }
export interface Images { image: string; heatmap: string | null }
export interface Source {
  live: boolean;
  cases(): Promise<Case[]>;
  thumbnail(c: Case): string;
  read(caseId: string | null, file: File | null, onEvent: (e: TraceEvent) => void): Promise<{ state: RunState; images: Images }>;
  signoff(state: RunState, decision: SignOff["decision"], note: string): Promise<RunState>;
}
