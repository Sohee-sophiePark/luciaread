import type { TraceEvent } from "../lib/types";

type S = "pending" | "active" | "done" | "warn" | "fail";
const STYLE: Record<S, string> = {
  pending: "border-slate-200 text-slate-500", active: "border-accent text-accent animate-pulse",
  done: "border-emerald-600 bg-emerald-50 text-emerald-900", warn: "border-amber-600 bg-amber-50 text-amber-900", fail: "border-red-600 bg-red-50 text-red-900",
};

/** Pipeline progress from trace events: gates, specialists, writer loop, sign-off. */
export function Stepper({ events, signed }: { events: TraceEvent[]; signed: boolean }) {
  const has = (t: string) => events.some((e) => e.type === t);
  const gate = (id: string): S => {
    const g = events.filter((e) => e.type === "gate_result" && e.payload.gate === id).at(-1);
    return g ? (g.payload.passed ? "done" : id === "G5" ? "warn" : "fail") : "pending";
  };
  const lane = (a: string): S => {
    const fin = events.find((e) => e.type === "agent_finished" && e.agent === a);
    if (fin) return fin.payload.degraded ? "warn" : "done";
    return events.some((e) => e.type === "agent_started" && e.agent === a) ? "active" : "pending";
  };
  const evals = events.filter((e) => e.type === "evaluator_verdict");
  const drafts = events.filter((e) => e.type === "draft_created").length;
  const sign: S = signed ? "done" : has("awaiting_signoff") ? "active" : has("needs_clinician_review") ? "warn" : "pending";
  const steps: [string, S, string?][] = [
    ["Input gate", gate("G0")],
    ["Modality gate", gate("G1")],
  ];
  const after: [string, S, string?][] = [
    ["Writer", drafts ? "done" : "pending", drafts ? `draft ${drafts}` : undefined],
    ["Output gates", gate("G5")],
    ["Evaluator", evals.length ? (evals.at(-1)!.payload.passed ? "done" : "warn") : "pending", evals.length ? `round ${evals.length}` : undefined],
    ["Clinician sign-off", sign],
  ];
  return (
    <ol aria-label="Pipeline" className="flex flex-wrap items-center gap-1.5 text-xs">
      {steps.map(([l, s]) => <Pill key={l} label={l} status={s} />)}
      <li className="flex flex-col gap-0.5 rounded-lg border border-slate-200 bg-white p-1">
        {["model", "quality", "visual"].map((a) => <Pill key={a} label={`${a} specialist`} status={lane(a)} small />)}
      </li>
      {after.map(([l, s, d]) => <Pill key={l} label={l} status={s} detail={d} />)}
    </ol>
  );
}

function Pill({ label, status, detail, small }: { label: string; status: S; detail?: string; small?: boolean }) {
  const Tag = small ? "span" : "li";
  return <Tag className={`block rounded-full border bg-white ${small ? "px-2 py-0 text-[11px]" : "px-2.5 py-1"} ${STYLE[status]}`}>{label}{detail && <span className="ml-1 opacity-80">· {detail}</span>}</Tag>;
}
