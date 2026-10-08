import { useState } from "react";
import { source } from "../lib/source";
import type { RunState, SignOff } from "../lib/types";

const fmt = (v: number, unit: string) => (unit === "pct" ? `${v.toFixed(1)}%` : unit === "px" ? `${v.toFixed(0)} px` : v.toFixed(1));

/** The draft read: code-set label, probability and triage; writer text; flags; sign-off. */
export function ReadCard({ state, onSignoff }: { state: RunState; onSignoff: (d: SignOff["decision"], note: string) => void }) {
  const [note, setNote] = useState("");
  const f = state.final;
  if (state.status === "REJECTED" || state.status === "FAILED" || state.status === "DEGRADED") {
    return (
      <div className="rounded-xl border border-slate-300 bg-slate-50 p-4 text-sm">
        <p className="font-semibold text-slate-900">{state.status === "REJECTED" ? "Not accepted" : "Read stopped"}</p>
        <p className="mt-1 text-slate-700">{state.message ?? state.error}</p>
      </div>
    );
  }
  const review = state.triage === "needs_review";
  const flags = Object.values(state.flags).filter((x) => x.severity === "review");
  return (
    <article className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full bg-slate-900 px-2.5 py-1 font-semibold text-white">Model output: {state.label}</span>
        {f && <span className="rounded-full border border-slate-300 px-2.5 py-1 text-slate-800">calibrated probability {f.probability_pct.toFixed(1)}%</span>}
        <span className={`rounded-full px-2.5 py-1 font-semibold ${review ? "bg-amber-100 text-amber-950" : "bg-emerald-100 text-emerald-950"}`}>
          {review ? "Needs human review" : "Routine queue"}
        </span>
      </div>
      {f ? (
        <>
          <h2 className="font-serif text-xl leading-snug text-slate-900">{f.headline}</h2>
          <p className="text-sm leading-relaxed text-slate-700">{f.summary}</p>
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800">{f.key_points.map((k, i) => <li key={i}>{k}</li>)}</ul>
          <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700"><span className="font-semibold">Review note. </span>{f.review_note}</p>
        </>
      ) : <p className="text-sm text-slate-700">{state.message}</p>}
      {flags.length > 0 && (
        <ul aria-label="Review flags" className="space-y-1 text-xs">
          {flags.map((x) => <li key={x.flag_id} className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1 text-amber-950"><b>{x.flag_id}</b> · {x.message}</li>)}
        </ul>
      )}
      <details className="text-xs text-slate-700">
        <summary className="cursor-pointer text-sm text-accent">How this read was made</summary>
        <table className="mt-2 w-full text-left">
          <thead><tr className="text-slate-500"><th className="py-1 font-normal">Metric</th><th className="font-normal">Value</th><th className="font-normal">Tool</th></tr></thead>
          <tbody>{Object.values(state.metrics).map((m) => <tr key={m.key} className="border-t border-slate-100"><td className="py-1 pr-2">{m.label}</td><td className="pr-2">{fmt(m.value, m.unit)}</td><td>{m.source_tool}</td></tr>)}</tbody>
        </table>
        <p className="mt-2">Revisions: {state.revision_count} · evaluator rounds: {state.eval_verdicts.length} · gates: {state.gate_results.map((g) => `${g.gate} ${g.passed ? "pass" : "fail"}`).join(", ")}</p>
      </details>
      {f && <p className="text-xs leading-relaxed text-slate-600">{f.disclaimer}</p>}
      {state.signoff ? (
        <p className={`rounded-lg p-3 text-sm ${state.signoff.decision === "sign_off" ? "bg-emerald-50 text-emerald-950" : "bg-slate-100 text-slate-800"}`}>
          {state.signoff.decision === "sign_off"
            ? source.live ? "Signed off. A signed report was written." : "Signed off. In this public demo nothing is stored."
            : "Returned to the queue. Nothing was signed."}
          {state.signoff.note && ` Note: ${state.signoff.note}`}
        </p>
      ) : (state.status === "AWAITING_SIGNOFF" || state.status === "NEEDS_CLINICIAN_REVIEW") && (
        <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 pt-3">
          <label className="sr-only" htmlFor="note">Note</label>
          <input id="note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note (optional)" className="min-w-40 flex-1 rounded-lg border border-slate-300 p-2 text-sm" />
          <button onClick={() => onSignoff("sign_off", note)} className="rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white hover:bg-teal-800">Sign off</button>
          <button onClick={() => onSignoff("return", note)} className="rounded-lg border border-slate-400 px-3 py-2 text-sm text-slate-800 hover:bg-slate-50">Return</button>
        </div>
      )}
    </article>
  );
}
