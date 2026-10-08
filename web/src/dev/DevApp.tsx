import { useEffect, useState } from "react";
import type { RunState, TraceEvent } from "../lib/types";

interface RunRow { run_id: string; case_id: string; status: string; label: string | null; triage: string | null; budget: Record<string, number> }

/** Developer console (local only): runs, full trace, gates, evaluator verdicts, cost. */
export function DevApp() {
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [sel, setSel] = useState<{ state: RunState; events: TraceEvent[]; cost_usd: number } | null>(null);
  useEffect(() => { fetch("/api/dev/runs").then((r) => r.json()).then(setRuns); }, []);
  const open = (id: string) => fetch(`/api/dev/runs/${id}`).then((r) => r.json()).then(setSel);
  return (
    <div className="grid min-h-screen grid-cols-[320px_minmax(0,1fr)] gap-4 p-4 text-xs">
      <ul className="space-y-1 overflow-auto">
        {runs.map((r) => (
          <li key={r.run_id}><button onClick={() => open(r.run_id)} className="w-full rounded border border-slate-200 bg-white p-2 text-left hover:border-accent">
            <b>{r.case_id}</b> {r.status} {r.label ?? ""} {r.triage ?? ""}<br /><span className="text-slate-500">{r.run_id} · {r.budget.llm_calls ?? 0} calls · {r.budget.tokens ?? 0} tok</span>
          </button></li>
        ))}
      </ul>
      {sel && (
        <div className="space-y-3 overflow-auto">
          <p className="text-sm"><b>{sel.state.run_id}</b> · {sel.state.status} · cost ${sel.cost_usd.toFixed(4)}</p>
          <pre className="rounded bg-white p-2">{JSON.stringify({ gates: sel.state.gate_results, evaluator: sel.state.eval_verdicts }, null, 1)}</pre>
          <table className="w-full bg-white">
            <tbody>{sel.events.map((e) => (
              <tr key={e.seq} className={`border-t border-slate-100 align-top ${e.level === "warn" ? "bg-amber-50" : e.level === "error" ? "bg-red-50" : ""}`}>
                <td className="p-1">{e.seq}</td><td className="p-1">{e.t_ms}ms</td><td className="p-1">{e.agent}</td><td className="p-1">{e.type}</td>
                <td className="p-1"><code className="break-all">{JSON.stringify(e.payload)}</code></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}
