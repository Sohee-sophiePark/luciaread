import type { Case, Images, RunState, Source, TraceEvent } from "./types";

const json = async (url: string, init?: RequestInit) => {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `${r.status}`);
  return r.json();
};

/** Live or local replay: the FastAPI backend on the laptop. */
export const liveSource: Source = {
  live: true,
  cases: () => json("/api/cases"),
  thumbnail: (c) => `/api/cases/${c.id}/image.png`,
  async read(caseId, file, onEvent) {
    const init: RequestInit = { method: "POST" };
    if (file) { const f = new FormData(); f.append("file", file); init.body = f; }
    const { run_id } = await json(file ? "/api/reads/upload" : `/api/reads/case/${caseId}`, init);
    await new Promise<void>((resolve) => {
      const es = new EventSource(`/api/runs/${run_id}/events`);
      es.onmessage = (m) => {
        const e = JSON.parse(m.data) as TraceEvent;
        onEvent(e);
        if (e.type === "run_completed") { es.close(); resolve(); }
      };
      es.onerror = () => { es.close(); resolve(); };
    });
    const state: RunState = await json(`/api/runs/${run_id}`);
    const images: Images = { image: `/api/runs/${run_id}/image.png`, heatmap: state.label ? `/api/runs/${run_id}/heatmap.png` : null };
    return { state, images };
  },
  signoff: (state, decision, note) =>
    json(`/api/runs/${state.run_id}/signoff`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, note }) }),
};

interface Recorded { case: Case; image: string; heatmap: string | null; events: TraceEvent[]; state: RunState }
let data: Promise<{ cases: Recorded[] }> | null = null;
const load = (): Promise<{ cases: Recorded[] }> => (data ??= fetch("./static-data.json").then((r) => r.json()));
const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));
const SHOWN = /^(gate_result|route_decided|agent_started|agent_finished|triage_set|draft_created|evaluator_verdict|revision_requested|awaiting_signoff|needs_clinician_review|run_rejected|injection_flagged|visual_concern|run_completed)$/;

/** Public demo: recorded reads bundled as JSON, replayed step by step; no backend, no model calls. */
export const staticSource: Source = {
  live: false,
  cases: async () => (await load()).cases.map((r) => r.case),
  thumbnail: (c) => `./cases/${c.file}`,
  async read(caseId, _file, onEvent) {
    const rec = (await load()).cases.find((r) => r.case.id === caseId);
    if (!rec) throw new Error("This public demo replays recorded cases only. Uploads run on your own laptop.");
    for (const e of rec.events) {
      if (!SHOWN.test(e.type)) continue;
      if (e.type === "run_completed") break;
      await pause(300);
      onEvent(e);
    }
    const s = rec.state;
    const state = s.signoff ? { ...s, signoff: null, status: "AWAITING_SIGNOFF" as const } : s;
    return { state, images: { image: rec.image, heatmap: rec.heatmap } };
  },
  async signoff(state, decision, note) {
    return { ...state, status: decision === "sign_off" ? "SIGNED_OFF" : "RETURNED", signoff: { decision, note, at: new Date().toISOString() } };
  },
};

export const source: Source = import.meta.env.VITE_STATIC ? staticSource : liveSource;
