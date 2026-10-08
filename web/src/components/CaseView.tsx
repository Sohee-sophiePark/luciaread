import { useEffect, useState } from "react";
import { source } from "../lib/source";
import type { Case, Images, RunState, TraceEvent } from "../lib/types";
import { ReadCard } from "./ReadCard";
import { Stepper } from "./Stepper";
import { Viewer } from "./Viewer";

/** One read: dark image viewer beside the light report panel. */
export function CaseView({ caseId, file }: { caseId: string | null; file: File | null }) {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [state, setState] = useState<RunState | null>(null);
  const [images, setImages] = useState<Images | null>(null);
  const [meta, setMeta] = useState<Case | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    if (caseId) source.cases().then((cs) => live && setMeta(cs.find((c) => c.id === caseId) ?? null));
    source.read(caseId, file, (e) => live && setEvents((p) => [...p, e]))
      .then((r) => { if (live) { setState(r.state); setImages(r.images); } })
      .catch((e) => live && setError(String(e.message ?? e)));
    return () => { live = false; };
  }, [caseId, file]);
  const preview = file ? URL.createObjectURL(file) : meta ? source.thumbnail(meta) : null;
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <section aria-label="Image" className="rounded-2xl bg-slate-950 p-3 text-slate-200">
        <Viewer src={images?.image ?? preview} heatmap={images?.heatmap ?? null} alt={meta?.title ?? "Uploaded image"} />
        {meta && (
          <p className="mt-3 text-xs leading-relaxed text-slate-300">
            <span className="font-semibold text-slate-100">{meta.id} · {meta.title}.</span>{" "}
            {meta.source ?? "No dataset image."}{meta.changes && <> Changes: {meta.changes}.</>}
          </p>
        )}
      </section>
      <section aria-label="Draft read" className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4">
        <a href="#/" className="text-sm text-accent underline-offset-2 hover:underline">← All cases</a>
        <Stepper events={events} signed={state?.status === "SIGNED_OFF"} />
        {error && <p className="text-sm text-red-700">{error}</p>}
        {!state && !error && <p className="text-sm text-slate-600" aria-live="polite">Reading… {events.at(-1)?.type.replaceAll("_", " ")}</p>}
        {state && <ReadCard state={state} onSignoff={async (d, n) => setState(await source.signoff(state, d, n))} />}
      </section>
    </div>
  );
}
