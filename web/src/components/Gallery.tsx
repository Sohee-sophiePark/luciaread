import { useEffect, useState } from "react";
import { source } from "../lib/source";
import type { Case } from "../lib/types";

/** Case picker; in live mode also a local upload (stays on this laptop). */
export function Gallery({ onUpload }: { onUpload: ((f: File) => void) | null }) {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { source.cases().then(setCases).catch((e) => setError(String(e.message ?? e))); }, []);
  return (
    <section aria-labelledby="pick">
      <h1 id="pick" className="font-serif text-xl text-slate-900">Pick a case</h1>
      <p className="mt-1 max-w-2xl text-sm text-slate-600">
        Lucia's agents check the image, run two trained classifiers, describe the image, and draft a read. Code computes
        every number and decides triage; a clinician signs off or returns the draft.
      </p>
      {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
      <ul className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cases?.map((c) => (
          <li key={c.id}>
            <a href={`#/case/${c.id}`} className="group block overflow-hidden rounded-xl border border-slate-200 bg-white hover:border-accent">
              <div className="aspect-[4/3] bg-slate-950">
                <img src={source.thumbnail(c)} alt={c.title} loading="lazy" className="h-full w-full object-contain opacity-90 group-hover:opacity-100" />
              </div>
              <div className="p-3">
                <span className="text-xs font-semibold text-accent">{c.id}</span>
                <p className="text-sm text-slate-800">{c.title}</p>
              </div>
            </a>
          </li>
        ))}
      </ul>
      {onUpload && (
        <label className="mt-6 block max-w-md rounded-xl border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-700">
          Upload a JPEG or PNG (local live mode). Do not upload real patient images.
          <input type="file" accept="image/png,image/jpeg" className="mt-2 block text-sm" onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
        </label>
      )}
    </section>
  );
}
