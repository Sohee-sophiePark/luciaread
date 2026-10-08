import { useEffect, useState } from "react";
import { source } from "../lib/source";
import type { Case } from "../lib/types";
import { DropBox } from "./DropBox";

const sha256 = async (f: File) =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", await f.arrayBuffer()))].map((b) => b.toString(16).padStart(2, "0")).join("");

/** Case picker, drop box and downloadable test images. Live mode uploads; otherwise a dropped test image opens its recorded read. */
export function Gallery({ live, onUpload }: { live: boolean; onUpload: (f: File) => void }) {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  useEffect(() => { source.cases().then(setCases).catch((e) => setError(String(e.message ?? e))); }, []);
  const tests = cases?.filter((c) => c.kind === "test") ?? [];
  const onFile = async (f: File) => {
    setNote("");
    if (!["image/png", "image/jpeg"].includes(f.type)) return setNote("Please use a JPEG or PNG image.");
    if (live) return onUpload(f);
    const digest = await sha256(f);
    const hit = cases?.find((c) => c.sha256 === digest);
    if (hit) location.hash = `#/case/${hit.id}`;
    else setNote("This public demo can only read the images on this page; nothing was uploaded. Run LuciaRead locally to read your own images.");
  };
  return (
    <section aria-labelledby="pick">
      <h1 id="pick" className="font-serif text-xl text-slate-900">Pick a case</h1>
      <p className="mt-1 max-w-2xl text-sm text-slate-600">
        Lucia's agents check the image, run two trained classifiers, describe the image, and draft a read. Code computes
        every number and decides triage; a clinician signs off or returns the draft.
      </p>
      {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
      <ul className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cases?.filter((c) => !c.kind).map((c) => (
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
      <DropBox live={live} onFile={onFile} />
      {note && <p role="status" className="mt-2 text-sm text-amber-900">{note}</p>}
      <Strip id="tests" title="Test images" items={tests}
        intro="Unseen images from the test split, one per class, picked at random. Download one and drop it above; the label shown is the dataset's, so you can compare it with the model's read." />
    </section>
  );
}

/** Downloadable images to drop on the page. */
function Strip({ id, title, intro, items }: { id: string; title: string; intro: string; items: Case[] }) {
  if (!items.length) return null;
  return (
    <section aria-labelledby={id} className="mt-6">
      <h2 id={id} className="font-serif text-lg text-slate-900">{title}</h2>
      <p className="mt-1 text-sm text-slate-600">{intro}</p>
      <ul className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {items.map((c) => (
          <li key={c.id} className="overflow-hidden rounded-lg border border-slate-200 bg-white">
            <img src={source.thumbnail(c)} alt={c.title} loading="lazy" className="aspect-square w-full bg-slate-950 object-contain" />
            <div className="p-2 text-xs">
              <p className="text-slate-800">{c.title.replace("Test image: ", "")}</p>
              <a href={source.thumbnail(c)} download={`LuciaRead-${c.id}.png`} className="mt-1 inline-block font-semibold text-accent underline-offset-2 hover:underline">
                Download {c.id}
              </a>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
