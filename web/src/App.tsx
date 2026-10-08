import { useEffect, useState } from "react";
import { CaseView } from "./components/CaseView";
import { Gallery } from "./components/Gallery";
import { source } from "./lib/source";

const useHash = () => {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => { const f = () => setHash(location.hash); addEventListener("hashchange", f); return () => removeEventListener("hashchange", f); }, []);
  return hash;
};

export default function App() {
  const hash = useHash();
  const [mode, setMode] = useState(source.live ? "" : "DEMO");
  const [upload, setUpload] = useState<File | null>(null);
  useEffect(() => { if (source.live) fetch("/api/health").then((r) => r.json()).then((h) => setMode(h.run_mode === "live" ? "LIVE" : "REPLAY")); }, []);
  const caseId = hash.match(/^#\/case\/([A-Za-z0-9_-]+)/)?.[1] ?? null;
  const onUpload = (f: File) => { setUpload(f); location.hash = "#/upload"; };
  return (
    <div className="min-h-screen">
      <div role="note" className="bg-amber-100 px-4 py-1.5 text-center text-xs text-amber-950">
        Research demo · not medical advice · not a diagnosis · images: Kermany et al. 2018, CC BY 4.0
      </div>
      <div className="mx-auto max-w-6xl px-4 py-5 sm:px-6">
        <header className="mb-6 flex flex-wrap items-center gap-3">
          <a href="#/" className="font-serif text-2xl text-slate-900">LuciaRead</a>
          {mode && <span className={`rounded-full px-2 py-0.5 text-xs font-semibold text-white ${mode === "LIVE" ? "bg-emerald-700" : "bg-slate-700"}`}
            title={mode === "LIVE" ? "Reads call the model now" : "Reads replay recorded runs"}>{mode}</span>}
          <span className="text-sm text-slate-600">Draft image reads for clinician review: chest X-ray and retinal OCT</span>
        </header>
        {caseId || (hash === "#/upload" && upload)
          ? <CaseView key={caseId ?? upload!.name} caseId={caseId} file={caseId ? null : upload} />
          : <Gallery onUpload={source.live && mode === "LIVE" ? onUpload : null} />}
        <footer className="mt-10 border-t border-slate-200 pt-4 text-xs leading-relaxed text-slate-600">
          Images: Kermany D, Zhang K, Goldbaum M (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for
          Classification", Mendeley Data V2, doi:10.17632/rscbjbr9sj.2, licensed CC BY 4.0. Changes are noted per case.
          Classifiers trained on one public pediatric dataset; results are not externally validated.
        </footer>
      </div>
    </div>
  );
}
