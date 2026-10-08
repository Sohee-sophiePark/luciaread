import { useState } from "react";

/** Drop zone for one JPEG or PNG; also opens the file picker on click or keyboard. */
export function DropBox({ live, onFile }: { live: boolean; onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  const take = (f?: File) => f && onFile(f);
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}
      className={`mt-6 block cursor-pointer rounded-xl border-2 border-dashed p-5 text-center focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent ${over ? "border-accent bg-accent-soft" : "border-slate-300 bg-white hover:border-accent"}`}
    >
      <span className="block text-sm font-semibold text-slate-900">Drop a chest X-ray or retinal OCT image here, or click to choose</span>
      <span className="mt-1 block text-xs text-slate-600">
        {live ? "Runs a live read on this laptop. JPEG or PNG. Do not use real patient images."
          : "This public demo reads the images on this page. Run LuciaRead locally to read your own images."}
      </span>
      <input type="file" accept="image/png,image/jpeg" className="sr-only" onChange={(e) => take(e.target.files?.[0])} />
    </label>
  );
}
