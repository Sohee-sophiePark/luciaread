import { useState } from "react";

/** Scan with an optional Grad-CAM overlay and opacity control. */
export function Viewer({ src, heatmap, alt }: { src: string | null; heatmap: string | null; alt: string }) {
  const [show, setShow] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  if (!src) return <div className="aspect-square w-full animate-pulse rounded-xl bg-slate-900" />;
  return (
    <figure>
      <div className="relative mx-auto w-full max-w-[560px]">
        <img src={src} alt={alt} className="block w-full rounded-lg" />
        {heatmap && show && <img src={heatmap} alt="" aria-hidden className="pointer-events-none absolute inset-0 h-full w-full rounded-lg" style={{ opacity }} />}
      </div>
      {heatmap && (
        <figcaption className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-300">
          <label className="flex items-center gap-1.5"><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> Grad-CAM overlay</label>
          <label className="flex items-center gap-1.5">Opacity
            <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} disabled={!show} className="w-28 accent-teal-500" />
          </label>
          <span className="text-slate-400">Where the model's evidence concentrates; not proof it is right.</span>
        </figcaption>
      )}
    </figure>
  );
}
