import type { SiteBase, SiteResult } from "../api/types";
import { signed } from "../lib/format";
import { layerAngle } from "../lib/logogram";
import { selectionOfSite } from "../lib/sites";
import { useStore } from "../store/app";
import { Logogram } from "./Logogram";
import s from "./LogogramDial.module.css";

/**
 * A run's logogram as a dial: the glyph with its layers numbered around it. Each number opens the
 * strongest component of that layer, so the shape of the ink leads straight to the evidence.
 */
export function LogogramDial({
  seed,
  profile,
  sites,
  results,
  size = 196,
}: {
  seed: string;
  profile: number[];
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  size?: number;
}) {
  const select = useStore((st) => st.select);
  const n = profile.length;
  const every = n <= 16 ? 1 : n <= 32 ? 2 : 4;
  const strongestIn = (layer: number) => {
    let best: SiteBase | null = null;
    let value = 0;
    for (const site of sites) {
      if (site.layer !== layer) continue;
      const v = results[site.index]?.effect.mean;
      if (v !== null && v !== undefined && Math.abs(v) >= Math.abs(value)) {
        best = site;
        value = v;
      }
    }
    return best ? { site: best, value } : null;
  };
  // The glyph's ring sits at radius 30 of 100; labels go just outside the largest swell.
  const box = size * 1.5;
  const center = box / 2;
  const scale = size / 100;
  return (
    <figure className={s.dial} style={{ width: box }}>
      <div className={s.face} style={{ width: box, height: box }}>
        <div className={s.glyph} style={{ left: (box - size) / 2, top: (box - size) / 2 }}>
          <Logogram seed={seed} profile={profile} size={size} title="This run's logogram" />
        </div>
        {profile.map((_, layer) => {
          if (layer % every !== 0) return null;
          const angle = layerAngle(layer, n);
          const r = (30 + 31) * scale;
          const strongest = strongestIn(layer);
          return (
            <button
              key={layer}
              type="button"
              className={s.label}
              style={{ left: center + r * Math.cos(angle), top: center + r * Math.sin(angle) }}
              disabled={!strongest}
              onClick={() => strongest && select(selectionOfSite(strongest.site))}
              aria-label={strongest ? `Layer ${layer}: strongest ${strongest.site.label}, ${signed(strongest.value, 3)}` : `Layer ${layer}: not measured`}
              title={strongest ? `${strongest.site.label} ${signed(strongest.value, 2)}` : undefined}
            >
              {layer}
            </button>
          );
        })}
      </div>
      <figcaption className={s.caption}>
        This run's logogram. Layers run clockwise from the top; the ink swells outward where a layer's strongest
        effect is positive and inward where it is negative. Select a layer number to inspect it.
      </figcaption>
    </figure>
  );
}
