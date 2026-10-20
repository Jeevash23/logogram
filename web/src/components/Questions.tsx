import { guideUrl } from "../lib/help";
import { methodsOf, QUESTIONS, type QuestionId } from "../lib/questions";
import { Button } from "./ui";
import s from "./Questions.module.css";

/**
 * What Logogram can find out: the four questions its methods answer, each with a drawing, what it
 * finds out, its methods, and where to begin. The first screens and Start here show it. A question
 * a bundled example asks can open it, where `onExample` is given; every question links to the page
 * of the documentation that works through it.
 */
export function QuestionList({ onExample, exampleLabel, compact }: { onExample?: (id: string) => void; exampleLabel?: (id: string) => string; compact?: boolean }) {
  return (
    <dl className={`${s.list} ${compact ? s.compact : ""}`}>
      {QUESTIONS.map((q) => (
        <div key={q.id} className={s.item}>
          <dt className={s.title}>
            <Pictogram id={q.id} />
            {q.title}
          </dt>
          <dd className={s.text}>{q.text}</dd>
          {!compact && <dd className={s.methods}>{methodsOf(q.id)}</dd>}
          <dd className={s.links}>
            {q.example && onExample && (
              <Button size="small" onClick={() => onExample(q.example!)}>
                {exampleLabel?.(q.example) ?? "Open the example"}
              </Button>
            )}
            <a className={s.link} href={guideUrl(q.guide)} target="_blank" rel="noopener noreferrer">
              {q.guide.title}
              <span className="visually-hidden"> (opens in a new tab)</span>
              <span aria-hidden="true"> ↗</span>
            </a>
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** Each question, drawn in ink: a cell found on the map; a path between parts; a direction an
 * activation is read along; a point pushed along a direction. */
function Pictogram({ id }: { id: QuestionId }) {
  return (
    <svg className={s.pictogram} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      {id === "where" && (
        <>
          {[0, 1, 2].flatMap((r) =>
            [0, 1, 2].map((c) => <rect key={`${r}${c}`} className={s.cell} x={4 + c * 9} y={4 + r * 9} width={7} height={7} rx={1} />),
          )}
          <rect className={s.ink} x={13} y={13} width={7} height={7} rx={1} />
          <circle className={s.stroke} cx={16.5} cy={16.5} r={7.5} />
        </>
      )}
      {id === "flow" && (
        <>
          <path className={s.stroke} d="M8 24 C 8 15, 16 17, 16 10 M16 10 C 16 17, 24 15, 24 24" />
          <path className={s.stroke} d="M8 24 L 24 24" opacity={0.4} />
          <circle className={s.ink} cx={8} cy={24} r={3} />
          <circle className={s.ink} cx={24} cy={24} r={3} />
          <circle className={s.ink} cx={16} cy={9} r={3} />
        </>
      )}
      {id === "represent" && (
        <>
          <path className={s.stroke} d="M4 26 L 28 10" opacity={0.45} />
          <path className={s.stroke} d="M8 24 L 18 7 M 14.5 8.2 L 18 7 L 18.8 10.6" />
          <path className={s.stroke} d="M18 7 L 22.4 13.6" strokeDasharray="2 2" />
          <circle className={s.ink} cx={22.4} cy={13.6} r={2.6} />
        </>
      )}
      {id === "steer" && (
        <>
          <circle className={s.stroke} cx={9} cy={21} r={3} />
          <path className={s.stroke} d="M12 19 L 23 12 M 19 11.6 L 23 12 L 21.6 15.6" />
          <circle className={s.ink} cx={25} cy={11} r={3} />
        </>
      )}
    </svg>
  );
}
