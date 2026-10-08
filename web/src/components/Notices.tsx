import { useStore } from "../store/app";
import { Icon } from "./ui";
import s from "./Notices.module.css";

export function Notices() {
  const notices = useStore((st) => st.notices);
  const dismiss = useStore((st) => st.dismiss);
  if (notices.length === 0) return null;
  return (
    <div className={s.stack} aria-live="polite">
      {notices.map((n) => (
        <div key={n.id} className={`${s.notice} ${n.tone === "error" ? s.error : ""}`} role={n.tone === "error" ? "alert" : "status"}>
          <Icon name={n.tone === "error" ? "alert" : "info"} size={15} />
          <div className={s.text}>{n.text}</div>
          <button type="button" className={s.close} onClick={() => dismiss(n.id)} aria-label="Dismiss">
            <Icon name="close" size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
