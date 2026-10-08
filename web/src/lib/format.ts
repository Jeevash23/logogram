// Number, size and time formatting. Numbers use a true minus sign and tabular figures (CSS).

const MINUS = "−";

export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const s = Math.abs(x).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return x < 0 ? MINUS + s : s;
}

export function signed(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const s = Math.abs(x).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return (x < 0 ? MINUS : "+") + s;
}

export function ci(lo: number | null | undefined, hi: number | null | undefined, digits = 2): string {
  return `${num(lo, digits)} to ${num(hi, digits)}`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function prob(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  if (x < 0.001) return "<0.001";
  return x.toFixed(3);
}

export function bytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const gb = 1024 ** 3;
  if (n >= gb) return `${(n / gb).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(n / 1024 ** 2))} MB`;
}

export function count(n: number): string {
  return n.toLocaleString("en-US");
}

export function params(n: number | null | undefined): string {
  if (!n) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  return `${Math.round(n / 1e6)}M`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m} min ${s.toString().padStart(2, "0")} s`;
}

export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (now - t) / 1000);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  const d = new Date(t);
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function clock(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Show token whitespace explicitly: a leading space as a middle dot, newlines as ⏎. */
export function visibleToken(token: string): string {
  return token.replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/^ /, "·").replace(/ /g, "·");
}

export function shortRevision(rev: string | null | undefined): string {
  return rev ? rev.slice(0, 7) : "latest";
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${count(n)} ${n === 1 ? one : many}`;
}
