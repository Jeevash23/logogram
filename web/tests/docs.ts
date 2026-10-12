import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// The repository's files as the app's links name them: Markdown pages and their headings' anchors,
// and the web app's sources, for checks that links and walkthrough controls lead somewhere.

const ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** A file of the repository, from its root, such as docs/getting-started.md. */
export function readRepo(path: string): string {
  return readFileSync(join(ROOT, path), "utf8");
}

/** The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens. */
export function slug(heading: string): string {
  return heading
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\- _]/gu, "")
    .replace(/ /g, "-");
}

/** Every heading's anchor in a Markdown file of the repository (repeats numbered as GitHub does). */
export function anchorsOf(path: string): string[] {
  const text = readRepo(path).replace(/```[\s\S]*?```/g, "");
  const seen = new Map<string, number>();
  return [...text.matchAll(/^#{1,6} (.+)$/gm)].map((m) => {
    const base = slug(m[1].replace(/[`*]/g, ""));
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    return n === 0 ? base : `${base}-${n}`;
  });
}

/** The text of every source file of the web app. */
export function webSources(): string {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (/\.tsx?$/.test(name)) out.push(readFileSync(path, "utf8"));
    }
  };
  walk(join(ROOT, "web/src"));
  return out.join("\n");
}
