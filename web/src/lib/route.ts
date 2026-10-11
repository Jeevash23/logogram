// Where the workbench is, kept in the URL's fragment: the view, and the run it shows or the two
// runs it compares. Back and Forward move between views, a reload comes back to the same place,
// and the link can be kept while the server runs. The fragment never carries the session token.

import { useStore, VIEWS, type View } from "../store/app";

export type Route =
  | { screen: "projects" }
  | { screen: "workbench"; view: View; run?: string; compare?: [string, string] };

const VIEW_IDS = new Set<string>(VIEWS.map((v) => v.id));

/** The fragment for a route: #/projects, #/<view>, #/<view>/<run> or #/compare/<run>/<run>. */
export function formatRoute(route: Route): string {
  if (route.screen === "projects") return "#/projects";
  const parts: string[] = [route.view];
  if (route.compare) parts.push(...route.compare);
  else if (route.run) parts.push(route.run);
  return `#/${parts.map(encodeURIComponent).join("/")}`;
}

/** The route a fragment names, or null for one that names none. */
export function parseRoute(hash: string): Route | null {
  if (!hash.startsWith("#/")) return null;
  let parts: string[];
  try {
    parts = hash.slice(2).split("/").filter(Boolean).map(decodeURIComponent);
  } catch {
    return null; // a malformed escape
  }
  const [first, ...rest] = parts;
  if (first === "projects" && rest.length === 0) return { screen: "projects" };
  if (!first || !VIEW_IDS.has(first)) return null;
  const view = first as View;
  if (view === "compare" && rest.length === 2) return { screen: "workbench", view, compare: [rest[0], rest[1]] };
  if (rest.length === 1) return { screen: "workbench", view, run: rest[0] };
  return rest.length === 0 ? { screen: "workbench", view } : null;
}

/** The route the store is at, or null where there is no place worth keeping. */
export function currentRoute(): Route | null {
  const st = useStore.getState();
  if (st.screen === "projects") return { screen: "projects" };
  if (st.screen !== "workbench" || !st.project) return null;
  const [a, b] = st.compareIds;
  if (st.view === "compare" && a && b) return { screen: "workbench", view: "compare", compare: [a, b] };
  return { screen: "workbench", view: st.view, run: st.activeRunId ?? undefined };
}

let following = false;

/** Move the store to a route: Back, Forward, a reload, or a link. Runs that no longer exist are
 * left out, and the URL is then corrected to where the workbench really is. */
export async function followRoute(route: Route | null): Promise<void> {
  const st = useStore.getState();
  if (!route) return;
  following = true;
  try {
    if (route.screen === "projects") {
      if (st.screen === "workbench") st.goto("projects");
      return;
    }
    if (!st.project) return; // nothing to show the view in yet
    if (st.screen !== "workbench") st.goto("workbench");
    const known = new Set(useStore.getState().runs.map((r) => r.id));
    if (route.compare) {
      const [a, b] = route.compare;
      if (known.has(a) && known.has(b)) st.setCompare(a, b);
      st.setView("compare");
    } else if (route.run && route.run !== st.activeRunId && known.has(route.run)) {
      await st.openRun(route.run, route.view);
    } else if (route.view !== useStore.getState().view) {
      st.setView(route.view);
    }
  } finally {
    following = false;
    write("replace");
  }
}

let ready = false;
let scheduled: "push" | "replace" | null = null;

function write(mode: "push" | "replace"): void {
  if (!ready) return;
  const route = currentRoute();
  if (!route) return;
  const hash = formatRoute(route);
  if (hash === window.location.hash) return;
  const url = `${window.location.pathname}${window.location.search}${hash}`;
  if (mode === "push") window.history.pushState(null, "", url);
  else window.history.replaceState(window.history.state, "", url);
}

/** Keep the URL and the store in step. A new view is a new history entry; a run opened in the
 * same view, the steps of opening a project, and following Back or Forward replace the current
 * one. Call once, after the app has booted (the project opened at the URL's place: pendingRoute). */
export function startRoutes(): () => void {
  const unsubscribe = useStore.subscribe((st, prev) => {
    const placeChanged = st.screen !== prev.screen || st.view !== prev.view;
    const runChanged = st.activeRunId !== prev.activeRunId || st.compareIds !== prev.compareIds;
    if (!placeChanged && !runChanged) return;
    const mode = placeChanged && !following && !st.entering ? "push" : "replace";
    // One action changes several fields (a view, then its run): one entry for all of them, written
    // before anything else can happen (a Back pressed at once must find it).
    if (scheduled === null) {
      scheduled = mode;
      queueMicrotask(() => {
        const m = scheduled ?? "replace";
        scheduled = null;
        write(m);
      });
    } else if (mode === "push") scheduled = "push";
  });
  const onPop = () => void followRoute(parseRoute(window.location.hash));
  window.addEventListener("popstate", onPop);
  ready = true;
  write("replace");
  return () => {
    ready = false;
    unsubscribe();
    window.removeEventListener("popstate", onPop);
  };
}
