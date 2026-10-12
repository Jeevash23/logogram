// Runs the browser fixture doesn't have, made from its finished run: a run that failed with a
// traceback, one running in another process, one that ran with other versions. The server's own
// answers are fetched and changed, so everything else stays as the server says.

import type { Page, Route } from "@playwright/test";

import type { Comparison, ContinuationsFile, Environment, RunDetail, RunListing, Spec } from "../src/api/types";

export interface FakeRun {
  id: `fake-${string}`;
  name: string;
  status: "running" | "failed" | "finished" | "cancelled";
  elsewhere?: boolean;
  error?: string;
  traceback?: string;
  /** What its spec says other than the fixture run's. */
  spec?: Partial<Spec>;
  /** What the steered model said, as a steering run records it. */
  continuations?: ContinuationsFile;
}

/** A traceback as a failed run keeps it: this machine's folders are already placeholders. */
export const TRACEBACK = [
  "Traceback (most recent call last):",
  '  File "<logogram>/runner.py", line 412, in run_spec',
  "    backend = (provider or default_provider(on_event))(spec.model)",
  '  File "<site-packages>/transformer_lens/model_bridge/sources/transformers.py", line 301, in boot',
  "    config = AutoConfig.from_pretrained(model_name, **kwargs)",
  "OSError: Can't load the configuration of <huggingface cache>/models--example/snapshots/abc.",
].join("\n");

async function fixtureRun(route: Route): Promise<RunListing> {
  const url = new URL(route.request().url());
  const response = await route.fetch({ url: `${url.origin}/api/runs` });
  const runs = (await response.json()) as RunListing[];
  const run = runs.find((r) => r.status === "finished" && !r.id.startsWith("fake-"));
  if (!run) throw new Error("The browser fixture has no finished run.");
  return run;
}

function listingOf(fake: FakeRun, base: RunListing): RunListing {
  const finished = fake.status === "finished";
  return {
    ...base,
    id: fake.id,
    name: fake.name,
    status: fake.status,
    elsewhere: !!fake.elsewhere,
    error: fake.error ?? null,
    finished: finished ? base.finished : null,
    wall_time_s: finished ? base.wall_time_s : null,
    profile: finished ? base.profile : null,
    derived_from: null,
    experiment: fake.spec?.experiment ?? base.experiment,
    scope: fake.spec?.scope ?? base.scope,
    partial: null,
  };
}

/** Answer the history and these runs' details as if they were runs of the open project. */
export async function fakeRuns(page: Page, fakes: () => FakeRun[]): Promise<void> {
  await page.route("**/api/runs", async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const response = await route.fetch();
    const runs = (await response.json()) as RunListing[];
    const base = runs.find((r) => r.status === "finished");
    if (!base) return route.fulfill({ response });
    await route.fulfill({ response, json: [...fakes().map((f) => listingOf(f, base)), ...runs] });
  });
  await page.route(/\/api\/runs\/fake-[^/?]+(\/sites\/\d+)?$/, async (route) => {
    const url = new URL(route.request().url());
    const [, , , id, ...rest] = url.pathname.split("/");
    const fake = fakes().find((f) => f.id === id);
    if (!fake) return route.fallback();
    const base = await fixtureRun(route);
    const response = await route.fetch({ url: `${url.origin}/api/runs/${encodeURIComponent(base.id)}${rest.length ? `/${rest.join("/")}` : ""}` });
    if (rest.length) return route.fulfill({ response }); // a site's evidence: the fixture run's
    const detail = (await response.json()) as RunDetail;
    const manifest = detail.manifest ? { ...detail.manifest, run_id: id } : null;
    const ended = fake.status === "finished";
    await route.fulfill({
      response,
      json: {
        ...detail,
        id,
        spec: { ...detail.spec, ...fake.spec, name: fake.name },
        summary: ended && detail.summary ? { ...detail.summary, run_id: id, name: fake.name } : null,
        manifest: manifest && {
          ...manifest,
          status: fake.status,
          error: fake.error ?? null,
          finished_at: ended ? manifest.finished_at : undefined,
          wall_time_s: ended ? manifest.wall_time_s : undefined,
        },
        listing: listingOf(fake, base),
        traceback: fake.traceback ?? null,
        predictions: null,
        continuations: fake.continuations ?? null,
      } satisfies RunDetail,
    });
  });
}

/** This server's environment, as if it had other versions (from GET /api/environment). */
export async function otherVersions(page: Page, versions: Record<string, string>): Promise<void> {
  await page.route("**/api/environment", async (route) => {
    const response = await route.fetch();
    const env = (await response.json()) as Environment;
    await route.fulfill({ response, json: { ...env, versions: { ...env.versions, ...versions } } satisfies Environment });
  });
}

/** Comparisons, as if the two runs' manifests recorded these differences in where they ran. */
export async function comparedAcrossMachines(page: Page, differences: Comparison["environment_differences"]): Promise<void> {
  await page.route("**/api/compare?**", async (route) => {
    const response = await route.fetch();
    const comparison = (await response.json()) as Comparison;
    await route.fulfill({ response, json: { ...comparison, environment_differences: differences } });
  });
}
