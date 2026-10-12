import { test, expect } from "@playwright/test";
import { formatRoute, parseRoute, type Route } from "../src/lib/route";

test("a place in the workbench survives a trip through the URL", () => {
  const places: Route[] = [
    { screen: "projects" },
    { screen: "workbench", view: "explore" },
    { screen: "workbench", view: "results", run: "2026-10-11 without BOS/1" },
    { screen: "workbench", view: "compare", compare: ["a run", "b#run"] },
    { screen: "workbench", view: "compare", run: "one run, before a second is chosen" },
    { screen: "workbench", view: "sweep", sweep: "20261012-101500-bootstrap-seed-3fa9" },
    { screen: "workbench", view: "sweep" },
    { screen: "workbench", view: "experiment" },
  ];
  for (const place of places) expect(parseRoute(formatRoute(place))).toEqual(place);
  expect(formatRoute({ screen: "workbench", view: "results", run: "x" })).toBe("#/results/x");
  expect(formatRoute({ screen: "workbench", view: "sweep", sweep: "s1" })).toBe("#/sweep/s1");
});

test("fragments that name no place are ignored", () => {
  for (const hash of ["", "#", "#/", "#/nowhere", "#/results/a/b", "#/compare/a/b/c", "#/sweep/a/b", "#/projects/x", "#/results/%E0%A4%A", "results"]) {
    expect(parseRoute(hash), hash).toBeNull();
  }
});
