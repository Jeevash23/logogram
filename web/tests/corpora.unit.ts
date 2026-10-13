import { test, expect } from "@playwright/test";

import type { CorpusTop, DirectionListing, ModelInfo, RepresentationPoint, TopExample } from "../src/api/types";
import {
  datasetFacts,
  directionModelText,
  directionProblem,
  exampleBound,
  fieldsOf,
  formatOf,
  foundElsewhere,
  guessField,
  histogramBars,
  nameFromFile,
  nameProblem,
  positionWords,
  readingText,
  siteName,
  siteWords,
  splitRow,
  wholeNumber,
} from "../src/lib/corpora";
import {
  axisDomain,
  classCounts,
  classMarks,
  explainedText,
  exportSites,
  exportSize,
  keyboardOrder,
  MARKED_CLASSES,
  nearestPoint,
  pointLabel,
  pointText,
  scatterScale,
} from "../src/lib/representations";

// Corpora and representations: reading an uploaded file's fields, naming things, describing what
// was read, and the scatter's marks, scales and keyboard order.

test("a file's format comes from its name, and Parquet and compressed files aren't uploaded", () => {
  expect(formatOf("reviews.CSV")).toBe("csv");
  expect(formatOf("rows.ndjson")).toBe("jsonl");
  expect(formatOf("notes.txt")).toBe("lines");
  expect(formatOf("notes.txt", "paragraphs")).toBe("paragraphs");
  expect(formatOf("anything")).toBe("lines");
  expect(formatOf("train.parquet")).toBeNull();
  expect(formatOf("train.jsonl.gz")).toBeNull();
});

test("an uploaded file's fields are read from its first rows", () => {
  expect(fieldsOf('{"text": "a", "label": 1}\n{"text": "b", "source": "x"}\n', "jsonl")).toEqual(["text", "label", "source"]);
  expect(fieldsOf(JSON.stringify([{ sentence: "a" }, { sentence: "b", stars: 5 }]), "json")).toEqual(["sentence", "stars"]);
  // A .json file of JSON Lines reads as JSON Lines.
  expect(fieldsOf('{"a": 1}\n{"b": 2}\n', "json")).toEqual(["a", "b"]);
  expect(fieldsOf('﻿review,"stars, out of 5"\n"good",5\n', "csv")).toEqual(["review", "stars, out of 5"]);
  expect(fieldsOf("text\tlabel\nx\ty\n", "tsv")).toEqual(["text", "label"]);
  expect(fieldsOf("one\ntwo\n", "lines")).toEqual([]);
  expect(splitRow('a,"b ""quoted"", c",d', ",")).toEqual(["a", 'b "quoted", c', "d"]);
});

test("the text and label fields are guessed from their names", () => {
  expect(guessField(["id", "Sentence", "label"], "text")).toBe("Sentence");
  expect(guessField(["id", "body"], "text")).toBe("body");
  expect(guessField(["id", "words"], "text")).toBe("id");
  expect(guessField(["text", "stars"], "label")).toBe("stars");
  expect(guessField(["text"], "label")).toBeNull();
});

test("names are made from files and checked as the server checks them", () => {
  expect(nameFromFile("My reviews (2024).csv")).toBe("My-reviews-2024");
  expect(nameFromFile("folder/.hidden.txt")).toBe("hidden");
  expect(nameFromFile("???.txt")).toBe("corpus");
  const taken = new Set(["reviews"]);
  expect(nameProblem("reviews", taken)).toContain("already exists");
  expect(nameProblem("", taken)).toContain("Give the corpus a name");
  expect(nameProblem("a/b", taken)).toContain("Use letters, digits");
  expect(nameProblem("pc1-resid_post-6", taken, "direction")).toBeNull();
  expect(wholeNumber(" 12 ")).toBe(12);
  expect(wholeNumber("-3")).toBe(-3);
  expect(wholeNumber("1.5")).toBeNull();
  expect(wholeNumber("")).toBeNull();
});

test("a dataset offers the labels and the meta fields every prompt has", () => {
  const facts = datasetFacts([
    { positions: { S2: [1, 2], end: [3, 4] }, meta: { template: "went", pattern: "ABBA", extra: [1] } },
    { positions: { S2: [1, 2] }, meta: { template: "home", pattern: "BABA" } },
  ]);
  expect(facts).toEqual({ labels: ["S2"], metaFields: ["template", "pattern"] });
  expect(datasetFacts([])).toEqual({ labels: [], metaFields: [] });
});

test("sites and positions read as the server writes them", () => {
  expect(siteWords({ kind: "resid_post", layer: 6 })).toBe("the residual stream leaving layer 6");
  expect(siteWords({ kind: "mlp_out", layer: 0 })).toBe("layer 0's MLP output");
  expect(siteName({ kind: "attn_out", layer: 3 })).toBe("attn_out:3");
  expect(positionWords({ kind: "index", index: -2 })).toBe("token -2 from the end");
  expect(positionWords({ kind: "label", label: "S2" })).toBe("the token labelled S2");
});

const info = { id: "org/model", revision: "abc1234def", source: "hub", sha256: null, d_model: 32 } as unknown as ModelInfo;
const direction = (model: DirectionListing["model"], width = 32): DirectionListing => ({ name: "d", error: null, model, width, dimensions: 1 });

test("a direction found on another model is marked, and one of another width can't be read", () => {
  expect(foundElsewhere(direction({ id: "org/model", source: "hub", revision: "abc1234def", sha256: null }), info)).toBe(false);
  expect(foundElsewhere(direction({ id: "org/model", source: "hub", revision: null, sha256: null }), info)).toBe(false);
  expect(foundElsewhere(direction({ id: "org/model", source: "hub", revision: "other", sha256: null }), info)).toBe(true);
  expect(foundElsewhere(direction({ id: "other/model", source: "hub", revision: null, sha256: null }), info)).toBe(true);
  expect(foundElsewhere(direction({ id: "mine", source: "local", revision: null, sha256: "f".repeat(64) }), info)).toBe(true);
  expect(directionModelText(direction({ id: "org/model", source: "hub", revision: "abc1234def", sha256: null }))).toBe("org/model at abc1234");
  expect(directionProblem(direction({ id: "org/model", source: "hub", revision: null, sha256: null }, 48), info)).toContain("48 wide");
  expect(directionProblem({ name: "x", error: "It can't be read." }, info)).toBe("It can't be read.");
});

const example = (values: (number | null)[]): TopExample => ({ text: 0, id: null, label: null, position: 1, value: 1, start: 0, tokens: values.map(() => "x"), values, open: null });

test("examples are shaded up to the largest magnitude they show, and a reading says what it read", () => {
  expect(exampleBound([example([null, 2, -1])], [example([-5, 0])])).toBe(5);
  expect(exampleBound([example([null, 0])])).toBe(1);
  const result = {
    tokens: 1200,
    texts_read: 10,
    truncated: 3,
    active_share: 0.25,
    settings: { max_tokens: 128 },
  } as unknown as CorpusTop;
  expect(readingText(result)).toBe("Over 1,200 tokens of 10 texts, 3 of them cut to 128 tokens, it is above zero at 25.0% of them.");
  const bars = histogramBars({ edges: [0, 1, 2, 4], counts: [2, 0, 4] }, 100, 50);
  expect(bars.map((b) => [b.x, b.w, b.h])).toEqual([
    [0, 25, 25],
    [25, 25, 0],
    [50, 50, 50],
  ]);
});

const point = (row: number, coords: number[], mark: string | null = null, side: "clean" | "corrupt" | null = null): RepresentationPoint => ({
  row,
  index: row,
  side,
  mark,
  text: `text ${row}`,
  coords,
});

test("each class gets an ink shape, filled then hollow, and the rest are dots", () => {
  const marks = classMarks(["a", "b", "c", "d", "e", "f", "g", "h", "i"]);
  expect(MARKED_CLASSES).toBe(8);
  expect(marks.get("a")).toEqual({ shape: "circle", filled: true });
  expect(marks.get("d")).toEqual({ shape: "diamond", filled: true });
  expect(marks.get("e")).toEqual({ shape: "circle", filled: false });
  expect(marks.has("i")).toBe(false);
  const counts = classCounts([point(0, [0], "a"), point(1, [0], "a"), point(2, [0], null)]);
  expect([...counts.entries()]).toEqual([
    ["a", 2],
    [null, 1],
  ]);
});

test("the scatter's scale keeps zero in view, and the keyboard and the pointer find points", () => {
  expect(axisDomain([1, 3])).toEqual([-0.18, 3.18]);
  expect(axisDomain([0, 0])).toEqual([-1, 1]);
  const points = [point(0, [2, 0]), point(1, [-1, 1]), point(2, [-1, -1]), point(3, [0, 0])];
  const margin = { left: 0, right: 0, top: 0, bottom: 0 };
  const scale = scatterScale(points, 0, 1, 100, 100, margin);
  expect(scale.x(scale.xDomain[0])).toBe(0);
  expect(scale.y(scale.yDomain[0])).toBe(100);
  expect(keyboardOrder(points, 0, 1)).toEqual([2, 1, 3, 0]);
  const at = (i: number) => [scale.x(points[i].coords[0]), scale.y(points[i].coords[1])];
  const [x, y] = at(3);
  expect(nearestPoint(points, scale, 0, 1, x + 2, y - 1)).toBe(3);
  expect(nearestPoint(points, scale, 0, 1, x + 40, y + 40)).toBeNull();
});

test("a point and an export are described in words", () => {
  expect(pointText(point(3, [0], "clean", "clean"))).toBe("Prompt 3 (clean): text 3");
  expect(pointLabel(point(3, [0], "ABBA", "corrupt"))).toBe("Prompt 3 (corrupt) · ABBA");
  expect(pointText(point(3, [0], "ABBA", "corrupt"))).toBe("Prompt 3 (corrupt) · ABBA: text 3");
  expect(pointText({ ...point(7, [0], "pos"), text: "x".repeat(200) }, 10)).toBe("Text 7 · pos: xxxxxxxxxx…");
  expect(explainedText(0.4567)).toBe("46%");
  expect(explainedText(0.0123)).toBe("1.2%");
  expect(exportSites({ kind: "resid_post", layer: 2 }, false, 4)).toEqual([{ kind: "resid_post", layer: 2 }]);
  expect(exportSites({ kind: "resid_post", layer: 2 }, true, 3).map((s) => s.layer)).toEqual([0, 1, 2]);
  expect(exportSize(1000, 768, 1)).toBe("3.1 MB");
  expect(exportSize(10, 32, 2)).toBe("3 KB");
});
