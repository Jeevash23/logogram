import { test, expect } from "@playwright/test";

import type { PromptRecord } from "../src/api/types";
import {
  deriveRequest,
  draftOf,
  editCounts,
  editedRecords,
  editsText,
  hasEdits,
  nameProblem,
  noEdits,
  recordOf,
  sameRecord,
  spanText,
  suggestName,
  withEdit,
  withRemoved,
  type RecordDraft,
} from "../src/lib/promptEdits";
import { PAGE_SIZE, foundByMeta, labelsText, matchRows, matchText, pageOfRow, pageRows, rangeText, rowAfterKey, wordDiff } from "../src/lib/promptRows";

const ioi = (i: number, name = "Mary"): PromptRecord => ({
  clean: `When ${name} and John went to the store number ${i}, John gave a drink to`,
  corrupt: `When ${name} and John went to the store number ${i}, ${name} gave a drink to`,
  answer: ` ${name}`,
  distractor: " John",
  meta: { template: i % 2 ? "went" : "home", pattern: "ABBA" },
});

test("a search finds text in either prompt, the answers and meta, ignoring case", () => {
  const records: PromptRecord[] = [
    { clean: "The war lasted from 1732 to 17", corrupt: "The war lasted from 1701 to 17", answer: ["33", "34"], distractor: ["30", "31"] },
    { clean: "def f(alpha, beta):", corrupt: "def f(gamma, delta):", answer: " beta", distractor: " alpha", meta: { template: "docstring", checked: true } },
    { clean: "When Mary and John went", corrupt: "When Anna and John went", answer: " Mary", distractor: " John" },
  ];
  expect(matchRows(records, "")).toEqual([0, 1, 2]);
  expect(matchRows(records, "   ")).toEqual([0, 1, 2]);
  expect(matchRows(records, "1732")).toEqual([0]); // the clean prompt
  expect(matchRows(records, "anna")).toEqual([2]); // the corrupt prompt, whatever the case
  expect(matchRows(records, "34")).toEqual([0]); // a token of an answer's set
  expect(matchRows(records, "31")).toEqual([0]); // and of a distractor's
  expect(matchRows(records, "  MARY ")).toEqual([2]); // trimmed
  expect(matchRows(records, "DOCSTRING")).toEqual([1]); // meta's values
  expect(matchRows(records, "checked: true")).toEqual([1]); // and its keys
  expect(matchRows(records, "gamma")).toEqual([1]);
  // Fields are searched one at a time: no match runs from one into the next.
  expect(matchRows(records, "17 the war")).toEqual([]);
  expect(matchRows(records, "nothing like this")).toEqual([]);
  // A row found by its meta alone shows it, since the table doesn't.
  expect(foundByMeta(records[1], "docstring")).toBe(true);
  expect(foundByMeta(records[1], "beta")).toBe(false);
  expect(foundByMeta(records[1], "")).toBe(false);
  expect(foundByMeta(records[2], "docstring")).toBe(false);
});

test("pages hold a hundred rows each and say where they are", () => {
  const rows = Array.from({ length: 2000 }, (_, i) => i);
  expect(PAGE_SIZE).toBe(100);
  const second = pageRows(rows, 1);
  expect(second).toMatchObject({ page: 1, pages: 20, total: 2000, start: 100, end: 200 });
  expect(second.rows[0]).toBe(100);
  expect(second.rows.at(-1)).toBe(199);
  expect(rangeText(second, false)).toBe("Rows 101–200 of 2,000");
  expect(rangeText(pageRows(rows, 0), false)).toBe("Rows 1–100 of 2,000");

  // The last page holds what is left; a page past it or before the first shows the nearest.
  const odd = Array.from({ length: 250 }, (_, i) => i);
  expect(pageRows(odd, 2)).toMatchObject({ page: 2, pages: 3, start: 200, end: 250 });
  expect(pageRows(odd, 2).rows).toHaveLength(50);
  expect(pageRows(odd, 7).page).toBe(2);
  expect(pageRows(odd, -3).page).toBe(0);
  expect(pageRows(odd, Number.NaN).page).toBe(0);
  expect(rangeText(pageRows(odd, 2), false)).toBe("Rows 201–250 of 250");

  // Every row is on some page, once.
  const seen = Array.from({ length: pageRows(odd, 0).pages }, (_, page) => pageRows(odd, page).rows).flat();
  expect(seen).toEqual(odd);

  expect(pageRows([], 0)).toMatchObject({ rows: [], page: 0, pages: 1, total: 0 });
  expect(rangeText(pageRows([], 0), false)).toBe("No rows");
  expect(rangeText(pageRows([], 0), true)).toBe("No matches");
});

test("a search looks through every row, and its matches are paged in turn", () => {
  const records = Array.from({ length: 2000 }, (_, i) => ioi(i, i % 500 === 7 ? "Zelda" : "Mary"));
  const matches = matchRows(records, "zelda");
  // Found far past the first page.
  expect(matches).toEqual([7, 507, 1007, 1507]);
  const page = pageRows(matches, 0);
  expect(page.rows).toEqual([7, 507, 1007, 1507]);
  expect(rangeText(page, true)).toBe("Matches 1–4 of 4");
  expect(matchText(matches.length, records.length)).toBe("4 of 2,000 rows match");
  expect(matchText(1, 2000)).toBe("1 of 2,000 rows matches");
  expect(matchText(1, 1)).toBe("1 of 1 row matches");
  expect(matchText(0, 2000)).toBe("No rows match");

  const many = matchRows(records, "store number 1"); // 1, 10–19, 100–199, 1000–1999
  expect(many).toHaveLength(1 + 10 + 100 + 1000);
  const third = pageRows(many, 2);
  expect(rangeText(third, true)).toBe("Matches 201–300 of 1,111");
  expect(pageOfRow(many, 1999)).toBe(11);
  expect(pageOfRow(many, 1)).toBe(0);
  expect(pageOfRow(many, 2)).toBeNull();
});

test("the keyboard moves a row, a page, or to either end, and no further", () => {
  expect(rowAfterKey("ArrowDown", 5, 150)).toBe(6);
  expect(rowAfterKey("ArrowUp", 5, 150)).toBe(4);
  expect(rowAfterKey("ArrowUp", 0, 150)).toBe(0);
  expect(rowAfterKey("ArrowDown", 149, 150)).toBe(149);
  expect(rowAfterKey("PageDown", 5, 150)).toBe(105);
  expect(rowAfterKey("PageDown", 105, 150)).toBe(149);
  expect(rowAfterKey("PageUp", 105, 150)).toBe(5);
  expect(rowAfterKey("PageUp", 5, 150)).toBe(0);
  expect(rowAfterKey("Home", 77, 150)).toBe(0);
  expect(rowAfterKey("End", 77, 150)).toBe(149);
  for (const key of ["Enter", " ", "a", "Tab", "toString", "constructor"]) expect(rowAfterKey(key, 5, 150)).toBeNull();
  expect(rowAfterKey("ArrowDown", 0, 0)).toBeNull();
});

test("edits replace and remove prompt pairs without touching the file's", () => {
  const records = [ioi(0), ioi(1), ioi(2), ioi(3)];
  const before = JSON.stringify(records);
  let edits = noEdits("abc");
  expect(hasEdits(edits)).toBe(false);

  const changed = { ...records[1], answer: [" Mary", " Mia"] };
  edits = withEdit(edits, 1, changed, records[1]);
  edits = withRemoved(edits, [3], true);
  expect(editCounts(edits)).toEqual({ edited: 1, removed: 1 });
  expect(editsText(edits)).toBe("1 prompt pair edited and 1 removed");
  expect(editedRecords(records, edits)[1]).toEqual(changed);
  expect(JSON.stringify(records)).toBe(before);

  // Editing a pair back to the file's own is no edit; meta's order doesn't count.
  const reordered = { ...records[1], meta: { pattern: "ABBA", template: "went" } };
  expect(sameRecord(reordered, records[1])).toBe(true);
  expect(editCounts(withEdit(edits, 1, reordered, records[1]))).toEqual({ edited: 0, removed: 1 });

  // A removed pair keeps its edits until it is restored, but they aren't sent.
  edits = withRemoved(edits, [1], true);
  expect(editsText(edits)).toBe("2 prompt pairs removed");
  expect(deriveRequest("datasets/ioi.jsonl", edits, " ioi-edited ")).toEqual({
    dataset: "datasets/ioi.jsonl",
    dataset_sha256: "abc",
    name: "ioi-edited",
    remove: [1, 3],
    replace: {},
  });
  edits = withRemoved(edits, [1, 3], false);
  expect(editsText(edits)).toBe("1 prompt pair edited");
  expect(deriveRequest("datasets/ioi.jsonl", edits, "x")).toMatchObject({ remove: [], replace: { 1: changed } });
  expect(withRemoved(noEdits("abc"), [5, 2, 5, 9], true).removed).toEqual([2, 5, 9]);
});

test("a new dataset is offered a name that isn't taken, and refused one the server would refuse", () => {
  const taken = new Set(["ioi.jsonl", "ioi-edited.jsonl", "ioi-edited-2.jsonl"]);
  expect(suggestName("ioi.jsonl", new Set())).toBe("ioi-edited");
  expect(suggestName("ioi.jsonl", taken)).toBe("ioi-edited-3");
  expect(suggestName("datasets/greater-than.jsonl", taken)).toBe("greater-than-edited");
  // A run's saved copy is named by its hash.
  expect(suggestName(`${"0a".repeat(32)}.jsonl`, taken)).toBe("prompts-edited");
  const long = suggestName(`${"n".repeat(100)}.jsonl`, new Set());
  expect(long).toHaveLength(100);
  expect(long.endsWith("-edited")).toBe(true);
  expect(nameProblem(long, new Set())).toBeNull();

  expect(nameProblem("ioi-edited-3", taken)).toBeNull();
  expect(nameProblem("ioi-edited-3.jsonl", taken)).toBeNull();
  expect(nameProblem("ioi-edited", taken)).toBe("ioi-edited.jsonl already exists. Choose another name.");
  expect(nameProblem("  ", taken)).toBe("Give the new dataset a name.");
  for (const bad of ["../ioi", "a/b", ".hidden", "-x", "has space", "x".repeat(101)]) {
    expect(nameProblem(bad, taken)).toContain("Use letters, digits");
  }
});

test("the editor's fields hold every field of a prompt pair, and say what to fix", () => {
  const clean = "When Mary and John went to the store, John gave a drink to";
  const record: PromptRecord = {
    clean,
    corrupt: clean.replace("John gave", "Mary gave"),
    answer: [" Mary", " Mia"],
    distractor: " John Smith",
    positions: { IO: [5, 9], end: [clean.length - 2, clean.length] },
    id: "ioi-00001",
    meta: { template: "went", nested: { a: [1, 2] } },
  };
  const draft = draftOf(record);
  expect(draft.meta).toBe('{"template": "went", "nested": {"a":[1,2]}}');
  expect(draft.answer).toMatchObject({ kind: "set", members: " Mary\n Mia" });
  expect(draft.distractor).toMatchObject({ kind: "text", text: " John Smith" });
  expect(spanText(clean, draft.positions[0])).toBe("Mary");
  expect(recordOf(draft)).toEqual({ record, problem: null });

  // Optional fields left empty are left out.
  const bare: PromptRecord = { clean: "a b", corrupt: "a c", answer: " x", distractor: " y" };
  expect(recordOf(draftOf(bare))).toEqual({ record: bare, problem: null });
  // A set from a token: blank lines don't count, spaces do.
  const asSet: RecordDraft = { ...draftOf(bare), answer: { kind: "set", text: " x", members: " x\n\n x \n" } };
  expect(recordOf(asSet).record?.answer).toEqual([" x", " x "]);

  const problem = (patch: Partial<RecordDraft>) => recordOf({ ...draft, ...patch }).problem;
  expect(problem({ clean: "" })).toBe("Write the clean prompt.");
  expect(problem({ corrupt: "" })).toBe("Write the corrupt prompt.");
  expect(problem({ distractor: { kind: "text", text: "", members: "" } })).toBe("Write the distractor.");
  expect(problem({ answer: { kind: "set", text: "", members: "\n\n" } })).toBe("List the tokens of the answer's set, one per line.");
  expect(problem({ answer: { kind: "set", text: "", members: " Mary\n Mia\n Mary" } })).toBe("The answer's set lists “·Mary” twice. List each token once.");
  expect(problem({ positions: [{ label: "", start: "0", end: "1" }] })).toBe("Name each position, or remove it.");
  expect(problem({ positions: [{ label: "last", start: "0", end: "1" }] })).toContain("“last” can't name a position");
  expect(problem({ positions: [{ label: "-2", start: "0", end: "1" }] })).toContain("can't name a position");
  expect(problem({ positions: [{ label: "S", start: "0", end: "1" }, { label: "S", start: "1", end: "2" }] })).toBe("Two positions are called “S”. Give each its own name.");
  expect(problem({ positions: [{ label: "S", start: "", end: "1" }] })).toContain("needs a start and an end");
  expect(problem({ positions: [{ label: "S", start: "1.5", end: "3" }] })).toContain("needs a start and an end");
  expect(problem({ positions: [{ label: "S", start: "4", end: "4" }] })).toBe("Position “S” must end after it starts, at 0 or later.");
  expect(problem({ positions: [{ label: "S", start: "0", end: "99" }] })).toBe(`Position “S” ends at character 99, but the clean prompt has ${clean.length}.`);
  expect(problem({ meta: "{template: went}" })).toContain("Meta isn't valid JSON");
  expect(problem({ meta: "[1, 2]" })).toContain("Meta must be a JSON object");
  expect(problem({ meta: "  " })).toBeNull();
  expect(spanText(clean, { label: "S", start: "0", end: "99" })).toBeNull();
  // Any other label is the dataset's own, even one an object already has.
  const own = recordOf({ ...draft, positions: [{ label: "constructor", start: "0", end: "4" }, { label: "__proto__", start: "5", end: "9" }] });
  expect(JSON.stringify(own.record?.positions)).toBe('{"constructor":[0,4],"__proto__":[5,9]}');
});

test("prompts of different lengths mark only the words they don't share, and show their labels", () => {
  const clean = "When Mary and John went to the store, John gave a drink to";
  const corrupt = "When Mary and John went to the store, Bob Kelly gave a drink to";
  const marked = (text: string, other: string) => wordDiff(text, other).filter((w) => w.differs).map((w) => w.text);
  // Compared in place, every word after the extra one would differ.
  expect(marked(clean, corrupt)).toEqual(["John"]);
  expect(marked(corrupt, clean)).toEqual(["Bob", "Kelly"]);
  expect(wordDiff(corrupt, clean).map((w) => w.text).join("")).toBe(corrupt);
  // Prompts of as many words are compared in place, as they always were.
  expect(marked("When Mary and John", "When John and Mary")).toEqual(["Mary", "John"]);

  expect(labelsText(corrupt, { end: [61, 63], S2: [38, 47], IO: [5, 9] })).toBe("IO “Mary” · S2 “Bob Kelly” · end “to”");
  expect(labelsText(clean, undefined)).toBe("");
});

test("the editor labels the corrupt prompt's positions too, and says what to fix", () => {
  const clean = "When Mary and John went to the store, John gave a drink to";
  const corrupt = "When Mary and John went to the store, Bob Kelly gave a drink to";
  const record: PromptRecord = {
    clean,
    corrupt,
    answer: " Mary",
    distractor: " John",
    positions: { S2: [38, 42], end: [56, 58] },
    corrupt_positions: { S2: [38, 47], end: [61, 63] },
  };
  const draft = draftOf(record);
  expect(draft.labelCorrupt).toBe(true);
  expect(draft.positions[0]).toMatchObject({ label: "S2", corruptStart: "38", corruptEnd: "47" });
  expect(recordOf(draft)).toEqual({ record, problem: null });
  // Unlabelling the corrupt prompt leaves its positions out.
  const { corrupt_positions: _, ...unlabelled } = record;
  expect(recordOf({ ...draft, labelCorrupt: false })).toEqual({ record: unlabelled, problem: null });
  // Labelled with no positions: matched at the last token only.
  expect(recordOf({ ...draft, positions: [] }).record?.corrupt_positions).toEqual({});

  const problem = (patch: Partial<RecordDraft["positions"][number]>) =>
    recordOf({ ...draft, positions: [{ ...draft.positions[0], ...patch }] }).problem;
  expect(problem({ corruptStart: "" })).toContain("needs a start and an end in the corrupt prompt too");
  expect(problem({ corruptStart: "40", corruptEnd: "40" })).toBe("Position “S2” must end after it starts in the corrupt prompt, at 0 or later.");
  expect(problem({ corruptEnd: "99" })).toBe(`Position “S2” ends at character 99 of the corrupt prompt, which has ${corrupt.length}.`);
  expect(spanText(corrupt, { label: "S2", start: "38", end: "47", corruptStart: "", corruptEnd: "" })).toBe("Bob Kelly");
});
