import { test, expect } from "@playwright/test";
import {
  COALESCE_MS,
  continuesEdit,
  EMPTY_HISTORY,
  HISTORY_LIMIT,
  pushHistory,
  redoHistory,
  sameForm,
  undoHistory,
} from "../src/lib/formHistory";
import { DEFAULT_FORM, FORM_REPLACED, useStore, type FormState } from "../src/store/app";
import type { ProjectInfo } from "../src/api/types";

const project = (id: string): ProjectInfo => ({ session_id: id, name: id, path: `projects/${id}`, datasets: [], description: "" });
const form = (patch: Partial<FormState>): FormState => ({ ...DEFAULT_FORM, ...patch });

function fresh() {
  const store = useStore.getState();
  store.leaveProject();
  useStore.setState({ project: project("form"), screen: "workbench", view: "experiment" });
  return store;
}

test("history keeps at most the last hundred steps, and a new change drops what was undone", () => {
  let history = EMPTY_HISTORY;
  for (let i = 0; i < HISTORY_LIMIT + 50; i++) history = pushHistory(history, form({ statSeed: i }));
  expect(history.past).toHaveLength(HISTORY_LIMIT);
  expect(history.past[0].statSeed).toBe(50);
  expect(history.past.at(-1)?.statSeed).toBe(HISTORY_LIMIT + 49);

  const current = form({ statSeed: 1000 });
  const undone = undoHistory(history, current);
  expect(undone?.form.statSeed).toBe(HISTORY_LIMIT + 49);
  expect(undone?.history.future).toEqual([current]);
  const redone = redoHistory(undone!.history, undone!.form);
  expect(redone?.form).toEqual(current);
  expect(pushHistory(undone!.history, undone!.form).future).toEqual([]);
});

test("steps that would change nothing are skipped, and a suggested name is not a change", () => {
  const a = form({ notes: "first" });
  const b = form({ notes: "first", name: "Patch clean→corrupt · L1 H2" });
  expect(sameForm(a, b)).toBe(true);
  expect(sameForm(a, { ...b, nameEdited: true })).toBe(false);
  // Key order doesn't matter: forms loaded from a spec and forms built in the page compare equal.
  expect(sameForm(form({ baseline: { kind: "resample", pool: "corrupt", donors: 3, seed: 1 } }),
    form({ baseline: { seed: 1, donors: 3, pool: "corrupt", kind: "resample" } }))).toBe(true);
  const history = { past: [form({ notes: "older" }), a], future: [] };
  expect(undoHistory(history, b)?.form.notes).toBe("older");
  expect(undoHistory({ past: [a], future: [] }, b)).toBeNull();
});

test("rapid edits to the same field are one step; other fields and pauses start a new one", () => {
  expect(continuesEdit({ fields: "notes", at: 1000 }, "notes", 1000 + COALESCE_MS - 1)).toBe(true);
  expect(continuesEdit({ fields: "notes", at: 1000 }, "notes", 1000 + COALESCE_MS)).toBe(false);
  expect(continuesEdit({ fields: "notes", at: 1000 }, "name,nameEdited", 1001)).toBe(false);
  expect(continuesEdit(null, "notes", 1001)).toBe(false);

  const store = fresh();
  try {
    for (const notes of ["T", "Th", "The", "The heads"]) store.setForm({ notes });
    expect(useStore.getState().formHistory.past).toHaveLength(1);
    store.setForm({ bootstrap: 2000 });
    store.setForm({ kind: "ablation" });
    expect(useStore.getState().formHistory.past).toHaveLength(3);

    store.undoForm();
    expect(useStore.getState().form.kind).toBe("activation_patching");
    store.undoForm();
    expect(useStore.getState().form.bootstrap).toBe(1000);
    expect(useStore.getState().form.notes).toBe("The heads");
    store.undoForm();
    expect(useStore.getState().form).toEqual(DEFAULT_FORM);
    store.undoForm(); // nothing left: no change
    expect(useStore.getState().form).toEqual(DEFAULT_FORM);

    store.redoForm();
    store.redoForm();
    expect(useStore.getState().form.bootstrap).toBe(2000);
    expect(useStore.getState().formHistory.future).toHaveLength(1);
    // A suggested name follows the form without a step of its own, and keeps redo available.
    store.setForm({ name: "Patch clean→corrupt · layer × head" }, { record: false });
    expect(useStore.getState().formHistory.future).toHaveLength(1);
    // A new edit after undoing drops the steps that were undone.
    store.setForm({ ci: 0.99 });
    expect(useStore.getState().formHistory.future).toEqual([]);
  } finally {
    store.leaveProject();
  }
});

test("patching a component replaces the form but keeps the notes, with a notice to undo it", () => {
  const store = fresh();
  try {
    store.setForm({ notes: "Name movers should matter here.", kind: "ablation", baseline: { kind: "zero" } });
    const before = useStore.getState().form;
    store.prefillExperiment("activation_patching", { layer: 9, part: "head", head: 6 });
    const after = useStore.getState();
    expect(after.form.notes).toBe("Name movers should matter here.");
    expect(after.form.kind).toBe("activation_patching");
    expect(after.form.scope).toEqual({ kind: "sites", sites: [{ kind: "head", layer: 9, head: 6, position: { kind: "all" } }] });
    const notice = after.notices.find((n) => n.text === FORM_REPLACED);
    expect(notice?.action?.label).toBe("Undo");

    notice?.action?.run();
    expect(useStore.getState().form).toEqual(before);
    // The undo is itself a step: the replaced form is one Undo away again.
    store.undoForm();
    expect(useStore.getState().form.kind).toBe("activation_patching");
  } finally {
    store.leaveProject();
  }
});

test("undoing a staged configuration puts the sites back in the tray", () => {
  const store = fresh();
  try {
    useStore.setState({ view: "explore" });
    store.stageSelection({ layer: 1, part: "head", head: 0 });
    store.stageSelection({ layer: 2, part: "mlp" });
    const staged = useStore.getState().stagedSites;
    store.configureStaged();
    expect(useStore.getState().stagedSites).toEqual([]);
    expect(useStore.getState().form.scope).toEqual({ kind: "sites", sites: staged });
    useStore.getState().notices.find((n) => n.text === FORM_REPLACED)?.action?.run();
    expect(useStore.getState().form.scope).toEqual(DEFAULT_FORM.scope);
    expect(useStore.getState().stagedSites).toEqual(staged);
  } finally {
    store.leaveProject();
  }
});
