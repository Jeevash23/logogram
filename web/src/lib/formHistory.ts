// Undo and redo for the experiment form. Every change pushes the form it replaces; rapid edits
// to the same fields, such as typing, are one step.

import type { FormState } from "../store/app";

/** Steps kept; the oldest go first. */
export const HISTORY_LIMIT = 100;
/** Edits to the same fields closer together than this belong to one step, as typing does. */
export const COALESCE_MS = 1000;

export interface FormHistory {
  /** Earlier forms, oldest first. */
  past: FormState[];
  /** Forms undone, most recently undone last. */
  future: FormState[];
}

export const EMPTY_HISTORY: FormHistory = { past: [], future: [] };

/** Equality of plain data, whatever the key order (the form holds only JSON values). */
export function sameData(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => sameData(x, b[i]));
  const ka = Object.keys(a).filter((k) => (a as Record<string, unknown>)[k] !== undefined);
  const kb = Object.keys(b).filter((k) => (b as Record<string, unknown>)[k] !== undefined);
  return ka.length === kb.length && ka.every((k) => sameData((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** Whether two forms describe the same experiment. A suggested name follows the rest of the form,
 * so it doesn't count; a name the user typed does. */
export function sameForm(a: FormState, b: FormState): boolean {
  return sameData(a.nameEdited ? a : { ...a, name: "" }, b.nameEdited ? b : { ...b, name: "" });
}

/** Remember the form a change is about to replace. A new change drops what was undone. */
export function pushHistory(history: FormHistory, previous: FormState, limit = HISTORY_LIMIT): FormHistory {
  const past = [...history.past, previous];
  return { past: past.length > limit ? past.slice(past.length - limit) : past, future: [] };
}

/** The form before the current one, skipping steps that would change nothing. */
export function undoHistory(history: FormHistory, current: FormState): { history: FormHistory; form: FormState } | null {
  let i = history.past.length - 1;
  while (i >= 0 && sameForm(history.past[i], current)) i--;
  if (i < 0) return null;
  return { form: history.past[i], history: { past: history.past.slice(0, i), future: [...history.future, current] } };
}

/** The form last undone, skipping steps that would change nothing. */
export function redoHistory(history: FormHistory, current: FormState): { history: FormHistory; form: FormState } | null {
  let i = history.future.length - 1;
  while (i >= 0 && sameForm(history.future[i], current)) i--;
  if (i < 0) return null;
  return { form: history.future[i], history: { past: [...history.past, current], future: history.future.slice(0, i) } };
}

/** Forget a saved draft that has now run: no form in the history fills its folder again. */
export function forgetDraft(history: FormHistory, id: string): FormHistory {
  const drop = (form: FormState) => (form.draftId === id ? { ...form, draftId: null } : form);
  return { past: history.past.map(drop), future: history.future.map(drop) };
}

/** The most recent edit: which fields it changed, and when. */
export interface LastEdit {
  fields: string;
  at: number;
}

/** The fields an edit changes, as one comparable name. */
export function editFields(patch: Partial<FormState>): string {
  return Object.keys(patch).sort().join(",");
}

/** Whether an edit continues the previous one: the same fields, changed again within COALESCE_MS. */
export function continuesEdit(last: LastEdit | null, fields: string, at: number): boolean {
  return last !== null && last.fields === fields && at - last.at >= 0 && at - last.at < COALESCE_MS;
}
