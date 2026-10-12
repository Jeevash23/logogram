import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { ChatTemplateSpec, ChatWrapped } from "../api/types";
import { Callout, Field, Segmented, Spinner, TextArea } from "../components/ui";
import { CHAT_CHOICES, chatTemplateFor, hasChatTemplate, shownBreaks, systemMessage, type ChatChoice } from "../lib/chat";
import { modelName } from "../lib/hooks";
import { useStore } from "../store/app";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

const WHAT_EACH_DOES: Record<ChatChoice, string> = {
  none: "Each prompt is read as plain text, as it is written.",
  user_turn:
    "Each prompt is the person's message; the template then opens the model's reply, so the answer is the first token of the reply (often without a leading space).",
  assistant_prefix:
    "Each prompt is the start of the model's reply to the message below; the model goes on from it, so the answer continues the prompt.",
};

/**
 * How the experiment's prompts are read: as plain text, or wrapped in the loaded model's own chat
 * template, as the person's message or as the start of the model's reply. Offered when the model
 * ships a template (or a saved spec uses one), with the first prompt shown as the model reads it.
 */
export function ChatTemplateSettings({
  value,
  onChange,
  firstPrompt,
}: {
  value: ChatTemplateSpec | null;
  onChange: (value: ChatTemplateSpec | null) => void;
  /** The first prompt of the dataset, shown wrapped. */
  firstPrompt: string | null;
}) {
  const info = useStore((st) => st.model.info);
  if (!hasChatTemplate(info) && value === null) return null;
  const choice: ChatChoice = value?.kind ?? "none";
  return (
    <div className={e.stack}>
      <Field label="Chat template" help={WHAT_EACH_DOES[choice]}>
        <Segmented label="Chat template" value={choice} onChange={(c) => onChange(chatTemplateFor(c, value))} options={CHAT_CHOICES} />
      </Field>
      {value?.kind === "assistant_prefix" && (
        <Field label="The person's message the reply answers" help="The same message comes before every prompt.">
          <TextArea
            rows={2}
            maxLength={4000}
            value={value.user}
            onChange={(ev) => onChange({ ...value, user: ev.target.value })}
            placeholder="Continue the sentence."
            aria-invalid={!value.user.trim()}
          />
        </Field>
      )}
      {value && (
        <Field label="System message" help="Optional. Left empty, there is none (some templates then write a default of their own).">
          <TextArea
            rows={1}
            maxLength={4000}
            value={value.system ?? ""}
            onChange={(ev) => onChange({ ...value, system: systemMessage(ev.target.value) })}
          />
        </Field>
      )}
      {value && <WrappedPrompt template={value} text={firstPrompt} />}
    </div>
  );
}

/** The first prompt as the loaded model reads it in its template: the template's text faint, the
 * prompt underlined, line breaks shown. */
function WrappedPrompt({ template, text }: { template: ChatTemplateSpec; text: string | null }) {
  const model = useStore((st) => st.model);
  const ready = model.state === "ready";
  const [wrapped, setWrapped] = useState<ChatWrapped | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The start of a reply answers a message: without one there is nothing to show yet.
  const unanswered = template.kind === "assistant_prefix" && !template.user.trim();
  const key = JSON.stringify([template, text, ready, model.info?.id, model.info?.revision, model.info?.sha256]);
  useEffect(() => {
    setError(null);
    if (!ready || !text || unanswered) {
      setWrapped(null);
      return;
    }
    let cancelled = false;
    // Typing a message renders again once it pauses.
    const t = window.setTimeout(() => {
      api.chatWrap({ chat_template: template, text }).then(
        (out) => !cancelled && setWrapped(out),
        (err: Error) => {
          if (cancelled) return;
          setWrapped(null);
          setError(err.message);
        },
      );
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // The key holds everything the request depends on.
  }, [key]);

  if (!ready) return <p className={s.small}>Load the model to see the text it reads.</p>;
  if (!text) return <p className={s.small}>Choose prompts to see the first one as the model reads it.</p>;
  if (unanswered) return <p className={s.small}>Write the person's message to see the first prompt as the model reads it.</p>;
  if (error) return <Callout tone="error" title="The chat template can't be used">{error}</Callout>;
  if (!wrapped) return <Spinner label="Wrapping the first prompt" />;
  const { start, end } = wrapped;
  return (
    <figure className={e.wrapped} aria-labelledby="wrapped-caption">
      <figcaption id="wrapped-caption" className={s.small}>
        What {modelName(model.info?.id)} reads for the first prompt: the clean prompt is underlined, the rest is its chat template's
        (SHA-256 {wrapped.sha256.slice(0, 12)}…). ⏎ marks a line break.
      </figcaption>
      <p className={e.wrappedText}>
        {start === null || end === null ? (
          shownBreaks(wrapped.text)
        ) : (
          <>
            <span className={e.templateText}>{shownBreaks(wrapped.text.slice(0, start))}</span>
            <mark className={e.promptText}>{shownBreaks(wrapped.text.slice(start, end))}</mark>
            <span className={e.templateText}>{shownBreaks(wrapped.text.slice(end))}</span>
          </>
        )}
      </p>
      {wrapped.problem && <Callout tone="error" title="The template changes this prompt">{wrapped.problem}</Callout>}
    </figure>
  );
}
