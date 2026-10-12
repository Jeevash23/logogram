import * as RadixDialog from "@radix-ui/react-dialog";
import * as RadixTooltip from "@radix-ui/react-tooltip";
import {
  createContext,
  forwardRef,
  useContext,
  useId,
  useRef,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type KeyboardEvent,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { Logogram } from "../Logogram";
import { Icon, type IconName } from "./Icon";
import s from "./ui.module.css";

export { Icon } from "./Icon";

function cx(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(" ");
}

export { cx };

// -- buttons -----------------------------------------------------------------------------------

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "small" | "medium" | "large";
  icon?: IconName;
  iconAfter?: IconName;
  /** What the button does, or why it can't be used yet, shown on hover and on keyboard focus. A
   * disabled button with a tip stays focusable, so its reason can be read; clicking it does
   * nothing. */
  tip?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "medium", icon, iconAfter, className, children, type, tip, disabled, onClick, ...rest },
  ref,
) {
  const explained = !!disabled && tip !== undefined && tip !== null && tip !== false;
  const button = (
    <button
      ref={ref}
      type={type ?? "button"}
      className={cx(
        s.button,
        variant === "primary" && s.primary,
        variant === "ghost" && s.ghost,
        size === "small" && s.small,
        size === "large" && s.large,
        !children && s.iconOnly,
        className,
      )}
      disabled={explained ? undefined : disabled}
      aria-disabled={explained || undefined}
      onClick={explained ? (e) => e.preventDefault() : onClick}
      {...rest}
    >
      {icon && <Icon name={icon} size={size === "small" ? 14 : 16} />}
      {children}
      {iconAfter && <Icon name={iconAfter} size={size === "small" ? 14 : 16} />}
    </button>
  );
  return tip === undefined || tip === null || tip === false ? button : <Tip text={tip}>{button}</Tip>;
});

// -- form controls -----------------------------------------------------------------------------

// A Field's label is tied to the first text control inside it through this id.
const FieldId = createContext<string | undefined>(undefined);

export function Field({
  label,
  help,
  children,
  htmlFor,
  className,
}: {
  label: ReactNode;
  help?: ReactNode;
  children: ReactNode;
  htmlFor?: string;
  className?: string;
}) {
  const generated = useId();
  const id = htmlFor ?? generated;
  const helpId = `${id}-help`;
  return (
    <div className={cx(s.field, className)}>
      <label className={s.fieldLabel} htmlFor={id}>
        {label}
      </label>
      <FieldId.Provider value={id}>{children}</FieldId.Provider>
      {help && (
        <div className={s.fieldHelp} id={helpId}>
          {help}
        </div>
      )}
    </div>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, id, ...rest }, ref) {
    const fieldId = useContext(FieldId);
    return <input ref={ref} id={id ?? fieldId} className={cx(s.input, className)} {...rest} />;
  },
);

export function TextArea({ className, id, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const fieldId = useContext(FieldId);
  return <textarea id={id ?? fieldId} className={cx(s.textarea, className)} {...rest} />;
}

export function Select({ className, children, id, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  const fieldId = useContext(FieldId);
  return (
    <select id={id ?? fieldId} className={cx(s.select, className)} {...rest}>
      {children}
    </select>
  );
}

/**
 * A radio group's keyboard (WAI-ARIA): one tab stop, on the checked option or else the first
 * that can be chosen; the arrow keys move to the next or previous option, wrapping around, and
 * choose it; Home and End go to the first and last. Disabled options are skipped. Returns the
 * props for the option at an index.
 */
export function useRadioGroup<T>(
  options: { value: T; disabled?: boolean }[],
  selected: T | null | undefined,
  onChange: (value: T) => void,
) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const enabled = options.map((o) => !o.disabled);
  const checked = options.findIndex((o) => o.value === selected);
  const stop = checked >= 0 && enabled[checked] ? checked : enabled.indexOf(true);
  const step = (from: number, by: 1 | -1) => {
    const n = options.length;
    for (let k = 1; k <= n; k++) {
      const i = (((from + by * k) % n) + n) % n;
      if (enabled[i]) return i;
    }
    return from;
  };
  return (index: number) => ({
    ref: (el: HTMLButtonElement | null) => {
      refs.current[index] = el;
    },
    tabIndex: index === stop ? 0 : -1,
    onKeyDown: (e: KeyboardEvent<HTMLButtonElement>) => {
      let next: number;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = step(index, 1);
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = step(index, -1);
      else if (e.key === "Home") next = enabled.indexOf(true);
      else if (e.key === "End") next = enabled.lastIndexOf(true);
      else return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      if (next < 0 || next === index) return;
      refs.current[next]?.focus();
      onChange(options[next].value);
    },
  });
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  /** reason: why a disabled option can't be chosen, a sentence shown under the control. */
  options: { value: T; label: ReactNode; disabled?: boolean; reason?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  const radio = useRadioGroup(options, value, onChange);
  const noteId = useId();
  // Several options can share one reason: it is said once.
  const reasons = [...new Set(options.filter((o) => o.disabled && o.reason).map((o) => o.reason))];
  const group = (
    <div className={s.segmented} role="radiogroup" aria-label={label} aria-describedby={reasons.length ? noteId : undefined}>
      {options.map((o, i) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={s.segment}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          {...radio(i)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
  if (reasons.length === 0) return group;
  return (
    <div className={s.segmentedWithNote}>
      {group}
      <p id={noteId} className={s.segmentedNote}>
        {reasons.join(" ")}
      </p>
    </div>
  );
}

export function Choices<T extends string>({
  value,
  options,
  onChange,
  label,
  columns = 1,
}: {
  value: T | null;
  options: { value: T; title: ReactNode; detail?: ReactNode; disabled?: boolean }[];
  onChange: (v: T) => void;
  label: string;
  columns?: number;
}) {
  const radio = useRadioGroup(options, value, onChange);
  return (
    <div
      className={s.choices}
      role="radiogroup"
      aria-label={label}
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {options.map((o, i) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={s.choice}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          {...radio(i)}
        >
          <span className={s.radioDot} />
          <span className={s.choiceTitle}>{o.title}</span>
          {o.detail && <span className={s.choiceDetail}>{o.detail}</span>}
        </button>
      ))}
    </div>
  );
}

export function Checkbox({
  checked,
  onChange,
  children,
  disabled,
  top,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  disabled?: boolean;
  /** Beside the first line of a label of several lines, rather than its middle. */
  top?: boolean;
}) {
  return (
    <label className={cx(s.check, top && s.checkTop)} style={disabled ? { opacity: 0.45 } : undefined}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{children}</span>
    </label>
  );
}

// -- display -----------------------------------------------------------------------------------

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className={s.kbd}>{children}</kbd>;
}

/** A bar of progress; null while how far along it is isn't known. */
export function Progress({ value, label }: { value: number | null; label: string }) {
  return (
    <div
      className={s.progress}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value === null ? undefined : Math.round(value * 100)}
    >
      {value === null ? (
        <div className={s.progressIndeterminate} />
      ) : (
        <div className={s.progressBar} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
      )}
    </div>
  );
}

/** Work in progress: an ink ring writing itself, over and over. */
export function Spinner({ label = "Working" }: { label?: string }) {
  return (
    <span className={s.spinner} role="status" aria-label={label}>
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="5.6" pathLength={100} />
      </svg>
    </span>
  );
}

/** An empty state: a quiet logogram, what's missing, and the way forward. */
export function Empty({
  title,
  children,
  action,
  seed,
  align = "center",
}: {
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  /** Seeds the logogram; defaults to the title. */
  seed?: string;
  align?: "center" | "start";
}) {
  return (
    <div className={cx(s.empty, align === "start" && s.emptyStart)}>
      <Logogram seed={seed ?? (typeof title === "string" ? title : "empty")} size={58} className={s.emptyGlyph} haze={false} />
      <div className={s.emptyTitle}>{title}</div>
      {children && <div className={s.emptyText}>{children}</div>}
      {action && <div className={s.emptyAction}>{action}</div>}
    </div>
  );
}

export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "error";
  title?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className={cx(s.callout, tone === "error" && s.calloutError)} role={tone === "error" ? "alert" : "note"}>
      <Icon name={tone === "error" ? "alert" : "info"} />
      <div>
        {title && <div className={s.calloutTitle}>{title}</div>}
        {children && <div className={s.calloutBody}>{children}</div>}
      </div>
    </div>
  );
}

// -- overlays ----------------------------------------------------------------------------------

export function Tip({ text, children, side = "top" }: { text: ReactNode; children: ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <RadixTooltip.Root delayDuration={350}>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content className={s.tooltip} side={side} sideOffset={6} collisionPadding={8}>
          {text}
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}

export const TooltipProvider = RadixTooltip.Provider;

/**
 * For a dialog's onEscapeKeyDown: Escape in a field that uses it itself, marked
 * data-owns-escape (a search it clears, a title being edited that it leaves as it was), stays
 * with the field instead of closing the dialog.
 */
export function escapeStaysInFields(event: globalThis.KeyboardEvent): void {
  const target = event.target;
  if (target instanceof Element && target.closest("[data-owns-escape]")) event.preventDefault();
}

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  wide,
  onEscapeKeyDown,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  /** Called before Escape closes the dialog; preventing its default keeps it open. */
  onEscapeKeyDown?: (event: globalThis.KeyboardEvent) => void;
}) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className={s.overlay} />
        <RadixDialog.Content className={cx(s.dialog, wide && s.dialogWide)} onEscapeKeyDown={onEscapeKeyDown}>
          <div className={s.dialogHeader}>
            <div>
              <RadixDialog.Title className={s.dialogTitle}>{title}</RadixDialog.Title>
              {description ? (
                <RadixDialog.Description className={s.dialogDescription}>{description}</RadixDialog.Description>
              ) : (
                <RadixDialog.Description className="visually-hidden">{String(title)}</RadixDialog.Description>
              )}
            </div>
            <RadixDialog.Close asChild>
              <Button variant="ghost" size="small" icon="close" aria-label="Close" />
            </RadixDialog.Close>
          </div>
          <div className={s.dialogBody}>{children}</div>
          {footer && <div className={s.dialogFooter}>{footer}</div>}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}

/**
 * Ask before an action that can't be undone. The safe choice has the focus, so Enter, Space and
 * Escape all keep things as they are; the action is the solid ink button.
 */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  children,
  keepLabel,
  confirmLabel,
  onConfirm,
  returnFocus,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  children: ReactNode;
  keepLabel: string;
  confirmLabel: string;
  onConfirm: () => void;
  /** Where the keyboard goes when the question closes, for one asked from a menu or a list. */
  returnFocus?: () => void;
}) {
  const keep = useRef<HTMLButtonElement>(null);
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className={cx(s.overlay, s.confirmOverlay)} />
        <RadixDialog.Content
          role="alertdialog"
          className={cx(s.dialog, s.confirm)}
          onOpenAutoFocus={(e) => {
            e.preventDefault();
            keep.current?.focus();
          }}
          onCloseAutoFocus={(e) => {
            if (!returnFocus) return;
            e.preventDefault();
            returnFocus();
          }}
        >
          <div className={s.dialogHeader}>
            <div>
              <RadixDialog.Title className={s.dialogTitle}>{title}</RadixDialog.Title>
              <RadixDialog.Description className={s.dialogDescription}>{children}</RadixDialog.Description>
            </div>
          </div>
          <div className={s.dialogFooter}>
            <RadixDialog.Close asChild>
              <Button ref={keep}>{keepLabel}</Button>
            </RadixDialog.Close>
            <Button variant="primary" onClick={onConfirm}>
              {confirmLabel}
            </Button>
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}

export const menuClasses = {
  menu: s.menu,
  item: s.menuItem,
  shortcut: s.menuShortcut,
  label: s.menuLabel,
  separator: s.menuSeparator,
};
