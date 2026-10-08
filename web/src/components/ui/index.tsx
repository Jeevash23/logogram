import * as RadixDialog from "@radix-ui/react-dialog";
import * as RadixTooltip from "@radix-ui/react-tooltip";
import {
  createContext,
  forwardRef,
  useContext,
  useId,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { Icon, type IconName } from "./Icon";
import s from "./ui.module.css";

export { Icon, Mark } from "./Icon";

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
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "medium", icon, iconAfter, className, children, type, ...rest },
  ref,
) {
  return (
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
      {...rest}
    >
      {icon && <Icon name={icon} size={size === "small" ? 14 : 16} />}
      {children}
      {iconAfter && <Icon name={iconAfter} size={size === "small" ? 14 : 16} />}
    </button>
  );
});

export function IconButton({
  icon,
  label,
  size = "medium",
  ...rest
}: Omit<ButtonProps, "children" | "icon"> & { icon: IconName; label: string }) {
  return (
    <Tip text={label}>
      <Button variant="ghost" size={size} icon={icon} aria-label={label} {...rest} />
    </Tip>
  );
}

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

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className={s.segmented} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={s.segment}
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
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
  return (
    <div
      className={s.choices}
      role="radiogroup"
      aria-label={label}
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={s.choice}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
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
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={s.check} style={disabled ? { opacity: 0.45 } : undefined}>
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

export function Chip({ k, children, title }: { k?: ReactNode; children: ReactNode; title?: string }) {
  return (
    <span className={s.chip} title={title}>
      {k && <span className={s.chipKey}>{k}</span>}
      <span>{children}</span>
    </span>
  );
}

export function Progress({ value }: { value: number | null }) {
  return (
    <div
      className={s.progress}
      role="progressbar"
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

export function Spinner({ label = "Working" }: { label?: string }) {
  return <span className={s.spinner} role="status" aria-label={label} />;
}

export function Empty({
  title,
  children,
  action,
}: {
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className={s.empty}>
      <div className={s.emptyTitle}>{title}</div>
      {children && <div className={s.emptyText}>{children}</div>}
      {action}
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

export function Section({
  title,
  aside,
  children,
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={s.section}>
      <div className={s.sectionHeader}>
        <h4 className={s.sectionTitle}>{title}</h4>
        {aside}
      </div>
      {children}
    </section>
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

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  wide,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className={s.overlay} />
        <RadixDialog.Content className={cx(s.dialog, wide && s.dialogWide)}>
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

export const menuClasses = {
  menu: s.menu,
  item: s.menuItem,
  shortcut: s.menuShortcut,
  label: s.menuLabel,
  separator: s.menuSeparator,
};
