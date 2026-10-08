// A small set of 16px stroke icons drawn for Logogram.

import type { SVGProps } from "react";

const PATHS: Record<string, string> = {
  check: "M3.5 8.5l3 3 6-7",
  close: "M4 4l8 8M12 4l-8 8",
  chevronLeft: "M10 3.5L5.5 8 10 12.5",
  chevronRight: "M6 3.5L10.5 8 6 12.5",
  chevronDown: "M3.5 6L8 10.5 12.5 6",
  chevronUp: "M3.5 10L8 5.5 12.5 10",
  play: "M5 3.5v9l7.5-4.5z",
  stop: "M4.5 4.5h7v7h-7z",
  plus: "M8 3v10M3 8h10",
  minus: "M3 8h10",
  copy: "M5.5 5.5V3.2c0-.4.3-.7.7-.7h6.6c.4 0 .7.3.7.7v6.6c0 .4-.3.7-.7.7h-2.3M3.2 5.5h6.6c.4 0 .7.3.7.7v6.6c0 .4-.3.7-.7.7H3.2c-.4 0-.7-.3-.7-.7V6.2c0-.4.3-.7.7-.7z",
  folder: "M2 4.5c0-.6.4-1 1-1h3l1.5 1.5H13c.6 0 1 .4 1 1v6.5c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1z",
  file: "M4 2.5h5l3 3v8H4zM9 2.5v3h3",
  search: "M7 11.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9zM10.3 10.3L13.5 13.5",
  alert: "M8 2.5l6 10.5H2zM8 6.5v3M8 11.2v.3",
  info: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM8 7.2v4M8 4.9v.3",
  sun: "M8 10.8a2.8 2.8 0 1 0 0-5.6 2.8 2.8 0 0 0 0 5.6zM8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1",
  moon: "M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5z",
  refresh: "M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3",
  flag: "M4 14V2.5M4 3h7.5l-1.5 2.75L11.5 8.5H4",
  arrowRight: "M3 8h10M9 4l4 4-4 4",
  arrowLeft: "M13 8H3M7 4L3 8l4 4",
  grid: "M2.5 2.5h4.5v4.5H2.5zM9 2.5h4.5v4.5H9zM2.5 9h4.5v4.5H2.5zM9 9h4.5v4.5H9z",
  more: "M3.5 8h.01M8 8h.01M12.5 8h.01",
  command: "M6 6V4.5A1.5 1.5 0 1 0 4.5 6H6zm0 0v4m0-4h4m-4 4v1.5A1.5 1.5 0 1 1 4.5 10H6zm0 0h4m0-4V4.5A1.5 1.5 0 1 1 11.5 6H10zm0 0v4m0 0v1.5a1.5 1.5 0 1 0 1.5-1.5H10z",
  download: "M8 2.5v8M4.5 7L8 10.5 11.5 7M3 13.5h10",
  layers: "M8 2.5l5.5 3L8 8.5l-5.5-3zM2.5 8L8 11l5.5-3M2.5 10.5L8 13.5l5.5-3",
  eye: "M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8zM8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
  split: "M8 2.5v11M2.5 3.5h11v9h-11z",
  diff: "M5 2.5v6M2 5.5h6M9 11.5h5",
  cpu: "M4.5 4.5h7v7h-7zM6.5 2v2.5M9.5 2v2.5M6.5 11.5V14M9.5 11.5V14M2 6.5h2.5M2 9.5h2.5M11.5 6.5H14M11.5 9.5H14",
};

export type IconName = keyof typeof PATHS;

interface Props extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 16, ...rest }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

/** The Logogram mark: an ink ring, nearly closed. */
export function Mark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <path
        fill="currentColor"
        d="M16 3.2c7.3 0 12.8 5.6 12.8 12.6 0 7.2-5.7 13-12.9 13-6.8 0-12.1-5-12.6-11.6l2.4-.4c.5 5.3 4.8 9.4 10.2 9.4 5.8 0 10.4-4.6 10.4-10.4 0-5.6-4.4-10.1-10.1-10.1-3.1 0-5.6 1.2-7.4 3.3L6.9 7.4C9.2 4.8 12.4 3.2 16 3.2Z"
      />
      <path fill="currentColor" d="M4.6 12.1c.4-1.3 1-2.5 1.8-3.6l1.3 1.1c-.5.9-.9 1.8-1.2 2.9Z" />
    </svg>
  );
}
