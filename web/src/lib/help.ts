// Where help lives: the guide (the README, as GitHub shows it) and the license, in the repository
// pyproject.toml names. Opening one of these links is the reader's own action; the app fetches
// nothing from them. tests/help.unit.ts checks them against pyproject.toml and the README.

export const REPOSITORY = "https://github.com/Jeevash23/logogram";
export const LICENSE = "MIT";
export const LICENSE_URL = `${REPOSITORY}/blob/main/LICENSE`;
export const CHANGELOG_URL = `${REPOSITORY}/blob/main/CHANGELOG.md`;
/** The issue form for problems. The report never goes in the address: the person pastes it. */
export const NEW_ISSUE_URL = `${REPOSITORY}/issues/new?template=bug_report.yml`;

/** The README's sections the Help menu opens, by their anchors on GitHub. */
export const GUIDE: { title: string; anchor: string }[] = [
  { title: "Quickstart", anchor: "quickstart" },
  { title: "Running on a remote GPU", anchor: "running-on-a-remote-gpu" },
  { title: "Troubleshooting", anchor: "troubleshooting" },
  { title: "The spec", anchor: "the-spec" },
  { title: "Command line", anchor: "command-line" },
];

export function guideUrl(anchor: string): string {
  return `${REPOSITORY}#${anchor}`;
}

/** The most of an error message a problem report asks the server to read (it keeps less). */
export const ERROR_SENT = 4000;

/** The last error, as the problem report is asked to quote it. */
export function errorForReport(error: string | null): string | null {
  const text = error?.trim();
  return text ? text.slice(0, ERROR_SENT) : null;
}
