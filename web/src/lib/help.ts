// Where help lives: the documentation in docs/ and the README, as GitHub shows them, and the
// license, in the repository pyproject.toml names. Opening one of these links is the reader's own
// action; the app fetches nothing from them. tests/help.unit.ts checks them against pyproject.toml
// and the files they name.

export const REPOSITORY = "https://github.com/Jeevash23/logogram";
export const LICENSE = "MIT";
export const LICENSE_URL = `${REPOSITORY}/blob/main/LICENSE`;
export const CHANGELOG_URL = `${REPOSITORY}/blob/main/CHANGELOG.md`;
/** The issue form for problems. The report never goes in the address: the person pastes it. */
export const NEW_ISSUE_URL = `${REPOSITORY}/issues/new?template=bug_report.yml`;

/** A page of the repository, as GitHub shows it: a file and, if given, one of its headings. */
export interface DocLink {
  title: string;
  /** The file, from the repository's root, such as docs/getting-started.md. */
  path: string;
  /** A heading's anchor, as GitHub writes it. */
  anchor?: string;
}

/** The documentation the Help menu opens, on GitHub. */
export const GUIDE: DocLink[] = [
  { title: "Getting started", path: "docs/getting-started.md" },
  { title: "Concepts", path: "docs/README.md", anchor: "concepts" },
  { title: "How-to guides", path: "docs/README.md", anchor: "how-to-guides" },
  { title: "Reference", path: "docs/README.md", anchor: "reference" },
  { title: "Running on a remote GPU", path: "README.md", anchor: "running-on-a-remote-gpu" },
  { title: "Troubleshooting", path: "README.md", anchor: "troubleshooting" },
];

/** Where a page of the repository is on GitHub. The README is the repository's own page. */
export function guideUrl(link: Pick<DocLink, "path" | "anchor">): string {
  const hash = link.anchor ? `#${link.anchor}` : "";
  return link.path === "README.md" ? `${REPOSITORY}${hash}` : `${REPOSITORY}/blob/main/${link.path}${hash}`;
}

/** The most of an error message a problem report asks the server to read (it keeps less). */
export const ERROR_SENT = 4000;

/** The last error, as the problem report is asked to quote it. */
export function errorForReport(error: string | null): string | null {
  const text = error?.trim();
  return text ? text.slice(0, ERROR_SENT) : null;
}
