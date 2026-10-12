import { useId } from "react";

import { CHANGELOG_URL, LICENSE, LICENSE_URL, REPOSITORY } from "../lib/help";
import { useStore } from "../store/app";
import { BRAND_SEED, Logogram } from "./Logogram";
import { UpdateDetails } from "./UpdateNotice";
import { Dialog } from "./ui";
import s from "./HelpDialogs.module.css";

/** A link that opens in a new tab, and says so. */
function External({ href, children }: { href: string; children: string }) {
  return (
    <a className={s.link} href={href} target="_blank" rel="noopener noreferrer">
      {children}
      <span className="visually-hidden"> (opens in a new tab)</span>
      <span aria-hidden="true"> ↗</span>
    </a>
  );
}

/** About Logogram: the version, the license, the source, and updates. */
export function AboutDialog() {
  const open = useStore((st) => st.helpDialog === "about");
  const version = useStore((st) => st.version);
  const released = useStore((st) => st.update?.released);
  const updatesId = useId();
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (!value) useStore.setState({ helpDialog: null });
      }}
      title="About Logogram"
      description="A local workbench for causal experiments inside language models."
    >
      <div className={s.about}>
        <Logogram seed={BRAND_SEED} size={52} detail={0.85} weight={1.2} haze={false} className={s.glyph} />
        <dl className={s.rows}>
          <div className={s.row}>
            <dt>Version</dt>
            <dd className="num">
              {version}
              {released ? `, released ${released}` : ""}
            </dd>
          </div>
          <div className={s.row}>
            <dt>License</dt>
            <dd>
              <External href={LICENSE_URL}>{LICENSE}</External>
            </dd>
          </div>
          <div className={s.row}>
            <dt>Source code</dt>
            <dd>
              <External href={REPOSITORY}>{REPOSITORY.replace("https://", "")}</External>
            </dd>
          </div>
          <div className={s.row}>
            <dt>Changes</dt>
            <dd>
              <External href={CHANGELOG_URL}>What changed in each version</External>
            </dd>
          </div>
        </dl>
      </div>
      <section className={s.updates} aria-labelledby={updatesId}>
        <h3 className={s.heading} id={updatesId}>
          Updates
        </h3>
        <UpdateDetails />
      </section>
      <p className={s.note}>
        Everything runs on this computer. Logogram goes online only to download models from Hugging Face when you
        ask, and to ask pypi.org for the newest version number when you check or allow daily checks.
      </p>
    </Dialog>
  );
}
