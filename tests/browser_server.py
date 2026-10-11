"""Offline, temporary workbench for browser checks. Never uses the user's projects or models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Another port lets two checkouts run the browser checks at once.
PORT = int(os.environ.get("LOGOGRAM_BROWSER_PORT", "8877"))
# Runs with this name wait until they are cancelled, so the checks can queue runs behind one.
HELD_RUN = "Held run"


def hold_runs() -> None:
    """Make runs named HELD_RUN wait, before they start, until they are cancelled (or for five
    minutes at most, so a check that forgets to cancel one can't stall the server)."""
    from logogram import runner

    run_spec = runner.run_spec

    def held(spec: Any, project: Any, **kwargs: Any) -> Any:
        cancel = kwargs.get("cancel")
        if spec.name == HELD_RUN and cancel is not None:
            cancel.wait(300)
        return run_spec(spec, project, **kwargs)

    runner.run_spec = held


def main() -> None:
    import uvicorn

    import logogram.project as projects
    from conftest import _build_tiny_model, build_tiny_architecture, make_spec
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
    from logogram.datasets import write_dataset
    from logogram.ioi import generate_ioi
    from logogram.runner import configure_determinism, run_spec

    # Another tiny architecture family from conftest.TINY_ARCHITECTURES, for checking by eye.
    arch = os.environ.get("LOGOGRAM_FIXTURE_ARCH", "gpt2")
    with tempfile.TemporaryDirectory(prefix="logogram-browser-") as temporary:
        root = Path(temporary)
        projects.config_dir = lambda: root / "config"
        projects.default_projects_parent = lambda: root / "projects"
        from logogram.server.app import create_app
        from logogram.server.security import SecurityConfig

        folder = root / "model"
        folder.mkdir()
        if arch == "gpt2":
            _build_tiny_model(folder)
        else:
            build_tiny_architecture(folder, arch)
        configure_determinism()
        backend = TransformerLensBackend.from_bridge(
            boot_local(folder, device="cpu", dtype="float32"),
            model_id=f"tiny-{arch}",
            revision="test",
            dtype="float32",
            process_weights=True,
        )
        project = projects.Project.create(root / "projects", "Browser fixture")
        write_dataset(project.dataset_file("ioi.jsonl"), generate_ioi(8, seed=0))
        spec = make_spec(
            name="Without BOS",
            model={"id": f"tiny-{arch}", "revision": "test", "device": "cpu"},
            tokenization={"prepend_bos": False},
            dataset={"path": "datasets/ioi.jsonl", "limit": 5},
            execution={"batch_size": 2},
        )
        outcome = run_spec(spec, project, backend=backend)
        assert outcome.status == "finished", outcome.manifest.get("error")
        # The fixture credential is scoped to this short-lived loopback server only.
        app = create_app(SecurityConfig(token="browser-test-token", port=PORT))
        state = app.state.logogram
        state.backend = backend
        state.model_status = {"state": "ready"}
        state.model_ref = spec.model
        # The browser's color scheme picks the theme, so the checks run in white and in black.
        state.update_settings(system_check_seen=True, theme="system")
        state.open_project(project)
        if os.environ.get("LOGOGRAM_FIXTURE_SAE"):
            # A tiny SAE that reconstructs layer 1's residual stream exactly, for checking by eye.
            from test_sae import exact_sae

            state.sae = exact_sae(root / "sae", backend.info.d_model)
            state.sae_status = {"state": "ready"}
        hold_runs()
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
