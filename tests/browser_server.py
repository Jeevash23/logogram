"""Offline, temporary workbench for browser checks. Never uses the user's projects or models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def main() -> None:
    import uvicorn

    import logogram.project as projects
    from conftest import _build_tiny_model, make_spec
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
    from logogram.datasets import write_dataset
    from logogram.ioi import generate_ioi
    from logogram.runner import configure_determinism, run_spec

    with tempfile.TemporaryDirectory(prefix="logogram-browser-") as temporary:
        root = Path(temporary)
        projects.config_dir = lambda: root / "config"
        projects.default_projects_parent = lambda: root / "projects"
        from logogram.server.app import create_app
        from logogram.server.security import SecurityConfig

        folder = root / "model"
        folder.mkdir()
        _build_tiny_model(folder)
        configure_determinism()
        backend = TransformerLensBackend.from_bridge(
            boot_local(folder, device="cpu", dtype="float32"),
            model_id="tiny-gpt2",
            revision="test",
            dtype="float32",
            process_weights=True,
        )
        project = projects.Project.create(root / "projects", "Browser fixture")
        write_dataset(project.dataset_file("ioi.jsonl"), generate_ioi(8, seed=0))
        spec = make_spec(
            name="Without BOS",
            tokenization={"prepend_bos": False},
            dataset={"path": "datasets/ioi.jsonl", "limit": 5},
            execution={"batch_size": 2},
        )
        outcome = run_spec(spec, project, backend=backend)
        assert outcome.status == "finished", outcome.manifest.get("error")
        # The fixture credential is scoped to this short-lived loopback server only.
        app = create_app(SecurityConfig(token="browser-test-token", port=8877))
        state = app.state.logogram
        state.backend = backend
        state.model_status = {"state": "ready"}
        state.model_ref = spec.model
        state.update_settings(system_check_seen=True)
        state.open_project(project)
        uvicorn.run(app, host="127.0.0.1", port=8877, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
