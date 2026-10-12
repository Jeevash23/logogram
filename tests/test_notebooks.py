"""The notebooks in notebooks/: each is a valid notebook with every output cleared (outputs would
carry the machine it ran on), opens with its Open in Colab badge and the install cell, and calls
only the public Python API. They download real models, so they aren't run here: run them on real
weights before changing them (see notebooks/README.md)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import logogram

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
COLAB = "https://colab.research.google.com/github/Jeevash23/logogram/blob/main/notebooks/{}"
# The nbformat 4.5 schema's cell id.
CELL_ID = re.compile(r"^[a-zA-Z0-9-_]{1,64}$")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _python(cell: dict) -> str:
    """A code cell as Python: IPython's magics and shell lines commented out."""
    lines = _source(cell).split("\n")
    return "\n".join(
        f"# {line}" if line.lstrip().startswith(("%", "!")) else line for line in lines
    )


def test_there_are_notebooks():
    assert [p.name for p in NOTEBOOKS] == [
        "attribution-patching.ipynb",
        "ioi-head-patching.ipynb",
        "sae-features.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_each_notebook_is_valid_and_has_no_outputs(path: Path):
    nb = _read(path)
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert set(nb["metadata"]) <= {"kernelspec", "language_info"}, "notebook metadata"
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids) and all(CELL_ID.match(i) for i in ids)
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert cell["metadata"] == {}, f"{cell['id']} keeps metadata from a run"
        assert isinstance(cell["source"], (str, list))
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"{path.name}: clear the outputs of {cell['id']}"
            assert cell["execution_count"] is None, f"{path.name}: {cell['id']} was run"
            ast.parse(_python(cell), filename=f"{path.name}:{cell['id']}")
        else:
            assert "outputs" not in cell and "execution_count" not in cell
    # One trailing newline, two-space-free JSON as Jupyter writes it: a diff shows real changes.
    assert path.read_text(encoding="utf-8").endswith("}\n")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_each_notebook_opens_in_colab_and_installs_logogram(path: Path):
    first, second = _read(path)["cells"][:2]
    assert first["cell_type"] == "markdown"
    badge = _source(first).split("\n")[0]
    assert badge.startswith("[![Open in Colab]("), badge
    assert badge.endswith(f"]({COLAB.format(path.name)})"), badge
    assert second["cell_type"] == "code" and _source(second).strip() == "%pip install logogram"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_each_notebook_calls_only_the_public_api(path: Path):
    public = set(logogram._API)
    code = "\n".join(_python(c) for c in _read(path)["cells"] if c["cell_type"] == "code")
    tree = ast.parse(code)
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    ]
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "logogram" in imported, "import logogram as lg"
    assert not [m for m in modules if m and m.startswith("logogram")], "from logogram import ..."
    assert not [n for n in imported if n.startswith("logogram.")], "import logogram.something"
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "lg"
    }
    assert used, "the notebook uses the API as lg"
    assert used <= public, f"not public: {sorted(used - public)}"
    # Names that start with an underscore are private, on any object.
    private = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_")
    }
    assert not private, f"private names: {sorted(private)}"


def test_the_docs_list_every_notebook():
    readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        assert f"({path.name})" in readme, f"notebooks/README.md doesn't list {path.name}"
        assert COLAB.format(path.name) in readme
