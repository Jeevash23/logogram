# Notebooks

Short notebooks that use Logogram's [Python API](../docs/reference/python-api.md) on GPT-2 small.
Each runs through the same code as the app and `logogram run`, so its runs land in a project you
can open in the app.

| Notebook | What it does | |
|---|---|---|
| [ioi-head-patching.ipynb](ioi-head-patching.ipynb) | Patches every head of GPT-2 small on the IOI example, reads the statistics behind each effect, prompt by prompt, and checks the result the other way round | [Open in Colab](https://colab.research.google.com/github/Jeevash23/logogram/blob/main/notebooks/ioi-head-patching.ipynb) |
| [attribution-patching.ipynb](attribution-patching.ipynb) | Estimates every head at once with attribution patching, verifies the strongest by patching, and shows where one gradient goes wrong and integrated gradients don't | [Open in Colab](https://colab.research.google.com/github/Jeevash23/logogram/blob/main/notebooks/attribution-patching.ipynb) |
| [sae-features.ipynb](sae-features.ipynb) | Loads a published SAE for GPT-2 small, reads its features on a prompt and what one writes, estimates which matter for the answer, and patches one | [Open in Colab](https://colab.research.google.com/github/Jeevash23/logogram/blob/main/notebooks/sae-features.ipynb) |

## Running them

On your own computer, start JupyterLab with Logogram beside it:

```bash
uvx --from jupyterlab --with logogram jupyter lab notebooks/
```

In a clone of this repository, `uv run --with jupyterlab jupyter lab notebooks/` uses the code of
the clone instead. The first cell, `%pip install logogram`, is for Colab; skip it where Logogram
is installed. The notebooks copy the IOI example into your projects folder
(`Documents/Logogram/ioi-example`), as the app does, and add their runs to it: `logogram open` on
that folder shows them in the app.
GPT-2 small (about 0.5 GB) and the SAE (about 150 MB) download once from Hugging Face. On a CPU
each notebook takes a few minutes.

## Changing them

The notebooks are committed with their outputs cleared, since outputs record the machine they
ran on. Run a notebook end to end on the real weights before committing a change to it, then
clear its outputs (in Jupyter, **Edit → Clear Outputs of All Cells**, or
`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`). `tests/test_notebooks.py`
checks that each one is a valid notebook without outputs, opens with its Colab badge and the
install cell, and calls only the public API.
