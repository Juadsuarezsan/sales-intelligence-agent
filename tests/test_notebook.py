from pathlib import Path

import nbformat

from scripts import run_notebook


def test_notebook_builds_and_executes_offline(tmp_path: Path) -> None:
    out = tmp_path / "demo.ipynb"
    assert run_notebook.main([str(out)]) == 0
    nb = nbformat.read(str(out), as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 6
    assert all(c.execution_count for c in code_cells)
    first_output = code_cells[0].outputs[0]["text"]
    assert "model: claude-sonnet-4-5-20250929" in first_output
    assert any("mode: fallback" in (c.outputs[0]["text"] if c.outputs else "") for c in code_cells)


def test_notebook_build_only(tmp_path: Path) -> None:
    out = tmp_path / "demo.ipynb"
    assert run_notebook.main([str(out), "--no-exec"]) == 0
    nb = nbformat.read(str(out), as_version=4)
    assert all(not c.outputs for c in nb.cells if c.cell_type == "code")
