#!/usr/bin/env python3
"""Check the Colab notebooks for the failure modes the test suite cannot see.

The notebooks are not part of the installed package, so pytest never imports
them. That let a real break ship: TMol 0.3.0 made ``one2three`` private and
dropped its submodule re-exports, the package's imports were migrated behind
``tmol_mutscan._compat``, and three import lines *inside the notebooks* were
missed. One aborted a benchmark run at the first complex; the others broke the
scan notebook's verification cell.

Checks, cheapest first:

1. Valid JSON and nbformat 4.
2. Every code cell parses.
3. No stored outputs or execution counts, so diffs stay readable.
4. No direct ``tmol.<submodule>`` imports -- those must go through
   ``tmol_mutscan._compat``, which is what absorbs TMol's API moves. This
   single rule is what would have caught the break above, and it needs no
   dependencies.
5. Every Colab badge URL resolves to a file that exists in the repository.
6. With ``--execute-imports``: actually run each notebook's module-level
   imports against the installed TMol. This is the check that proves the
   notebooks work on the release in front of you.

Exit status is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import types
from pathlib import Path

NOTEBOOK_RUNTIME_MODULES = {"IPython", "ipywidgets", "google"}
"""Modules a notebook host provides, not this package. Missing ones mean the
checking environment is incomplete rather than the notebook being wrong."""

BADGE_RE = re.compile(
    r"colab\.research\.google\.com/github/[^/]+/[^/]+/blob/[^/]+/(?P<path>[^\s)\"']+\.ipynb)"
)


class Problems:
    """Collects failures so one run reports everything, not just the first."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, where: str, message: str) -> None:
        self.items.append(f"{where}: {message}")

    def ok(self) -> bool:
        return not self.items


def load_notebook(path: Path, problems: Problems) -> dict | None:
    try:
        notebook = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.add(str(path), f"not readable as JSON ({exc})")
        return None
    if notebook.get("nbformat") != 4:
        problems.add(str(path), f"nbformat is {notebook.get('nbformat')!r}, expected 4")
    if not isinstance(notebook.get("cells"), list):
        problems.add(str(path), "has no 'cells' list")
        return None
    return notebook


def code_cells(notebook: dict):
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            yield index, cell, "".join(cell.get("source", []))


def check_syntax_and_outputs(path: Path, notebook: dict, problems: Problems) -> None:
    for index, cell, source in code_cells(notebook):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            problems.add(f"{path} cell {index}", f"syntax error line {exc.lineno}: {exc.msg}")
        if cell.get("outputs"):
            problems.add(f"{path} cell {index}", "has stored outputs; clear them before committing")
        if cell.get("execution_count") is not None:
            problems.add(f"{path} cell {index}", "has an execution_count; clear outputs before committing")


def check_no_direct_tmol_submodule_imports(path: Path, notebook: dict, problems: Problems) -> None:
    """`import tmol` is fine; `from tmol.pose import ...` is not.

    TMol has moved these names between releases. Routing them through
    tmol_mutscan._compat is what keeps the notebooks working across versions.
    """
    for index, cell, source in code_cells(notebook):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue                       # already reported
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tmol."):
                problems.add(
                    f"{path} cell {index}",
                    f"imports from TMol directly ('from {node.module} import ...'); "
                    "use tmol_mutscan._compat instead",
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tmol."):
                        problems.add(
                            f"{path} cell {index}",
                            f"imports a TMol submodule directly ('import {alias.name}'); "
                            "use tmol_mutscan._compat instead",
                        )


def check_badges(repo_root: Path, problems: Problems) -> None:
    for document in sorted(repo_root.rglob("*.md")):
        if ".git" in document.parts:
            continue
        for match in BADGE_RE.finditer(document.read_text()):
            target = repo_root / match.group("path")
            if not target.is_file():
                problems.add(
                    str(document.relative_to(repo_root)),
                    f"Colab badge points at {match.group('path')}, which does not exist",
                )


def module_level_imports(source: str) -> str:
    """Only top-level imports; conditional ones live inside if/def bodies."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    statements = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return "\n".join(ast.unparse(node) for node in statements)


def execute_imports(path: Path, notebook: dict, problems: Problems) -> None:
    """Run each notebook's module-level imports against the installed TMol."""
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=lambda *a, **k: None, upload=lambda: {})
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules.setdefault("google", google)
    sys.modules["google.colab"] = colab
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass

    for index, _cell, source in code_cells(notebook):
        code = module_level_imports(source)
        if not code:
            continue
        try:
            exec(compile(code, f"<{path.name} cell {index}>", "exec"), {"__name__": "__main__"})
        except ModuleNotFoundError as exc:
            hint = ""
            if (exc.name or "").split(".")[0] in NOTEBOOK_RUNTIME_MODULES:
                hint = (
                    f" -- {exc.name} is part of a notebook runtime rather than this "
                    "package; install it (pip install ipython) to run this check"
                )
            problems.add(
                f"{path} cell {index}",
                f"module-level imports failed: {type(exc).__name__}: {exc}{hint}",
            )
        except Exception as exc:  # noqa: BLE001 - any failure is a finding
            problems.add(
                f"{path} cell {index}",
                f"module-level imports failed: {type(exc).__name__}: {exc}",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path, help="defaults to colab/*.ipynb")
    parser.add_argument(
        "--execute-imports",
        action="store_true",
        help="also run each notebook's module-level imports (needs the package installed)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    paths = args.notebooks or sorted((repo_root / "colab").glob("*.ipynb"))
    if not paths:
        print("no notebooks found", file=sys.stderr)
        return 1

    problems = Problems()
    for path in paths:
        notebook = load_notebook(path, problems)
        if notebook is None:
            continue
        check_syntax_and_outputs(path, notebook, problems)
        check_no_direct_tmol_submodule_imports(path, notebook, problems)
        if args.execute_imports:
            execute_imports(path, notebook, problems)
        print(f"checked {path} ({len(notebook['cells'])} cells)")

    check_badges(repo_root, problems)

    if problems.ok():
        extra = " and executed their imports" if args.execute_imports else ""
        print(f"\nOK: {len(paths)} notebook(s) checked{extra}; Colab badges resolve")
        return 0
    print(f"\n{len(problems.items)} problem(s):", file=sys.stderr)
    for item in problems.items:
        print(f"  - {item}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
