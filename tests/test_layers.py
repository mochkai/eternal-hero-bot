"""Enforce the architecture refactor layer import allow-list (spec §3).

The test parses imports with ``ast`` and checks them against the allow-list for
packages that exist at the current migration step.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "eternal_hero_bot"
"""Package root for the bot."""

STDLIB = set(sys.stdlib_module_names) | {"__future__"}
"""Module names treated as standard library."""

# Spec §3 allow-list. Values are allowed top-level import targets: stdlib is always
# allowed; "numpy" allows numpy/numpy.*; "eternal_hero_bot.<pkg>" allows that package.
# "ANY_INTERNAL" means any eternal_hero_bot package (used for runtime/cli/debug).
# Besides numpy, "device" may use PIL for PNG decoding (spec §2.2: Pillow stays for I/O;
# §5.1: AdbDevice is unchanged); no layer may import a higher/sibling project layer.
LAYER_ALLOW: dict[str, set[str]] = {
    "core": {"numpy"},
    "device": {"eternal_hero_bot.core", "numpy", "PIL"},
    "perception": {"eternal_hero_bot.core", "eternal_hero_bot.assets_lib", "numpy"},
    "brain": {"eternal_hero_bot.core"},
    "skills": {"eternal_hero_bot.core", "eternal_hero_bot.action.navigation"},
    "action": {
        "eternal_hero_bot.core",
        "eternal_hero_bot.assets_lib",
        "eternal_hero_bot.device",
        "numpy",
    },
    "recording": {"eternal_hero_bot.core"},
    "runtime": {"ANY_INTERNAL"},
    "cli": {"ANY_INTERNAL"},
    "debug": {"ANY_INTERNAL"},
}
"""Layer boundary allow-list, adapted as packages/modules appear."""


def _iter_python_files() -> list[Path]:
    """Return all Python files under the package root."""
    return sorted(SRC_ROOT.rglob("*.py"))


def _layer_for(path: Path) -> str | None:
    """Determine the spec layer name for a Python file inside a layer package.

    Only files inside an existing layer package directory (for example ``core/``)
    are enforced. Top-level legacy modules are mid-migration and may depend on
    third-party packages until moved in later steps.

    Args:
        path: Python file under ``SRC_ROOT``.

    Returns:
        The layer name, or ``None`` if the file is not inside a known layer package.
    """
    rel = path.relative_to(SRC_ROOT)
    parts = rel.with_suffix("").parts
    if not parts:
        return None
    top = parts[0]
    if top in LAYER_ALLOW and (SRC_ROOT / top).is_dir():
        return top
    return None


def _resolve_import_from(path: Path, node: ast.ImportFrom) -> list[str]:
    """Resolve an ``from ... import ...`` node to absolute module names.

    Args:
        path: Source file containing the import.
        node: AST import-from node.

    Returns:
        Absolute imported module names.
    """
    if node.level == 0:
        base = node.module or ""
        return [base] if base else []
    rel = path.relative_to(SRC_ROOT)
    pkg_parts = list(rel.with_suffix("").parts)
    if path.name == "__init__.py":
        pkg_parts = list(rel.parent.parts)
    # level 1 means current package; walk up level-1.
    up = node.level - 1
    if up > len(pkg_parts):
        return []
    base_parts = pkg_parts[: len(pkg_parts) - up]
    if node.module:
        base_parts.extend(node.module.split("."))
    return [".".join(base_parts)] if base_parts else []


def _imports_for(path: Path) -> list[str]:
    """Collect absolute imported module names from a file.

    Args:
        path: Python source file.

    Returns:
        Imported module names.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.extend(_resolve_import_from(path, node))
    return modules


def _is_allowed(module: str, layer: str) -> bool:
    """Check whether an imported module is allowed for a layer.

    Args:
        module: Absolute imported module name.
        layer: Importing layer name.

    Returns:
        True if allowed.
    """
    top = module.split(".", 1)[0]
    if top in STDLIB:
        return True
    allowed = LAYER_ALLOW[layer]
    if "ANY_INTERNAL" in allowed and top == "eternal_hero_bot":
        return True
    if module == "numpy" or module.startswith("numpy."):
        return "numpy" in allowed
    if top != "eternal_hero_bot":
        # Third-party deps are rejected unless explicitly in the allow-list.
        return top in allowed or module in allowed
    for rule in allowed:
        if rule in {"numpy", "ANY_INTERNAL"}:
            continue
        if module == rule or module.startswith(rule + "."):
            return True
    # Allow imports within the same layer package.
    return module == f"eternal_hero_bot.{layer}" or module.startswith(f"eternal_hero_bot.{layer}.")


def test_layer_packages_present() -> None:
    """Step 2 requires `core/`; step 3 requires `device/`, `perception/`, `action/`."""
    assert (SRC_ROOT / "core").is_dir()
    for name in ("arrays.py", "config.py", "state.py", "actions.py", "geometry.py"):
        assert (SRC_ROOT / "core" / name).is_file(), name
    # Step 3 (spec §3, §11): `AdbDevice` -> device/, vision/tracker -> perception/,
    # pathing -> action/navigation.py. `Device` protocol lives in device/base.py.
    assert (SRC_ROOT / "device").is_dir()
    for name in ("base.py", "adb.py"):
        assert (SRC_ROOT / "device" / name).is_file(), name
    assert (SRC_ROOT / "perception").is_dir()
    for name in ("vision.py", "tracker.py"):
        assert (SRC_ROOT / "perception" / name).is_file(), name
    assert (SRC_ROOT / "action").is_dir()
    assert (SRC_ROOT / "action" / "navigation.py").is_file()


def test_layer_boundaries() -> None:
    """Every module's imports satisfy the spec layer allow-list."""
    violations: list[str] = []
    for path in _iter_python_files():
        layer = _layer_for(path)
        if layer is None or layer not in LAYER_ALLOW:
            continue
        for module in _imports_for(path):
            if not module:
                continue
            if not _is_allowed(module, layer):
                rel = path.relative_to(SRC_ROOT.parent.parent)
                violations.append(f"{rel}: {layer} may not import {module}")
    assert not violations, "Layer boundary violations:\n" + "\n".join(sorted(violations))
