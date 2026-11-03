"""`shpe new`: create a correctly placed, registered and tested component from a template.

A template kind is a folder ``scaffold/templates/<kind>/`` holding a ``template.toml`` and the files it lists.
``template.toml`` has:

    description = "..."
    parts = ["subpackage", "name"]      # how the CLI target "a/b" is split into placeholders
    identifier = ["name"]               # parts that must be lowercase python identifiers
    [allowed]                           # optional: allowed values for a part
    subpackage = ["returns", ...]
    [[files]]
    source = "feature.py.tmpl"          # file in the kind's folder
    path = "src/shpe_research/features/{{subpackage}}/{{name}}.py"   # target, relative to the repo root

Placeholders in paths and file contents: every part, plus ``{{class_name}}`` (CamelCase of ``name``) and, for
studies, ``{{id}}``, ``{{slug}}`` and ``{{title}}``. Shared ``__init__.py`` files are never edited: the command
prints the import line to add. No market data is read and no timestamps are involved.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TEAM_ALIASES = {"statarb": "stat-arb"}


class ScaffoldError(Exception):
    """A scaffold request that cannot be carried out; nothing is written when this is raised."""


def list_kinds() -> list[str]:
    """Template kinds found under ``scaffold/templates/`` (folders holding a ``template.toml``), sorted."""
    return sorted(p.name for p in TEMPLATES.iterdir() if (p / "template.toml").is_file())


def _load(kind: str) -> dict:
    if kind not in list_kinds():
        raise ScaffoldError(f"unknown kind {kind!r}; available kinds: {', '.join(list_kinds())}")
    return tomllib.loads((TEMPLATES / kind / "template.toml").read_text(encoding="utf-8"))


def _fill(text: str, values: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def _values(kind: str, spec: dict, target: str) -> dict[str, str]:
    parts = spec["parts"]
    pieces = target.strip("/").split("/")
    if len(pieces) != len(parts) or not all(pieces):
        raise ScaffoldError(f"{kind} target must look like {'/'.join(f'<{p}>' for p in parts)}, got {target!r}")
    values = dict(zip(parts, pieces, strict=True))
    if "team" in values:
        values["team"] = _TEAM_ALIASES.get(values["team"], values["team"])
    for part, allowed in spec.get("allowed", {}).items():
        if values[part] not in allowed:
            raise ScaffoldError(f"unknown {part} {values[part]!r}; allowed {part}s: {', '.join(allowed)}")
    for part in spec.get("identifier", []):
        if not _IDENTIFIER.match(values[part]):
            raise ScaffoldError(f"{part} {values[part]!r} must be lowercase letters, digits and underscores")
    if "slug" in values:
        if not _SLUG.match(values["slug"]):
            raise ScaffoldError(f"slug {values['slug']!r} must be lowercase letters, digits and hyphens")
        head, _, rest = values["slug"].partition("-")
        values["id"] = head
        values["title"] = (rest or head).replace("-", " ")
    if "name" in values:
        values["class_name"] = "".join(w.capitalize() for w in values["name"].split("_"))
    return values


def scaffold(kind: str, target: str, root: Path | str = ".", dry_run: bool = False) -> list[Path]:
    """Create the files for ``kind`` at ``target`` under ``root`` and return their paths.

    Inputs: a template kind (see ``list_kinds``), a target such as ``returns/demo`` and the repo root. Validates
    everything first: an unknown kind, subpackage or team, a bad name, or any target file that already exists
    raises ``ScaffoldError`` and writes nothing (existing files are never overwritten). With ``dry_run`` the paths
    are returned and nothing is written.
    """
    spec = _load(kind)
    values = _values(kind, spec, target)
    root = Path(root)
    plan = [
        (root / _fill(f["path"], values), _fill((TEMPLATES / kind / f["source"]).read_text(encoding="utf-8"), values))
        for f in spec["files"]
    ]
    clashes = [str(path) for path, _ in plan if path.exists()]
    if clashes:
        raise ScaffoldError("refusing to overwrite existing files: " + ", ".join(clashes))
    if not dry_run:
        for path, content in plan:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    return [path for path, _ in plan]


def _new(args: argparse.Namespace) -> int:
    try:
        paths = scaffold(args.kind, args.target, root=args.root, dry_run=args.dry_run)
    except ScaffoldError as exc:
        print(f"shpe new: {exc}", file=sys.stderr)
        return 2
    verb = "Would write" if args.dry_run else "Wrote"
    for path in paths:
        print(f"{verb} {path}")
    spec = _load(args.kind)
    if "next_steps" in spec and not args.dry_run:
        print(_fill(spec["next_steps"], _values(args.kind, spec, args.target)))
    return 0


def _add_new(sub) -> None:
    """Register ``shpe new <kind> <target> [--dry-run] [--root DIR]`` on the CLI subparsers."""
    p = sub.add_parser("new", help="scaffold a feature, model or study from a template (C1j)")
    p.add_argument("kind", help="template kind: " + ", ".join(list_kinds()))
    p.add_argument("target", help="feature <subpackage>/<name>; model <name>; study <team>/<id>-<slug>")
    p.add_argument("--dry-run", action="store_true", help="print the paths without writing")
    p.add_argument("--root", type=Path, default=Path("."), help="repo root to write into (default: current directory)")
    p.set_defaults(handler=_new)
