"""Write docs/contracts/weekly_config.v1.schema.json from the pydantic model.

    uv run python scripts/export_weekly_config_schema.py          # (re)write
    uv run python scripts/export_weekly_config_schema.py --check  # exit 1 if the checked-in file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from shpe_research.contracts.weekly_config import json_schema

OUT = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "weekly_config.v1.schema.json"


def render() -> str:
    schema = json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "weekly_config.v1.schema.json"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        return 0 if OUT.exists() and OUT.read_text(encoding="utf-8") == text else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
