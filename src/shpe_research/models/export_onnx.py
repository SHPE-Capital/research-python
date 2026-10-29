"""TODO(M3): export models to ONNX for the trading platform.  Level: intermediate.

The platform runs models in-process with onnxruntime-node (strategies may not
call external services). Export must carry everything needed to reproduce a
prediction exactly:

    model.onnx        the fitted pipeline INCLUDING preprocessing (scaler), via
                      skl2onnx for sklearn models or torch.onnx for PyTorch
    manifest.json     the existing ModelManifest (feature order is the contract)
    golden.parquet    ~200 feature rows from the validation split with the Python
                      model's predictions; the platform's parity test (task P2)
                      loads these and must match within `tolerance`

Needs `uv sync --extra ml`.

Done when:
    - tests/todo/test_models_todo.py::test_onnx_export passes: onnxruntime
      predictions on golden rows match model.predict within 1e-5.
    - `shpe export <run_dir>/model` writes the three files to <run_dir>/export/.
"""

from __future__ import annotations

from pathlib import Path

from shpe_research.contracts.model import Model


def export_onnx(model: Model, golden_rows, out_dir: str | Path, tolerance: float = 1e-5) -> Path:
    raise NotImplementedError("TODO(M3): ONNX export with golden parity rows")
