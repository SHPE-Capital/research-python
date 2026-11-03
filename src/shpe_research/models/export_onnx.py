"""Export a fitted model to ONNX for the trading platform (M3).

The platform runs models in-process with onnxruntime-node (strategies may not
call external services). An export carries everything needed to reproduce a
prediction exactly:

    model.onnx        the fitted pipeline INCLUDING preprocessing (scaler), so
                      the platform feeds raw features. Input is one float32
                      tensor `[n, n_features]` in manifest feature order.
    manifest.json     the `ModelManifest` fields plus `feature_dtypes`,
                      `onnx_input`, `onnx_output`, `tolerance`, `threshold`
                      (null until a study sets one) and `model_id`.
    golden.parquet    the golden feature rows (manifest order) with the Python
                      model's score in column `score`. The platform parity test
                      (P2) loads these and must match within `tolerance`.

Timing: export happens after fit; golden rows are validation-split rows, so no
row used here is known later than the model's decision time.

Supported models: those built on `SklearnModel` (a fitted `pipeline_`),
converted with skl2onnx. Other model families raise NotImplementedError.
Needs `uv sync --extra ml`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.contracts.model import MANIFEST_FILE, Model

MODEL_FILE = "model.onnx"
GOLDEN_FILE = "golden.parquet"
SCORE_COLUMN = "score"
ONNX_INPUT = "features"


def _convert(model: Model, n_features: int) -> bytes:
    """ONNX bytes for the model's fitted sklearn pipeline (scaler included)."""
    pipeline = getattr(model, "pipeline_", None)
    if pipeline is None:
        raise NotImplementedError(
            f"{model.model_type}: ONNX export supports scikit-learn pipeline models only"
        )
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    final = pipeline.steps[-1][1]
    options = {id(final): {"zipmap": False}} if model.task == "classification" else None
    onx = convert_sklearn(
        pipeline,
        initial_types=[(ONNX_INPUT, FloatTensorType([None, n_features]))],
        options=options,
        target_opset=17,
    )
    return onx.SerializeToString()


def _onnx_scores(onnx_bytes: bytes, X: np.ndarray, task: str) -> np.ndarray:
    """Scores from onnxruntime: P(class 1) for classification, the prediction for regression."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: X.astype(np.float32)})
    if task == "classification":
        return np.asarray(outputs[1], dtype=float)[:, 1]
    return np.asarray(outputs[0], dtype=float).reshape(-1)


def export_onnx(model: Model, golden_rows: pd.DataFrame, out_dir: str | Path, tolerance: float = 1e-5) -> Path:
    """Write `model.onnx`, `manifest.json` and `golden.parquet` into `out_dir`; return `out_dir`.

    Inputs: a fitted `model`; `golden_rows`, a frame holding every manifest
    feature column (extra columns are ignored, order is taken from the
    manifest); `tolerance`, the largest allowed absolute score gap.
    Raises ValueError (and writes nothing) if onnxruntime does not reproduce
    `model.predict` within `tolerance`, or if the golden rows contain NaN.
    """
    if model.manifest is None:
        raise RuntimeError(f"{model.name}: export before fit()")
    names = list(model.manifest.feature_names)
    missing = [c for c in names if c not in golden_rows.columns]
    if missing:
        raise ValueError(f"golden rows are missing features {missing}")
    golden = golden_rows[names].reset_index(drop=True)
    if golden.isna().any().any():
        raise ValueError("golden rows contain NaN; drop or impute them before export")

    onnx_bytes = _convert(model, len(names))
    expected = model.predict(golden)
    got = _onnx_scores(onnx_bytes, golden.to_numpy(np.float32), model.task)
    max_diff = float(np.max(np.abs(got - expected))) if len(golden) else 0.0
    if not max_diff <= tolerance:
        raise ValueError(
            f"{model.name}: onnxruntime differs from model.predict by {max_diff:.3g}, above tolerance {tolerance:g}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / MODEL_FILE).write_bytes(onnx_bytes)
    manifest = asdict(model.manifest)
    manifest.update(
        model_id=model.manifest.name,
        feature_dtypes=["float32"] * len(names),
        onnx_input=ONNX_INPUT,
        onnx_output="probabilities[:, 1]" if model.task == "classification" else "variable",
        threshold=None,
        tolerance=tolerance,
        golden_max_abs_diff=max_diff,
    )
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str))
    golden.assign(**{SCORE_COLUMN: expected}).to_parquet(out / GOLDEN_FILE, index=False)
    return out


def _add_export(sub, handler) -> None:
    """Register `shpe export <run_dir>/model`; `handler` (in cli.py) builds the golden rows."""
    p = sub.add_parser("export", help="export a run's model to ONNX with golden rows")
    p.add_argument("model_dir", help="<run_dir>/model")
    p.add_argument("--golden-rows", type=int, default=200)
    p.add_argument("--tolerance", type=float, default=1e-5)
    p.set_defaults(handler=handler)
