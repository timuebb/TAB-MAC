# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
from pathlib import Path

__all__ = [
    "hbosski",
    "knnski",
    "lodaski",
    "lofski",
    "ocsvmski",
    "pcaodetectorski",
    "isolationforestski",
    "cblofski",
]

_MODEL_CLASSES = {
    "hbosski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.HBOS_skinterface",
        "HBOSSKI",
    ),
    "knnski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.KNN_skinterface",
        "KNNSKI",
    ),
    "lodaski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.LODA_skinterface",
        "LODASKI",
    ),
    "lofski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.LOF_skinterface",
        "LOFSKI",
    ),
    "ocsvmski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.OCSVM_skinterface",
        "OCSVMSKI",
    ),
    "pcaodetectorski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.PCAODetector_skinterface",
        "PCAODetectorSKI",
    ),
    "isolationforestski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.IsolationForest_skinterface",
        "IsolationForestSKI",
    ),
    "cblofski": (
        "ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.CBLOF_skinterface",
        "CBLOFSKI",
    ),
}


def _ensure_third_party_path() -> None:
    third_party_path = Path(__file__).resolve().parent / "third_party"
    path_text = str(third_party_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


class TodsModelAdapter:
    """
    Adapt TODS sklearn-style wrappers to the benchmark detect_* interface.
    """

    def __init__(self, model_name: str, model_class: object, model_args: dict):
        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.model_name = model_name

    def detect_fit(self, series, label) -> object:
        self.model = self.model_class(**self.model_args)
        self.model.fit(series.values)
        return self.model

    def detect_score(self, train):
        prediction_score = self.model.predict_score(train.values).reshape(-1)
        return prediction_score, prediction_score

    def detect_label(self, train):
        prediction_labels = self.model.predict(train.values).reshape(-1)
        return prediction_labels, prediction_labels

    def __repr__(self):
        return self.model_name


def generate_model_factory(model_name: str, model_class: object, required_args: dict) -> object:
    def model_factory(**kwargs) -> object:
        return TodsModelAdapter(model_name, model_class, kwargs)

    return {"model_factory": model_factory, "required_hyper_params": required_args}


def __getattr__(name: str):
    if name not in _MODEL_CLASSES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, class_name = _MODEL_CLASSES[name]
    _ensure_third_party_path()
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    model_info = generate_model_factory(model_class.__name__, model_class, {})
    globals()[name] = model_info
    return model_info
