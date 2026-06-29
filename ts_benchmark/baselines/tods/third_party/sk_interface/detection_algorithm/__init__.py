"""Lazy exports for TODS sklearn-style detection interfaces."""

from importlib import import_module


_EXPORTS = {
    "ABODSKI": "ABOD_skinterface",
    "AutoEncoderSKI": "AutoEncoder_skinterface",
    "AutoRegODetectorSKI": "AutoRegODetector_skinterface",
    "CBLOFSKI": "CBLOF_skinterface",
    "COFSKI": "COF_skinterface",
    "DeepLogSKI": "DeepLog_skinterface",
    "HBOSSKI": "HBOS_skinterface",
    "IsolationForestSKI": "IsolationForest_skinterface",
    "KDiscordODetectorSKI": "KDiscordODetector_skinterface",
    "KNNSKI": "KNN_skinterface",
    "LODASKI": "LODA_skinterface",
    "LOFSKI": "LOF_skinterface",
    "LSTMODetectorSKI": "LSTMODetector_skinterface",
    "MatrixProfileSKI": "MatrixProfile_skinterface",
    "Mo_GaalSKI": "Mo_Gaal_skinterface",
    "OCSVMSKI": "OCSVM_skinterface",
    "PCAODetectorSKI": "PCAODetector_skinterface",
    "SODSKI": "SOD_skinterface",
    "So_GaalSKI": "So_Gaal_skinterface",
    "SystemWiseDetectionSKI": "SystemWiseDetection_skinterface",
    "TelemanomSKI": "Telemanom_skinterface",
    "VariationalAutoEncoderSKI": "VariationalAutoEncoder_skinterface",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"tods.sk_interface.detection_algorithm.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
