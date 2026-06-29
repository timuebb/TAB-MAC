"""Lazy exports for vendored TODS detection primitives."""

from importlib import import_module


_EXPORTS = {
    "AutoRegODetectorPrimitive": "AutoRegODetect",
    "DeepLogPrimitive": "DeepLog",
    "EnsemblePrimitive": "Ensemble",
    "KDiscordODetectorPrimitive": "KDiscordODetect",
    "LSTMODetectorPrimitive": "LSTMODetect",
    "MatrixProfilePrimitive": "MatrixProfile",
    "PCAODetectorPrimitive": "PCAODetect",
    "ABODPrimitive": "PyodABOD",
    "AutoEncoderPrimitive": "PyodAE",
    "CBLOFPrimitive": "PyodCBLOF",
    "COFPrimitive": "PyodCOF",
    "HBOSPrimitive": "PyodHBOS",
    "IsolationForestPrimitive": "PyodIsolationForest",
    "KNNPrimitive": "PyodKNN",
    "LODAPrimitive": "PyodLODA",
    "LOFPrimitive": "PyodLOF",
    "Mo_GaalPrimitive": "PyodMoGaal",
    "OCSVMPrimitive": "PyodOCSVM",
    "SODPrimitive": "PyodSOD",
    "So_GaalPrimitive": "PyodSoGaal",
    "VariationalAutoEncoderPrimitive": "PyodVAE",
    "SystemWiseDetectionPrimitive": "SystemWiseDetection",
    "TelemanomPrimitive": "Telemanom",
    "XGBODPrimitive": "PyodXGBOD",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"tods.detection_algorithm.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
