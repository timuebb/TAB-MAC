from importlib import import_module

__all__ = [
    "VAR_model",
    "LOF",
    "DCdetector",
    "AnomalyTransformer",
    "ModernTCN",
    "DualTF",
    "TFAD",
    "TranAD",
    "MatrixProfile",
    "LeftSTAMPi",
    "KMeans",
    "DWT_MLEAD",
    "SAND",
    "Torsk",
    "EIF",
    "ContraAD",
    "Series2Graph",
    "CATCH",
    "CrossAD",
]

_MODEL_IMPORTS = {
    "LOF": "ts_benchmark.baselines.self_impl.LOF.lof",
    "VAR_model": "ts_benchmark.baselines.self_impl.VAR.VAR",
    "DCdetector": "ts_benchmark.baselines.self_impl.DCdetector.DCdetector",
    "AnomalyTransformer": "ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer",
    "ModernTCN": "ts_benchmark.baselines.self_impl.ModernTCN.ModernTCN",
    "DualTF": "ts_benchmark.baselines.self_impl.DualTF.DualTF",
    "TFAD": "ts_benchmark.baselines.self_impl.TFAD.TFAD",
    "MatrixProfile": "ts_benchmark.baselines.self_impl.MatrixProfile.MatrixProfile",
    "TranAD": "ts_benchmark.baselines.self_impl.TranAD.TranAD",
    "LeftSTAMPi": "ts_benchmark.baselines.self_impl.LeftSTAMPi.LeftSTAMPi",
    "KMeans": "ts_benchmark.baselines.self_impl.KMeans.KMeans",
    "DWT_MLEAD": "ts_benchmark.baselines.self_impl.DWT_MLEAD.DWTMLEAD",
    "SAND": "ts_benchmark.baselines.self_impl.SAND.SAND",
    "Torsk": "ts_benchmark.baselines.self_impl.torsk.torsk",
    "EIF": "ts_benchmark.baselines.self_impl.eif.eif",
    "ContraAD": "ts_benchmark.baselines.self_impl.ContraAD.ContraAD",
    "Series2Graph": "ts_benchmark.baselines.self_impl.Series2Graph.Series2Graph",
    "CATCH": "ts_benchmark.baselines.self_impl.CATCH.CATCH",
    "CrossAD": "ts_benchmark.baselines.self_impl.CrossAD.CrossAD",
}


def __getattr__(name):
    module_name = _MODEL_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
