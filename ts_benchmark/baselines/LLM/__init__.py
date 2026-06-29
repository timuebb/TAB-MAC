# -*- coding: utf-8 -*-
from importlib import import_module

__all__ = [
    "GPT4TSModel",
    "UniTimeModel",
    "CALFModel",
    "LLMMixerModel"
]

_MODEL_IMPORTS = {
    "GPT4TSModel": "ts_benchmark.baselines.LLM.model.GPT4TS_model",
    "UniTimeModel": "ts_benchmark.baselines.LLM.model.UniTime_model",
    "CALFModel": "ts_benchmark.baselines.LLM.model.CALF_model",
    "LLMMixerModel": "ts_benchmark.baselines.LLM.model.LLMMixer_model",
}


def __getattr__(name):
    module_name = _MODEL_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
