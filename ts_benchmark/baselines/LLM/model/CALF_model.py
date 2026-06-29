import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/CALF")

from ts_benchmark.baselines.LLM.submodules.CALF import CALF
from ts_benchmark.utils.device import get_device

class CALFModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        # config.pred_len = config.horizon
        device = get_device()
        self.model = CALF.Model(config, device)
       
    def forward(self, x_enc):        
        output = self.model(x_enc)
        return output['outputs_time']
