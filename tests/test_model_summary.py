import torch.nn as nn

from noiselora_sv.utils.model_summary import summarize_model_parameters


class TinyCRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(2, 3)
        self.msnrh = nn.Linear(3, 4)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(2, 2)
        self.teacher = nn.Linear(2, 2)
        self.noise_network = TinyCRN()


def test_parameter_summary_does_not_double_count_msnrh():
    summary = summarize_model_parameters(TinyModel())
    assert summary["crn_excluding_ms_nrh"] == 9
    assert summary["ms_nrh"] == 16
