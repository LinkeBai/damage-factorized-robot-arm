from torch import nn

from scripts.build_bt_dpwm_z78_compute_failure_ledger import parameter_count


def test_parameter_count_includes_all_module_parameters():
    module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=False))
    assert parameter_count(module) == 24
