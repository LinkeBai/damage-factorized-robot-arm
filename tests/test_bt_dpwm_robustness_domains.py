import pytest

from scripts.evaluate_bt_dpwm_safe_adapt_z49 import register_robustness_domains


def test_register_robustness_domains_builds_inline_test_domains():
    domains = register_robustness_domains([{
        "name": "unit_robust_payload",
        "topologies": ["D2", "D4"],
        "physics": {"payload_mass_delta_kg": 0.02, "seed": 901},
    }])
    assert [domain.domain_id for domain in domains] == [
        "D2__unit_robust_payload", "D4__unit_robust_payload"]
    assert all(domain.residual.payload_mass_delta_kg == 0.02 for domain in domains)


def test_register_robustness_domains_rejects_duplicate_pairs():
    with pytest.raises(ValueError, match="unique"):
        register_robustness_domains([{
            "name": "unit_robust_duplicate",
            "topologies": ["D3", "D3"],
            "physics": {"control_delay_steps": 2, "seed": 902},
        }])
