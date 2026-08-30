from scripts.run_panda_scripted_grasp_baseline import run


def test_scripted_grasp_reports_physical_evidence_without_weld():
    result = run(settle_steps=10, stage_steps=2)
    assert result["uses_weld"] is False
    assert result["uses_handwritten_grasp_flag"] is False
    assert result["finger_contact_steps"] >= 0
    assert result["bilateral_contact_steps"] >= 0
    assert isinstance(result["success"], bool)
