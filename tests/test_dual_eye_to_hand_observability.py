from scripts.evaluate_dual_eye_to_hand_observability import evaluate


def test_dual_eye_to_hand_protocol_covers_both_robots_and_views():
    result = evaluate(height=120, width=160)
    assert len(result["rows"]) == 18
    assert {row["robot"] for row in result["rows"]} == {"genkiarm", "panda"}
    assert all(len(row["cameras"]) == 2 for row in result["rows"])
    assert result["scope"] == "observability_only_not_visual_world_model_accuracy"
