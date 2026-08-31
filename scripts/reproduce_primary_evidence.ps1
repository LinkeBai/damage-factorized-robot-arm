param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Repository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repository

$ConfirmationArchive = "data\primary_candidates\confirmation_d3_query_seed91031.npz"
if (-not (Test-Path -LiteralPath $ConfirmationArchive)) {
    Write-Output "Confirmation archive missing; regenerating frozen D3 seed 91031 archive."
    & $Python scripts\generate_primary_sequence_candidates.py `
        --phase confirmation `
        --seed 91031 `
        --episodes-per-lock 40 `
        --replans 5 `
        --candidates 128 `
        --segments 5 `
        --steps-per-segment 10 `
        --output $ConfirmationArchive
}

& $Python scripts\summarize_primary_three_seed.py `
    --run-root runs\icra_primary_decision_full_w10_128eval_strict_v2 `
    --output results\final\primary-strict-development-3seed-summary.json

& $Python scripts\summarize_decision_loss_ablation.py `
    --weight10-root runs\icra_primary_decision_full_w10_128eval_strict_v2 `
    --weight0-root runs\icra_primary_no_decision_w0_128eval_strict_v2 `
    --output results\final\primary-decision-loss-ablation-3seed.json

& $Python scripts\summarize_global_matched_ablation.py `
    --global-root runs\icra_primary_global_matched_w10_128eval_strict_v2 `
    --ipwm-root runs\icra_primary_decision_full_w10_128eval_strict_v2 `
    --output results\final\primary-global-matched-ablation-3seed.json

& $Python scripts\summarize_projection_ablation.py `
    --no-projection-root runs\icra_primary_decision_full_w10_no_projection_128eval_strict_v2 `
    --projected-root runs\icra_primary_decision_full_w10_128eval_strict_v2 `
    --output results\final\primary-projection-ablation-3seed.json

& $Python scripts\summarize_primary_compute.py `
    --ipwm-root runs\icra_primary_decision_full_w10_128eval_strict_v2 `
    --global-root runs\icra_primary_global_matched_w10_128eval_strict_v2 `
    --output results\final\primary-compute-cost.json

& $Python scripts\verify_primary_candidate_protocol.py `
    --phase confirmation `
    --candidate-file $ConfirmationArchive `
    --output results\final\confirmation-d3-query-seed91031-audit.json

& $Python scripts\summarize_global_matched_ablation.py `
    --global-root runs\icra_confirmation_d3_query_global_w10 `
    --ipwm-root runs\icra_confirmation_d3_query_selective_w10 `
    --seeds 7,17,27 `
    --output results\final\confirmation-d3-query-seed91031-summary.json

& $Python scripts\audit_primary_evidence_contract.py
& $Python scripts\audit_primary_environment.py
& $Python scripts\audit_large_advantage_metrics.py
& $Python scripts\audit_goal_completion.py 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Output "Goal completion remains fail-closed until real-robot and final-score evidence exist."
}
& $Python scripts\audit_icra_pdf.py paper\main.pdf `
    --source paper\main.tex `
    --output results\final\icra-pdf-anonymity-audit.json

& $Python -m json.tool results\final\primary-result-provenance-ledger.json | Out-Null
& $Python -m pytest `
    tests\test_analyze_real_robot_grasp.py `
    tests\test_analyze_real_robot_push.py `
    tests\test_primary_evidence_audit.py `
    tests\test_icra_pdf_audit.py `
    tests\test_real_robot_push_schedule.py `
    tests\test_real_robot_preflight.py `
    tests\test_real_robot_level_a_schedule.py `
    tests\test_build_real_robot_feasibility_assets.py `
    tests\test_real_robot_schedule_completion.py `
    tests\test_goal_completion_audit.py `
    tests\test_build_real_robot_paper_assets.py `
    tests\test_primary_environment_audit.py `
    tests\test_large_advantage_metric_audit.py `
    tests\test_decision_focused.py `
    tests\test_primary_candidate_protocol.py `
    tests\test_block_triangular_dpwm.py `
    tests\test_selective_intervention_rollout.py `
    -q

Write-Output "Primary and post-freeze D3-query evidence regenerated; focused tests passed."
