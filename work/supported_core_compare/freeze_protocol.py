"""Register one fixed matched comparison before any formal fitting/testing."""
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, OUT, PROTOCOL, sha, write


def main():
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    sources = [ROOT/'sim/assets/arm_push.xml',
               ROOT/'src/robotarm/envs/checked_push_reset.py',
               ROOT/'src/robotarm/envs/supported_push_reset.py',
               ROOT/'src/robotarm/envs/constraint_lock.py',
               ROOT/'scripts/ipwm_scale_train.py',
               ROOT/'scripts/audit_ipwm_targeted_advantage.py',
               ROOT/'config/experiment/icra_primary_d2d4_decision_development_3seed_v1.yaml',
               ROOT/'runs/icra_primary_decision_full_w10_128eval_strict_v2/seed27/model.pt',
               ROOT/'runs/icra_primary_global_matched_w10_128eval_strict_v2/seed27/model.pt',
               ROOT/'runs/g2_bt_dpwm_meta_train_z32/seed27_v1/baseline_model.pt']
    sources += sorted((ROOT/'src/robotarm/models').glob('*.py'))
    training = dict(seeds=[7, 17, 27], methods=['ipwm', 'carrier', 'global'],
        epochs=10, batch_size=512, learning_rate=.0003, gradient_clip=5.,
        horizon_schedule=[10]*5+[25]*5, common_pretraining_seed=27,
        selection='Minimum validation H25 object XY MSE including initialization; no test selection.',
        windows='Every pool trajectory once per epoch; independent seeded contiguous window, no replacement.',
        loss_scales=[.1]*5+[1.]*5+[.03]*2+[.1]*2,
        identity='Three adaptation repetitions from one fixed pretrained source; not independent pretraining.')
    spec = dict(version='supported-core-v1', frozen=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='One corrected-initialization/support matched comparison, all methods/seeds/results retained.',
        display_names={'ipwm': 'LockPusher', 'carrier': 'Carrier', 'global': 'Global'},
        model_revision='supported-contact-reset-development-v1', engine='MuJoCo CPU',
        data_seed=91362026, reset_seed=9132026, steps=50, segments=5,
        segment_steps=10, action_limit=.8, workers=4, shard_size=250,
        locks=[0, 1, 2, 3, 4], profiles=['nominal', 'high_damping', 'weak_motor', 'mixed'],
        counts={'pool': 50000, 'validation': 2000, 'test': 6000},
        data_scope='Balanced within-condition independent resets; test split is not unseen-physics generalization.',
        initial_state='Geometry-checked zero-velocity reset with gravity-loaded block_z support and actual lock angle.',
        control='Original raw motor inputs, gravity, damping, friction; no PD/gravity compensation change.',
        training=training, seeds=training['seeds'], methods=training['methods'],
        evaluation={'horizons': [10, 25, 50], 'start_step': 0,
                    'metrics': ['object_xy_rmse_mm', 'object_velocity_rmse_m_s',
                                'free_q_rmse_rad', 'free_v_rmse_rad_s',
                                'pusher_xy_rmse_mm', 'reference_deviation', 'lock_deviation']},
        planning={'problems_per_cell': 120, 'locks': [0, 1, 2, 3, 4],
                  'distance_bands_m': [[.04, .065], [.065, .09]],
                  'candidate_budget': 128, 'horizon_steps': 50, 'segments': 5,
                  'replans': 5, 'executed_steps_per_replan': 10, 'success_radius_m': .03,
                  'seed_repeat': {'7': 0, '17': 1, '27': 2},
                  'goal_direction': 'Initial pusher-to-object planar direction plus uniform [-pi/4,pi/4].',
                  'goal_rng_seed': 91362026,
                  'reset_index': 'band*120+problem_index; profile=problem_index%4.',
                  'independent_problems': 1200,
                  'metrics': ['initial_distance_m', 'terminal_distance_m', 'progress_m',
                              'relative_terminal_error', 'success_at_30mm'],
                  'termination': 'Execute all five replans, score final state; no favorable early stopping.'},
        hard_gates=['all initial geometric/support/limit audits pass', 'all recorded states finite',
                    'exact count/identity/hash integrity and split separation',
                    'all nine model choices frozen before any model test evaluation'],
        dynamic_audit='Record every trajectory, contact/force chronology, support, z/vz and dynamic gaps; no outcome filtering.',
        limitations=['14-D model input omits block_z/vz, full simulated state is archived.',
                     'Fixed object orientation excludes rotational pushing.',
                     'Legacy pretrained source retained identically; this is transfer adaptation, not retraining the full chain.',
                     'The three fitted variants do not alone isolate selective recurrence.',
                     'Corrected simulation is not a real-hardware matched baseline.'],
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sources},
        test_access='Physics-only dataset audits allowed; model test evaluation only after all nine selections are sealed.',
        stopping='One fixed comparison; no seed replacement, budget extension, or outcome-based rerun.')
    write(PROTOCOL, spec)
    write(OUT/'protocol-frozen.json', {'protocol_sha256': sha(PROTOCOL), 'created_utc': spec['created_utc']})
    print(PROTOCOL)
    print(sha(PROTOCOL))


if __name__ == '__main__':
    main()
