"""Matched supported-push control with CPU execution and queued GPU prediction.

All methods receive identical reset/goal and raw-command candidate identities.
Each owns its realized MuJoCo state and replans from that state. Candidate
scoring is terminal object-to-goal distance, with no oracle selection or added
safety filter. This comparison does not isolate the effect of state isolation.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import mujoco
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / 'src')]
from work.supported_core_compare.common import OUT, PROTOCOL, load_protocol, make_reset, read, sha, state14, write
from work.supported_core_compare import common
from work.supported_core_compare.data import METRIC_NAMES, contact_records, frame_diagnostics
from work.supported_core_compare.evaluate import METHODS, SEEDS, completed_result, require_selection_freeze, verify_implementation
from work.supported_core_compare.train import load_selected
from robotarm.envs import checked_push_reset as checked
from robotarm.envs import supported_push_reset as supported

CONTACT_COLUMNS = ('trajectory_index', 'observation_step', 'internal_substep', 'geom1_id', 'geom2_id',
                   'distance_m', 'normal_force_N', 'tangent_force_norm_N')
GOAL_DEFINITION = (
    'Initial world-XY vector from the center of pusher_geom to the block body '
    'center, rotated by uniform [-pi/4, pi/4]. Goal radius is uniform in the '
    'assigned distance band. This replaces the archived global +X convention '
    'prospectively to match the repaired initial pose.'
)


def array_sha(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes()).hexdigest()


def planning_spec(protocol):
    spec = protocol['planning']
    expected = {'problems_per_cell': 120, 'candidate_budget': 128,
                'horizon_steps': 50, 'segments': 5, 'replans': 5,
                'executed_steps_per_replan': 10, 'success_radius_m': .03}
    for key, value in expected.items():
        if spec[key] != value:
            raise RuntimeError(f'Unexpected frozen planning setting: {key}')
    if spec['distance_bands_m'] != [[.04, .065], [.065, .09]]:
        raise RuntimeError('Unexpected goal bands')
    if protocol['steps'] != 50 or protocol['segment_steps'] != 10 or protocol['action_limit'] != .8:
        raise RuntimeError('Unexpected raw-action protocol')
    return spec


def goal_from_geometry(model, data, lock, profile, band, index, protocol):
    spec = planning_spec(protocol)
    pi = protocol['profiles'].index(profile)
    entropy = [int(spec['goal_rng_seed']), 505, lock, pi, band, index]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    radius = float(rng.uniform(*spec['distance_bands_m'][band]))
    offset = float(rng.uniform(-np.pi / 4, np.pi / 4))
    pusher_xy = data.geom_xpos[model.geom('pusher_geom').id, :2].copy()
    object_xy = data.body('block').xpos[:2].copy()
    vector = object_xy - pusher_xy
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise RuntimeError('Undefined initial pusher-to-object goal direction')
    angle = float(np.arctan2(vector[1], vector[0]) + offset)
    direction = np.array([np.cos(angle), np.sin(angle)])
    goal = object_xy + radius * direction
    if not np.isfinite(goal).all():
        raise RuntimeError('Nonfinite goal')
    return goal, {'goal_rng_entropy': entropy, 'initial_pusher_geom_xy_m': pusher_xy.tolist(),
                  'initial_object_xy_m': object_xy.tolist(), 'goal_xy_m': goal.tolist(),
                  'goal_radius_m': radius, 'base_direction_unit_xy': (vector / norm).tolist(),
                  'goal_direction_offset_rad': offset, 'goal_angle_world_rad': angle}


def setup(lock, band, protocol, *, count=120, split='planning'):
    models, datas, goals, records = [], [], [], []
    for index in range(count):
        profile = protocol['profiles'][index % len(protocol['profiles'])]
        # The formal mapping uses 120 even if an independent development smoke is smaller.
        model, data, record = make_reset(lock, profile, split, band * 120 + index)
        mujoco.mj_forward(model, data)
        audit = supported.inspect_reset(model, data, lock)
        if not audit['passed']:
            raise RuntimeError(f'Invalid planning initialization: {record["reset_id"]}: {audit}')
        goal, goal_record = goal_from_geometry(model, data, lock, profile, band, index, protocol)
        record = {**record, **goal_record, 'lock_index': lock, 'profile': profile,
                  'band': band, 'problem_index': index, 'split': split,
                  'initial_state_sha256': array_sha(np.concatenate((data.qpos, data.qvel, [record['lock_angle_rad']])))}
        models.append(model); datas.append(data); goals.append(goal); records.append(record)
    if len({id(model) for model in models}) != len(models):
        raise RuntimeError('Every realized trajectory must own its MuJoCo model/lock parameters')
    return models, datas, np.asarray(goals), records


def candidate_commands(protocol, repeat, lock, band, replan, count, candidates):
    entropy = [int(protocol['data_seed']), 606, repeat, lock, band, replan]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    actions = rng.uniform(-protocol['action_limit'], protocol['action_limit'],
                          (count, candidates, protocol['planning']['segments'], 5)).astype(np.float32)
    actions[:, :, :, lock] = 0.
    return actions, entropy


@torch.no_grad()
def select(model, states, candidates, goals, lock, diagnosed_angles,
           *, device='cuda', batch_size=2048, horizon=50, segment_steps=10):
    n, count, segments, dof = candidates.shape
    if dof != 5 or horizon != segments * segment_steps:
        raise RuntimeError('Candidate shape/horizon mismatch')
    initial = np.repeat(np.asarray(states, dtype=np.float32), count, axis=0)
    sequence = candidates.reshape(n * count, segments, 5)
    fixed_angles = np.repeat(diagnosed_angles, count)
    outputs = []
    for start in range(0, len(initial), batch_size):
        stop = min(start + batch_size, len(initial))
        x = torch.as_tensor(initial[start:stop], dtype=torch.float32, device=device)
        actions = torch.as_tensor(sequence[start:stop], dtype=torch.float32, device=device)
        mask = torch.zeros((len(x), 5), device=device); mask[:, lock] = 1.
        angles = torch.zeros_like(mask)
        angles[:, lock] = torch.as_tensor(fixed_angles[start:stop], dtype=torch.float32, device=device)
        hidden = None
        for step in range(horizon):
            x, hidden = model.step(x, actions[:, step // segment_steps], mask, angles, hidden)
            if not torch.isfinite(x).all():
                raise RuntimeError(f'Nonfinite candidate forecast at step {step + 1}')
        outputs.append(x[:, 10:12].cpu().numpy())
    predicted = np.concatenate(outputs).reshape(n, count, 2)
    costs = np.linalg.norm(predicted.astype(np.float64) - goals[:, None, :], axis=-1)
    return np.argmin(costs, axis=1), costs, predicted


def _run_cell(model, model_hash, seed, method, lock, band, protocol,
              *, device, formal=True, count=120, candidate_count=128, destination=None,
              selection_registry=None):
    spec = planning_spec(protocol)
    repeat = int(spec['seed_repeat'][str(seed)])
    if formal and (count != 120 or candidate_count != 128):
        raise RuntimeError('Formal budgets cannot be changed')
    identity = {'protocol_sha256': sha(PROTOCOL), 'script_sha256': sha(__file__),
                'model_sha256': model_hash, 'seed': seed, 'method': method,
                'repeat': repeat, 'lock_index': lock, 'band': band,
                'scope': 'formal' if formal else 'development_interface_smoke'}
    if formal:
        # The --all worker verifies the large fitting-data identity once before
        # loading any model. Standalone cell calls still perform the full audit.
        frozen = require_selection_freeze() if selection_registry is None else selection_registry
        verify_implementation()
        if frozen != read(OUT / 'training-complete.json'):
            raise RuntimeError('Selection registry changed during the planning worker')
        if frozen['models'][f'{method}/seed{seed}'] != model_hash:
            raise RuntimeError('Planning model does not match the frozen selection')
        identity.update(implementation_freeze_sha256=sha(OUT / 'implementation-frozen.json'),
                        selection_freeze_sha256=sha(OUT / 'training-complete.json'))
    folder = Path(destination) if destination else OUT / 'planning' / method / f'seed{seed}' / f'D{lock + 1}-B{band}'
    if completed_result(folder, identity):
        return read(folder / 'complete.json')
    temporary = folder.with_name(folder.name + f'.incomplete-{os.getpid()}-{uuid.uuid4().hex}')
    temporary.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    current_step = 0
    try:
        ms, ds, goals, resets = setup(lock, band, protocol, count=count,
                                    split='planning' if formal else 'development')
        write(temporary / 'resets.json', resets)
        total_steps = spec['replans'] * spec['executed_steps_per_replan']
        observation_dt = .005
        integration_dt = float(ms[0].opt.timestep)
        internal_steps = int(round(observation_dt / integration_dt))
        if internal_steps < 1 or not np.isclose(internal_steps * integration_dt, observation_dt, rtol=0., atol=1e-12):
            raise RuntimeError('5 ms observation period must contain an integer number of physics substeps')
        if any(not np.isclose(m.opt.timestep, integration_dt, rtol=0., atol=1e-15) for m in ms):
            raise RuntimeError('Every planning world must use the same integration timestep')
        states = np.full((count, total_steps + 1, 14), np.nan, dtype=np.float64)
        full_qpos = np.full((count, total_steps + 1, ms[0].nq), np.nan)
        full_qvel = np.full((count, total_steps + 1, ms[0].nv), np.nan)
        times = np.full((count, total_steps + 1), np.nan)
        diagnostics = np.full((count, total_steps + 1, len(METRIC_NAMES)), np.nan)
        substep_diagnostics = np.full((count, total_steps, internal_steps, len(METRIC_NAMES)), np.nan)
        substep_times = np.full((count, total_steps, internal_steps), np.nan)
        substep_qpos = np.full((count, total_steps, internal_steps, ms[0].nq), np.nan)
        substep_qvel = np.full((count, total_steps, internal_steps, ms[0].nv), np.nan)
        controls = np.full((count, total_steps, 5), np.nan)
        initial = np.stack([state14(m, d) for m, d in zip(ms, ds)])
        angles = np.array([record['lock_angle_rad'] for record in resets])
        contact_events = []
        first_contact = np.full(count, -1, dtype=np.int64)
        first_contact_substep = np.full(count, -1, dtype=np.int64)
        contact_steps = np.zeros(count, dtype=np.int64)
        contact_internal_steps = np.zeros(count, dtype=np.int64)
        max_before_contact = np.zeros(count)
        selected_indices, selected_sequences, all_costs, all_predictions = [], [], [], []
        candidate_hashes, candidate_entropies = [], []

        def capture_frame(index, step, values=None):
            m, d = ms[index], ds[index]
            # Forward updates xpos after integration; contacts used for transition
            # chronology were captured before this call.
            if values is None:
                mujoco.mj_forward(m, d)
                values = frame_diagnostics(m, d, lock, initial[index])
            states[index, step] = state14(m, d)
            full_qpos[index, step], full_qvel[index, step] = d.qpos, d.qvel
            times[index, step] = d.time
            diagnostics[index, step] = [values[name] for name in METRIC_NAMES]
            if not values['finite'] or not np.isfinite(states[index, step]).all():
                raise RuntimeError(f'Nonfinite realized state: problem {index}, step {step}')
            if first_contact[index] < 0:
                max_before_contact[index] = max(max_before_contact[index], values['object_displacement_m'])

        for index in range(count):
            capture_frame(index, 0)
        prediction_seconds = 0.; execution_seconds = 0.
        for replan in range(spec['replans']):
            candidates, entropy = candidate_commands(protocol, repeat, lock, band, replan, count, candidate_count)
            candidate_hashes.append(array_sha(candidates)); candidate_entropies.append(entropy)
            prediction_began = time.perf_counter()
            chosen, costs, predicted = select(model, states[:, current_step], candidates,
                goals, lock, angles, device=device, horizon=spec['horizon_steps'],
                segment_steps=protocol['segment_steps'])
            prediction_seconds += time.perf_counter() - prediction_began
            picked = candidates[np.arange(count), chosen]
            selected_indices.append(chosen); selected_sequences.append(picked)
            all_costs.append(costs); all_predictions.append(predicted)
            execution_began = time.perf_counter()
            for observation in range(spec['executed_steps_per_replan']):
                current_step += 1
                for index, (m, d) in enumerate(zip(ms, ds)):
                    eq = m.equality('fault_lock_' + checked.JOINTS[lock]).id
                    if m.eq_data[eq, 0] != angles[index] or not d.eq_active[eq] or np.count_nonzero(d.eq_active) != 1:
                        raise RuntimeError(f'Wrong per-world lock identity for problem {index}')
                    action = picked[index, 0]
                    if action[lock] != 0. or np.max(abs(action)) > protocol['action_limit'] + 1e-7:
                        raise RuntimeError('Invalid selected raw command')
                    d.ctrl[:] = action
                    controls[index, current_step - 1] = action
                    observed_touch = False
                    callback_indices = []
                    last_values = None
                    start_time = float(d.time)

                    def record_substep(internal_model, internal_data, internal_index):
                        nonlocal observed_touch, last_values
                        if internal_model is not m or internal_data is not d:
                            raise RuntimeError('Physics callback changed the trajectory-owned world')
                        if internal_index != len(callback_indices) or internal_index >= internal_steps:
                            raise RuntimeError('Internal physics callback must use consecutive zero-based indices')
                        callback_indices.append(internal_index)
                        if m.eq_data[eq, 0] != angles[index] or not d.eq_active[eq] or np.count_nonzero(d.eq_active) != 1:
                            raise RuntimeError(f'Wrong lock identity during internal step for problem {index}')
                        # Solver forces must be read before any forward call.
                        touching = False
                        for contact in contact_records(m, d):
                            contact_events.append([index, current_step, internal_index + 1,
                                contact['geom1'], contact['geom2'], contact['distance_m'],
                                contact['normal_force_N'], contact['tangent_force_norm_N']])
                            names = {contact['geom1_name'], contact['geom2_name']}
                            touching |= ('block_geom' in names and bool(names & {'tool_geom', 'pusher_geom'})
                                         and contact['normal_force_N'] > 1e-9)
                        if touching:
                            observed_touch = True
                            contact_internal_steps[index] += 1
                            if first_contact[index] < 0:
                                first_contact[index] = current_step
                                first_contact_substep[index] = internal_index + 1
                        mujoco.mj_forward(m, d)
                        last_values = frame_diagnostics(m, d, lock, initial[index])
                        substep_diagnostics[index, current_step - 1, internal_index] = [last_values[name] for name in METRIC_NAMES]
                        substep_times[index, current_step - 1, internal_index] = d.time
                        substep_qpos[index, current_step - 1, internal_index] = d.qpos
                        substep_qvel[index, current_step - 1, internal_index] = d.qvel
                        if not last_values['finite']:
                            raise RuntimeError(f'Nonfinite internal state: problem {index}, observation {current_step}, substep {internal_index + 1}')
                        if first_contact[index] < 0:
                            max_before_contact[index] = max(max_before_contact[index], last_values['object_displacement_m'])

                    common.step(m, d, on_substep=record_substep)
                    if len(callback_indices) != internal_steps or last_values is None:
                        raise RuntimeError('An internal integration step was not audited')
                    if not np.isclose(d.time - start_time, observation_dt, rtol=0., atol=1e-10):
                        raise RuntimeError('Executed physics time differs from the 5 ms observation protocol')
                    if observed_touch:
                        contact_steps[index] += 1
                    capture_frame(index, current_step, values=last_values)
            execution_seconds += time.perf_counter() - execution_began
            print(f'planning {method} seed{seed} D{lock + 1} B{band} replan {replan + 1}/5', flush=True)
        initial_distance = np.linalg.norm(states[:, 0, 10:12] - goals, axis=1)
        terminal_distance = np.linalg.norm(states[:, -1, 10:12] - goals, axis=1)
        if not np.isfinite(terminal_distance).all() or np.any(initial_distance <= 0):
            raise RuntimeError('Invalid task metric; no outcomes may be dropped')
        if not np.isfinite(substep_diagnostics).all() or not np.isfinite(substep_times).all():
            raise RuntimeError('Missing/nonfinite internal diagnostic records')
        if not np.allclose(times[:, -1] - times[:, 0], total_steps * observation_dt, rtol=0., atol=1e-9):
            raise RuntimeError('The physical episode must remain 50 observations / 0.25 seconds')
        rows = dict(reset_id=np.array([r['reset_id'] for r in resets]),
            initial_state_sha256=np.array([r['initial_state_sha256'] for r in resets]),
            locked_joint=np.full(count, lock, dtype=np.int64),
            profile=np.array([r['profile'] for r in resets]), band=np.full(count, band, dtype=np.int64),
            goal_xy_m=goals, initial_distance_m=initial_distance, terminal_distance_m=terminal_distance,
            progress_m=initial_distance - terminal_distance, relative_terminal_error=terminal_distance / initial_distance,
            success_at_30mm=terminal_distance < spec['success_radius_m'],
            object_net_displacement_m=np.linalg.norm(states[:, -1, 10:12] - states[:, 0, 10:12], axis=1),
            states=states, full_qpos=full_qpos, full_qvel=full_qvel, time_s=times,
            diagnostic_metrics=diagnostics, diagnostic_metric_names=np.array(METRIC_NAMES),
            substep_diagnostic_metrics=substep_diagnostics, substep_time_s=substep_times,
            substep_full_qpos=substep_qpos, substep_full_qvel=substep_qvel,
            contacts=np.asarray(contact_events, dtype=np.float64).reshape(-1, len(CONTACT_COLUMNS)),
            contact_columns=np.array(CONTACT_COLUMNS), controls=controls,
            first_tool_object_contact_step=first_contact, tool_object_contact_step_count=contact_steps,
            first_tool_object_contact_substep=first_contact_substep,
            tool_object_contact_internal_step_count=contact_internal_steps,
            max_object_displacement_before_tool_contact_m=max_before_contact,
            chosen_indices=np.stack(selected_indices, axis=1),
            chosen_segment_commands=np.stack(selected_sequences, axis=1),
            predicted_candidate_terminal_distance_m=np.stack(all_costs, axis=1),
            predicted_candidate_terminal_xy_m=np.stack(all_predictions, axis=1),
            candidate_array_sha256=np.array(candidate_hashes),
            candidate_rng_entropy=np.array(candidate_entropies, dtype=np.int64),
            goal_angle_world_rad=np.array([r['goal_angle_world_rad'] for r in resets]),
            goal_direction_offset_rad=np.array([r['goal_direction_offset_rad'] for r in resets]),
            goal_base_direction_unit_xy=np.array([r['base_direction_unit_xy'] for r in resets]))
        np.savez_compressed(temporary / 'rows.npz', **rows)
        report = {'method': method, 'seed': seed, 'repeat': repeat, 'lock_index': lock, 'band': band,
            'identity': identity, 'independent_problems': count, 'candidate_budget': candidate_count,
            'horizon_steps': spec['horizon_steps'], 'replans': spec['replans'], 'executed_steps': total_steps,
            'observation_timestep_s': observation_dt, 'integration_timestep_s': integration_dt,
            'internal_steps_per_observation': internal_steps,
            'executed_internal_steps': total_steps * internal_steps,
            'goal_definition': GOAL_DEFINITION, 'goal_rng_definition': '[goal_rng_seed,505,lock,profile_id,band,problem_index]',
            'candidate_rng_definition': '[data_seed,606,repeat,lock,band,replan]',
            'candidate_hash_definition': 'SHA256(dtype.str + repr(shape) + contiguous raw bytes)',
            'controller_definition': 'Raw commands; minimum predicted terminal object distance; execute first 10-step segment, then replan from realized state.',
            'all_episodes_retained': True, 'no_favorable_early_stop': True,
            'contact_definition': 'Each internal transition solver contact is captured after mj_step and before mj_forward. observation_step and internal_substep are 1-based; trajectory_index is 0-based. Tool/object classification requires normal force >1e-9 N; all rows retained.',
            'dynamic_extrema_definition': 'Use initial diagnostic_metrics[:,0] plus every substep_diagnostic_metrics entry; observation-only extrema can miss between-frame contact/limit peaks.',
            'diagnostic_metric_names': list(METRIC_NAMES), 'contact_columns': list(CONTACT_COLUMNS),
            'joint_order': [ms[0].joint(j).name for j in range(ms[0].njnt)],
            'geom_names_by_id': [ms[0].geom(j).name for j in range(ms[0].ngeom)],
            'state_projection': '14-D learned input omits block z/vz; full qpos/qvel and support dynamics archived.',
            'scope': identity['scope'], 'seconds': time.perf_counter() - began,
            'prediction_seconds': prediction_seconds, 'execution_and_diagnostics_seconds': execution_seconds,
            'rows_sha256': sha(temporary / 'rows.npz'), 'resets_sha256': sha(temporary / 'resets.json')}
        write(temporary / 'complete.json', report)
        temporary.rename(folder)
        return report
    except BaseException as error:
        # Keep incomplete evidence; a missing complete.json can never count as a run.
        partial = {name: value for name, value in locals().items()
                   if name in ('states', 'full_qpos', 'full_qvel', 'times', 'diagnostics', 'controls',
                               'substep_diagnostics', 'substep_times', 'substep_qpos', 'substep_qvel')}
        if 'contact_events' in locals():
            partial['contacts'] = np.asarray(contact_events, dtype=np.float64).reshape(-1, len(CONTACT_COLUMNS))
        if partial:
            np.savez_compressed(temporary / 'partial.npz', **partial)
        write(temporary / 'failure.json', {'identity': identity, 'step': current_step,
              'error': repr(error), 'traceback': traceback.format_exc(),
              'seconds': time.perf_counter() - began})
        raise


def run_development_smoke(model, *, seed=7, method='ipwm', lock=0, band=0, device='cuda'):
    """Root may call explicitly; 4 development resets, 8 candidates, no test read."""
    if seed not in SEEDS or method not in METHODS or lock not in range(5) or band not in (0, 1):
        raise ValueError('Invalid smoke identity')
    destination = OUT / 'smoke' / 'planning' / f'{time.time_ns()}-{method}-seed{seed}-D{lock + 1}-B{band}'
    return _run_cell(model.eval(), 'development-build-no-selected-model', seed, method, lock, band,
                     load_protocol(), device=device, formal=False, count=4,
                     candidate_count=8, destination=destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--lock', type=int, choices=range(1, 6), help='User-facing lock number 1..5')
    parser.add_argument('--band', type=int, choices=(0, 1))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.all and any(value is None for value in (args.method, args.seed, args.lock, args.band)):
        parser.error('Use --all or --method --seed --lock --band')
    if args.all and any(value is not None for value in (args.method, args.seed, args.lock, args.band)):
        parser.error('--all cannot be combined with cell selectors')
    torch.set_num_threads(2)
    frozen = require_selection_freeze()
    protocol = load_protocol()
    for seed in SEEDS if args.all else (args.seed,):
        for method in METHODS if args.all else (args.method,):
            model, digest = load_selected(method, seed, args.device)
            for lock in range(5) if args.all else (args.lock - 1,):
                for band in (0, 1) if args.all else (args.band,):
                    _run_cell(model, digest, seed, method, lock, band, protocol, device=args.device,
                              selection_registry=frozen)
            del model


if __name__ == '__main__':
    main()
