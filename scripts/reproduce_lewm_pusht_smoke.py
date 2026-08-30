"""Small, explicitly non-paper-scale LeWM PushT checkpoint reproduction.

This validates the released checkpoint, latent CEM planner, and closed-loop
control path on locally collected upstream PushT episodes.  It is a diagnostic
smoke test, not a reproduction of the paper's full 50-episode protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument(
        "--policy", choices=("lewm", "random"), default="lewm"
    )
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--cem-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_model(checkpoint_dir: Path) -> torch.nn.Module:
    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = instantiate(config)
    state = torch.load(
        checkpoint_dir / "weights.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA for CEM rollout.")

    # LanceDB misclassifies Windows backslash paths as table names.
    dataset = swm.data.LanceDataset(args.dataset.as_posix())
    sample_pixels = dataset.load_chunk([0], [0], [1])[0]["pixels"]
    image_shape = tuple(int(x) for x in sample_pixels.shape[-2:])

    processors = {}
    for key in ("action", "proprio", "state"):
        scaler = StandardScaler().fit(dataset.get_col_data(key))
        processors[key] = scaler
        if key != "action":
            processors[f"goal_{key}"] = scaler

    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )

    if args.policy == "lewm":
        model = load_model(args.checkpoint_dir).to("cuda").eval()
        model.requires_grad_(False)
        solver = swm.solver.CEMSolver(
            model=model,
            batch_size=1,
            num_samples=args.num_samples,
            var_scale=1.0,
            n_steps=args.cem_steps,
            topk=max(4, args.num_samples // 10),
            device="cuda",
            seed=args.seed,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
            process=processors,
            transform={"pixels": image_transform, "goal": image_transform},
        )
    else:
        policy = swm.policy.RandomPolicy()

    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {
            "method": "_set_goal_state",
            "args": {"goal_state": {"value": "goal_state"}},
        },
    ]
    successes = []
    for episode_id in range(args.episode_start, args.episode_start + args.episodes):
        world = swm.World(
            "swm/PushT-v1",
            num_envs=1,
            max_episode_steps=2 * args.eval_budget,
            image_shape=image_shape,
        )
        world.set_policy(policy)
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[episode_id],
            start_steps=[0],
            goal_offset=args.goal_offset,
            eval_budget=args.eval_budget,
            callables=callables,
        )
        successes.append(bool(np.asarray(metrics["episode_successes"])[0]))
    result = {
        "scope": "upstream-smoke-not-paper-reproduction",
        "checkpoint": str(args.checkpoint_dir),
        "dataset": str(args.dataset),
        "policy": args.policy,
        "episodes": args.episodes,
        "episode_start": args.episode_start,
        "goal_offset": args.goal_offset,
        "eval_budget": args.eval_budget,
        "num_samples": args.num_samples,
        "cem_steps": args.cem_steps,
        "metrics": {
            "success_rate": float(np.mean(successes) * 100.0),
            "episode_successes": successes,
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
