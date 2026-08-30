"""Z29: deploy one development-seed physical reaction across frozen scaffolds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--development-seed", type=int, default=7)
    parser.add_argument("--seeds", default="7,17,27")
    args = parser.parse_args()
    development = torch.load(
        args.source_root / f"seed{args.development_seed}_v1" / "model.pt",
        map_location="cpu",
    )
    reaction = {name: value for name, value in development.items()
                if name.startswith("reaction_adapter.")}
    if not reaction:
        raise ValueError("development checkpoint has no reaction adapter")
    for seed in (int(value) for value in args.seeds.split(",")):
        source = torch.load(args.source_root / f"seed{seed}_v1" / "model.pt",
                            map_location="cpu")
        source.update({name: value.clone() for name, value in reaction.items()})
        output = args.output_root / f"seed{seed}_v1"
        output.mkdir(parents=True, exist_ok=True)
        torch.save(source, output / "model.pt")
        (output / "transfer.json").write_text(json.dumps({
            "seed": seed, "development_seed": args.development_seed,
            "transferred_tensors": sorted(reaction),
        }, indent=2), encoding="utf-8")
        print(f"[Z29] seed={seed} <- physical adapter seed={args.development_seed}")


if __name__ == "__main__":
    main()
