"""Pre-flight check for a multi-dataset pretraining run. Reads only metadata and file listings (no decoding).

For every dataset in the training list that passes the fps filter, it checks that each episode in
meta/episodes.jsonl has its data parquet and one video per camera key, and that episode stats exist.
It also reports robot types, because `make_policy` does not merge normalization stats across robot
types (see policies/factory.py): with more than one, the stats of an arbitrary type are used.

Usage:
    python examples/preflight_check.py --root /data/scratch/nkim4724/datasets \
        --list examples/all_datasets_relative.txt --min-fps 30 --max-fps 30
Exits non-zero if any listed dataset is missing metadata or files.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def check_dataset(root: Path, repo_id: str) -> dict:
    ds = root / repo_id
    info_path = ds / "meta" / "info.json"
    if not info_path.is_file():
        return {"repo_id": repo_id, "error": "missing meta/info.json"}
    info = json.loads(info_path.read_text())
    episodes = [json.loads(line) for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    stats_path = ds / "meta" / "episodes_stats.jsonl"
    n_stats = sum(1 for line in stats_path.read_text().splitlines() if line.strip()) if stats_path.is_file() else 0
    video_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "video"]
    missing = []
    for ep in episodes:
        idx = ep["episode_index"]
        chunk = idx // info["chunks_size"]
        data_file = ds / info["data_path"].format(episode_chunk=chunk, episode_index=idx)
        if not data_file.is_file():
            missing.append(str(data_file.relative_to(ds)))
        for key in video_keys:
            video_file = ds / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=idx)
            if not video_file.is_file():
                missing.append(str(video_file.relative_to(ds)))
    return {
        "repo_id": repo_id,
        "fps": info["fps"],
        "robot_type": info.get("robot_type") or "<empty>",
        "frames": info["total_frames"],
        "episodes": len(episodes),
        "episode_stats": n_stats,
        "cameras": sorted(video_keys),
        "action_shape": tuple(info["features"].get("action", {}).get("shape", ())),
        "state_shape": tuple(info["features"].get("observation.state", {}).get("shape", ())),
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path)
    parser.add_argument("--min-fps", type=int, default=1)
    parser.add_argument("--max-fps", type=int, default=100)
    args = parser.parse_args()

    repo_ids = [r.strip() for r in args.list.read_text().split(",") if r.strip()]
    results = [check_dataset(args.root, r) for r in repo_ids]
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    kept = [r for r in ok if args.min_fps <= r["fps"] <= args.max_fps]

    print(f"{len(repo_ids)} listed, {len(errors)} unreadable, {len(kept)} pass fps {args.min_fps}-{args.max_fps}, "
          f"{sum(r['frames'] for r in kept) / 1e6:.2f}M frames")
    for r in errors:
        print(f"  UNREADABLE {r['repo_id']}: {r['error']}")

    bad = [r for r in kept if r["missing"] or r["episode_stats"] != r["episodes"]]
    print(f"\nFiles: {len(kept) - len(bad)} complete, {len(bad)} with problems")
    for r in bad:
        print(f"  {r['repo_id']}: {len(r['missing'])} missing files, "
              f"{r['episode_stats']}/{r['episodes']} episode stats; first missing: {r['missing'][:2]}")

    by_type = defaultdict(list)
    for r in kept:
        by_type[r["robot_type"]].append(r)
    print(f"\nRobot types ({len(by_type)}):")
    for rt, rs in sorted(by_type.items(), key=lambda kv: -sum(r["frames"] for r in kv[1])):
        shapes = Counter((r["state_shape"], r["action_shape"]) for r in rs)
        print(f"  {rt}: {len(rs)} datasets, {sum(r['frames'] for r in rs) / 1e6:.2f}M frames, "
              f"(state, action) shapes {dict(shapes)}")

    cams = Counter(tuple(r["cameras"]) for r in kept)
    print("\nCamera key sets:", {", ".join(k): v for k, v in cams.most_common()})
    return 1 if errors or bad else 0


if __name__ == "__main__":
    sys.exit(main())
