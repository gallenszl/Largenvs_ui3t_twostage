"""Pack each Objaverse object directory into a single uncompressed .tar.

Source layout:  <root>/<shard>/<hash>/{000..024}.png, {000..024}_depth.png, transforms.json
Output layout:  <tar_root>/<shard>/<hash>.tar  (member names: ./000.png ... ./transforms.json)

Run via Slurm CPU job; do NOT run on the login node (8 GiB cgroup cap).
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from multiprocessing import Pool


EXPECTED_FILES = (
    [f"{i:03d}.png" for i in range(25)]
    + [f"{i:03d}_depth.png" for i in range(25)]
    + ["transforms.json"]
)
EXPECTED_MEMBERS = frozenset(f"./{f}" for f in EXPECTED_FILES)


def pack_one(args):
    obj_name, src_root, tar_root = args
    src_dir = os.path.join(src_root, obj_name)
    dst_tar = os.path.join(tar_root, obj_name + ".tar")

    if os.path.exists(dst_tar):
        try:
            with tarfile.open(dst_tar, "r:") as t:
                names = set(t.getnames())
            if EXPECTED_MEMBERS.issubset(names):
                return ("skip", obj_name, None)
        except Exception:
            pass

    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(src_dir, f))]
    if missing:
        return ("missing_src", obj_name, missing)

    os.makedirs(os.path.dirname(dst_tar), exist_ok=True)
    tmp_tar = f"{dst_tar}.tmp.{os.getpid()}"
    try:
        with tarfile.open(tmp_tar, "w") as t:
            for fname in EXPECTED_FILES:
                t.add(os.path.join(src_dir, fname), arcname=f"./{fname}")
        os.rename(tmp_tar, dst_tar)
        return ("packed", obj_name, None)
    except Exception as e:
        if os.path.exists(tmp_tar):
            try:
                os.unlink(tmp_tar)
            except OSError:
                pass
        return ("error", obj_name, repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", default="data/objaverse_v1_in_lvis_25v.txt")
    parser.add_argument("--root-path", default="/home/z50057756/FluffyElephant")
    parser.add_argument("--tar-root", default="/home/z50057756/FluffyElephant_tar")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
    )
    parser.add_argument("--limit", type=int, default=0, help="if >0, only process first N entries")
    parser.add_argument("--audit-out", default="", help="JSON file to write per-status counts and the lists of missing/error objects")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    with open(args.list_file, "r") as f:
        objects = [l.strip() for l in f if l.strip()]
    if args.limit > 0:
        objects = objects[: args.limit]

    print(f"[tar_objaverse] list_file={args.list_file} n={len(objects)} workers={args.num_workers}", flush=True)
    print(f"[tar_objaverse] root={args.root_path}", flush=True)
    print(f"[tar_objaverse] tar_root={args.tar_root}", flush=True)
    os.makedirs(args.tar_root, exist_ok=True)

    task_args = [(o, args.root_path, args.tar_root) for o in objects]
    counts = {"packed": 0, "skip": 0, "missing_src": 0, "error": 0}
    missing_list = []
    error_list = []

    t0 = time.time()
    with Pool(args.num_workers) as pool:
        for i, (status, obj, info) in enumerate(pool.imap_unordered(pack_one, task_args, chunksize=8), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "missing_src":
                missing_list.append({"object": obj, "missing": info})
            elif status == "error":
                error_list.append({"object": obj, "error": info})
            if i % args.progress_every == 0 or i == len(task_args):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                print(
                    f"[tar_objaverse] {i}/{len(task_args)}  "
                    f"packed={counts['packed']} skip={counts['skip']} "
                    f"missing={counts['missing_src']} error={counts['error']}  "
                    f"{rate:.1f}/s  elapsed={elapsed:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[tar_objaverse] done in {elapsed:.0f}s: {counts}", flush=True)

    if args.audit_out:
        audit = {
            "list_file": args.list_file,
            "root_path": args.root_path,
            "tar_root": args.tar_root,
            "total": len(task_args),
            "counts": counts,
            "missing": missing_list,
            "errors": error_list,
            "elapsed_sec": elapsed,
        }
        os.makedirs(os.path.dirname(args.audit_out) or ".", exist_ok=True)
        with open(args.audit_out, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"[tar_objaverse] audit written to {args.audit_out}", flush=True)

    if counts["error"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
