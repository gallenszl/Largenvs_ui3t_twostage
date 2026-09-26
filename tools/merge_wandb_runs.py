#!/usr/bin/env python3
"""Create clean canonical W&B runs from checkpoint-selected run fragments.

The tool is deliberately append-only: it creates a new canonical run, verifies
its scalar history, and only then tags source runs as fragments.  It never
rewinds or deletes a source run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import wandb
import yaml


INTERNAL_KEYS = {"_step", "_runtime", "_timestamp", "_wandb"}
SEMANTIC_STEP_KEYS = ("forward_pass_step", "iter")
FORBIDDEN_RUN_ID_CHARS = frozenset("/\\#?%:")
CANONICAL_ID_FILENAME = ".wandb_canonical_run_id"
CANONICAL_METADATA_FILENAME = ".wandb_canonical_merge.json"
CANONICAL_PENDING_FILENAME = ".wandb_canonical_run_id.pending"
LINEAGE_ARTIFACT_TYPE = "merge-lineage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--api-key-path", default="configs/api_keys.yaml", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--mark-sources", action="store_true")
    return parser.parse_args()


def load_api_key(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        api_key = yaml.safe_load(handle).get("wandb")
    if not api_key:
        raise ValueError(f"No W&B API key in {path}")
    os.environ["WANDB_API_KEY"] = api_key


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"entity", "project", "experiments"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest missing keys: {sorted(missing)}")
    return manifest


def scalar_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return not isinstance(value, float) or math.isfinite(value)
    return False


def integer_step(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    step = int(value)
    return step if float(value) == step else None


def semantic_step(row: dict[str, Any]) -> int | None:
    for key in SEMANTIC_STEP_KEYS:
        step = integer_step(row.get(key))
        if step is not None:
            return step
    # Before the custom-step patch all wandb.log calls passed the training step
    # as SDK step.  Rows without forward_pass_step (notably grad warnings) can
    # therefore safely fall back to _step.  New code always emits the custom key.
    return integer_step(row.get("_step"))


def coalesce_segment(
    rows: list[dict[str, Any]], start_step: int, end_step: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    provenance: dict[tuple[int, str], tuple[float, int]] = {}
    conflicts = 0
    used_rows = 0
    for row_index, row in enumerate(rows):
        step = semantic_step(row)
        if step is None or step < start_step or step > end_step:
            continue
        # Preserve ``iter`` as a scalar while using forward_pass_step as the
        # custom x-axis. Legacy source runs also use it as a semantic fallback.
        payload = selected.setdefault(
            step, {"forward_pass_step": step, "iter": step}
        )
        contributed = False
        for key, value in row.items():
            if key in INTERNAL_KEYS or key in SEMANTIC_STEP_KEYS:
                continue
            if value is None or not scalar_value(value):
                continue
            timestamp = row.get("_timestamp")
            timestamp_order = (
                float(timestamp)
                if isinstance(timestamp, (int, float)) and math.isfinite(timestamp)
                else 0.0
            )
            value_order = (timestamp_order, row_index)
            previous_order = provenance.get((step, key))
            if previous_order is not None and payload[key] != value:
                conflicts += 1
            # A remote run ID can contain several resumed Python sessions.  The
            # final successful session has the latest timestamp and must replace
            # earlier abandoned replay values at the same semantic step.
            if previous_order is None or value_order >= previous_order:
                payload[key] = value
                provenance[(step, key)] = value_order
            contributed = True
        if contributed:
            used_rows += 1
    return selected, {
        "history_rows": len(rows),
        "used_rows": used_rows,
        "selected_steps": len(selected),
        "conflicts_resolved_last_write_wins": conflicts,
    }


def validation_metrics(checkpoint_dir: Path, expected_scenes: int) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for eval_dir in sorted(checkpoint_dir.glob("eval_iter_*")):
        match = re.fullmatch(r"eval_iter_(\d+)", eval_dir.name)
        if match is None:
            continue
        step = int(match.group(1))
        metric_paths = sorted(eval_dir.glob("*/metrics.json"))
        if len(metric_paths) != expected_scenes:
            raise ValueError(
                f"{eval_dir}: expected {expected_scenes} metrics.json files, "
                f"found {len(metric_paths)}"
            )
        values: dict[str, list[float]] = {}
        for metric_path in metric_paths:
            with metric_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)["summary"]
            for key, value in summary.items():
                if key == "scene_name":
                    continue
                if not scalar_value(value):
                    raise ValueError(f"Non-scalar validation value {metric_path}: {key}")
                values.setdefault(key, []).append(float(value))
        if any(len(items) != expected_scenes for items in values.values()):
            raise ValueError(f"Incomplete validation metric in {eval_dir}")
        result[step] = {
            f"val/{key}": sum(items) / len(items) for key, items in values.items()
        }
    return result


def strip_resume_ckpt(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    training = config.get("training")
    if isinstance(training, dict):
        training.pop("resume_ckpt", None)
    return config


def canonical_payload(
    api: wandb.Api,
    project_path: str,
    experiment: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    checkpoint_dir = Path(experiment["checkpoint_dir"])
    expected_final_step = int(experiment["expected_final_step"])
    final_ckpt = checkpoint_dir / f"ckpt_{expected_final_step:016d}.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Final checkpoint is missing: {final_ckpt}")

    merged: dict[int, dict[str, Any]] = {}
    source_config: dict[str, Any] | None = None
    source_config_sha256: str | None = None
    segment_reports = []
    source_ids = []
    previous_end = 0
    for segment in experiment["segments"]:
        run_id = segment["run_id"]
        start_step = int(segment["start_step"])
        end_step = int(segment["end_step"])
        if start_step != previous_end + 1:
            raise ValueError(
                f"{experiment['name']}: non-contiguous manifest at {run_id}: "
                f"expected {previous_end + 1}, got {start_step}"
            )
        run = api.run(f"{project_path}/{run_id}")
        if run.state == "running":
            raise RuntimeError(f"Refusing to merge running source {run_id}")
        rows = list(run.scan_history())
        selected, report = coalesce_segment(rows, start_step, end_step)
        overlap = merged.keys() & selected.keys()
        if overlap:
            raise ValueError(f"Overlapping selected steps: {sorted(overlap)[:5]}")
        merged.update(selected)
        candidate_config = strip_resume_ckpt(run.config)
        candidate_config_sha256 = hashlib.sha256(
            json.dumps(candidate_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if source_config is None:
            source_config = candidate_config
            source_config_sha256 = candidate_config_sha256
        elif candidate_config != source_config:
            raise ValueError(
                f"{experiment['name']}: source config changed between fragments; "
                f"first_sha256={source_config_sha256}, "
                f"{run_id}_sha256={candidate_config_sha256}"
            )
        source_ids.append(run_id)
        segment_reports.append(
            {
                "run_id": run_id,
                "source_state": run.state,
                "start_step": start_step,
                "end_step": end_step,
                **report,
            }
        )
        previous_end = end_step

    if previous_end != expected_final_step:
        raise ValueError(
            f"{experiment['name']}: manifest ends at {previous_end}, "
            f"expected {expected_final_step}"
        )

    local_val = validation_metrics(
        checkpoint_dir, int(experiment.get("expected_validation_scenes", 64))
    )
    expected_validation_steps = {
        int(step) for step in experiment["expected_validation_steps"]
    }
    if set(local_val) != expected_validation_steps:
        raise ValueError(
            f"{experiment['name']}: validation steps differ: "
            f"expected={sorted(expected_validation_steps)}, "
            f"actual={sorted(local_val)}"
        )
    for step, metrics in local_val.items():
        if step > expected_final_step:
            continue
        payload = merged.setdefault(
            step, {"forward_pass_step": step, "iter": step}
        )
        for key in [key for key in payload if key.startswith("val/")]:
            del payload[key]
        payload.update(metrics)

    if not merged or min(merged) != 1 or max(merged) != expected_final_step:
        raise ValueError(
            f"{experiment['name']}: selected range is "
            f"{min(merged) if merged else None}..{max(merged) if merged else None}"
        )
    if source_config is None:
        raise ValueError(f"{experiment['name']}: no source config")

    discarded_reports = []
    for run_id in experiment.get("discarded_run_ids", []):
        run = api.run(f"{project_path}/{run_id}")
        if run.state == "running":
            raise RuntimeError(f"Refusing to mark running discarded branch {run_id}")
        if run.name != experiment["name"]:
            raise ValueError(
                f"Discarded run {run_id} has unexpected display name {run.name!r}"
            )
        discarded_reports.append({"run_id": run_id, "source_state": run.state})

    rows = [merged[step] for step in sorted(merged)]
    digest = hashlib.sha256(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")
    ).hexdigest()
    report = {
        "experiment": experiment["name"],
        "canonical_name": experiment["canonical_name"],
        "source_run_ids": source_ids,
        "discarded_run_ids": [item["run_id"] for item in discarded_reports],
        "discarded_runs": discarded_reports,
        "segments": segment_reports,
        "selected_history_steps": len(rows),
        "semantic_step_min": min(merged),
        "semantic_step_max": max(merged),
        "validation_steps_from_local_64_scene": sorted(local_val),
        "payload_sha256": digest,
        "source_config_sha256": source_config_sha256,
    }
    source_config["wandb_merge"] = {
        "canonical": True,
        "source_run_ids": source_ids,
        "lineage": [
            {
                "run_id": item["run_id"],
                "start_step": item["start_step"],
                "end_step": item["end_step"],
            }
            for item in segment_reports
        ],
        "payload_sha256": digest,
    }
    return merged, report, source_config


def write_report(
    checkpoint_dir: Path,
    report: dict[str, Any],
    payload: dict[int, dict[str, Any]],
) -> tuple[Path, Path]:
    report_dir = checkpoint_dir / "wandb_merge"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "merge_report.json"
    payload_path = report_dir / "canonical_history.jsonl"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with payload_path.open("w", encoding="utf-8") as handle:
        for step in sorted(payload):
            handle.write(json.dumps(payload[step], sort_keys=True) + "\n")
    return report_path, payload_path


def atomic_write_text(path: Path, text: str) -> None:
    """Durably publish a small sidecar with replace semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def canonical_upload_lock(checkpoint_dir: Path):
    """Serialize canonical creation/recovery for one checkpoint directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_dir / f"{CANONICAL_ID_FILENAME}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_claim(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        claim = json.load(handle)
    claim["run_id"] = validate_run_id(claim.get("run_id", ""))
    for key in ("canonical_name", "payload_sha256"):
        if not isinstance(claim.get(key), str) or not claim[key]:
            raise ValueError(f"Invalid canonical claim {path}: missing {key}")
    return claim


def get_or_create_claim(
    checkpoint_dir: Path, canonical_name: str, payload_sha256: str
) -> tuple[str, bool]:
    """Get a stable canonical ID; caller must hold ``canonical_upload_lock``."""
    verified_path = checkpoint_dir / CANONICAL_ID_FILENAME
    metadata_path = checkpoint_dir / CANONICAL_METADATA_FILENAME
    pending_path = checkpoint_dir / CANONICAL_PENDING_FILENAME
    expected = {
        "canonical_name": canonical_name,
        "payload_sha256": payload_sha256,
    }

    verified_id = None
    if verified_path.exists():
        verified_id = validate_run_id(verified_path.read_text(encoding="utf-8"))
        if not metadata_path.exists():
            raise ValueError(f"Verified ID exists without metadata: {verified_path}")
        metadata = read_claim(metadata_path)
        if metadata["run_id"] != verified_id:
            raise ValueError(f"Canonical ID/metadata mismatch in {checkpoint_dir}")
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(
                    f"Verified canonical claim changed {key}: "
                    f"persisted={metadata[key]!r}, current={value!r}"
                )

    if pending_path.exists():
        claim = read_claim(pending_path)
        for key, value in expected.items():
            if claim[key] != value:
                raise ValueError(
                    f"Pending canonical claim changed {key}: "
                    f"persisted={claim[key]!r}, current={value!r}"
                )
        if verified_id is not None and claim["run_id"] != verified_id:
            raise ValueError(f"Pending/verified canonical ID mismatch in {checkpoint_dir}")
        return claim["run_id"], verified_id is not None

    if verified_id is not None:
        return verified_id, True

    claim = {
        "run_id": validate_run_id(wandb.util.generate_id()),
        **expected,
    }
    atomic_write_text(pending_path, json.dumps(claim, indent=2, sort_keys=True) + "\n")
    return claim["run_id"], False


def publish_verified_claim(
    checkpoint_dir: Path,
    canonical_id: str,
    canonical_name: str,
    payload_sha256: str,
) -> None:
    """Atomically publish verification metadata and then the user-facing ID."""
    metadata = {
        "run_id": canonical_id,
        "canonical_name": canonical_name,
        "payload_sha256": payload_sha256,
        "status": "verified",
    }
    atomic_write_text(
        checkpoint_dir / CANONICAL_METADATA_FILENAME,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        checkpoint_dir / CANONICAL_ID_FILENAME,
        f"{canonical_id}\n",
    )


def validate_run_id(run_id: str) -> str:
    run_id = run_id.strip()
    if not run_id or set(run_id) & FORBIDDEN_RUN_ID_CHARS:
        raise ValueError(f"Invalid generated W&B run ID: {run_id!r}")
    return run_id


def define_step_metrics() -> None:
    wandb.define_metric("forward_pass_step")
    for metric_name in (
        "train/*",
        "val/*",
        "grad_norm",
        "grad_norm_details/*",
        "lr",
        "iter_time",
        "epoch",
        "param_update_step",
        "iter",
    ):
        wandb.define_metric(metric_name, step_metric="forward_pass_step")


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
    return expected == actual


def verify_remote(
    api: wandb.Api,
    project_path: str,
    run_id: str,
    expected: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    last_error = None
    # Public API indexing is asynchronous; allow up to roughly five minutes.
    for attempt in range(60):
        try:
            api.flush()
            remote = api.run(f"{project_path}/{run_id}")
            rows = list(remote.scan_history())
            actual, _ = coalesce_segment(rows, min(expected), max(expected))
            if set(actual) != set(expected):
                raise AssertionError(
                    f"step mismatch expected={len(expected)} actual={len(actual)}"
                )
            checked_values = 0
            for step, payload in expected.items():
                for key, value in payload.items():
                    if key not in actual[step] or not values_equal(value, actual[step][key]):
                        raise AssertionError(
                            f"value mismatch step={step} key={key}: "
                            f"expected={value!r}, actual={actual[step].get(key)!r}"
                        )
                    checked_values += 1
            return {
                "remote_history_steps": len(actual),
                "remote_scalar_values_checked": checked_values,
            }
        except Exception as exc:  # remote indexing can lag briefly after finish
            last_error = exc
            if attempt < 59:
                time.sleep(5)
    raise RuntimeError(f"Remote verification failed for {run_id}: {last_error}")


def validate_remote_prefix(
    rows: list[dict[str, Any]], expected: dict[int, dict[str, Any]]
) -> int:
    """Return the completed strict-prefix length or fail on divergent history."""
    actual, _ = coalesce_segment(rows, min(expected), max(expected))
    expected_steps = sorted(expected)
    actual_steps = sorted(actual)
    if actual_steps != expected_steps[: len(actual_steps)]:
        raise RuntimeError(
            "Existing canonical history is not a strict semantic-step prefix: "
            f"expected_prefix={expected_steps[:len(actual_steps)]}, actual={actual_steps}"
        )
    for step, actual_payload in actual.items():
        expected_payload = expected[step]
        if set(actual_payload) != set(expected_payload):
            raise RuntimeError(
                f"Existing canonical row has different keys at step {step}: "
                f"expected={sorted(expected_payload)}, actual={sorted(actual_payload)}"
            )
        for key, value in expected_payload.items():
            if not values_equal(value, actual_payload[key]):
                raise RuntimeError(
                    f"Existing canonical row diverged at step={step}, key={key}: "
                    f"expected={value!r}, actual={actual_payload[key]!r}"
                )
    return len(actual_steps)


def mark_sources(
    api: wandb.Api,
    project_path: str,
    source_ids: list[str],
    discarded_ids: list[str],
    canonical_id: str,
) -> None:
    marker = f"merged-into:{canonical_id}"
    for source_id in source_ids:
        run = api.run(f"{project_path}/{source_id}")
        run.tags = tuple(
            sorted(set(run.tags or ()) | {"fragment", "selected-lineage", marker})
        )
        note = f"Scalar history was canonically merged into run {canonical_id}."
        if note not in (run.notes or ""):
            run.notes = ((run.notes or "").rstrip() + "\n\n" + note).strip()
        run.update()
    for source_id in discarded_ids:
        run = api.run(f"{project_path}/{source_id}")
        run.tags = tuple(
            sorted(set(run.tags or ()) | {"fragment", "discarded-branch", marker})
        )
        note = (
            f"Excluded checkpoint branch; canonical scalar history is run {canonical_id}."
        )
        if note not in (run.notes or ""):
            run.notes = ((run.notes or "").rstrip() + "\n\n" + note).strip()
        run.update()


def upload_experiment(
    api: wandb.Api,
    entity: str,
    project: str,
    experiment: dict[str, Any],
    payload: dict[int, dict[str, Any]],
    report: dict[str, Any],
    config: dict[str, Any],
    report_path: Path,
    payload_path: Path,
    mark_source_runs: bool,
) -> str:
    project_path = f"{entity}/{project}"
    checkpoint_dir = Path(experiment["checkpoint_dir"])
    with canonical_upload_lock(checkpoint_dir):
        canonical_id, already_verified = get_or_create_claim(
            checkpoint_dir,
            experiment["canonical_name"],
            report["payload_sha256"],
        )
        report["canonical_run_id"] = canonical_id
        report["canonical_claim_was_verified"] = already_verified
        report_path, payload_path = write_report(checkpoint_dir, report, payload)

        existing = list(
            api.runs(
                project_path,
                filters={"display_name": experiment["canonical_name"]},
            )
        )
        foreign_ids = [run.id for run in existing if run.id != canonical_id]
        if foreign_ids:
            raise RuntimeError(
                f"Canonical display name belongs to other run(s): {foreign_ids}"
            )
        remote = next((run for run in existing if run.id == canonical_id), None)
        completed_prefix = 0
        if remote is not None:
            completed_prefix = validate_remote_prefix(
                list(remote.scan_history()), payload
            )

        history_incomplete = completed_prefix < len(payload)
        # A previous process can upload the full scalar prefix and then fail
        # while creating the artifact or summary. An unverified claim must
        # therefore reopen the same run even when no history rows are missing.
        needs_upload_session = history_incomplete or not already_verified
        if needs_upload_session:
            if already_verified and history_incomplete:
                raise RuntimeError(
                    f"Verified canonical {canonical_id} no longer has complete history"
                )
            notes = (
                "Canonical scalar history rebuilt from checkpoint-selected W&B fragments. "
                "Source runs are retained; system metrics and terminal logs remain there. "
                f"Sources: {', '.join(report['source_run_ids'])}."
            )
            run = wandb.init(
                entity=entity,
                project=project,
                id=canonical_id,
                resume="allow",
                name=experiment["canonical_name"],
                tags=["canonical", "merged"],
                notes=notes,
                config=config,
                reinit=True,
            )
            define_step_metrics()
            expected_steps = sorted(payload)
            for step in expected_steps[completed_prefix:]:
                wandb.log(payload[step])
            artifact = wandb.Artifact(
                name=f"{experiment['artifact_name']}-{canonical_id}",
                type=LINEAGE_ARTIFACT_TYPE,
            )
            artifact.add_file(str(report_path), name="merge_report.json")
            artifact.add_file(str(payload_path), name="canonical_history.jsonl")
            run.log_artifact(artifact)
            run.summary["merge/payload_sha256"] = report["payload_sha256"]
            run.summary["merge/source_run_count"] = len(report["source_run_ids"])
            run.finish()

        verification = verify_remote(api, project_path, canonical_id, payload)
        report.update(verification)
        write_report(checkpoint_dir, report, payload)
        publish_verified_claim(
            checkpoint_dir,
            canonical_id,
            experiment["canonical_name"],
            report["payload_sha256"],
        )
        if mark_source_runs:
            mark_sources(
                api,
                project_path,
                report["source_run_ids"],
                report["discarded_run_ids"],
                canonical_id,
            )
        return canonical_id


def main() -> None:
    args = parse_args()
    load_api_key(args.api_key_path)
    manifest = read_manifest(args.manifest)
    entity = manifest["entity"]
    project = manifest["project"]
    project_path = f"{entity}/{project}"
    selected_names = set(args.only)
    api = wandb.Api(timeout=90)

    experiments = [
        item
        for item in manifest["experiments"]
        if not selected_names or item["name"] in selected_names
    ]
    if selected_names - {item["name"] for item in experiments}:
        raise ValueError(f"Unknown --only values: {sorted(selected_names)}")

    for experiment in experiments:
        payload, report, config = canonical_payload(api, project_path, experiment)
        report_path, payload_path = write_report(
            Path(experiment["checkpoint_dir"]), report, payload
        )
        print(
            f"{experiment['name']}: {report['semantic_step_min']}.."
            f"{report['semantic_step_max']}, {report['selected_history_steps']} history "
            f"steps, sources={report['source_run_ids']}, sha256={report['payload_sha256']}"
        )
        if args.apply:
            canonical_id = upload_experiment(
                api,
                entity,
                project,
                experiment,
                payload,
                report,
                config,
                report_path,
                payload_path,
                args.mark_sources,
            )
            print(f"  uploaded canonical run: {project_path}/{canonical_id}")
        else:
            print(f"  dry-run report: {report_path}")


if __name__ == "__main__":
    main()
