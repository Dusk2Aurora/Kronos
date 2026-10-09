"""Bounded operational recovery of the user-interrupted B5 training loop.

prepare-recovery never fits; train-remaining never evaluates supplemental TEST.
The original protocol, training claim, journal prefix and five fits are retained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.frozen.experiment_03_1 import run as original

np, pd, torch, model = original.np, original.pd, original.torch, original.model
RUN = original.RUN
MODELS = RUN / "models"
SHUTDOWN = RUN / "provenance/shutdown_20261009/shutdown_manifest.json"
AMENDMENT = ROOT / "research/frozen/experiment_03_1/OPERATIONAL_RESUME_20261009.md"
SEAL = RUN / "preflight/recovery_seal_20261009.json"
CLAIM = MODELS / "recovery_claim_20261009.json"
CODE_REVIEW = RUN / "preflight/recovery_code_review.json"
BACKUP_MANIFEST = RUN / "provenance/pre_recovery_tool_versions/manifest.json"
JOURNAL = MODELS / "fit_events.jsonl"
INTERRUPTED = "w16_wd0.001_s43"
FIT_KWARGS = dict(max_epochs=120, patience=15, batch_size=64, lr=0.001)
CANDIDATES = [(w, wd, s, f"w{w}_wd{wd}_s{s}")
              for w in (16, 32) for wd in (0.0001, 0.001) for s in (17, 29, 43)]
OUTPUTS = ["selected.json", "all_candidate_validation_predictions.csv",
           "training_resources.json", "candidate_audit.json",
           "interrupted_attempt_resources.json", "recovered_candidate_audit.json"]
PUBLICATION_HELPERS = {f"research/delivery/frozen_risk_03_1_{name}.py"
                       for name in ("review", "finalize", "report")}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def relative(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def verify_hashes(files):
    for name, digest in files.items():
        require(original.sha(ROOT / name) == digest, f"Changed sealed file: {name}")


def shutdown_bindings(manifest):
    """Only publication helper originals may be checked at their sealed archive."""
    backup = original.read(BACKUP_MANIFEST)
    bindings = {}
    archive_root = (RUN / "provenance/pre_recovery_tool_versions").resolve()
    for name, digest in manifest["source_and_artifact_sha256"].items():
        if name in PUBLICATION_HELPERS:
            entry = backup[name]
            archived = (ROOT / entry["archived_path"]).resolve()
            require(archived.parent == archive_root and entry["sha256"] == digest,
                    f"Invalid publication-original archive: {name}")
            bindings[relative(archived)] = digest
        else:
            bindings[name] = digest
    return bindings


def events():
    raw = JOURNAL.read_bytes()
    require(raw.endswith(b"\n"), "Journal lacks complete final newline")
    return raw, [json.loads(line) for line in raw.splitlines()]


def check_original_journal():
    raw, rows = events()
    require(len(rows) == 11, "Recovery requires exactly the archived 11 events")
    for i, (_, _, _, ident) in enumerate(CANDIDATES[:5]):
        require(rows[2*i]["candidate"] == ident and rows[2*i]["state"] == "STARTED",
                f"Unexpected original start: {ident}")
        require(rows[2*i+1]["candidate"] == ident and rows[2*i+1]["state"] == "COMPLETED",
                f"Unexpected original completion: {ident}")
    require(rows[-1]["candidate"] == INTERRUPTED and rows[-1]["state"] == "STARTED",
            "Unexpected original interrupted candidate")
    return raw, rows


def checkpoint_metadata(path, width, wd, seed, audit):
    payload = model.load_b5(path, device="cpu")
    meta = payload["metadata"]
    require(audit["status"] == "PASS" and audit["reload_exact"] is True, "Bad existing audit")
    require(meta == audit["metadata"], f"Checkpoint/audit metadata differ: {path}")
    expected = dict(width=width, weight_decay=wd, seed=seed, **FIT_KWARGS)
    expected.update(receptive_field=257, history_shape=[256, 11], ordinary_dim=33,
                    dtype="float32", selection_loss_dtype="float64", finite_checks_passed=True)
    for key, value in expected.items():
        require(meta[key] == value, f"Unexpected {key}: {path}")
    require(np.isfinite(meta["target_scale"]) and meta["target_scale"] > 0,
            "Invalid checkpoint TRAIN scale")
    require(np.isfinite(meta["best_valid_raw_qlike"]), "Nonfinite saved VALID loss")
    require(all(torch.isfinite(t).all().item() for t in payload["state_dict"].values()),
            "Nonfinite saved weights")
    return payload


def absent_outputs():
    for name in OUTPUTS:
        require(not (MODELS / name).exists(), f"Existing output: {name}")
    for _, _, _, ident in CANDIDATES[5:]:
        for name in (f"{ident}.pt", f"history_{ident}.csv", f"audit_{ident}.json"):
            require(not (MODELS / name).exists(), f"Existing remaining-fit artifact: {name}")
    require(not (RUN / "predictions/evaluation_claim.json").exists(), "TEST evaluation already claimed")
    require(not (RUN / "predictions/test_predictions_only.csv").exists(), "B5 TEST predictions exist")


def check_shutdown():
    c = original.cfg()
    require(c["model"]["width_candidates"] == [16, 32] and
            c["model"]["weight_decay_candidates"] == [0.0001, 0.001] and
            c["model"]["seeds"] == [17, 29, 43], "Scientific candidate grid changed")
    for key, value in dict(max_epochs=120, patience=15, batch_size=64,
                           learning_rate=0.001, budget_fits=12).items():
        require(c["training"][key] == value, f"Scientific training setting changed: {key}")
    original.lock_check()
    manifest = original.read(SHUTDOWN)
    require(manifest["status"] == "PAUSED_FOR_USER_SHUTDOWN" and
            manifest["protocol_sha256"] == original.CONFIG_SHA and
            manifest["completed_candidate_count"] == 5 and
            manifest["interrupted_candidate"] == INTERRUPTED and
            manifest["interrupted_disk_checkpoint_available"] is False and
            manifest["raw_fit_events_count"] == 11, "Unexpected shutdown manifest")
    verify_hashes(shutdown_bindings(manifest))
    require(original.verify_parent() == manifest["parent_delivery_sha256"], "Parent seal changed")
    require(original.read(RUN / "preflight/model_tests.json")["status"] == "PASS", "Model tests missing")
    claim = original.read(MODELS / "training_claim.json")
    require(claim["state"] == "CLAIMED" and claim["budget_fits"] == 12 and
            claim["protocol_sha256"] == original.CONFIG_SHA, "Original claim changed")
    raw, rows = check_original_journal()
    require([e["candidate"] for e in manifest["completed_candidates"]] ==
            [x[3] for x in CANDIDATES[:5]], "Unexpected archived fit order")
    for i, (width, wd, seed, ident) in enumerate(CANDIDATES[:5]):
        entry = manifest["completed_candidates"][i]
        path = MODELS / f"{ident}.pt"
        require(relative(path) == entry["checkpoint"] and original.sha(path) == entry["sha256"],
                f"Archived checkpoint changed: {ident}")
        audit = original.read(MODELS / f"audit_{ident}.json")
        checkpoint_metadata(path, width, wd, seed, audit)
        history = original.loadcsv(MODELS / f"history_{ident}.csv")
        require(history.epoch.tolist() == list(range(1, audit["resources"]["epochs_run"]+1)),
                f"Incomplete history: {ident}")
        require(np.isfinite(history[["train_scaled_objective", "train_raw_qlike", "valid_raw_qlike"]]).all().all(),
                f"Nonfinite existing history: {ident}")
        require(entry["best_epoch"] == audit["metadata"]["best_epoch"] == rows[2*i+1]["best_epoch"] and
                entry["epochs_run"] == audit["resources"]["epochs_run"] == rows[2*i+1]["epochs_run"] and
                entry["elapsed_seconds"] == audit["resources"]["elapsed_seconds"], "Archived resource mismatch")
    absent_outputs()
    return manifest, raw


def prepare_recovery():
    require(not CLAIM.exists() and not SEAL.exists(), "Recovery already prepared or claimed")
    manifest, raw = check_shutdown()
    require(AMENDMENT.is_file() and AMENDMENT.stat().st_size > 0, "Operational amendment required")
    review = original.read(CODE_REVIEW)
    require(review["status"] == "PASS" and review["source_sha256"] == original.sha(Path(__file__).resolve()),
            "Independent recovery code review must PASS for this exact source")
    paths = [SHUTDOWN, AMENDMENT, Path(__file__).resolve(),
             CODE_REVIEW, BACKUP_MANIFEST,
             ROOT / "research/frozen/experiment_03_1/run.py",
             ROOT / "research/frozen/experiment_03_1/b5_model.py",
             RUN / "preflight/pretrain_review_seal.json", RUN / "preflight/source_seal.json",
             MODELS / "training_claim.json", JOURNAL]
    for _, _, _, ident in CANDIDATES[:5]:
        paths.extend(MODELS / name for name in (f"{ident}.pt", f"history_{ident}.csv", f"audit_{ident}.json"))
    hashes = shutdown_bindings(manifest)
    hashes.update({relative(p): original.sha(p) for p in paths})
    original.write(SEAL, dict(status="PASS", at_utc=original.now(), protocol_sha256=original.CONFIG_SHA,
                   files_sha256=hashes, journal_prefix_bytes=len(raw),
                   journal_prefix_sha256=original.sha(JOURNAL), completed_candidates=[x[3] for x in CANDIDATES[:5]],
                   remaining_candidates=[x[3] for x in CANDIDATES[5:]], candidate_fits=12,
                   fit_attempts_including_user_interruption=13, interrupted_attempts=1,
                   scientific_selection_unchanged=True, amendment=relative(AMENDMENT),
                   parent_delivery_sha256=manifest["parent_delivery_sha256"]))
    print("B5_RECOVERY_PREPARED_NO_FITS", original.sha(SEAL), flush=True)


def append_event(value):
    with JOURNAL.open("ab") as f:
        f.write((json.dumps(value, allow_nan=False)+"\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def validation_arrays():
    # Only TRAIN/VALID arrays reach model fitting or inference in this entry point.
    a, labels = original.arrays()
    require(set(a["roles"]) == {"train", "validation", "TEST_Q2", "TEST_Q3"}, "Unknown roles")
    tr, va = a["roles"] == "train", a["roles"] == "validation"
    require((int(tr.sum()), int(va.sum())) == (2369, 92), "Unexpected TRAIN/VALID counts")
    ids = a["ids"][va].copy()
    result = (a["windows"][tr], a["ordinary"][tr], labels.RV_raw.to_numpy()[tr],
              a["windows"][va], a["ordinary"][va], labels.RV_raw.to_numpy()[va], ids)
    del a, labels
    return result


def train_remaining():
    seal = original.read(SEAL)
    require(seal["status"] == "PASS" and seal["protocol_sha256"] == original.CONFIG_SHA,
            "Invalid recovery seal")
    verify_hashes(seal["files_sha256"])
    manifest, prefix = check_shutdown()
    require(len(prefix) == seal["journal_prefix_bytes"] and original.sha(JOURNAL) ==
            seal["journal_prefix_sha256"], "Original journal prefix changed")
    original.write(CLAIM, dict(state="CLAIMED", at_utc=original.now(), recovery_seal_sha256=original.sha(SEAL),
                   protocol_sha256=original.CONFIG_SHA, remaining_fits=7, candidate_fits=12,
                   fit_attempts_including_user_interruption=13, interrupted_attempts=1))
    append_event(dict(candidate=INTERRUPTED, state="INTERRUPTED", at_utc=manifest["at_utc"],
                      recorded_at_utc=original.now(), reason=manifest["user_request"],
                      shutdown_manifest_sha256=original.sha(SHUTDOWN), epochs_run=None,
                      disk_checkpoint_available=False))
    original.write(MODELS / "interrupted_attempt_resources.json", dict(
        candidate=INTERRUPTED, status="USER_SHUTDOWN_PARTIAL_ATTEMPT",
        shutdown_manifest_sha256=original.sha(SHUTDOWN), interrupted_at_utc=manifest["at_utc"],
        pre_pause_partial_fit_wall_seconds_approx=manifest["pre_pause_partial_fit_wall_seconds_approx"],
        partial_fit_epochs_run=None, gpu_compute_seconds=None, pause_idle_seconds=None,
        wall_time_is_not_gpu_compute=True, pause_idle_excluded_from_compute=True,
        included_in_completed_training_resources=False, matched_compute_budget_claim=False))
    xt, ot, yt, xv, ov, yv, valid_ids = validation_arrays()
    scale = float(np.median(np.maximum(yt, model.FLOOR)))
    predictions, metadata, resources, recovered = {}, {}, [], []
    for width, wd, seed, ident in CANDIDATES[:5]:
        audit = original.read(MODELS / f"audit_{ident}.json")
        payload = checkpoint_metadata(MODELS / f"{ident}.pt", width, wd, seed, audit)
        require(payload["metadata"]["target_scale"] == scale, "Saved TRAIN target scale changed")
        model.configure(seed, "cuda")
        payload["model"].to("cuda")
        pv = model.predict_b5(payload, xv, ov)
        repeat = model.predict_b5(payload, xv, ov)
        require(np.array_equal(pv["raw_score"], repeat["raw_score"]), "Recovered VALID repeat differs")
        loss = model.raw_qlike(yv, pv["prediction"])
        require(abs(loss-audit["metadata"]["best_valid_raw_qlike"]) <= 1e-11,
                f"Recovered VALID loss mismatch: {ident}")
        predictions[ident], metadata[ident] = pv["prediction"], payload["metadata"]
        resources.append(dict(audit["resources"], width=width, weight_decay=wd, seed=seed))
        recovered.append(dict(candidate=ident, status="PASS", valid_raw_qlike_recomputed=loss,
                              saved_valid_raw_qlike=audit["metadata"]["best_valid_raw_qlike"],
                              absolute_difference=abs(loss-audit["metadata"]["best_valid_raw_qlike"]),
                              repeated_raw_score_exact=True, original_resources_preserved=True))
        del payload, pv, repeat
        torch.cuda.empty_cache()
    for width, wd, seed, ident in CANDIDATES[5:]:
        append_event(dict(candidate=ident, state="STARTED", at_utc=original.now(),
                          operational_shutdown_retry=ident == INTERRUPTED))
        try:
            payload = model.fit_b5(xt, ot, yt, xv, ov, yv, width, wd, seed, "cuda", **FIT_KWARGS)
            # Exclusive reservation: save_b5's own path write is not exclusive.
            with (MODELS / f"{ident}.pt").open("xb") as f:
                torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in
                                           payload["model"].state_dict().items()}, metadata=payload["metadata"]), f)
            reload = model.load_b5(MODELS / f"{ident}.pt", device="cuda")
            pv = model.predict_b5(payload, xv, ov)
            reloaded = model.predict_b5(reload, xv, ov)
            require(np.array_equal(pv["raw_score"], reloaded["raw_score"]), "Reload scores differ")
            require(abs(model.raw_qlike(yv, pv["prediction"])-payload["metadata"]["best_valid_raw_qlike"])
                    <= 1e-11, "New fit best VALID loss mismatch")
            with (MODELS / f"history_{ident}.csv").open("x", encoding="utf-8", newline="") as f:
                pd.DataFrame(payload["history"]).to_csv(f, index=False, float_format="%.17g")
            original.write(MODELS / f"audit_{ident}.json", dict(status="PASS", metadata=payload["metadata"],
                           resources=payload["resources"], reload_exact=True))
            predictions[ident], metadata[ident] = pv["prediction"], payload["metadata"]
            resources.append(dict(payload["resources"], width=width, weight_decay=wd, seed=seed))
            append_event(dict(candidate=ident, state="COMPLETED", at_utc=original.now(),
                              best_epoch=payload["metadata"]["best_epoch"],
                              epochs_run=payload["resources"]["epochs_run"]))
            print("B5_RECOVERY_CANDIDATE", ident, "VALID", payload["metadata"]["best_valid_raw_qlike"], flush=True)
            del payload, reload, pv, reloaded
            torch.cuda.empty_cache()
        except BaseException as e:
            append_event(dict(candidate=ident, state="FAILED", error=repr(e), at_utc=original.now()))
            raise
    raw, rows = events()
    require(raw.startswith(prefix), "Journal original byte prefix changed")
    require(len(rows) == 26 and Counter(e["state"] for e in rows) ==
            {"STARTED": 13, "COMPLETED": 12, "INTERRUPTED": 1}, "Recovery event count mismatch")
    require([e["candidate"] for e in rows if e["state"] == "COMPLETED"] ==
            [x[3] for x in CANDIDATES], "Completed candidate order mismatch")
    # Recheck all immutable recovery bindings; the journal alone was appended.
    verify_hashes({k: v for k, v in seal["files_sha256"].items() if k != relative(JOURNAL)})
    scores, allpred = [], []
    for width in (16, 32):
        for wd in (0.0001, 0.001):
            ids = [f"w{width}_wd{wd}_s{s}" for s in (17, 29, 43)]
            scores.append(dict(width=width, weight_decay=wd,
                               validation_qlike=model.raw_qlike(yv, np.mean([predictions[i] for i in ids], axis=0)),
                               seed_metadata=[metadata[i] for i in ids]))
            for ident in ids:
                allpred.append(pd.DataFrame(dict(opportunity_id=valid_ids, candidate=ident,
                                                 prediction_RV=predictions[ident])))
    selected = dict(min(scores, key=lambda v: (v["validation_qlike"], v["width"], -v["weight_decay"])))
    selected.update(structure_scores=scores, selection_basis="VALID-only mean RV ensemble")
    original.write(MODELS / "selected.json", selected)
    with (MODELS / "all_candidate_validation_predictions.csv").open("x", encoding="utf-8", newline="") as f:
        pd.concat(allpred, ignore_index=True).to_csv(f, index=False, float_format="%.17g")
    original.write(MODELS / "training_resources.json", resources)
    original.write(MODELS / "recovered_candidate_audit.json", dict(status="PASS", candidates=recovered,
                   validation_N=92, test_prediction_generated=False, recovery_seal_sha256=original.sha(SEAL)))
    original.write(MODELS / "candidate_audit.json", dict(status="PASS", candidate_fits=12,
                   fit_attempts_including_user_interruption=13, interrupted_attempts=1,
                   max_possible_epochs=1440, actual_total_epochs=sum(r["epochs_run"] for r in resources),
                   actual_total_epochs_scope="12 completed fits only; interrupted partial epoch count unknown",
                   interrupted_partial_epochs_unknown=True, validation_only_selection=True,
                   selected_sha256=original.sha(MODELS / "selected.json"),
                   no_fits_after_supplement_test_evaluation=True, original_journal_prefix_preserved=True,
                   journal_event_counts=dict(Counter(e["state"] for e in rows)),
                   recovery_seal_sha256=original.sha(SEAL), interruption_resources=relative(MODELS / "interrupted_attempt_resources.json")))
    print("B5_RECOVERY_TRAINING_COMPLETE", selected["width"], selected["weight_decay"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-recovery", "train-remaining"])
    args = parser.parse_args()
    try:
        {"prepare-recovery": prepare_recovery, "train-remaining": train_remaining}[args.command]()
    except BaseException as e:
        original.write(RUN / "provenance" / f"failure_recovery_{args.command}_{time.time_ns()}.json",
                       dict(state="FAILED", command=args.command, error=repr(e),
                            traceback=traceback.format_exc(), at_utc=original.now()))
        raise


if __name__ == "__main__":
    main()
