"""Reproducible command-line workflow; expensive experts run in separate envs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .io import read_json, read_jsonl, write_json, write_jsonl, load_detectors


def _evidence_args(parser, *, detectors=True):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observations", required=True)
    if detectors:
        parser.add_argument("--detectors", required=True)


def build_parser():
    parser = argparse.ArgumentParser(prog="ontoforensics", description="Typed evidence fusion for partial speech forgery research")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run simulated cached-observation pipeline; not a benchmark")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--with-rdf", action="store_true")
    imp = commands.add_parser("import-partialedit", help="Import official PartialEdit v1.1 + VCTK originals")
    imp.add_argument("--root", required=True)
    imp.add_argument("--csv", required=True)
    imp.add_argument("--bona-fide-root")
    imp.add_argument("--speaker-splits")
    imp.add_argument("--mic", choices=["mic1", "mic2"])
    imp.add_argument("--subsets", nargs="+", choices=["E1", "E2", "E1-Codec", "E2-Codec"])
    imp.add_argument("--exclude-codec", action="store_true")
    imp.add_argument("--non-strict", action="store_true", help="Report skipped failures explicitly; default fails on errors")
    imp.add_argument("--output", required=True)
    split = commands.add_parser("split-dev", help="Reserve calibration speakers from development only")
    split.add_argument("--manifest", required=True)
    split.add_argument("--fraction", type=float, default=.5)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--output", required=True)
    pilot = commands.add_parser("select-pilot", help="Select source groups within fixed splits; preserves all variants")
    pilot.add_argument("--manifest", required=True)
    pilot.add_argument("--groups-per-split", type=int, default=100)
    pilot.add_argument("--seed", type=int, default=17)
    pilot.add_argument("--min-duration-sec", type=float, default=0.)
    pilot.add_argument("--max-duration-sec", type=float)
    pilot.add_argument("--require-paired", action="store_true", help="Require both labels within every retained source group")
    pilot.add_argument("--output", required=True)
    prep = commands.add_parser("prepare-audio", help="Create common mono 16 kHz PCM16 copies without trimming")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--audio-dir", required=True)
    prep.add_argument("--output", required=True)
    val = commands.add_parser("validate", help="Check manifest, observations, scope, and provenance")
    _evidence_args(val)
    val.add_argument("--check-audio", action="store_true")
    quality = commands.add_parser("annotate-quality", help="Measure local RMS and clipping from audio")
    _evidence_args(quality)
    quality.add_argument("--output", required=True)
    merge = commands.add_parser("merge-observations", help="Merge expert caches with explicit provenance validation")
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--detectors", required=True)
    merge.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Fit fusion on train; calibrate on calibration; threshold on dev")
    _evidence_args(train)
    train.add_argument("--mode", choices=["mean", "flat_lr", "flat_mlp", "relation_lr"], default="relation_lr")
    train.add_argument("--relations", help="Relations JSON; disabled flags provide ablations")
    train.add_argument("--step-sec", type=float, default=.04)
    train.add_argument("--context-sec", type=float, default=.4)
    train.add_argument("--target-fpr", type=float, default=.05)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--output", required=True)
    for name in ("predict", "evaluate"):
        cmd = commands.add_parser(name)
        _evidence_args(cmd, detectors=False)
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--output", required=True)
        if name == "evaluate":
            cmd.add_argument("--split", default="test", choices=["train", "dev", "calibration", "test"])
            cmd.add_argument("--bootstrap", type=int, default=0)
        else:
            cmd.add_argument("--rejection-confidence", type=float, default=.6)
    graph = commands.add_parser("export-graph", help="Export typed observation/provenance graph, not a probability engine")
    _evidence_args(graph)
    graph.add_argument("--output", required=True)
    shacl = commands.add_parser("validate-rdf", help="Validate the exported RDF with SHACL")
    shacl.add_argument("--graph", required=True)
    return parser


def dispatch(args):
    from .data import load_manifest, validate_manifest, write_manifest
    if args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.output, seed=args.seed, with_rdf=args.with_rdf)
        return {"simulation_only": True, "recordings": result["recordings"],
                "modes_completed": list(result["models"]), "summary": str(Path(args.output)/"demo_summary.json")}
    if args.command == "import-partialedit":
        from .data import import_partialedit
        records = import_partialedit(args.root, args.csv, bona_fide_root=args.bona_fide_root,
                                     speaker_splits=args.speaker_splits, mic=args.mic,
                                     include_codec=not args.exclude_codec, strict=not args.non_strict,
                                     subsets=args.subsets)
        write_manifest(args.output, records)
        report_path = str(Path(args.output).with_suffix(".import_report.json"))
        write_json(report_path, records.report)
        return {"recordings": len(records), "manifest": args.output, "report": report_path}
    if args.command == "split-dev":
        from .data import split_development
        records = split_development(load_manifest(args.manifest), calibration_fraction=args.fraction, seed=args.seed)
        write_manifest(args.output, records)
        return validate_manifest(records)
    if args.command == "select-pilot":
        import numpy as np
        records = load_manifest(args.manifest)
        validate_manifest(records)
        if args.groups_per_split < 1:
            raise ValueError("groups-per-split must be positive")
        if args.min_duration_sec < 0 or (args.max_duration_sec is not None and args.max_duration_sec <= args.min_duration_sec):
            raise ValueError("invalid duration limits")
        by_source = {}
        for record in records:
            by_source.setdefault(record["source_recording_id"], []).append(record)
        excluded = {}
        for group, members in by_source.items():
            if any(r["duration_sec"] < args.min_duration_sec or
                   (args.max_duration_sec is not None and r["duration_sec"] > args.max_duration_sec) for r in members):
                excluded[group] = "at least one variant outside declared expert duration limits"
            elif args.require_paired and {r.get("label") for r in members} != {0, 1}:
                excluded[group] = "source group lacks a bona fide/edited pair"
        rng = np.random.default_rng(args.seed)
        keep = set()
        for split in sorted({r["split"] for r in records}):
            groups = sorted({r["source_recording_id"] for r in records if r["split"] == split and r["source_recording_id"] not in excluded})
            keep.update(rng.choice(groups, size=min(len(groups), args.groups_per_split), replace=False).tolist())
        selected = [r for r in records if r["source_recording_id"] in keep]
        write_manifest(args.output, selected)
        report = validate_manifest(selected)
        report.update(seed=args.seed, maximum_source_groups_per_split=args.groups_per_split,
                      sampling="random source groups after declared eligibility checks; all available variants retained",
                      require_bona_fide_edited_pair=args.require_paired,
                      min_duration_sec=args.min_duration_sec, max_duration_sec=args.max_duration_sec,
                      excluded_source_groups=excluded,
                      caveat="pilot subset only; not the official full benchmark")
        write_json(str(Path(args.output).with_suffix(".selection_report.json")), report)
        return report
    if args.command == "prepare-audio":
        from .audio_quality import prepare_audio
        records = prepare_audio(load_manifest(args.manifest), args.audio_dir)
        write_manifest(args.output, records)
        return {"recordings": len(records), "sample_rate": 16000, "channels": 1,
                "output": args.output, "no_trimming": True}
    if args.command == "validate-rdf":
        from .ontology import validate_rdf
        return validate_rdf(args.graph)
    if args.command == "merge-observations":
        from .ontology import validate_observations, deduplicate_observations
        records = load_manifest(args.manifest)
        validate_manifest(records)
        observations = [row for path in args.inputs for row in read_jsonl(path)]
        detectors = load_detectors(args.detectors)
        report = validate_observations(observations, records, detectors)
        canonical, dedup_report = deduplicate_observations(observations)
        write_jsonl(args.output, canonical)
        write_json(str(Path(args.output).with_suffix(".merge_report.json")), {"validation": report, "deduplication": dedup_report})
        return {"written_observations": len(canonical), "output": args.output, "deduplication": dedup_report}
    records, observations = load_manifest(args.manifest), read_jsonl(args.observations)
    if args.command in ("predict", "evaluate"):
        from .pipeline import predict, evaluate
        model = read_json(args.model)
        if args.command == "evaluate":
            return evaluate(model, records, observations, args.output, split=args.split, bootstrap=args.bootstrap)
        result = predict(model, records, observations, output=args.output,
                         rejection_confidence=args.rejection_confidence)
        return {"recordings": len(result["predictions"]), "output": args.output,
                "review": sum(p["decision"] == "review" for p in result["predictions"])}
    detectors = load_detectors(args.detectors)
    if args.command == "validate":
        from .ontology import validate_observations, deduplicate_observations
        return {"manifest": validate_manifest(records, check_audio=args.check_audio),
                "observations": validate_observations(observations, records, detectors),
                "deduplication": deduplicate_observations(observations)[1]}
    if args.command == "annotate-quality":
        from .audio_quality import annotate_quality
        from .ontology import validate_observations
        validate_manifest(records, check_audio=True)
        validate_observations(observations, records, detectors)
        result = annotate_quality(records, observations)
        validate_observations(result, records, detectors)
        write_jsonl(args.output, result)
        return {"observations": len(result), "measured": ["rms_db", "clipping_fraction"], "output": args.output}
    if args.command == "export-graph":
        from .ontology import export_graph
        return export_graph(records, observations, detectors, args.output)
    if args.command == "train":
        from .pipeline import train
        from .features import read_relations
        return train(records, observations, detectors, args.output, mode=args.mode,
                     relations=read_relations(args.relations), step_sec=args.step_sec,
                     context_sec=args.context_sec, target_fpr=args.target_fpr, seed=args.seed)[1]
    raise ValueError(f"unknown command {args.command}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
