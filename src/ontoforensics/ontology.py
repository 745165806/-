"""Typed evidence validation, conservative deduplication, and RDF export.

The core needs only the Python standard library. RDF/SHACL validation is an
optional extra. An Observation is an evidence *record* (prov:Entity/sosa:Result),
not a sosa:Observation activity or a fact that an audio segment is forged.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

NS = "https://example.org/ontoforensics#"
INSTANCE_NS = "https://example.org/ontoforensics/id/"
PROV = "http://www.w3.org/ns/prov#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD = "http://www.w3.org/2001/XMLSchema#"
STATUS = frozenset({"valid", "missing", "not_applicable", "failed"})
SCORE_TYPES = frozenset({"logit", "margin", "probability"})
QUALITY_FIELDS = frozenset({"rms_db", "clipping_fraction", "estimated_snr_db"})
OBSERVATION_FIELDS = frozenset({
    "observation_id", "recording_id", "detector_id", "run_id", "start_sec",
    "end_sec", "score", "status", "provenance_id", "derived_from", "quality",
})
TIME_TOLERANCE = 1e-8


def _identifier(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a nonempty string")
    return value


def _finite(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: {field} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{context}: {field} cannot be represented as a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: {field} must be finite; NaN/Infinity are invalid")
    return number


def _entries(value: Any, id_field: str, envelope: str) -> list[dict[str, Any]]:
    """Accept a list, an ID-keyed mapping, or the documented JSON envelope."""
    if isinstance(value, Mapping):
        if envelope in value and isinstance(value[envelope], list):
            value = value[envelope]
        elif id_field in value:
            value = [value]
        else:
            value = list(value.values())
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{envelope} must be a list or a mapping of records")
    out = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"each {envelope} entry must be an object")
        out.append(dict(item))
    return out


def _index(value: Any, id_field: str, envelope: str) -> dict[str, dict[str, Any]]:
    result = {}
    for item in _entries(value, id_field, envelope):
        identifier = _identifier(item.get(id_field), id_field, envelope)
        if identifier in result:
            raise ValueError(f"duplicate {id_field}: {identifier!r}")
        result[identifier] = item
    return result


def _basic_observation(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError("each observation must be an object")
    obj = copy.deepcopy(dict(item))
    oid = _identifier(obj.get("observation_id"), "observation_id", "observation")
    context = f"observation {oid!r}"
    unknown = set(obj) - OBSERVATION_FIELDS
    if unknown:
        raise ValueError(f"{context}: unexpected fields {sorted(unknown)}; keep labels/metadata in the manifest")
    for field in ("recording_id", "detector_id", "run_id"):
        _identifier(obj.get(field), field, context)
    if obj.get("status") not in STATUS:
        raise ValueError(f"{context}: status must be one of {sorted(STATUS)}")
    if obj["status"] == "valid":
        obj["score"] = _finite(obj.get("score"), "score", context)
    else:
        if obj.get("score") is not None:
            raise ValueError(f"{context}: a non-valid observation must have score=null/absent, never zero")
        obj["score"] = None
    for field in ("start_sec", "end_sec"):
        if obj.get(field) is not None:
            obj[field] = _finite(obj[field], field, context)
        else:
            obj[field] = None
    if (obj["start_sec"] is None) != (obj["end_sec"] is None):
        raise ValueError(f"{context}: start_sec and end_sec must both be supplied or both be null")
    if obj["start_sec"] is not None:
        if obj["start_sec"] < 0 or obj["end_sec"] <= obj["start_sec"]:
            raise ValueError(f"{context}: require 0 <= start_sec < end_sec")
    for field in ("provenance_id", "derived_from"):
        if obj.get(field) is not None:
            _identifier(obj[field], field, context)
        else:
            obj[field] = None
    quality = obj.get("quality", {})
    if quality is None:
        quality = {}
    if not isinstance(quality, Mapping):
        raise ValueError(f"{context}: quality must be an object")
    unknown_quality = set(quality) - QUALITY_FIELDS
    if unknown_quality:
        raise ValueError(f"{context}: unsupported quality fields {sorted(unknown_quality)}")
    normalized_quality = {}
    for field, value in quality.items():
        if value is None:
            continue  # an absent measurement has no RDF value or numerical evidence
        number = _finite(value, f"quality.{field}", context)
        if field == "clipping_fraction" and not 0 <= number <= 1:
            raise ValueError(f"{context}: clipping_fraction must lie in [0, 1]")
        normalized_quality[field] = number
    obj["quality"] = normalized_quality
    return obj


def _evidence_tuple(obj: Mapping[str, Any], include_detector: bool = True) -> tuple:
    """The measurement content; identifiers of copies/runs are intentionally absent."""
    fields = (obj["recording_id"], obj["start_sec"], obj["end_sec"],
              obj["status"], obj["score"], tuple(sorted(obj["quality"].items())))
    return ((obj["detector_id"],) + fields) if include_detector else fields


def deduplicate_observations(observations: Iterable[Mapping[str, Any]]) -> tuple[list[dict], dict]:
    """Collapse verified aliases and exact repeated detector outputs.

    ``derived_from`` is deliberately restricted to a copy-of-result relation in
    this baseline. It must reference an observation in this batch. A changed
    score is not a copy. A provenance_id identifies one *output*, not a WAV file,
    a whole run, a model, or a window. Exact content deduplication additionally
    requires the same detector_id. This does not infer statistical independence.
    """
    objects = [_basic_observation(obj) for obj in observations]
    by_id: dict[str, dict] = {}
    repeated_ids = 0
    for obj in objects:
        oid = obj["observation_id"]
        if oid in by_id:
            if obj != by_id[oid]:
                raise ValueError(f"conflicting records reuse observation_id {oid!r}")
            repeated_ids += 1
        else:
            by_id[oid] = obj

    # Iterative path traversal avoids recursion limits on long provenance chains.
    terminal: dict[str, str] = {}
    for oid in by_id:
        trail = []
        seen = set()
        current = oid
        while current not in terminal:
            if current in seen:
                raise ValueError(f"derived_from cycle involving observation {current!r}")
            seen.add(current)
            trail.append(current)
            source = by_id[current].get("derived_from")
            if source is None:
                terminal[current] = current
                break
            if source not in by_id:
                raise ValueError(f"observation {current!r}: unresolved derived_from {source!r}")
            current = source
        root = terminal[current]
        for member in trail:
            terminal[member] = root

    parent = {oid: oid for oid in by_id}

    def find(oid: str) -> str:
        while parent[oid] != oid:
            parent[oid] = parent[parent[oid]]
            oid = parent[oid]
        return oid

    def union(left: str, right: str) -> None:
        lroot, rroot = find(left), find(right)
        if lroot != rroot:
            parent[max(lroot, rroot)] = min(lroot, rroot)

    provenance_owner: dict[str, str] = {}
    for oid, obj in by_id.items():
        if obj["derived_from"] is not None:
            union(oid, obj["derived_from"])
        provenance_id = obj["provenance_id"]
        if provenance_id is not None:
            if provenance_id in provenance_owner:
                union(oid, provenance_owner[provenance_id])
            else:
                provenance_owner[provenance_id] = oid

    groups: dict[str, list[str]] = defaultdict(list)
    for oid in by_id:
        groups[find(oid)].append(oid)
    known_groups = []
    for members in groups.values():
        signatures = {_evidence_tuple(by_id[oid], include_detector=False) for oid in members}
        if len(signatures) != 1:
            raise ValueError(
                f"contradictory copies/provenance for observations {sorted(members)!r}: "
                "recording, interval, status, score, and measured quality must agree; "
                "a recomputed/calibrated result needs a new provenance identity"
            )
        roots = {terminal[oid] for oid in members}
        root_detectors = {by_id[oid]["detector_id"] for oid in roots}
        if len(root_detectors) > 1:
            raise ValueError(
                f"ambiguous provenance aliases across detectors: {sorted(members)!r}; "
                "use derived_from to identify the original output explicitly"
            )
        canonical = min(roots)
        known_groups.append((canonical, sorted(members)))

    # Content equality is conservative: never merge unrelated detector IDs merely
    # because they happen to emit the same numeric score.
    by_content: dict[tuple, list[tuple[str, list[str]]]] = defaultdict(list)
    for canonical, members in known_groups:
        by_content[_evidence_tuple(by_id[canonical])].append((canonical, members))
    final_groups = []
    for matching in by_content.values():
        canonical = min(item[0] for item in matching)
        members = sorted(oid for _, group in matching for oid in group)
        reasons = []
        if any(by_id[oid]["derived_from"] is not None for oid in members):
            reasons.append("derived_from")
        pcounts = Counter(by_id[oid]["provenance_id"] for oid in members if by_id[oid]["provenance_id"])
        if any(count > 1 for count in pcounts.values()):
            reasons.append("provenance_id")
        if len(matching) > 1:
            reasons.append("identical_content")
        final_groups.append({"canonical_id": canonical, "member_ids": members, "reasons": reasons})
    final_groups.sort(key=lambda group: group["canonical_id"])
    canonical_objects = [copy.deepcopy(by_id[group["canonical_id"]]) for group in final_groups]
    mapping = {oid: group["canonical_id"] for group in final_groups for oid in group["member_ids"]}
    report = {
        "input_count": len(objects),
        "unique_id_count": len(by_id),
        "canonical_count": len(canonical_objects),
        "removed_count": len(objects) - len(canonical_objects),
        "identical_repeated_ids": repeated_ids,
        "canonical_id_by_observation_id": mapping,
        "groups": [group for group in final_groups if len(group["member_ids"]) > 1],
        "note": "Duplicate counts and IDs are audit data, not predictive features.",
    }
    return canonical_objects, report


def validate_observations(observations: Iterable[Mapping[str, Any]], records: Any,
                          detectors: Any) -> dict:
    """Validate JSON evidence and its references; raise ValueError on bad input."""
    record_index = _index(records, "recording_id", "records")
    detector_index = _index(detectors, "detector_id", "detectors")
    durations = {}
    for rid, record in record_index.items():
        duration = _finite(record.get("duration_sec"), "duration_sec", f"recording {rid!r}")
        if duration <= 0:
            raise ValueError(f"recording {rid!r}: duration_sec must be positive")
        durations[rid] = duration
        sample_rate = record.get("sample_rate")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"recording {rid!r}: sample_rate must be a positive integer")
        label = record.get("label")
        if label is not None and (isinstance(label, bool) or label not in (0, 1)):
            raise ValueError(f"recording {rid!r}: label must be 0, 1, null, or absent")
    for did, detector in detector_index.items():
        context = f"detector {did!r}"
        _identifier(detector.get("family"), "family", context)
        if detector.get("score_type") not in SCORE_TYPES:
            raise ValueError(f"{context}: score_type must be one of {sorted(SCORE_TYPES)}")
        if not isinstance(detector.get("higher_is_fake"), bool):
            raise ValueError(f"{context}: higher_is_fake must be boolean")
        if detector.get("scope") not in {"local", "global"}:
            raise ValueError(f"{context}: scope must be local or global")
        digest = detector.get("checkpoint_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or
                                   any(ch not in "0123456789abcdefABCDEF" for ch in digest)):
            raise ValueError(f"{context}: checkpoint_sha256 must be a 64-character hexadecimal digest")
        if "synthetic" in detector and not isinstance(detector["synthetic"], bool):
            raise ValueError(f"{context}: synthetic must be boolean")
    objects = [_basic_observation(obj) for obj in observations]
    scope_counts = Counter()
    for obj in objects:
        context = f"observation {obj['observation_id']!r}"
        if obj["recording_id"] not in record_index:
            raise ValueError(f"{context}: unknown recording_id {obj['recording_id']!r}")
        if obj["detector_id"] not in detector_index:
            raise ValueError(f"{context}: unknown detector_id {obj['detector_id']!r}")
        detector = detector_index[obj["detector_id"]]
        scope = detector["scope"]
        scope_counts[scope] += 1
        start, end = obj["start_sec"], obj["end_sec"]
        duration = durations[obj["recording_id"]]
        if start is None:
            raise ValueError(f"{context}: {scope} observations require start_sec and end_sec; global bounds are [0,duration]")
        if start is not None and end > duration + TIME_TOLERANCE:
            raise ValueError(f"{context}: end_sec exceeds recording duration")
        if scope == "global" and start is not None:
            if abs(start) > TIME_TOLERANCE or abs(end - duration) > TIME_TOLERANCE:
                raise ValueError(f"{context}: global observations must target the whole recording with bounds [0,duration]")
        if obj["status"] == "valid" and detector["score_type"] == "probability":
            if not 0 <= obj["score"] <= 1:
                raise ValueError(f"{context}: a probability score must lie in [0, 1]")
    canonical_objects, dedup_report = deduplicate_observations(objects)
    measurement_keys = {}
    for obj in canonical_objects:
        key = (obj["recording_id"], obj["detector_id"], obj["start_sec"], obj["end_sec"])
        if key in measurement_keys:
            raise ValueError(
                f"multiple non-equivalent observations for one detector/target: {key!r}; "
                "the baseline needs one canonical measurement per target, not an implicit overwrite"
            )
        measurement_keys[key] = obj["observation_id"]
    by_id = {obj["observation_id"]: obj for obj in objects}
    for group in dedup_report["groups"]:
        settings = set()
        digests = set()
        for oid in group["member_ids"]:
            detector = detector_index[by_id[oid]["detector_id"]]
            settings.add((detector["family"], detector["score_type"], detector["scope"], detector["higher_is_fake"]))
            if detector.get("checkpoint_sha256"):
                digests.add(detector["checkpoint_sha256"].lower())
        if len(settings) > 1 or len(digests) > 1:
            raise ValueError(f"duplicate aliases disagree on detector semantics/checkpoint: {group['member_ids']!r}")
    return {
        "valid": True, "observation_count": len(objects),
        "recording_count": len(record_index), "detector_count": len(detector_index),
        "status_counts": dict(sorted(Counter(obj["status"] for obj in objects).items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "canonical_observation_count": dedup_report["canonical_count"],
        "duplicate_count": dedup_report["removed_count"],
    }


def _iri(kind: str, *values: Any) -> str:
    key = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return "<" + INSTANCE_NS + kind + "/" + quote(key, safe="") + ">"


def _term(uri: str) -> str:
    return "<" + uri + ">"


def _literal(value: Any, datatype: str | None = None) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
        datatype = datatype or "boolean"
    elif isinstance(value, float):
        text = repr(value)
        datatype = datatype or "double"
    elif isinstance(value, int):
        text = str(value)
        datatype = datatype or "integer"
    else:
        text = str(value)
    encoded = json.dumps(text, ensure_ascii=False)
    return encoded + ("^^" + _term(XSD + datatype) if datatype else "")


def export_graph(records: Any, observations: Iterable[Mapping[str, Any]],
                 detectors: Any, out_path: str | Path) -> dict:
    """Write deterministic Turtle using stdlib; never export supervision as evidence.

    Original and copied evidence records remain visible for audit. Only records
    marked isCanonical=true may enter a predictive feature builder. Semantic
    relationship edges are derived from canonical records and measured times.
    """
    objects = [_basic_observation(obj) for obj in observations]
    validation = validate_observations(objects, records, detectors)
    record_index = _index(records, "recording_id", "records")
    detector_index = _index(detectors, "detector_id", "detectors")
    canonical, dedup_report = deduplicate_observations(objects)
    canonical_ids = {obj["observation_id"] for obj in canonical}
    unique_objects = {obj["observation_id"]: obj for obj in objects}
    triples: set[tuple[str, str, str]] = set()

    def add(subject: str, predicate: str, obj: str) -> None:
        triples.add((subject, _term(predicate), obj))

    def type_(subject: str, name: str) -> None:
        add(subject, RDF + "type", _term(NS + name))

    for rid, record in record_index.items():
        node = _iri("recording", rid)
        type_(node, "AudioRecording")
        add(node, NS + "recordingId", _literal(rid))
        add(node, NS + "durationSec", _literal(float(record["duration_sec"])))
        add(node, NS + "sampleRate", _literal(record["sample_rate"]))
    for did, detector in detector_index.items():
        node = _iri("detector", did)
        family = _iri("family", detector["family"])
        type_(node, "DetectorModel")
        type_(family, "EvidenceFamily")
        add(node, NS + "detectorId", _literal(did))
        add(node, NS + "inEvidenceFamily", family)
        add(family, NS + "familyName", _literal(detector["family"]))
        add(node, NS + "scoreType", _literal(detector["score_type"]))
        add(node, NS + "scope", _literal(detector["scope"]))
        add(node, NS + "higherIsFake", _literal(detector["higher_is_fake"]))
        if detector.get("checkpoint_sha256"):
            add(node, NS + "checkpointSha256", _literal(detector["checkpoint_sha256"].lower()))
        if "synthetic" in detector:
            add(node, NS + "synthetic", _literal(detector["synthetic"]))
    targets = {}
    for oid, obj in unique_objects.items():
        recording = _iri("recording", obj["recording_id"])
        detector = detector_index[obj["detector_id"]]
        observation = _iri("observation", oid)
        # A raw run_id may name a batch. RDF records its specific invocation on
        # this recording/detector, avoiding an accidental many-input ambiguity.
        run = _iri("run", obj["recording_id"], obj["detector_id"], obj["run_id"])
        type_(run, "DetectorRun")
        add(run, NS + "runId", _literal(obj["run_id"]))
        add(run, NS + "usesDetector", _iri("detector", obj["detector_id"]))
        add(run, PROV + "used", recording)
        type_(observation, "Observation")
        type_(observation, "LocalObservation" if detector["scope"] == "local" else "GlobalObservation")
        add(observation, NS + "observationId", _literal(oid))
        add(observation, PROV + "wasGeneratedBy", run)
        add(observation, NS + "status", _literal(obj["status"]))
        add(observation, NS + "isCanonical", _literal(oid in canonical_ids))
        if detector["scope"] == "local":
            target = _iri("window", obj["recording_id"], obj["start_sec"], obj["end_sec"])
            type_(target, "TimeWindow")
            add(target, NS + "windowOf", recording)
            add(target, NS + "startSec", _literal(obj["start_sec"]))
            add(target, NS + "endSec", _literal(obj["end_sec"]))
        else:
            target = recording
        targets[oid] = target
        add(observation, NS + "hasTarget", target)
        if obj["score"] is not None:
            add(observation, NS + "score", _literal(obj["score"]))
        if obj["provenance_id"] is not None:
            add(observation, NS + "provenanceId", _literal(obj["provenance_id"]))
        if obj["derived_from"] is not None:
            add(observation, PROV + "wasDerivedFrom", _iri("observation", obj["derived_from"]))
        canonical_id = dedup_report["canonical_id_by_observation_id"][oid]
        if canonical_id != oid:
            add(observation, NS + "knownDuplicateOf", _iri("observation", canonical_id))
        for field, prop in (("rms_db", "rmsDb"), ("clipping_fraction", "clippingFraction"),
                            ("estimated_snr_db", "estimatedSnrDb")):
            if field in obj["quality"]:
                add(observation, NS + prop, _literal(obj["quality"][field]))

    # These joins are the same typed relations configured in relations.json.
    # No label, true generator, true codec or edited boundary is read here.
    family_groups = defaultdict(list)
    temporal_groups = defaultdict(list)
    for obj in canonical:
        detector = detector_index[obj["detector_id"]]
        oid = obj["observation_id"]
        family_groups[(obj["recording_id"], targets[oid], detector["family"])].append(oid)
        if detector["scope"] == "local":
            temporal_groups[(obj["recording_id"], obj["detector_id"])].append(obj)
    family_edges = 0
    temporal_edges = 0
    for ids in family_groups.values():
        for index, left in enumerate(sorted(ids)):
            for right in sorted(ids)[index + 1:]:
                add(_iri("observation", left), NS + "sameEvidenceFamilyAs", _iri("observation", right))
                add(_iri("observation", right), NS + "sameEvidenceFamilyAs", _iri("observation", left))
                family_edges += 2
    for group in temporal_groups.values():
        ordered = sorted(group, key=lambda obj: (obj["start_sec"], obj["end_sec"], obj["observation_id"]))
        for left, right in zip(ordered, ordered[1:]):
            if (left["start_sec"], left["end_sec"]) == (right["start_sec"], right["end_sec"]):
                continue
            add(_iri("observation", left["observation_id"]), NS + "nextTemporalObservation",
                _iri("observation", right["observation_id"]))
            temporal_edges += 1
    header = (
        "# OntoForensics evidence graph. Observations are result records, not authenticity facts.\n"
        "# Only isCanonical=true records may contribute predictive evidence.\n"
        f"@prefix of: <{NS}> .\n@prefix prov: <{PROV}> .\n"
        f"@prefix rdf: <{RDF}> .\n@prefix xsd: <{XSD}> .\n\n"
    )
    text = header + "".join(f"{s} {p} {o} .\n" for s, p, o in sorted(triples))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return {
        "exported": True, "path": str(destination.resolve()), "format": "turtle",
        "backend": "stdlib", "triple_count": len(triples),
        "canonical_observation_count": len(canonical),
        "duplicate_count": dedup_report["removed_count"],
        "same_family_edge_count": family_edges, "temporal_edge_count": temporal_edges,
        "validation": validation,
        "supervision_exported": False,
    }


def _ontology_resource(filename: str) -> Path:
    # A source checkout uses the canonical ontology/ files; an installed wheel
    # uses bundled resources/. Neither path depends on the working directory.
    candidates = (Path(__file__).resolve().parents[2] / "ontology" / filename,
                  Path(__file__).resolve().parent / "resources" / filename)
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError(
        f"ontology resource {filename!r} is missing; package ontology/*.ttl "
        "or ontoforensics/resources/*.ttl when building the distribution"
    )


def validate_rdf(graph_path: str | Path) -> dict:
    """Validate Turtle with local OWL/RDFS declarations and SHACL shapes.

    Missing optional packages return a clear skipped report, never a false pass.
    Invalid graphs raise ValueError, like validate_observations.
    """
    missing = [name for name in ("rdflib", "pyshacl") if importlib.util.find_spec(name) is None]
    if missing:
        return {
            "validated": False, "conforms": None, "status": "skipped",
            "missing_dependencies": missing,
            "install_command": "python -m pip install 'rdflib>=6' 'pyshacl>=0.25'",
            "message": "RDF was not checked: install the optional dependencies, then rerun validation.",
        }
    import rdflib
    from pyshacl import validate

    try:
        graph = rdflib.Graph().parse(str(graph_path), format="turtle")
        ontology = rdflib.Graph().parse(str(_ontology_resource("forensics.ttl")), format="turtle")
        shapes = rdflib.Graph().parse(str(_ontology_resource("shapes.ttl")), format="turtle")
    except Exception as exc:
        raise ValueError(f"could not parse RDF/ontology/shapes: {exc}") from exc
    for name in ("score", "startSec", "endSec", "durationSec", "rmsDb", "clippingFraction", "estimatedSnrDb"):
        for subject, value in graph.subject_objects(rdflib.URIRef(NS + name)):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"RDF {name} is not numeric for {subject}") from exc
            if not math.isfinite(number):
                raise ValueError(f"RDF {name} is not finite for {subject}")
    try:
        conforms, _, report = validate(
            data_graph=graph, shacl_graph=shapes, ont_graph=ontology,
            inference="rdfs", advanced=False, abort_on_first=False,
        )
    except Exception as exc:
        raise ValueError(f"SHACL validation could not complete: {exc}") from exc
    if not conforms:
        raise ValueError("RDF does not conform to ontology/shapes:\n" + str(report))
    return {"validated": True, "conforms": True, "status": "passed", "report": str(report)}
