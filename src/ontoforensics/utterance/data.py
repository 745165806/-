"""Read-only corpus adapters and speaker-disjoint experimental roles."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROLES = ['expert_select', 'fusion_train', 'fusion_select', 'calibration', 'threshold']

def identity(row):
    """Storage identity; embedded rows in the same shard are distinct recordings."""
    return (str(Path(row['path']).resolve()), row.get('storage', 'file'),
            str(row.get('row_index', '')) if row.get('storage', 'file') != 'file' else '')

def fingerprint(rows):
    payload = sorted((r['recording_id'], r['group_id'], r['role'], r.get('label'), identity(r)) for r in rows)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    temporary.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def audio(row):
    location = row['path']
    if row.get('storage', 'file') != 'file':
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
        index=int(row['row_index'])
        if index < 0: raise ValueError('Embedded audio row must be nonnegative')
        value=None
        if row['storage'] == 'parquet_embedded':
            shard=pq.ParquetFile(location)
            for group in range(shard.num_row_groups):
                count=shard.metadata.row_group(group).num_rows
                if index<count:
                    value=shard.read_row_group(group,columns=['audio'])['audio'][index].as_py()
                    break
                index-=count
        elif row['storage'] == 'arrow_embedded':
            with open(location, 'rb') as f:
                try: reader=ipc.open_stream(f)
                except Exception:
                    f.seek(0); reader=ipc.open_file(f)
                batches=(reader.get_batch(i) for i in range(reader.num_record_batches)) if hasattr(reader,'get_batch') else reader
                for batch in batches:
                    if index<batch.num_rows:
                        value=batch['audio'][index].as_py()
                        break
                    index-=batch.num_rows
        else:
            raise ValueError('Unknown storage type')
        if not isinstance(value, dict) or not value.get('bytes'):
            raise ValueError('Embedded audio bytes missing')
        location = io.BytesIO(value['bytes'])
    x, sr = sf.read(location, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty or nonfinite audio')
    g = math.gcd(sr, 16000)
    if sr != 16000:
        x = resample_poly(x, 16000 // g, sr // g).astype('float32')
    return x

def quality(x):
    return [float(20*np.log10(max(np.sqrt(np.mean(x*x)), 1e-10))),
            float(np.mean(np.abs(x) >= .999))]

def prepare(indexes, output, seed=17, check_audio=False):
    rows, excluded, seen = [], [], set()
    files = []
    for index in indexes:
        path = Path(index)
        if path.is_dir() or path.name == 'index.csv':
            catalog = path / 'index.csv' if path.is_dir() else path
            with catalog.open(newline='') as f:
                files.extend(catalog.parent / item['file'] for item in csv.DictReader(f))
        else:
            files.append(path)
    for index in files:
        index_hash = digest(index)
        with open(index, newline='') as f:
            for source in csv.DictReader(f):
                key = identity(source)
                reason = None
                if source['label'] not in ('0', '1'):
                    reason = 'unknown_label'
                elif not Path(source['path']).is_file():
                    reason = 'missing_file'
                elif key in seen:
                    reason = 'duplicate_location'
                if reason:
                    excluded.append({'id': source['id'], 'reason': reason})
                    continue
                seen.add(key)
                meta = json.loads(source['source_fields_json'] or '{}')
                dataset = source['dataset']
                speaker = meta.get('speaker_id') or meta.get('speaker')
                row = dict(recording_id=dataset+':'+source['id'], dataset=dataset,
                           path=source['path'], storage=source['storage'],
                           row_index=source['row_index'], label=int(source['label']),
                           speaker_id=speaker, group_id=dataset+':'+str(speaker or source['native_id']),
                           native_split=source['split'], role='test', index_sha256=index_hash)
                if dataset == 'ASVspoof2019_LA':
                    if source['split'] == 'train':
                        row['role'] = 'expert_train'
                    elif source['split'] == 'dev':
                        if not speaker:
                            raise ValueError('Source dev requires speaker IDs')
                        row['role'] = 'pending'
                if check_audio:
                    try:
                        x = audio(row)
                        row['audio_sha256'] = hashlib.sha256(x.tobytes()).hexdigest()
                    except Exception as e:
                        excluded.append({'id': row['recording_id'], 'reason': str(e)})
                        continue
                rows.append(row)
    groups = sorted({r['group_id'] for r in rows if r['role'] == 'pending'})
    np.random.default_rng(seed).shuffle(groups)
    # Largest remainder allocation, with at least one speaker per role.
    if len(groups) < 5:
        raise ValueError('At least five development speakers required')
    counts = np.ones(5, dtype=int)
    for _ in range(len(groups)-5):
        counts[np.argmax(np.array([.20,.35,.15,.15,.15])*len(groups)-counts)] += 1
    rng = np.random.default_rng(seed)
    labels_by_group = {g: set() for g in groups}
    for r in rows:
        if r['role'] == 'pending':
            labels_by_group[r['group_id']].add(r['label'])
    for attempt in range(10000):
        rng.shuffle(groups)
        mapping = dict(zip(groups, np.repeat(ROLES, counts)))
        if all(set().union(*(labels_by_group[g] for g in groups if mapping[g] == role)) == {0, 1} for role in ROLES):
            break
    else:
        raise ValueError('Cannot allocate both classes to all development roles')
    for r in rows:
        if r['role'] == 'pending':
            r['role'] = mapping[r['group_id']]
    validate(rows)
    write(output, rows)
    write(str(output)+'.excluded.jsonl', excluded)
    report = {'included': len(rows), 'excluded': len(excluded), 'speaker_allocation': mapping,
              'seed': seed, 'manifest_fingerprint': fingerprint(rows),
              'audio_content_checked': check_audio,
              'unknown_speaker_rows': sum(not r.get('speaker_id') for r in rows)}
    Path(str(output)+'.report.json').write_text(json.dumps(report, indent=2))
    return report

def validate(rows, require_roles=True, allow_unlabelled=False):
    ids, groups, contents = set(), {}, {}
    for r in rows:
        for name in ('recording_id', 'group_id', 'path', 'dataset', 'role'):
            if not isinstance(r.get(name), str) or not r[name]:
                raise ValueError('Missing or invalid '+name)
        if r['role'] not in ['expert_train'] + ROLES + ['test', 'unspecified']:
            raise ValueError('Unknown role: '+r['role'])
        if r['recording_id'] in ids:
            raise ValueError('Duplicate recording ID')
        ids.add(r['recording_id'])
        if not (allow_unlabelled and r.get('label') is None) and (type(r.get('label')) is not int or r['label'] not in (0, 1)):
            raise ValueError('Invalid binary label')
        for key, table in [(r['group_id'], groups),
                           (r.get('audio_sha256') or identity(r), contents)]:
            previous = table.setdefault(key, r['role'])
            if previous != r['role']:
                raise ValueError('Cross-role leakage: '+str(key))
    for role in (['expert_train'] + ROLES if require_roles else []):
        if {r['label'] for r in rows if r['role'] == role} != {0, 1}:
            raise ValueError('Role requires both classes: '+role)
