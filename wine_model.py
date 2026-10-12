"""Grouped train/validation/test ridge workflow without feature leakage."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
from audit_data import _audit_payload, COLUMNS
from wine_report_bundle import write_bundle

def solve(matrix, rhs):
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    n = len(a)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError('Singular regression system')
        a[col], a[pivot] = (a[pivot], a[col])
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for row in range(n):
            if row != col:
                scale = a[row][col]
                a[row] = [x - scale * y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]

def fit(rows, alpha):
    n = len(rows)
    means = [sum((r[j] for r in rows)) / n for j in range(11)]
    scales = [math.sqrt(sum(((r[j] - means[j]) ** 2 for r in rows)) / n) or 1 for j in range(11)]
    features = [[1] + [(r[j] - means[j]) / scales[j] for j in range(11)] for r in rows]
    gram = [[sum((r[i] * r[j] for r in features)) + (alpha if i == j and i > 0 else 0) for j in range(12)] for i in range(12)]
    rhs = [sum((x[j] * row[-1] for x, row in zip(features, rows))) for j in range(12)]
    return {'features': list(COLUMNS[:-1]), 'means': means, 'scales': scales, 'coefficients': solve(gram, rhs), 'alpha': alpha}

def predict(model, features):
    if model.get('features') != list(COLUMNS[:-1]) or len(features) != 11 or len(model.get('means', [])) != 11 or (len(model.get('scales', [])) != 11) or (len(model.get('coefficients', [])) != 12):
        raise ValueError('Invalid feature/model dimensions')
    if any((type(v) not in (int, float) or not math.isfinite(v) for seq in (features, model['means'], model['scales'], model['coefficients']) for v in seq)) or any((v <= 0 for v in model['scales'])):
        raise ValueError('Invalid model or feature values')
    return model['coefficients'][0] + sum((model['coefficients'][j + 1] * (features[j] - model['means'][j]) / model['scales'][j] for j in range(11)))

def metrics(model, rows):
    residuals = [predict(model, row[:-1]) - row[-1] for row in rows]
    return {'rows': len(rows), 'rmse': math.sqrt(sum((x * x for x in residuals)) / len(rows)), 'mae': sum((abs(x) for x in residuals)) / len(rows)}

def train(path, seed=17):
    raw = Path(path).read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('Dataset exceeds 16 MiB')
    audit = _audit_payload(raw)
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')), delimiter=';')
    header = [x.strip().lower().replace(' ', '_') for x in next(reader)]
    order = [header.index(x) for x in COLUMNS]
    rows = sorted({tuple((float(row[i]) for i in order)) for row in reader})
    groups = {}
    for row in rows:
        groups.setdefault(row[:-1], []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 15:
        raise ValueError('At least 15 unique feature groups required')
    first = int(len(keys) * 0.6)
    second = int(len(keys) * 0.8)
    partitions = [keys[:first], keys[first:second], keys[second:]]
    split = [[row for key in part for row in groups[key]] for part in partitions]
    candidates = [fit(split[0], alpha) for alpha in (0.01, 0.1, 1, 10)]
    validation = [metrics(model, split[1]) for model in candidates]
    selected = min(range(len(candidates)), key=lambda i: validation[i]['rmse'])
    model = candidates[selected]
    baseline = dict(model, coefficients=[sum((row[-1] for row in split[0])) / len(split[0])] + [0] * 11)
    return {'tool': 'wine-quality-model', 'source_sha256': audit['source_sha256'], 'seed': seed, 'model': model, 'split_groups': {name: {'groups': len(part), 'rows': len(data), 'feature_group_sha256': hashlib.sha256(json.dumps(part).encode()).hexdigest()} for name, part, data in zip(('train', 'validation', 'test'), partitions, split)}, 'validation_candidates': [dict(alpha=model['alpha'], **result) for model, result in zip(candidates, validation)], 'test': metrics(model, split[2]), 'training_mean_baseline_test': metrics(baseline, split[2]), 'preprocessing': 'Exact full-row deduplication; groups by all features exclude target; all scaling fit on training rows only; alpha selected on validation; test evaluated after selection.', 'scope': 'Local fixed-split ridge baseline. Exact feature groups cannot establish independence of wine provenance or general predictive quality.'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        report = train(a.dataset, a.seed)
        write_bundle(a.output, report, 'Wine-quality model development')
        print(json.dumps(report['test']))
    except (ValueError, OSError) as e:
        p.exit(2, str(e) + '\n')
if __name__ == '__main__':
    main()
