"""Validate the local wine CSV and report exact snapshot identity without raw rows."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

COLUMNS = ('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
           'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'quality')


def audit(path, group_summary=False):
    return _audit_payload(Path(path).read_bytes(), group_summary)


def _audit_payload(payload, group_summary=False):
    reader = csv.reader(io.StringIO(payload.decode('utf-8-sig')), delimiter=';')
    header = next(reader, [])
    normalized = [name.strip().lower().replace(' ', '_') for name in header]
    if len(set(normalized)) != len(normalized) or set(normalized) != set(COLUMNS):
        raise ValueError('Expected twelve distinct wine-quality columns')
    quality_index = normalized.index('quality')
    alcohol_index = normalized.index('alcohol')
    grouped = {}; deduplicated = {}
    unique = set(); distribution = Counter(); count = 0
    for line, row in enumerate(reader, 2):
        if len(row) != len(normalized):
            raise ValueError(f'Wrong field count at line {line}')
        try:
            values = tuple(float(value) for value in row)
        except ValueError as error:
            raise ValueError(f'Non-numeric value at line {line}') from error
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f'Non-finite or negative value at line {line}')
        quality = values[quality_index]
        if not quality.is_integer() or not 0 <= quality <= 10:
            raise ValueError(f'Quality must be an integer in [0, 10] at line {line}')
        grouped.setdefault(int(quality), []).append(values[alcohol_index])
        if values not in unique:
            deduplicated.setdefault(int(quality), []).append(values[alcohol_index])
        unique.add(values); distribution[int(quality)] += 1; count += 1
    if not count:
        raise ValueError('Empty dataset')
    result = {'status': 'passed', 'source_sha256': hashlib.sha256(payload).hexdigest(),
            'rows': count, 'unique_numeric_rows': len(unique), 'duplicate_rows': count - len(unique),
            'quality_distribution': dict(sorted(distribution.items())),
            'scope': 'Snapshot validation; duplicates are numeric row equality. No population representativeness or causal claim.'}
    if group_summary:
        import statistics
        summarize = lambda groups: {str(key): {'count': len(values), 'mean_alcohol': statistics.mean(values),
                                      'sample_sd_alcohol': statistics.stdev(values) if len(values)>1 else None}
                                   for key, values in sorted(groups.items())}
        result['alcohol_by_quality'] = {'all_rows': summarize(grouped), 'numeric_duplicates_removed': summarize(deduplicated),
                                       'scope': 'Descriptive group statistics before/after duplicate removal; no causal or predictive inference.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path, default=Path(__file__).parent / 'winequality-red.csv')
    parser.add_argument('--group-summary', action='store_true', help='Compare descriptive alcohol summaries before/after deduplication')
    parser.add_argument('--compare-with', type=Path, help='Check exact feature overlap against a second wine CSV split')
    args = parser.parse_args()
    try:
        result = audit(args.path, args.group_summary)
        if args.compare_with:
            result['split_overlap'] = split_overlap(args.path, args.compare_with)
            if result['source_sha256'] != result['split_overlap']['train_sha256']:
                raise ValueError('Input changed between audit and split comparison')
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f'Data audit failed: {error}\n')

def split_overlap(train_path, test_path):
    """Detect exact feature overlap even when a duplicate has another label.

    Report aggregate counts only. Exact matching cannot establish statistical
    independence or detect approximate/semantic duplicates.
    """
    payloads = [Path(path).read_bytes() for path in (train_path, test_path)]
    snapshots = [_audit_payload(payload) for payload in payloads]
    indexes = []
    row_counts = []
    full_rows = []
    for payload in payloads:
        reader = csv.reader(io.StringIO(payload.decode('utf-8-sig')), delimiter=';')
        header = [name.strip().lower().replace(' ', '_') for name in next(reader)]
        order = [header.index(name) for name in COLUMNS]
        groups = {};counts = Counter();complete = set()
        for row in reader:
            values = tuple(float(row[index]) for index in order)
            features, label = values[:-1], int(values[-1])
            groups.setdefault(features, set()).add(label)
            counts[features] += 1
            complete.add(values)
        indexes.append(groups);row_counts.append(counts);full_rows.append(complete)
    shared = indexes[0].keys() & indexes[1].keys()
    return {'status': 'potential_exact_feature_overlap' if shared else 'no_exact_feature_overlap',
            'train_sha256': snapshots[0]['source_sha256'], 'test_sha256': snapshots[1]['source_sha256'],
            'shared_unique_feature_vectors': len(shared),
            'test_rows_with_training_features': sum(row_counts[1][key] for key in shared),
            'shared_unique_full_rows': len(full_rows[0] & full_rows[1]),
            'shared_feature_groups_with_conflicting_labels': sum(len(indexes[0][key] | indexes[1][key]) > 1 for key in shared),
            'scope': 'Potential exact numerical feature leakage; the quality target is excluded from the feature identity. Shared features may be legitimate repeats. No raw rows exported, approximate-duplicate detection, or independence guarantee.'}


if __name__ == '__main__':
    main()
