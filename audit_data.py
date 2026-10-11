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
    payload = Path(path).read_bytes()
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
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.path, args.group_summary), indent=2))
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f'Data audit failed: {error}\n')

if __name__ == '__main__':
    main()
