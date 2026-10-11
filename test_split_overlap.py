from pathlib import Path
import tempfile
import unittest
from audit_data import COLUMNS, split_overlap


class SplitOverlapTests(unittest.TestCase):
    def write(self, path, rows, reverse=False):
        columns = list(COLUMNS)
        if reverse: columns.reverse()
        order = [COLUMNS.index(column) for column in columns]
        path.write_text(';'.join(columns) + '\n' + ''.join(
            ';'.join(str(row[index]) for index in order) + '\n' for row in rows))

    def test_feature_overlap_detected_with_changed_target_and_column_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);train,test=root/'train.csv',root/'test.csv'
            self.write(train, [[1] * 11 + [5]])
            self.write(test, [[1.0] * 11 + [6], [1] * 11 + [6]], reverse=True)
            result = split_overlap(train, test)
            self.assertEqual(result['shared_unique_feature_vectors'], 1)
            self.assertEqual(result['test_rows_with_training_features'], 2)
            self.assertEqual(result['shared_unique_full_rows'], 0)
            self.assertEqual(result['shared_feature_groups_with_conflicting_labels'], 1)

    def test_disjoint_features_and_invalid_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,test=root/'train.csv',root/'test.csv'
            self.write(train, [[1] * 11 + [5]])
            self.write(test, [[2] * 11 + [5]])
            self.assertEqual(split_overlap(train, test)['status'], 'no_exact_feature_overlap')
            self.write(test, [[float('nan')] * 11 + [5]])
            with self.assertRaises(ValueError):split_overlap(train, test)
