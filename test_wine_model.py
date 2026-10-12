import unittest
from wine_model import fit,predict,train
from pathlib import Path
class WineModelTests(unittest.TestCase):
    def test_train_only_scaling_and_ridge_prediction(self):
        rows=[tuple([float(i)]*11+[float(i)]) for i in range(20)]
        model=fit(rows,.01);self.assertEqual(model['means'],[9.5]*11)
        self.assertLess(abs(predict(model,[8.0]*11)-8),.001)
        with self.assertRaises(ValueError):predict(model,[float('nan')]*11)
    def test_seeded_workflow_reports_disjoint_group_splits(self):
        report=train(Path(__file__).parent/'winequality-red.csv')
        splits=report['split_groups'];self.assertEqual(sum(row['groups'] for row in splits.values()),1359)
        self.assertLess(report['test']['rmse'],report['training_mean_baseline_test']['rmse'])
        self.assertEqual(report['source_sha256'],train(Path(__file__).parent/'winequality-red.csv')['source_sha256'])
