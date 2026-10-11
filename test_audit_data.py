import tempfile, unittest
from pathlib import Path
from audit_data import audit, COLUMNS
class DataAuditTests(unittest.TestCase):
    def test_original(self):
        result=audit(Path(__file__).parent/'winequality-red.csv')
        self.assertEqual(result['rows'],1599)
        self.assertEqual(result['unique_numeric_rows'],1359)
    def test_invalid_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv'
            for value in ['nan','inf','-1','1.5','eleven']:
                path.write_text(';'.join(COLUMNS)+'\n'+';'.join(['1']*11+[value])+'\n')
                with self.assertRaises(ValueError):audit(path)
    def test_collision_and_width(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv'
            for payload in [';'.join(COLUMNS[:-1]+('alcohol',))+'\n',';'.join(COLUMNS)+'\n1;2\n']:
                path.write_text(payload)
                with self.assertRaises(ValueError):audit(path)
