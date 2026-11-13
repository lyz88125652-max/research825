import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from ml_pipeline.evaluation import write_csv
from ml_pipeline.reporting import generate_reports, score_predictions


class ReportingTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        # Matrix [[1,1,0], [0,2,0], [1,0,1]].
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.eye(3)[[0, 1, 1, 1, 0, 2]] * 0.8 + 0.2 / 3
        overall, classes = score_predictions(y, p)
        self.assertAlmostEqual(overall['accuracy'], 4 / 6)
        self.assertAlmostEqual(overall['sensitivity_macro'], 2 / 3)
        self.assertAlmostEqual(overall['specificity_macro'], 5 / 6)
        self.assertAlmostEqual(overall['mcc'], 6 / np.sqrt(12 * 11))
        self.assertEqual([r['specificity'] for r in classes], [0.75, 0.75, 1.0])
        self.assertEqual([r['tp'] for r in classes], [1, 2, 1])

    def test_workbook_oof_and_legacy_and_new_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model = run / 'cnn'
            model.mkdir()
            records = []
            for fold in (1, 2):
                for label in (1, 2, 3):
                    records.append({'fold': fold, 'true_label': label,
                                    **{f'prob_label_{c}': 0.8 if c == label else 0.1
                                       for c in (1, 2, 3)}})
                folder = model / f'fold_{fold}'
                folder.mkdir()
                history = [{'phase': 'selection', 'epoch': 1, 'train_loss': 0.4,
                            'validation_loss': 0.5},
                           {'phase': 'refit', 'epoch': 1, 'train_loss': 0.3,
                            'validation_loss': ''}]
                if fold == 2:
                    history[0]['train_eval_loss'] = 0.35
                    history[1]['train_eval_loss'] = ''
                write_csv(folder / 'history.csv', list(history[0]), history)
            write_csv(model / 'predictions.csv', list(records[0]), records)
            path, plots = generate_reports(run)
            workbook = load_workbook(path, read_only=True)
            self.assertEqual(workbook.sheetnames,
                             ['Summary', 'CV_folds', 'Class_CV', 'Class_metrics', 'Notes'])
            rows = list(workbook['Summary'].values)
            self.assertEqual([r[1] for r in rows[1:]], ['CV_mean', 'CV_std', 'OOF_pooled'])
            self.assertEqual(rows[1][rows[0].index('accuracy')], 1)
            self.assertEqual(rows[2][rows[0].index('accuracy')], 0)
            self.assertEqual(rows[3][2], 6)
            workbook.close()
            self.assertEqual(len(plots), 3)
            for plot in plots:
                self.assertTrue(plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()
