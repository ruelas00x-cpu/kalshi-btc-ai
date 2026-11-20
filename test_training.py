import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        import training
        self.m=training
        self.patcher=patch.object(training,'DATA_DIR',Path(self.tmp.name))
        self.patcher.start()
        self.p2=patch.object(training,'DB_FILE',Path(self.tmp.name)/'btc_research.sqlite3');self.p2.start()
        self.p3=patch.object(training,'MODEL_FILE',Path(self.tmp.name)/'candidate_model.joblib');self.p3.start()
        self.p4=patch.object(training,'METRICS_FILE',Path(self.tmp.name)/'training_metrics.json');self.p4.start()
    def tearDown(self):
        for p in [self.patcher,self.p2,self.p3,self.p4]:p.stop()
        self.tmp.cleanup()
    def test_no_label_from_invalid_result(self):
        self.assertFalse(self.m.save_settlement('t',123,'void'))
        self.assertFalse(self.m.save_settlement('t',123,''))
        self.assertTrue(self.m.save_settlement('t',123,'yes'))
        self.assertEqual(self.m.statistics()['settled_markets'],1)
    def test_missing_reference_not_logged(self):
        intel={'fresh':True,'last_closed_bar':123,'atr':12,'last_close':100,'range_low':90,'range_high':110}
        self.assertFalse(self.m.record_snapshot('t',650,intel,None,.5,.5))
        self.assertEqual(self.m.statistics()['observed_markets'],0)
    def test_chronological_candidate_only(self):
        import json
        for i in range(250):
            label=int(i%2==0)
            x={k:0.0 for k in self.m.FEATURES}
            x.update(elapsed_minute=3,margin_score=float((2*label-1)*4+(i%5)*.05),rsi=float(40+20*label),
                volume_multiple=1.2,atr_pct=.0007,ask_up=.6,ask_down=.42)
            with self.m.connect() as db:
                db.execute('INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)',
                  (f't{i}',3,1000+i*900,'2026-01-01T00:00:00+00:00',json.dumps(x),'UP' if label else 'DOWN','UP' if label else 'DOWN',.6,.42))
            self.m.save_settlement(f't{i}',1000+i*900+900,'yes' if label else 'no')
        stats=self.m.train()
        self.assertTrue(stats['trained'])
        self.assertEqual(stats['total_labeled_markets'],250)
        self.assertEqual(stats['holdout_count'],50)
        self.assertEqual(self.m.statistics()['model_status'],'CANDIDATE ONLY')
        self.assertTrue(self.m.MODEL_FILE.exists())

if __name__=='__main__': unittest.main()
