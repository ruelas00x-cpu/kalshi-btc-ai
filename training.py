"""Research-only offline ML training. Outcomes must be Kalshi official yes/no settlements.
All features predate settlement, one sample per contract; chronological holdout.
No trained predictions are used for any trading or dashboard decision.
"""
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get('DATA_DIR', './data'))
DB_FILE = DATA_DIR / 'btc_research.sqlite3'
MODEL_FILE = DATA_DIR / 'candidate_model.joblib'
METRICS_FILE = DATA_DIR / 'training_metrics.json'
FEATURES = ['elapsed_minute','margin_score','rsi','volume_multiple','atr_pct','ema_gap_atr',
            'spot_gap_atr','range_position','flow_net_fraction','flow_available',
            'ask_up','ask_down','range_regime']

def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_FILE, timeout=10)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS snapshots (
      ticker TEXT NOT NULL, elapsed_minute INTEGER NOT NULL, candle_time INTEGER NOT NULL,
      recorded_at TEXT NOT NULL, features TEXT NOT NULL, tv_signal TEXT, rule_signal TEXT,
      up_ask REAL, down_ask REAL,
      PRIMARY KEY(ticker, elapsed_minute))''')
    db.execute('''CREATE TABLE IF NOT EXISTS settlements (
      ticker TEXT PRIMARY KEY, close_ts REAL NOT NULL, outcome INTEGER NOT NULL,
      labeled_at TEXT NOT NULL, raw_result TEXT NOT NULL)''')
    return db

@contextmanager
def connect():
    db=_connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def make_features(intel, seconds_left, ref, up_ask, down_ask):
    """Only fields derived from completed candles; no future outcome features."""
    if ref is None or not intel.get('fresh') or intel.get('last_closed_bar') is None: return None
    atr = float(intel['atr']); price = float(intel['last_close'])
    if atr <= 0 or price <= 0: return None
    lo, hi = float(intel['range_low']), float(intel['range_high'])
    flow = intel.get('flow') or {}
    return {
       'elapsed_minute': max(0, min(14, int((900 - seconds_left) // 60))),
       'margin_score': float(intel['margin_score']),
       'rsi': float(intel['rsi']),
       'volume_multiple': float(intel['volume_multiple']),
       'atr_pct': atr / price,
       'ema_gap_atr': (float(intel['ema9'])-float(intel['ema20'])) / atr,
       'spot_gap_atr': (price-float(ref)) / atr,
       'range_position': (price-lo)/max(hi-lo,atr*.1),
       'flow_net_fraction': float(flow.get('net_fraction', 0)),
       'flow_available': 1 if intel.get('flow_status')=='fresh' else 0,
       'ask_up': float(up_ask),
       'ask_down': float(down_ask),
       'range_regime': 1 if intel.get('market_type') == 'RANGE' else 0,
    }

def record_snapshot(ticker, seconds_left, intel, reference, up_ask, down_ask, tv_call=None):
    # Exclude late windows, stale data and missing Kalshi executable quotes/threshold.
    if not (0 < seconds_left <= 780 and 0 < up_ask < 1 and 0 < down_ask < 1): return False
    values=make_features(intel,seconds_left,reference,up_ask,down_ask)
    if values is None:return False
    candle_time=int(intel['last_closed_bar'])
    # Never store candles whose close is later than this record's nominal 15m window.
    with connect() as db:
        r=db.execute('INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)',
          (ticker,values['elapsed_minute'],candle_time,datetime.now(timezone.utc).isoformat(),
           json.dumps(values,sort_keys=True), tv_call, intel.get('direction'),up_ask,down_ask))
        return r.rowcount==1

def unsettled_candidates(now_ts):
    with connect() as db:
        return db.execute('''SELECT s.ticker, MAX(s.candle_time) FROM snapshots s
          LEFT JOIN settlements o ON o.ticker=s.ticker
          WHERE o.ticker IS NULL GROUP BY s.ticker
          HAVING MAX(s.candle_time) < ? LIMIT 40''',(int(now_ts-120),)).fetchall()

def save_settlement(ticker, close_ts, result):
    # No assumptions about ambiguous, void or unsettled markets.
    result=str(result).strip().lower()
    if result not in ('yes','no'): return False
    with connect() as db:
        db.execute('INSERT OR IGNORE INTO settlements VALUES(?,?,?,?,?)',
            (ticker, close_ts, int(result=='yes'), datetime.now(timezone.utc).isoformat(), result))
    return True

def statistics():
    with connect() as db:
        markets=db.execute('SELECT COUNT(DISTINCT ticker) FROM snapshots').fetchone()[0]
        labels=db.execute('SELECT COUNT(*) FROM settlements').fetchone()[0]
        usable=db.execute('SELECT COUNT(DISTINCT s.ticker) FROM snapshots s JOIN settlements o ON s.ticker=o.ticker').fetchone()[0]
    metrics=None
    if METRICS_FILE.exists():
        try:metrics=json.loads(METRICS_FILE.read_text())
        except (ValueError,OSError): pass
    return {'observed_markets':markets,'settled_markets':labels,'trainable_markets':usable,
            'training_ready': usable>=250,'model_status':'CANDIDATE ONLY' if metrics else 'NOT TRAINED',
            'metrics':metrics,'live_model_enabled':False}

def train(min_markets=250, holdout_count=50):
    # One observation per ticker, prioritized 3m > 4m > 5m, no same-contract contamination.
    with connect() as db:
        rows=db.execute('''SELECT s.ticker,s.elapsed_minute,s.features,o.outcome,o.close_ts
          FROM snapshots s JOIN settlements o USING(ticker)
          WHERE s.elapsed_minute BETWEEN 2 AND 6
          ORDER BY o.close_ts,s.ticker,ABS(s.elapsed_minute-3),s.elapsed_minute''').fetchall()
    chosen={}
    for ticker, minute, feat, outcome, close in rows:
        if ticker not in chosen:chosen[ticker]=(json.loads(feat),int(outcome),float(close))
    entries=sorted(chosen.items(),key=lambda z:(z[1][2],z[0]))
    if len(entries)<max(min_markets, holdout_count*3):
        return {'trained':False,'reason':f'Need >= {max(min_markets,holdout_count*3)} eligible settled markets; have {len(entries)}'}
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    import joblib
    X=np.array([[f[k] for k in FEATURES] for _,(f,_,_) in entries],dtype=float)
    y=np.array([v[1] for _,v in entries],dtype=int)
    n=len(y); split=n-max(holdout_count,int(.2*n))
    if len(set(y[:split]))<2 or len(set(y[split:]))<2:
        return {'trained':False,'reason':'Not enough both-class examples in train/test chronological split'}
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000, C=.3))
    model.fit(X[:split],y[:split]); pred=model.predict_proba(X[split:])[:,1]
    base=float(y[:split].mean())
    brier=float(brier_score_loss(y[split:],pred))
    baseline_brier=float(brier_score_loss(y[split:],np.full(len(pred),base)))
    result={'trained':True,'trained_at':datetime.now(timezone.utc).isoformat(),
            'total_labeled_markets':n,'train_count':split,'holdout_count':n-split,
            'brier':round(brier,5),'baseline_brier':round(baseline_brier,5),
            'log_loss':round(float(log_loss(y[split:],pred,labels=[0,1])),5),
            'holdout_accuracy_at_0_5':round(float(((pred>=.5)==y[split:]).mean()),4),
            'holdout_up_rate':round(float(y[split:].mean()),4),
            'beats_baseline_brier': bool(brier<baseline_brier),
            'model_status':'CANDIDATE ONLY — NEVER USED FOR LIVE DECISIONS',
            'warning':'Exploratory chronologically held-out estimate; repeated retraining on the same holdout invalidates independent test claims.'}
    # Candidate saved for offline review only. NEVER connected to the dashboard's decision function.
    joblib.dump({'model':model,'features':FEATURES},MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    print(json.dumps(train(),indent=2))
