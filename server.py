"""Read-only Kalshi market + TradingView webhook bridge. NO ORDER PLACEMENT.
Run: pip install -r requirements.txt && export WEBHOOK_SECRET=... && uvicorn server:app --host 0.0.0.0 --port 8000
"""
import math
import secrets
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from threading import Lock

import requests
import json
import sqlite3
from collections import deque
from threading import Thread
from intelligence import evaluate
from training import record_snapshot, unsettled_candidates, save_settlement, statistics, train
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse

API = "https://external-api.kalshi.com/trade-api/v2"
SERIES = os.getenv("KALSHI_SERIES", "KXBTC15M")
SECRET = os.getenv("WEBHOOK_SECRET", "")
# This is only a *comparison input* — not a calibrated model estimate.
PROB = os.getenv("MODEL_WIN_PROBABILITY", "")
QTY = max(1, int(os.getenv("CONTRACT_COUNT", "1")))
FEE_MULT = Decimal(os.getenv("TAKER_FEE_MULTIPLIER", "1"))
MAX_AGE = 660  # Signal stays valid for current 15-minute market only
app = FastAPI(title="Kalshi BTC Read-Only Decision Monitor", docs_url=None, redoc_url=None, openapi_url=None)
security = HTTPBasic()

def require_password(credentials: HTTPBasicCredentials = Depends(security)):
    expected = os.getenv("DASHBOARD_PASSWORD", "")
    if not expected or not secrets.compare_digest(credentials.password.encode(), expected.encode()) or not secrets.compare_digest(credentials.username.encode(), b"btc"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid dashboard credentials", headers={"WWW-Authenticate": "Basic"})
    return True

lock = Lock()
last = {}
seen = set()
session = requests.Session()
DATA_PATH = os.getenv('DATA_PATH', os.path.join(os.getenv('DATA_DIR','./data'), 'btc_intelligence_observations.sqlite3'))
os.makedirs(os.path.dirname(os.path.abspath(DATA_PATH)), exist_ok=True)
flow_lock = Lock()
trade_events = deque(maxlen=6000)
flow_connected_at = None
candle_cache = {'at': 0, 'candles': [], 'error': None}
candle_lock = Lock()

def init_db():
    with sqlite3.connect(DATA_PATH) as db:
        db.execute('''CREATE TABLE IF NOT EXISTS observations (
          window TEXT NOT NULL, minute INTEGER NOT NULL, at_utc TEXT NOT NULL,
          tv_call TEXT, model_direction TEXT, model_state TEXT, score INTEGER,
          price REAL, reference REAL, kalshi_up_ask REAL, kalshi_down_ask REAL,
          features_json TEXT NOT NULL, PRIMARY KEY(window, minute))''')

def record_observation(k, t, intelligence):
    if not k or not intelligence.get('fresh'): return
    try:
        window = str(k['ticker']); minute = max(0, min(15, 15 - int(k['seconds_left'])//60))
        with sqlite3.connect(DATA_PATH, timeout=3) as db:
            db.execute('''INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
              (window, minute, datetime.now(timezone.utc).isoformat(),
               (t or {}).get('original_call'), intelligence.get('direction'), intelligence.get('state'),
               intelligence.get('margin_score'), intelligence.get('last_close'),
               k.get('kalshi_reference_strike'), (k.get('yes_up') or {}).get('ask'),
               (k.get('no_down') or {}).get('ask'), json.dumps(intelligence)))
    except (sqlite3.Error, ValueError, TypeError): pass

def trade_stream():
    """Optional public Coinbase trades stream. Reported trade side is exchange-specific."""
    import websocket
    def opened(ws):
        ws.send(json.dumps({'type':'subscribe','product_ids':['BTC-USD'],'channel':'market_trades'}))
    def message(ws, raw):
        global flow_connected_at
        try:
            data = json.loads(raw)
            if data.get('channel') != 'market_trades': return
            batch=[]
            for event in data.get('events',[]):
                for t in event.get('trades',[]):
                    if t.get('product_id')!='BTC-USD':continue
                    side=str(t.get('side','')).upper()
                    if side not in ('BUY','SELL'): continue
                    price=float(t['price']); size=float(t['size'])
                    if price<=0 or size<=0:continue
                    batch.append((time.time(),side,price*size))
            if batch:
                with flow_lock:
                    trade_events.extend(batch)
                    flow_connected_at=time.time()
        except (KeyError,TypeError,ValueError):return
    while True:
        try:
            ws=websocket.WebSocketApp('wss://advanced-trade-ws.coinbase.com',on_open=opened,on_message=message)
            ws.run_forever(ping_interval=20,ping_timeout=10)
        except Exception: pass
        time.sleep(8)

def recent_flow():
    now=time.time()
    with flow_lock:
        recent=[v for v in trade_events if now-v[0]<=60]
        stamp=flow_connected_at
    if not recent or stamp is None:return None
    return {'timestamp':stamp,'trades':len(recent),
            'buy_notional':sum(t[2] for t in recent if t[1]=='BUY'),
            'sell_notional':sum(t[2] for t in recent if t[1]=='SELL')}

def fetch_candles_cached():
    now=time.time()
    with candle_lock:
        if now-candle_cache['at']<22:return list(candle_cache['candles'])
    try:
        r = session.get('https://api.exchange.coinbase.com/products/BTC-USD/candles',
                        params={'granularity':60},headers={'User-Agent':'kalshi-btc-intelligence/1.0'},timeout=6)
        r.raise_for_status()
        data=r.json()
        clean=[{'time':int(v[0]),'low':float(v[1]),'high':float(v[2]),
                'open':float(v[3]),'close':float(v[4]),'volume':float(v[5])}
                for v in data if isinstance(v,list) and len(v)>=6]
        clean.sort(key=lambda row:row['time'])
        with candle_lock:candle_cache.update(at=now,candles=clean[-120:],error=None)
        return clean[-120:]
    except (requests.RequestException,ValueError,TypeError) as e:
        with candle_lock:candle_cache.update(at=now,error=type(e).__name__)
        return []

@app.on_event('startup')
def startup():
    init_db()
    if os.getenv('ENABLE_COINBASE_TRADE_FLOW','true').lower()=='true':
        Thread(target=trade_stream,daemon=True).start()
    if os.getenv('ENABLE_RESEARCH_POLLER','true').lower()=='true':
        Thread(target=research_poller,daemon=True).start()



def get_json(path, params=None):
    r = session.get(API + path, params=params, timeout=8)
    r.raise_for_status()
    return r.json()


def money(market, key):
    val = market.get(key + "_dollars")
    if val not in (None, ""):
        return float(val)
    old = market.get(key)
    return float(old) / 100 if old is not None else None


def dt(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def active_market():
    data = get_json("/markets", {"series_ticker": SERIES, "status": "open", "limit": 200})
    now = time.time()
    markets = [m for m in data.get("markets", []) if dt(m.get("close_time")) and dt(m["close_time"]) > now]
    # Find currently relevant 15-min window; reject future listings where possible.
    active = [m for m in markets if (dt(m.get("open_time")) or 0) <= now and dt(m["close_time"]) - now <= 900]
    if not active:
        return None
    return min(active, key=lambda m: dt(m["close_time"]))


def calc_fee(ask, count=QTY):
    # Kalshi standard quadratic taker fee, rounded UP to nearest cent per order.
    p = Decimal(str(ask))
    value = FEE_MULT * Decimal("0.07") * Decimal(count) * p * (1 - p)
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def side_data(ask):
    if ask is None or not (0 < ask < 1):
        return None
    fee = calc_fee(ask)
    cost = QTY * ask + fee
    breakeven = cost / QTY
    result = {"ask": round(ask, 4), "fee_total": fee, "all_in_cost": round(cost, 4),
              "break_even_probability": round(breakeven, 4), "win_profit": round(QTY - cost, 4),
              "lose_loss": round(-cost, 4)}
    try:
        p = float(PROB)
        if 0 <= p <= 1:
            result["manual_probability_assumption"] = p
            result["illustrative_ev_usd"] = round(p * QTY - cost, 4)
            # Do not report this as real model edge; uncalibrated assumptions are not actionable.
    except ValueError:
        pass
    return result


def best_depth(ticker):
    book = get_json("/markets/" + ticker + "/orderbook").get("orderbook_fp", {})
    data = {}
    for side in ("yes", "no"):
        bids = book.get(side + "_dollars") or []
        parsed = [(float(level[0]), float(level[1])) for level in bids if len(level) >= 2]
        data[side + "_best_bid"] = max((x[0] for x in parsed), default=None)
        data[side + "_bid_contracts"] = round(sum(x[1] for x in parsed), 2)
    data["contract_bid_depth_ratio"] = (round(data["yes_bid_contracts"] / data["no_bid_contracts"], 3)
                                           if data["no_bid_contracts"] > 0 else None)
    return data


@app.post("/webhook")
async def webhook(req: Request):
    if not SECRET or SECRET == "CHANGE_ME":
        raise HTTPException(503, "Configure WEBHOOK_SECRET first")
    payload = await req.json()
    if not secrets.compare_digest(str(payload.get("secret", "")), SECRET):
        raise HTTPException(403, "Invalid secret")
    try:
        ts = int(payload["bar_time_ms"]) / 1000
        score = int(payload["score"])
        price = float(payload["tv_price"])
        signal = str(payload["signal"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "Invalid payload")
    now = time.time()
    if abs(ts - now) > 120 or signal not in ("PENDING", "UP", "DOWN", "WAIT") or not (0 < price < 1e7) or not (-6 <= score <= 6):
        raise HTTPException(400, "Invalid/stale bar")
    slot = int((ts - 1) // 900)
    safe = {k:v for k,v in payload.items() if k != "secret"}
    safe["received_at"] = datetime.now(timezone.utc).isoformat()
    with lock:
        previous_slot = last.get("slot")
        if previous_slot is not None and slot < previous_slot:
            return {"accepted": False, "reason": "older window"}
        if previous_slot != slot:
            last.clear()
            last["slot"] = slot
            last["original_call"] = None
            last["call_at"] = None
        # Never rewrite the original directional call after it has fired.
        # One-minute UPDATE events can change the risk level, not the frozen entry signal.
        if last["original_call"] is None and signal in ("UP", "DOWN", "WAIT"):
            last["original_call"] = signal
            last["call_at"] = safe.get("decision_elapsed_minutes")
        last.update(safe)
    return {"accepted": True, "window": slot, "original_call": last.get("original_call")}

def research_tick():
    """Capture a timestamped 1m snapshot (no future bars) and label matured contracts."""
    now = time.time()
    try:
        m = active_market()
        if m:
            seconds_left = int(dt(m['close_time'])-now)
            ref = m.get('floor_strike')
            up = money(m,'yes_ask'); down = money(m,'no_ask')
            rows = fetch_candles_cached()
            with lock: tv = dict(last)
            if rows and up and down and ref is not None:
                intel=evaluate(rows,reference=ref,remaining_seconds=seconds_left,
                    tradingview=tv,flow=recent_flow())
                record_snapshot(m['ticker'],seconds_left,intel,ref,up,down,tv.get('original_call'))
    except (requests.RequestException,ValueError,TypeError,KeyError,sqlite3.Error):
        pass # Retry next cycle, never write synthetic candles or labels.
    for ticker, _ in unsettled_candidates(now):
        try:
            # Do not assign a result based on Coinbase price or dashboard guess.
            market = get_json('/markets/'+ticker).get('market',{})
            close=dt(market.get('close_time'))
            if close and close < now-60 and market.get('status') in ('settled','finalized'):
                save_settlement(ticker,close,market.get('result'))
        except requests.RequestException:
            try:
                market=get_json('/historical/markets/'+ticker).get('market',{})
                close=dt(market.get('close_time'))
                if close and close<now-60 and market.get('status') in ('settled','finalized'):
                    save_settlement(ticker,close,market.get('result'))
            except requests.RequestException: pass
        except (ValueError,TypeError,KeyError): pass

def research_poller():
    """Single-instance service poller. Keeps collecting when iPhone is closed."""
    next_train=0
    while True:
        try:
            research_tick()
            if os.getenv('AUTO_TRAIN_CANDIDATE','true').lower()=='true' and time.time()>next_train:
                train()
                next_train=time.time()+86400
        except Exception:
            pass
        time.sleep(50)

@app.get('/training-status')
def training_status(auth=Depends(require_password)):
    return statistics()

@app.get('/candles')
def candles(auth=Depends(require_password)):
    rows=fetch_candles_cached()
    if not rows:raise HTTPException(503,'Spot candle feed unavailable')
    return {'source':'Coinbase BTC-USD spot proxy','candles':rows[-90:]}

@app.get('/observations')
def observations(auth=Depends(require_password)):
    with sqlite3.connect(DATA_PATH) as db:
        rows=db.execute('SELECT window,minute,at_utc,tv_call,model_direction,model_state,score,price,reference,kalshi_up_ask,kalshi_down_ask FROM observations ORDER BY at_utc DESC LIMIT 100').fetchall()
    keys=['window','minute','at_utc','tv_call','model_direction','model_state','score','price','reference','kalshi_up_ask','kalshi_down_ask']
    return {'observations':[dict(zip(keys,r)) for r in rows],
            'warning':'Observations are not settled outcomes; performance cannot be computed from this data alone.'}

@app.get("/status")
def status(auth=Depends(require_password)):
    with lock:
        tv = dict(last)
    if tv:
        age = time.time() - dt(tv["received_at"])
        tv["age_seconds"] = round(age)
        tv["fresh"] = 0 <= age < 95
    response = {"timestamp": datetime.now(timezone.utc).isoformat(), "tradingview": tv or None,
                "message": "Comparison only: no trading orders, no calibrated probability model."}
    try:
        market = active_market()
        if not market:
            response["kalshi"] = None
            response["decision"] = "WAIT: no current 15-minute market verified"
            return response
        yes = money(market, "yes_ask")
        no = money(market, "no_ask")
        kd = {"ticker": market["ticker"], "title": market.get("title"), "close_time": market.get("close_time"),
              "kalshi_reference_strike": market.get("floor_strike"),
              "strike_type": market.get("strike_type"),
              "reference_is_verified": market.get("floor_strike") is not None,
              "rules_primary": market.get("rules_primary"),
              "seconds_left": round(dt(market["close_time"]) - time.time()),
              "yes_up": side_data(yes), "no_down": side_data(no),
              "volume": market.get("volume_fp", market.get("volume"))}
        try:
            kd["orderbook"] = best_depth(market["ticker"])
        except requests.RequestException as e:
            kd["orderbook_error"] = type(e).__name__
        response["kalshi"] = kd
        rows=fetch_candles_cached()
        try:
            intelligence=evaluate(rows,reference=kd.get('kalshi_reference_strike'),
                remaining_seconds=kd['seconds_left'],tradingview=tv,flow=recent_flow())
        except (KeyError,ValueError,TypeError,ZeroDivisionError) as e:
            intelligence={'state':'UNAVAILABLE','reason':'Engine could not analyze current candles','strategies':[],'warnings':[],'fresh':False}
        response['intelligence']=intelligence
        record_observation(kd,tv,intelligence)
        # Prevent stale direction carrying over to next 15m Kalshi market.
        tv_slot_end = (int((int(tv.get("bar_time_ms", 0))/1000 - 1) // 900) + 1) * 900 if tv else 0
        match_window = abs(tv_slot_end - dt(market["close_time"])) <= 65 if tv_slot_end else False
        if not tv or not tv.get("fresh") or not match_window:
            response["decision"] = "WAIT: TradingView call not current for this Kalshi contract"
        elif kd["seconds_left"] < 180:
            response["decision"] = "WAIT: window nearly over"
        elif tv.get("original_call") in (None, "WAIT"):
            response["decision"] = "WAIT: no confirmed directional call"
        elif tv["signal"] == "WAIT":
            response["decision"] = "WAIT: weak or conflicting BTC trend"
        else:
            selected = kd["yes_up"] if tv["original_call"] == "UP" else kd["no_down"]
            if not selected:
                response["decision"] = "WAIT: no executable ask available"
            else:
                response["decision"] = ("WATCH " + tv["original_call"] + ": market ask + break-even shown; "
                                        "NO VERIFIED EDGE without calibrated win probabilities")
        # Risk grade is a live REASSESSMENT, never a claim of a calibrated probability.
        initial = tv.get("original_call") if tv else None
        warnings = []
        if initial in ("UP", "DOWN") and tv.get("fresh") and match_window:
            aligned = int(tv.get("score", 0)) * (1 if initial == "UP" else -1)
            if aligned <= 0:
                warnings.append("Momentum score reversed against the original call")
            elif aligned <= 2:
                warnings.append("Momentum has weakened")
            if (initial == "UP" and (tv.get("exhaust_up") in (True, "true") or tv.get("down_momentum") in (True, "true"))) or (initial == "DOWN" and (tv.get("exhaust_down") in (True, "true") or tv.get("up_momentum") in (True, "true"))):
                warnings.append("One-minute reversal / exhaustion warning")
            strike = kd["kalshi_reference_strike"]
            if strike is not None:
                distance = float(tv.get("tv_price", 0)) - float(strike)
                response["spot_minus_kalshi_threshold"] = round(distance, 2)
                # TradingView proxy spot is not the official Kalshi CF Benchmarks index.
                try:
                    one_min_range = abs(float(tv.get("range_high")) - float(tv.get("range_low"))) / 24
                    threshold_buffer = max(20.0, one_min_range)
                except (ValueError, TypeError):
                    threshold_buffer = 35.0
                if abs(distance) < threshold_buffer:
                    warnings.append("Exchange BTC near Kalshi threshold; settlement risk elevated")
                if (initial == "UP" and distance < 0) or (initial == "DOWN" and distance > 0):
                    warnings.append("Exchange BTC now on opposite side of Kalshi threshold")
            if kd["seconds_left"] <= 180:
                warnings.append("Final 3 minutes: reversal and settlement risk")
        response["warnings"] = warnings
        response["risk_state"] = "CAUTION" if warnings else ("NORMAL" if response["decision"].startswith("WATCH ") else "PASS")
        if warnings and response["decision"].startswith("WATCH "):
            response["decision"] = "CAUTION: " + "; ".join(warnings)
        if intelligence.get('state') == 'CONFLICT' and response['decision'].startswith('WATCH '):
            response['decision']='CAUTION: Independent price-action analysis disagrees with TradingView'
            response['risk_state']='CAUTION'
        elif intelligence.get('state')=='UNAVAILABLE' and response['decision'].startswith('WATCH '):
            response['decision']='CAUTION: Independent market data is unavailable'
            response['risk_state']='CAUTION'
        return response
    except (requests.RequestException, ValueError, KeyError) as e:
        response["kalshi_error"] = type(e).__name__ + ": " + str(e)[:160]
        response["decision"] = "WAIT: live Kalshi data unavailable"
        return response


@app.get("/", response_class=HTMLResponse)
def dashboard(auth=Depends(require_password)):
    from pathlib import Path
    return Path(__file__).with_name("dashboard.html").read_text()

@app.get("/health")
def health():
    return {"ok": True}
