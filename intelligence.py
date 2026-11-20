"""Deterministic, causal BTC 1m feature engine. No ML training/verified edge.
All bar-based features exclude the currently forming candle by using the latest closed bar.
"""
from statistics import mean
from math import sqrt
from datetime import datetime, timezone


def _ema(xs, n):
    v = xs[0]
    k = 2 / (n + 1)
    out = []
    for x in xs:
        v += k * (x - v)
        out.append(v)
    return out


def _rsi(xs, period=14):
    if len(xs) <= period: return 50
    dif = [b-a for a,b in zip(xs[:-1],xs[1:])]
    gains=[max(d,0) for d in dif[-period:]]; losses=[max(-d,0) for d in dif[-period:]]
    g=mean(gains); l=mean(losses)
    return 100 if l==0 and g>0 else 50 if l==g==0 else 100-100/(1+g/l)


def evaluate(candles, *, reference=None, remaining_seconds=None, tradingview=None, flow=None, now=None):
    """Returns transparent strategy votes and warnings, NOT a trained win probability.

    Expects candles with time (UNIX start second), OHLCV. In-progress candle stripped
    by strict c['time'] + 60 <= now. Signals are only for observation.
    """
    now = now or datetime.now(timezone.utc).timestamp()
    closed = sorted([c for c in candles if c['time']+60 <= now],key=lambda c:c['time'])
    if len(closed)<35:
        return {"state":"UNAVAILABLE","reason":"Need >=35 completed one-minute candles","strategies":[],"warnings":[],"fresh":False}
    if now - closed[-1]['time'] > 155:
        return {"state":"UNAVAILABLE","reason":"Coinbase 1-minute candles are stale","strategies":[],"warnings":[],"fresh":False}
    p=[float(c['close']) for c in closed]; h=[float(c['high']) for c in closed]; l=[float(c['low']) for c in closed]; v=[float(c['volume']) for c in closed]
    opens=[float(c['open']) for c in closed]
    ema9=_ema(p,9); ema20=_ema(p,20); ema50=_ema(p,50)
    tr=[max(h[i]-l[i],abs(h[i]-p[i-1]),abs(l[i]-p[i-1])) for i in range(1,len(p))]
    atr=max(mean(tr[-14:]),p[-1]*0.00001)
    volmean=mean(v[-21:-1]); volratio=v[-1]/volmean if volmean else 0
    ret3=p[-1]-p[-4]; range_hi=max(h[-25:-1]); range_lo=min(l[-25:-1]); midpoint=(range_hi+range_lo)/2
    width=(range_hi-range_lo)/atr
    chop=width<5.0 or abs(ema9[-1]-ema20[-1])<0.15*atr
    rsi=_rsi(p)
    body=p[-1]-opens[-1]; upper=h[-1]-max(p[-1],opens[-1]); lower=min(p[-1],opens[-1])-l[-1]
    stretched=(p[-1]-ema9[-1])/atr
    entries=[]
    def vote(name, signal, why, weight=1):
        entries.append({'name':name,'direction':signal,'why':why,'weight':weight})
    vote('EMA trend', 'UP' if ema9[-1]>ema20[-1] and p[-1]>ema20[-1] else 'DOWN' if ema9[-1]<ema20[-1] and p[-1]<ema20[-1] else 'NEUTRAL','9/20 EMA alignment',2)
    vote('Structure', 'UP' if min(l[-5:-1])>min(l[-10:-5]) and max(h[-5:-1])>max(h[-10:-5]) else 'DOWN' if max(h[-5:-1])<max(h[-10:-5]) and min(l[-5:-1])<min(l[-10:-5]) else 'NEUTRAL','Recent higher highs/lows or lower highs/lows',2)
    vote('3-minute impulse', 'UP' if ret3>0.55*atr else 'DOWN' if ret3< -0.55*atr else 'NEUTRAL','3-minute close-to-close impulse',1)
    vote('Volume confirmation', 'UP' if body>0.22*atr and volratio>1.10 else 'DOWN' if body< -0.22*atr and volratio>1.10 else 'NEUTRAL','Body strength on above-average volume',1)
    # Prior rolling levels exclude latest candle: no repaint of breakout threshold.
    vote('Breakout / failure', 'UP' if p[-1]>range_hi+0.08*atr else 'DOWN' if p[-1]<range_lo-0.08*atr else 'NEUTRAL','Close outside previous 24-bar range',2)
    rej_up = lower>1.4*max(abs(body),0.05*atr) and p[-1]>opens[-1] and l[-1]<=range_lo+0.4*atr
    rej_down = upper>1.4*max(abs(body),0.05*atr) and p[-1]<opens[-1] and h[-1]>=range_hi-0.4*atr
    vote('Range rejection','UP' if rej_up else 'DOWN' if rej_down else 'NEUTRAL','Rejected range edge with candle wick',2)
    pull_up=min(l[-3:])<=ema9[-1]+0.25*atr and p[-1]>ema9[-1] and body>0
    pull_down=max(h[-3:])>=ema9[-1]-0.25*atr and p[-1]<ema9[-1] and body<0
    vote('Pullback defense','UP' if pull_up else 'DOWN' if pull_down else 'NEUTRAL','Reclaimed or rejected fast EMA after testing it',1)
    warnings=[]
    if abs(stretched)>2.2: warnings.append('Extended >2.2 ATR from EMA9; chasing risk')
    if rsi>75 and p[-1]<p[-2]:warnings.append('Overbought momentum rolling over')
    if rsi<25 and p[-1]>p[-2]:warnings.append('Oversold momentum reversing')
    if rej_up and stretched<0: warnings.append('Bullish wick at support')
    if rej_down and stretched>0: warnings.append('Bearish wick at resistance')
    if chop: warnings.append('Range/chop conditions: respect range edges')
    if reference is not None:
        try:
            diff=p[-1]-float(reference)
            if abs(diff)<1.5*atr: warnings.append('BTC spot proxy near Kalshi settlement threshold')
        except (ValueError,TypeError): diff=None
    else: diff=None
    if remaining_seconds is not None and remaining_seconds<=180:
        warnings.append('Final 3 minutes: late reversal/settlement risk')
    up=sum(e['weight'] for e in entries if e['direction']=='UP')
    down=sum(e['weight'] for e in entries if e['direction']=='DOWN')
    margin=up-down
    side='UP' if margin>=3 else 'DOWN' if margin<=-3 else 'PASS'
    flow_status='not connected'
    flow_data=None
    if flow:
        age=now-float(flow.get('timestamp',0))
        if 0<=age<=25 and flow.get('trades',0)>=10:
            net=float(flow.get('buy_notional',0))-float(flow.get('sell_notional',0))
            gross=float(flow.get('buy_notional',0))+float(flow.get('sell_notional',0))
            ratio=net/gross if gross else 0
            flow_data={'buy_notional_usd':round(float(flow.get('buy_notional',0))),
                       'sell_notional_usd':round(float(flow.get('sell_notional',0))),
                       'net_fraction':round(ratio,3),'observations':flow['trades'],'seconds_old':round(age)}
            flow_status='fresh'
            if side=='UP' and ratio< -0.18: warnings.append('Coinbase reported trade-side flow opposes UP')
            if side=='DOWN' and ratio>0.18: warnings.append('Coinbase reported trade-side flow opposes DOWN')
        else:flow_status='stale/insufficient'
    # Don't silently override the original TV call. Require agreement to promote it.
    tv_dir=tradingview.get('original_call') if tradingview and tradingview.get('fresh') else None
    agreement=(tv_dir in ('UP','DOWN') and side==tv_dir)
    opposite=(tv_dir in ('UP','DOWN') and side in ('UP','DOWN') and side!=tv_dir)
    if opposite: warnings.append('Independent candle analysis disagrees with TradingView')
    if tv_dir in ('UP','DOWN') and side=='PASS':warnings.append('Independent analysis lacks directional confirmation')
    if diff is not None and tv_dir in ('UP','DOWN') and ((tv_dir=='UP' and diff<0) or (tv_dir=='DOWN' and diff>0)):
        warnings.append('BTC spot proxy currently on losing side of Kalshi threshold')
    state='CONFIRMED' if agreement and not opposite and not (remaining_seconds is not None and remaining_seconds<120) else 'CONFLICT' if opposite else 'OBSERVE'
    return {'state':state,'fresh':True,'direction':side,'agreement_with_tv':agreement,'tv_direction':tv_dir,
            'margin_score':margin,'up_points':up,'down_points':down,
            'strategies':entries,'warnings':warnings,
            'market_type':'RANGE' if chop else 'TREND','rsi':round(rsi,1),'atr':round(atr,2),'volume_multiple':round(volratio,2),
            'range_high':round(range_hi,2),'range_low':round(range_lo,2),'range_mid':round(midpoint,2),
            'ema9':round(ema9[-1],2),'ema20':round(ema20[-1],2),'ema50':round(ema50[-1],2),
            'last_close':p[-1],'spot_minus_threshold':round(diff,2) if diff is not None else None,
            'flow_status':flow_status,'flow':flow_data,'last_closed_bar':closed[-1]['time'],
            'disclaimer':'Heuristic strategy votes, not trained probabilities. Coinbase price/flow is not Kalshi settlement index.'}
