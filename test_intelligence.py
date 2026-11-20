import unittest
from intelligence import evaluate

class IntelligenceTests(unittest.TestCase):
    def candles(self, drift=2):
        return [{'time':i*60,'open':90000+i*drift,'close':90000+(i+1)*drift,
                 'high':90003+(i+1)*drift,'low':89997+i*drift,'volume':2.0} for i in range(70)]
    def test_uptrend_uses_only_closed_bars(self):
        data=self.candles()
        s=evaluate(data,now=70*60+1,remaining_seconds=600)
        self.assertEqual(s['direction'],'UP')
        self.assertTrue(s['fresh'])
    def test_stale(self):
        s=evaluate(self.candles(),now=72*60+30)
        self.assertEqual(s['state'],'UNAVAILABLE')
    def test_confirmation_requires_current_tv(self):
        a=evaluate(self.candles(),now=70*60+1,tradingview={'original_call':'UP','fresh':True})
        b=evaluate(self.candles(),now=70*60+1,tradingview={'original_call':'DOWN','fresh':True})
        self.assertTrue(a['agreement_with_tv'])
        self.assertEqual(b['state'],'CONFLICT')
    def test_last_minute_warning(self):
        a=evaluate(self.candles(),now=70*60+1,remaining_seconds=45)
        self.assertTrue(any('Final 3 minutes' in x for x in a['warnings']))

if __name__=='__main__': unittest.main()
