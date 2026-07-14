from kalshi_mm.replay import ConservativeQueueSimulator
from kalshi_mm.strategy import KalshiQuote


class Bbo:
    bid = 0.49
    ask = 0.51
    bid_size = 10
    ask_size = 12


def test_queue_simulator_requires_trades_to_clear_queue_ahead():
    simulator = ConservativeQueueSimulator()
    simulator.replace(
        cancel_ids=[],
        quotes=[KalshiQuote("bid", 0.49, 5, 0.01)],
        bbo=Bbo(),
    )
    assert simulator.trade(ts_ms=1, price=0.49, count=9, taker_book_side="ask", fair_yes=0.5) == []
    fills = simulator.trade(ts_ms=2, price=0.49, count=4, taker_book_side="ask", fair_yes=0.5)
    assert len(fills) == 1
    assert fills[0].count == 3
