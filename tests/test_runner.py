from kalshi_mm.exchange import KalshiOrderBook
from kalshi_mm.runner import FillTracker, PaperExecutor, TickerWatch
from kalshi_mm.strategy import KalshiQuote, KalshiQuotePlan


def _book() -> KalshiOrderBook:
    book = KalshiOrderBook("T")
    book.apply(
        {
            "type": "orderbook_snapshot",
            "seq": 1,
            "msg": {
                "market_ticker": "T",
                "yes_dollars_fp": [["0.40", "5"]],
                "no_dollars_fp": [["0.60", "5"]],
            },
        }
    )
    return book


def test_paper_executor_fills_from_trade_tape_with_fees():
    executor = PaperExecutor(book=_book(), maker_fee_multiplier=1.0)
    plan = KalshiQuotePlan((), (KalshiQuote("bid", 0.49, 5, 0.01),), "post")
    executor.execute(plan, ticker="T", expiration_time=0)
    assert len(executor.resting()) == 1
    executor.on_trade_envelope(
        {
            "type": "trade",
            "msg": {
                "market_ticker": "T",
                "yes_price_dollars": "0.49",
                "count_fp": "10",
                "taker_book_side": "ask",
                "ts_ms": 1,
            },
        },
        "T",
    )
    assert executor.position == 5.0
    assert executor.fees_paid > 0.0
    assert executor.cash_flow < 0.0
    assert executor.fill_count == 1


def test_paper_executor_ignores_other_markets():
    executor = PaperExecutor(book=_book(), maker_fee_multiplier=0.0)
    executor.execute(KalshiQuotePlan((), (KalshiQuote("bid", 0.49, 5, 0.01),), "post"), ticker="T", expiration_time=0)
    executor.on_trade_envelope(
        {
            "type": "trade",
            "msg": {
                "market_ticker": "OTHER",
                "yes_price_dollars": "0.49",
                "count_fp": "10",
                "taker_book_side": "ask",
                "ts_ms": 1,
            },
        },
        "T",
    )
    assert executor.position == 0.0


def test_fill_tracker_estimates_position_and_reports_divergence():
    tracker = FillTracker("T")
    tracker.on_envelope({"type": "fill", "msg": {"market_ticker": "T", "count_fp": "2", "book_side": "bid"}})
    tracker.on_envelope({"type": "fill", "msg": {"market_ticker": "T", "count_fp": "1", "action": "sell", "side": "yes"}})
    tracker.on_envelope({"type": "fill", "msg": {"market_ticker": "OTHER", "count_fp": "9", "book_side": "bid"}})
    assert tracker.position() == 1.0
    assert tracker.sync(1.0) == 0.0
    assert tracker.sync(3.0) == 2.0
    assert tracker.position() == 3.0


def test_fill_tracker_no_side_buy_reduces_yes_position():
    tracker = FillTracker("T")
    tracker.on_envelope({"type": "fill", "msg": {"market_ticker": "T", "count_fp": "4", "action": "buy", "side": "no"}})
    assert tracker.position() == -4.0


def test_ticker_watch_requires_sustained_divergence():
    watch = TickerWatch("T", tolerance_ticks=2.0, sustain_s=1.0)
    watch.on_envelope(
        {"type": "ticker", "msg": {"market_ticker": "T", "yes_bid_dollars": "0.50", "yes_ask_dollars": "0.51", "ts_ms": 1_000}}
    )
    assert not watch.divergence_exceeded(book_bid=0.50, book_ask=0.51, tick=0.01, now_ms=1_000)
    # First divergent observation only arms the timer.
    assert not watch.divergence_exceeded(book_bid=0.40, book_ask=0.41, tick=0.01, now_ms=1_500)
    assert watch.divergence_exceeded(book_bid=0.40, book_ask=0.41, tick=0.01, now_ms=2_600)
    # Agreement disarms it again.
    assert not watch.divergence_exceeded(book_bid=0.50, book_ask=0.51, tick=0.01, now_ms=2_700)
    assert not watch.divergence_exceeded(book_bid=0.40, book_ask=0.41, tick=0.01, now_ms=2_800)
