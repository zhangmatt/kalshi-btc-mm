import json

from kalshi_mm.exchange import BrtiState, KalshiOrderBook
from kalshi_mm.websocket import KalshiWebSocket


class FakeSigner:
    def headers(self, *_args, **_kwargs):
        return {}


class FakeSocket:
    def __init__(self):
        self.sent = []

    def send(self, value):
        self.sent.append(json.loads(value))


def test_sequence_gap_requests_snapshot_and_disconnect_invalidates_book():
    book = KalshiOrderBook("T")
    stream = KalshiWebSocket(signer=FakeSigner(), market_ticker="T", order_book=book, brti=BrtiState())
    socket = FakeSocket()
    stream._on_message(
        socket,
        json.dumps(
            {
                "type": "orderbook_snapshot",
                "sid": 7,
                "seq": 1,
                "msg": {"market_ticker": "T", "yes_dollars_fp": [["0.49", "5"]], "no_dollars_fp": []},
            }
        ),
    )
    assert book.snapshot().valid
    stream._on_message(
        socket,
        json.dumps(
            {
                "type": "orderbook_delta",
                "sid": 7,
                "seq": 3,
                "msg": {"market_ticker": "T", "side": "yes", "price_dollars": "0.49", "delta_fp": "1"},
            }
        ),
    )
    assert socket.sent[-1]["params"]["sids"] == [7]
    assert socket.sent[-1]["params"]["action"] == "get_snapshot"
    stream._on_close(socket, 1006, "disconnect")
    assert not book.snapshot().valid


def test_subscriptions_cover_market_lifecycle_and_private_ids_do_not_collide():
    stream = KalshiWebSocket(
        signer=FakeSigner(),
        market_ticker="T",
        order_book=KalshiOrderBook("T"),
        brti=BrtiState(),
        include_private=True,
    )
    subscriptions = stream._subscriptions()
    assert len({value["id"] for value in subscriptions}) == len(subscriptions)
    assert any(value["params"]["channels"] == ["market_lifecycle_v2"] for value in subscriptions)
