from __future__ import annotations

from data import available_universes, universe_names, universe_symbols


def test_samples_extend_past_the_two_demo_names():
    assert universe_symbols("demo") == ["000001", "600519"]
    liquid = universe_symbols("liquid")
    assert "600036" in liquid
    assert "000858" in liquid
    assert "601318" in liquid
    assert "600900" in liquid
    assert len(liquid) > 2
    assert set(available_universes()) >= {"demo", "banks", "consumer", "liquid"}
    assert universe_names("banks")["601398"] == "工商银行"


def test_watch_etfs_stay_out_of_the_stock_samples():
    assert universe_symbols("watch") == ["159338", "513010"]
    assert universe_names("watch")["159338"] == "中证A500ETF"
    assert universe_names("watch")["513010"] == "恒生科技ETF"
    assert "watch" in available_universes()
    for sample in ("demo", "banks", "consumer", "extra", "liquid"):
        symbols = set(universe_symbols(sample))
        assert "159338" not in symbols
        assert "513010" not in symbols