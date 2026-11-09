"""静态研究样本。不是实时指数成分，也不会随行情软件自动更新。

`demo` 只有两只高流动性股票，仓库缓存里通常已有，离线能跑。
`banks`、`consumer` 是板块样本。`extra` 再放保险、电力、券商各一只。
`liquid` 是这三组合并后的静态名单，按常见大盘成交挑选，不是打分选股。第一次需要联网拉日线。
`watch` 是盯盘用的两只 ETF，不并进上面的股票池。
"""

from __future__ import annotations

from data.normalize import normalize_symbol

UNIVERSES: dict[str, dict[str, object]] = {
    "demo": {
        "title": "演示",
        "note": "平安银行、贵州茅台。离线缓存通常盖得住。",
        "names": {"000001": "平安银行", "600519": "贵州茅台"},
    },
    "banks": {
        "title": "银行",
        "note": "几只大盘银行，当作板块样本，不是指数调样清单。",
        "names": {
            "000001": "平安银行",
            "600036": "招商银行",
            "601166": "兴业银行",
            "601398": "工商银行",
        },
    },
    "consumer": {
        "title": "消费",
        "note": "白酒和乳品里的几只常见标的，同样只是静态样本。",
        "names": {
            "600519": "贵州茅台",
            "000858": "五粮液",
            "600887": "伊利股份",
            "000568": "泸州老窖",
        },
    },
    "extra": {
        "title": "补充",
        "note": "保险、电力、券商里各放一只常见大盘股，用来把 liquid 从银行和消费再铺开。按公开常见的成交活跃度挑选，不是打分选股，也不是为了靠近每月 3%。",
        "names": {
            "601318": "中国平安",
            "600900": "长江电力",
            "600030": "中信证券",
        },
    },
    "watch": {
        "title": "盯盘",
        "note": "中证A500ETF、恒生科技ETF。盯盘名单，不并进 demo 或 liquid。",
        "names": {"159338": "中证A500ETF", "513010": "恒生科技ETF"},
    },
}


def available_universes() -> list[str]:
    names = list(UNIVERSES)
    if "liquid" not in names:
        names.append("liquid")
    return names


def universe_names(name: str = "demo") -> dict[str, str]:
    if name == "liquid":
        merged: dict[str, str] = {}
        merged.update(universe_names("banks"))
        merged.update(universe_names("consumer"))
        merged.update(universe_names("extra"))
        return merged
    if name not in UNIVERSES:
        known = "、".join(available_universes())
        raise KeyError(f"没有名为 {name} 的样本。可选：{known}")
    raw = UNIVERSES[name]["names"]
    return {normalize_symbol(code): label for code, label in dict(raw).items()}


def universe_symbols(name: str = "demo") -> list[str]:
    return list(universe_names(name))


def universe_note(name: str = "demo") -> str:
    if name == "liquid":
        return (
            "银行、消费，再加上保险、电力、券商各一只。"
            "按常见大盘成交挑选的静态名单，不是指数成分，不是打分选股，也不是为了靠近每月 3%。"
        )
    if name not in UNIVERSES:
        universe_names(name)
    return str(UNIVERSES[name]["note"])
