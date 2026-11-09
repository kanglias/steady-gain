"""A 股日线行情，以及本地政策 / 宏观事件日历。"""

from data.bars import CacheMissError, DataError, FetchError, fetch_a_share_daily, get_bars
from data.benchmark import CSI300_CODE, CSI300_NAME, MISSING_CSI300_MESSAGE, csi300_excess, get_csi300, read_csi300_offline
from data.cache import default_cache_dir
from data.events import load_events
from data.universe import available_universes, universe_names, universe_note, universe_symbols

__all__ = [
    "CSI300_CODE",
    "CSI300_NAME",
    "MISSING_CSI300_MESSAGE",
    "csi300_excess",
    "CacheMissError",
    "DataError",
    "FetchError",
    "available_universes",
    "get_csi300",
    "read_csi300_offline",
    "default_cache_dir",
    "fetch_a_share_daily",
    "get_bars",
    "load_events",
    "universe_names",
    "universe_note",
    "universe_symbols",
]
