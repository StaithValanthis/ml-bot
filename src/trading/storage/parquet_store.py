from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trading.journal.ledger import LedgerEvent


class ParquetArchiveStore:
    """
    Append-style parquet archival writer.

    Baseline strategy:
    - one parquet file per event_type per UTC date
    - append implemented by read+concat+write for simplicity/reliability
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    async def write_event(self, event: LedgerEvent) -> None:
        date_key = event.timestamp.strftime("%Y-%m-%d")
        file_path = self._root_dir / f"{event.event_type}_{date_key}.parquet"
        row = {
            "event_type": event.event_type,
            "ts_utc": event.timestamp.isoformat(),
            "payload_json": _payload_to_json(event.payload),
        }
        new_table = pa.Table.from_pylist([row])
        if file_path.exists():
            existing = pq.read_table(file_path)
            combined = pa.concat_tables([existing, new_table], promote_options="default")
            pq.write_table(combined, file_path)
        else:
            pq.write_table(new_table, file_path)


def _payload_to_json(payload: dict[str, Any]) -> str:
    import json

    from trading.util.json_util import _json_default

    return json.dumps(payload, separators=(",", ":"), default=_json_default)
