"""Simple file-based cache for raw emails and parsed results."""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class EmailCache:
    """File-based cache storing raw HTML and parsed JSON per email ID."""

    def __init__(self, cache_dir: str = "data/email_cache"):
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.cache_dir / "raw"
        self.parsed_dir = self.cache_dir / "parsed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    def _raw_path(self, msg_id: str) -> Path:
        return self.raw_dir / f"{msg_id}.json"

    def _parsed_path(self, msg_id: str) -> Path:
        return self.parsed_dir / f"{msg_id}.json"

    # -- Raw emails --

    def has_raw(self, msg_id: str) -> bool:
        return self._raw_path(msg_id).exists()

    def save_raw(self, msg_id: str, data: Dict) -> None:
        with open(self._raw_path(msg_id), "w") as f:
            json.dump(data, f)

    def load_raw(self, msg_id: str) -> Optional[Dict]:
        path = self._raw_path(msg_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # -- Parsed results --

    def has_parsed(self, msg_id: str) -> bool:
        return self._parsed_path(msg_id).exists()

    def save_parsed(self, msg_id: str, data: Dict) -> None:
        with open(self._parsed_path(msg_id), "w") as f:
            json.dump(data, f)

    def load_parsed(self, msg_id: str) -> Optional[Dict]:
        path = self._parsed_path(msg_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_cached_ids(self) -> list:
        """List all cached raw email IDs."""
        return [p.stem for p in self.raw_dir.glob("*.json")]
