"""DayState and DashboardState dataclasses with JSON serialization."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any


@dataclass
class DayState:
    date: str               # ISO date string "YYYY-MM-DD"
    open: float
    high: float
    low: float
    close: float
    volume: float
    risk_score: float
    risk_regime: int        # 0=low 1=elevated 2=high 3=extreme
    range_stress: float
    anomaly_intensity: float
    change_momentum: float
    anomaly_score: float
    return_cusum_score: float
    anomaly_ewma_z: float
    p_bear: float
    p_balance: float
    p_bull: float
    predicted_regime: int   # 0=BEAR 1=BALANCE 2=BULL
    hmm_state: int
    hmm_bull_prob: float
    hmm_bear_prob: float
    corr_nq_10d: float
    corr_nq_20d: float
    corr_nq_60d: float
    corr_zn_10d: float
    corr_zn_20d: float
    corr_zn_60d: float
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class DashboardState:
    computed_at: str        # ISO datetime string
    as_of_date: str         # ISO date string
    today: DayState | None
    history: list[DayState] = field(default_factory=list)
    oos_predictions: list[int] = field(default_factory=list)
    oos_actuals: list[int] = field(default_factory=list)
    oos_probas: list[list[float]] = field(default_factory=list)
    oos_dates: list[str] = field(default_factory=list)
    intraday_signals: list[dict] = field(default_factory=list)
    model_accuracy: float = 0.0
    model_f1: float = 0.0
    refresh_duration_sec: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default)

    @classmethod
    def from_json(cls, s: str) -> DashboardState:
        data = json.loads(s)
        today_data = data.get("today")
        today = DayState(**today_data) if today_data else None
        history = [DayState(**h) for h in data.get("history", [])]
        return cls(
            computed_at=data["computed_at"],
            as_of_date=data["as_of_date"],
            today=today,
            history=history,
            oos_predictions=data.get("oos_predictions", []),
            oos_actuals=data.get("oos_actuals", []),
            oos_probas=data.get("oos_probas", []),
            oos_dates=data.get("oos_dates", []),
            intraday_signals=data.get("intraday_signals", []),
            model_accuracy=data.get("model_accuracy", 0.0),
            model_f1=data.get("model_f1", 0.0),
            refresh_duration_sec=data.get("refresh_duration_sec", 0.0),
            error=data.get("error"),
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def make_empty_state(error: str) -> DashboardState:
    """Return an error state for server startup."""
    return DashboardState(
        computed_at=datetime.utcnow().isoformat(),
        as_of_date=date.today().isoformat(),
        today=None,
        error=error,
    )
