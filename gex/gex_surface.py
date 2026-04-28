"""Dealer positioning model — GEX surface, key levels, and regime classification.

Given an options chain snapshot, builds a gamma/delta exposure grid across
spot levels to identify:
- Net GEX sign (positive = mean-reverting, negative = trending)
- Key levels (gamma flip point, max gamma strike, put wall, call wall)
- Hedging pressure for hypothetical spot moves
- Regime classification

Sign convention (dealer-short assumption):
  Call OI → positive GEX contribution (dealer hedge = counter-trend)
  Put OI → negative GEX contribution (dealer hedge = pro-trend)
  Net GEX > 0 → positive gamma → mean-reverting
  Net GEX < 0 → negative gamma → trending
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import norm


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class GEXKeyLevels:
    """Key gamma-derived price levels."""

    gamma_flip: Optional[float] = None
    max_gamma_strike: Optional[float] = None
    put_wall: Optional[float] = None
    call_wall: Optional[float] = None


@dataclass
class GEXRegime:
    """Gamma regime classification."""

    sign: int = 0  # +1 positive, -1 negative, 0 low/unknown
    label: str = "unknown"  # positive_gamma | negative_gamma | gamma_pinned | low_gamma
    net_gex: float = 0.0


# ── Core model ────────────────────────────────────────────────────────


class DealerPositionModel:
    """Model of dealer gamma/delta exposure across a grid of spot levels.

    Accepts an options chain DataFrame (from the historical parquet or a live
    Schwab snapshot) and computes vectorized exposure surfaces.
    """

    def __init__(
        self,
        chain_df: pd.DataFrame,
        spot: float,
        reference_spot: Optional[float] = None,
        r: float = 0.04,
        contract_mult: int = 100,
        grid_pct_range: Tuple[float, float] = (-0.05, 0.05),
        n_levels: int = 201,
    ):
        """
        Args:
            chain_df: Options chain with columns:
                strike, iv, open_interest, instrument_class (C/P),
                expiration (datetime), trade_date (datetime)
            spot: Current underlying spot price.
            reference_spot: Spot when IVs were observed (for IV mapping).
                            Defaults to *spot*.
            r: Risk-free rate.
            contract_mult: Contract multiplier (100 for standard).
            grid_pct_range: (lo, hi) percentage around spot for the grid.
            n_levels: Number of spot levels in the grid.
        """
        self.spot = spot
        self.reference_spot = reference_spot or spot
        self.r = r
        self.contract_mult = contract_mult

        self._chain = self._prepare_chain(chain_df)

        self.spot_levels = np.linspace(
            spot * (1 + grid_pct_range[0]),
            spot * (1 + grid_pct_range[1]),
            n_levels,
        )

        # Grids filled by _build()
        self._gex_grid: np.ndarray = np.array([])
        self._delta_grid: np.ndarray = np.array([])
        self._call_gex_grid: np.ndarray = np.array([])
        self._put_gex_grid: np.ndarray = np.array([])
        self._key_levels = GEXKeyLevels()
        self._regime = GEXRegime()

        self._build()

    # ── Chain preparation ─────────────────────────────────────────

    @staticmethod
    def _prepare_chain(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Normalise column names coming from different sources
        rename = {
            "instrument_class": "option_type",
            "contractType": "option_type",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "option_type" in df.columns:
            df["option_type"] = df["option_type"].astype(str).str.upper().str[0]

        # Drop rows with missing / zero IV or OI
        df = df[df["iv"].notna() & (df["iv"] > 0)].copy()
        df = df[df["open_interest"].notna() & (df["open_interest"] > 0)].copy()

        # Time-to-expiry in years
        if "expiration" in df.columns and "trade_date" in df.columns:
            exp = pd.to_datetime(df["expiration"])
            trade = pd.to_datetime(df["trade_date"])
            df["tte_years"] = (exp - trade).dt.total_seconds() / (365.25 * 86400)
            df["tte_years"] = df["tte_years"].clip(lower=1e-6)
            df["dte_days"] = (exp - trade).dt.days

        return df

    # ── Grid construction (vectorised) ────────────────────────────

    def _build(self):
        chain = self._chain
        n = len(self.spot_levels)

        if chain.empty or "tte_years" not in chain.columns:
            self._gex_grid = np.zeros(n)
            self._delta_grid = np.zeros(n)
            self._call_gex_grid = np.zeros(n)
            self._put_gex_grid = np.zeros(n)
            self._key_levels = GEXKeyLevels()
            self._regime = GEXRegime()
            return

        strikes = chain["strike"].values
        ivs = chain["iv"].values
        ois = chain["open_interest"].values
        option_types = chain["option_type"].values
        tte = chain["tte_years"].values

        is_call = option_types == "C"
        is_put = option_types == "P"

        # Broadcast: S (n_levels,1)  K (1,n_strikes)
        S = self.spot_levels[:, None]
        K = strikes[None, :]
        iv = np.maximum(ivs[None, :], 1e-5)
        T = np.maximum(tte[None, :], 1e-6)
        sqrt_T = np.sqrt(T)

        d1 = (np.log(S / K) + (self.r + 0.5 * iv ** 2) * T) / (iv * sqrt_T)

        # Gamma (identical for calls and puts)
        gamma = norm.pdf(d1) / (S * iv * sqrt_T)  # (n_levels, n_strikes)

        # Delta
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1.0

        # GEX = gamma * OI * mult * S^2 * 0.01  (dollar-gamma convention)
        gex_raw = gamma * ois[None, :] * self.contract_mult * S ** 2 * 0.01

        call_mask = is_call[None, :]
        put_mask = is_put[None, :]

        self._call_gex_grid = np.sum(gex_raw * call_mask, axis=1)
        self._put_gex_grid = -np.sum(gex_raw * put_mask, axis=1)
        self._gex_grid = self._call_gex_grid + self._put_gex_grid

        # Dealer delta (short position): -delta * OI * mult
        dealer_delta = np.where(
            call_mask,
            -delta_call * ois[None, :] * self.contract_mult,
            -delta_put * ois[None, :] * self.contract_mult,
        )
        self._delta_grid = np.sum(dealer_delta, axis=1)

        self._compute_key_levels()
        self._classify_regime()

    # ── Key level detection ───────────────────────────────────────

    def _compute_key_levels(self):
        levels = GEXKeyLevels()
        gex = self._gex_grid
        spots = self.spot_levels

        if len(gex) == 0:
            self._key_levels = levels
            return

        # Gamma flip: net GEX sign change nearest to current spot
        sign_changes = np.where(np.diff(np.sign(gex)))[0]
        if len(sign_changes) > 0:
            midpoints = (spots[sign_changes] + spots[sign_changes + 1]) / 2
            nearest = np.argmin(np.abs(midpoints - self.spot))
            i = sign_changes[nearest]
            denom = gex[i + 1] - gex[i]
            if abs(denom) > 1e-20:
                frac = -gex[i] / denom
                levels.gamma_flip = float(spots[i] + frac * (spots[i + 1] - spots[i]))
            else:
                levels.gamma_flip = float(midpoints[nearest])

        # Max absolute GEX level
        max_idx = int(np.argmax(np.abs(gex)))
        levels.max_gamma_strike = float(spots[max_idx])

        # Put wall: most negative put-GEX level below spot
        below = spots <= self.spot
        if np.any(below & (self._put_gex_grid < 0)):
            put_below = np.where(below, self._put_gex_grid, 0)
            levels.put_wall = float(spots[int(np.argmin(put_below))])

        # Call wall: most positive call-GEX level above spot
        above = spots >= self.spot
        if np.any(above & (self._call_gex_grid > 0)):
            call_above = np.where(above, self._call_gex_grid, 0)
            levels.call_wall = float(spots[int(np.argmax(call_above))])

        self._key_levels = levels

    # ── Regime classification ─────────────────────────────────────

    def _classify_regime(self):
        gex_at_spot = float(np.interp(self.spot, self.spot_levels, self._gex_grid))

        abs_gex = np.abs(self._gex_grid)
        positive_mask = abs_gex > 0
        median_gex = float(np.median(abs_gex[positive_mask])) if np.any(positive_mask) else 1.0

        regime = GEXRegime(net_gex=gex_at_spot)

        if abs(gex_at_spot) < median_gex * 0.1:
            regime.sign = 0
            regime.label = "low_gamma"
        elif gex_at_spot > 0:
            regime.sign = 1
            regime.label = "gamma_pinned" if gex_at_spot > median_gex * 3 else "positive_gamma"
        else:
            regime.sign = -1
            regime.label = "negative_gamma"

        self._regime = regime

    # ── Public API ────────────────────────────────────────────────

    @property
    def gex_grid(self) -> np.ndarray:
        """Net GEX at each spot level."""
        return self._gex_grid

    @property
    def delta_grid(self) -> np.ndarray:
        """Net dealer delta at each spot level."""
        return self._delta_grid

    @property
    def call_gex_grid(self) -> np.ndarray:
        return self._call_gex_grid

    @property
    def put_gex_grid(self) -> np.ndarray:
        return self._put_gex_grid

    @property
    def key_levels(self) -> GEXKeyLevels:
        return self._key_levels

    @property
    def regime(self) -> GEXRegime:
        return self._regime

    def gex_at(self, spot: float) -> float:
        """Interpolate net GEX at a single spot level."""
        return float(np.interp(spot, self.spot_levels, self._gex_grid))

    def delta_at(self, spot: float) -> float:
        """Interpolate net dealer delta at a single spot level."""
        return float(np.interp(spot, self.spot_levels, self._delta_grid))

    def hedge_flow(self, spot_from: float, spot_to: float) -> float:
        """Estimated dealer share flow for a spot move.

        Positive = dealer buying shares, negative = selling.
        """
        return -(self.delta_at(spot_to) - self.delta_at(spot_from))

    def gex_slope_at(self, spot: float, dx_pct: float = 0.001) -> float:
        """d(GEX)/dS at *spot* via central difference."""
        dx = spot * dx_pct
        return (self.gex_at(spot + dx) - self.gex_at(spot - dx)) / (2 * dx)

    def near_expiry_gamma_fraction(self, max_dte_days: int = 3) -> float:
        """Fraction of total absolute GEX from options expiring within *max_dte_days*."""
        chain = self._chain
        if chain.empty or "dte_days" not in chain.columns:
            return 0.0

        total_gex = abs(self.gex_at(self.spot))
        if total_gex < 1e-10:
            return 0.0

        near = chain[chain["dte_days"] <= max_dte_days]
        if near.empty:
            return 0.0

        # Lightweight sub-model (single grid point)
        sub = DealerPositionModel(
            near,
            self.spot,
            self.reference_spot,
            self.r,
            self.contract_mult,
            grid_pct_range=(0, 0),
            n_levels=1,
        )
        return min(abs(sub.gex_at(self.spot)) / total_gex, 1.0)


# ── Historical chain loader ──────────────────────────────────────────


def load_chain_for_date(
    trade_date: str,
    parquet_path: str = "gex/gamma_shares_combined.parquet",
) -> pd.DataFrame:
    """Load options chain for *trade_date* from the historical parquet.

    Falls back up to 3 business days if exact date is missing.
    Filters out already-expired options.
    """
    df = pd.read_parquet(parquet_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    target = pd.to_datetime(trade_date)

    chain = df[df["trade_date"] == target]
    if chain.empty:
        for i in range(1, 4):
            prev = target - pd.tseries.offsets.BDay(i)
            chain = df[df["trade_date"] == prev]
            if not chain.empty:
                break

    chain = chain[chain["expiration"] > target].copy()
    return chain
