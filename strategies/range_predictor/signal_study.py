"""Intraday range signal backtest and newsletter asymmetry study.

Uses 1-min bars + VectorizedTradeSimulator to simulate fade-at-boundary
trades and compare Model OOS vs Newsletter vs ATR baseline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.core.trade_simulator import (
    VectorizedTradeSimulator,
    _simulate_single_trade,
)
from strategies.range_predictor.analysis import _generate_oos_predictions
from strategies.range_predictor.newsletter.comparison import _filter_es_daily_newsletter
from strategies.range_predictor.features import _compute_atr, aggregate_to_daily


@dataclass
class SignalStudyConfig:
    stop_buffer_pts: float = 5.0
    target_frac: float = 0.5       # 0.5 = target midpoint of range
    min_range_width_pts: float = 8.0
    cooldown_bars: int = 15
    rth_only: bool = True
    n_folds: int = 5
    min_train_days: int = 100
    atr_mult: float = 0.7


# ── Data alignment helpers ─────────────────────────────────────────────

def _build_day_maps(df_1min: pd.DataFrame) -> Tuple[dict, dict]:
    """Build trading_day <-> date mappings from 1-min data.

    Returns:
        (td_to_date, date_to_td) dicts.
    """
    day_map = df_1min.groupby('trading_day')['dt'].first().dt.date
    td_to_date = day_map.to_dict()
    date_to_td = {v: k for k, v in td_to_date.items()}
    return td_to_date, date_to_td


def _build_predictions_by_day(
    pred_df: pd.DataFrame,
    date_to_td: dict,
) -> Dict[int, Tuple[float, float]]:
    """Map date-indexed predictions to {trading_day_int: (pred_high, pred_low)}."""
    result = {}
    for date_idx, row in pred_df.iterrows():
        d = pd.Timestamp(date_idx).date()
        td = date_to_td.get(d)
        if td is not None:
            result[td] = (row['pred_range_high'], row['pred_range_low'])
    return result


# ── Touch detection with cooldown ──────────────────────────────────────

def _find_touch_entries(
    df_1min: pd.DataFrame,
    predictions_by_day: Dict[int, Tuple[float, float]],
    config: SignalStudyConfig,
) -> pd.DataFrame:
    """Scan 1-min bars per day, find all touch entries with cooldown.

    For each trading day with predictions, walk bars sequentially.
    When price touches pred_high (bar high >= pred_high), enter a fade-high
    (short) trade. When price touches pred_low (bar low <= pred_low),
    enter a fade-low (long) trade.

    After a trade exits, impose a cooldown before re-entry on the same side.

    Returns DataFrame with columns:
        entry_idx, entry_price, stop, take, is_bull, side, trading_day
    """
    highs = df_1min['high'].values
    lows = df_1min['low'].values
    closes = df_1min['close'].values
    n_bars = len(highs)

    rth_mask = None
    if config.rth_only and 'ovn' in df_1min.columns:
        rth_mask = (df_1min['ovn'].values == 0)

    # Group bar indices by trading day
    td_groups: Dict[int, np.ndarray] = {}
    td_vals = df_1min['trading_day'].values
    for td in predictions_by_day:
        indices = np.where(td_vals == td)[0]
        if len(indices) > 0:
            td_groups[td] = indices

    entries: List[dict] = []

    for td, (pred_high, pred_low) in predictions_by_day.items():
        if td not in td_groups:
            continue

        day_indices = td_groups[td]
        width = pred_high - pred_low
        if width < config.min_range_width_pts:
            continue

        midpoint = (pred_high + pred_low) / 2.0
        target_reward = config.target_frac * width

        # Cooldown tracking: next allowed entry index per side
        next_allowed = {'fade_high': day_indices[0], 'fade_low': day_indices[0]}

        for global_i in day_indices:
            if rth_mask is not None and not rth_mask[global_i]:
                continue

            # ── Fade high (short) ──
            if (highs[global_i] >= pred_high
                    and global_i >= next_allowed['fade_high']):
                entry_price = pred_high
                stop = pred_high + config.stop_buffer_pts
                take = pred_high - target_reward  # short target is below
                is_bull = False

                # Simulate to find exit bar for cooldown
                exit_idx, _, _ = _simulate_single_trade(
                    highs, lows, closes,
                    global_i, entry_price, stop, take, is_bull,
                )
                entries.append({
                    'entry_idx': global_i,
                    'entry_price': entry_price,
                    'stop': stop,
                    'take': take,
                    'is_bull': is_bull,
                    'side': 'fade_high',
                    'trading_day': td,
                    'pred_high': pred_high,
                    'pred_low': pred_low,
                })
                next_allowed['fade_high'] = exit_idx + config.cooldown_bars

            # ── Fade low (long) ──
            if (lows[global_i] <= pred_low
                    and global_i >= next_allowed['fade_low']):
                entry_price = pred_low
                stop = pred_low - config.stop_buffer_pts
                take = pred_low + target_reward  # long target is above
                is_bull = True

                exit_idx, _, _ = _simulate_single_trade(
                    highs, lows, closes,
                    global_i, entry_price, stop, take, is_bull,
                )
                entries.append({
                    'entry_idx': global_i,
                    'entry_price': entry_price,
                    'stop': stop,
                    'take': take,
                    'is_bull': is_bull,
                    'side': 'fade_low',
                    'trading_day': td,
                    'pred_high': pred_high,
                    'pred_low': pred_low,
                })
                next_allowed['fade_low'] = exit_idx + config.cooldown_bars

    if not entries:
        return pd.DataFrame()

    return pd.DataFrame(entries)


# ── Trade simulation ──────────────────────────────────────────────────

def _simulate_all_trades(
    df_1min: pd.DataFrame,
    entries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run VectorizedTradeSimulator.simulate_batch on all entries.

    Returns entries_df with added columns:
        exit_idx, exit_price, exit_type, pnl
    """
    if entries_df.empty:
        return entries_df

    sim = VectorizedTradeSimulator(df_1min)
    exit_indices, exit_prices, exit_types, pnls = sim.simulate_batch(
        entry_indices=entries_df['entry_idx'].values,
        entry_prices=entries_df['entry_price'].values,
        stops=entries_df['stop'].values,
        takes=entries_df['take'].values,
        is_bulls=entries_df['is_bull'].values,
    )

    result = entries_df.copy()
    result['exit_idx'] = exit_indices
    result['exit_price'] = exit_prices
    result['exit_type'] = exit_types
    result['pnl'] = pnls
    return result


# ── Metrics ───────────────────────────────────────────────────────────

def _compute_trade_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute summary metrics from a trades DataFrame."""
    if trades_df.empty:
        return {
            'n_trades': 0, 'win_rate': np.nan, 'avg_pnl': np.nan,
            'total_pnl': 0.0, 'profit_factor': np.nan,
            'max_drawdown': 0.0,
            'pct_stop': np.nan, 'pct_take': np.nan, 'pct_close': np.nan,
        }

    pnls = trades_df['pnl'].values
    n = len(pnls)
    winners = pnls > 0
    losers = pnls < 0

    gross_profit = pnls[winners].sum() if winners.any() else 0.0
    gross_loss = abs(pnls[losers].sum()) if losers.any() else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Max drawdown from cumulative PnL curve
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    exit_types = trades_df['exit_type'].values
    return {
        'n_trades': n,
        'win_rate': winners.sum() / n,
        'avg_pnl': pnls.mean(),
        'total_pnl': pnls.sum(),
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'pct_stop': (exit_types == 'stop').sum() / n,
        'pct_take': (exit_types == 'take').sum() / n,
        'pct_close': (exit_types == 'close').sum() / n,
    }


def _build_source_results(
    trades_df: pd.DataFrame,
    regime_by_td: Dict[int, str],
) -> dict:
    """Organize metrics by side and regime."""
    overall = _compute_trade_metrics(trades_df)

    by_side = {}
    for side in ['fade_high', 'fade_low']:
        mask = trades_df['side'] == side
        by_side[side] = _compute_trade_metrics(trades_df[mask])

    by_regime = {}
    if regime_by_td and not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['regime'] = trades_df['trading_day'].map(regime_by_td)
        for regime in ['low_vol', 'normal', 'high_vol']:
            mask = trades_df['regime'] == regime
            if mask.sum() > 0:
                by_regime[regime] = _compute_trade_metrics(trades_df[mask])

    return {'overall': overall, 'by_side': by_side, 'by_regime': by_regime}


# ── ATR baseline predictions ──────────────────────────────────────────

def _build_atr_predictions(
    daily: pd.DataFrame,
    atr_mult: float = 0.7,
) -> pd.DataFrame:
    """Build ATR-based range predictions: prev_close +/- mult * ATR(14).

    Returns DataFrame indexed by date with pred_range_high, pred_range_low.
    """
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14)
    prev_close = daily['close'].shift(1)
    result = pd.DataFrame({
        'pred_range_high': prev_close + atr_mult * atr_14,
        'pred_range_low': prev_close - atr_mult * atr_14,
    }, index=daily.index)
    return result.dropna()


# ── Volatility regime mapping ─────────────────────────────────────────

def _compute_regime_by_td(
    daily: pd.DataFrame,
    date_to_td: dict,
) -> Dict[int, str]:
    """Compute volatility regime per trading day."""
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14)
    p20 = np.nanpercentile(atr_14.values, 20)
    p80 = np.nanpercentile(atr_14.values, 80)

    regime_by_td = {}
    for date_idx, atr_val in atr_14.items():
        d = pd.Timestamp(date_idx).date()
        td = date_to_td.get(d)
        if td is not None:
            if atr_val < p20:
                regime_by_td[td] = 'low_vol'
            elif atr_val > p80:
                regime_by_td[td] = 'high_vol'
            else:
                regime_by_td[td] = 'normal'
    return regime_by_td


# ── Newsletter asymmetry analysis ─────────────────────────────────────

def analyze_range_asymmetry(
    daily: pd.DataFrame,
    newsletter: Optional[pd.DataFrame] = None,
) -> dict:
    """Compare upside vs downside allocation across sources.

    Computes:
    - nl_upside = pred_high - prev_close  vs  nl_downside = prev_close - pred_low
    - Same for realized
    - Segmented by volatility regime
    """
    prev_close = daily['close'].shift(1)
    realized_upside = daily['high'] - prev_close
    realized_downside = prev_close - daily['low']

    # ATR for regime bucketing
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14)
    p20 = np.nanpercentile(atr_14.dropna().values, 20)
    p80 = np.nanpercentile(atr_14.dropna().values, 80)
    vol_regime = pd.Series('normal', index=daily.index)
    vol_regime[atr_14 < p20] = 'low_vol'
    vol_regime[atr_14 > p80] = 'high_vol'

    results = {
        'realized': {
            'avg_upside': float(realized_upside.mean()),
            'avg_downside': float(realized_downside.mean()),
            'bias': float(realized_downside.mean() - realized_upside.mean()),
        }
    }

    # By regime for realized
    results['realized_by_regime'] = {}
    for regime in ['low_vol', 'normal', 'high_vol']:
        mask = vol_regime == regime
        if mask.sum() < 10:
            continue
        results['realized_by_regime'][regime] = {
            'n': int(mask.sum()),
            'avg_upside': float(realized_upside[mask].mean()),
            'avg_downside': float(realized_downside[mask].mean()),
            'bias': float(realized_downside[mask].mean() - realized_upside[mask].mean()),
        }

    if newsletter is not None:
        nl = _filter_es_daily_newsletter(newsletter)
        nl = nl.set_index('date').sort_index()
        nl.index = pd.to_datetime(nl.index)

        common = nl.index.intersection(daily.index)
        if len(common) > 0:
            pc = prev_close.reindex(common)
            nl_up = nl.loc[common, 'nl_range_high'] - pc
            nl_down = pc - nl.loc[common, 'nl_range_low']

            results['newsletter'] = {
                'avg_upside': float(nl_up.mean()),
                'avg_downside': float(nl_down.mean()),
                'bias': float(nl_down.mean() - nl_up.mean()),
                'n': len(common),
            }

            # By regime
            results['newsletter_by_regime'] = {}
            for regime in ['low_vol', 'normal', 'high_vol']:
                mask = vol_regime.reindex(common) == regime
                if mask.sum() < 5:
                    continue
                idx = common[mask]
                results['newsletter_by_regime'][regime] = {
                    'n': int(mask.sum()),
                    'avg_upside': float(nl_up[idx].mean()),
                    'avg_downside': float(nl_down[idx].mean()),
                    'bias': float(nl_down[idx].mean() - nl_up[idx].mean()),
                }

    return results


# ── Printing ──────────────────────────────────────────────────────────

def _print_backtest_results(
    source_results: Dict[str, dict],
    config: SignalStudyConfig,
) -> None:
    """Print formatted comparison table."""
    print(f"\nINTRADAY RANGE SIGNAL BACKTEST")
    print(f"  Stop buffer: {config.stop_buffer_pts} pts | "
          f"Target: {config.target_frac*100:.0f}% of range width | "
          f"Cooldown: {config.cooldown_bars} bars | "
          f"{'RTH only' if config.rth_only else 'Incl OVN'}")
    print("=" * 72)

    sources = list(source_results.keys())
    header = f"{'OVERALL':<24}" + "".join(f"{s:>16}" for s in sources)
    print(header)
    print("-" * 72)

    def _fmt(v, fmt='.1f'):
        if isinstance(v, float) and np.isnan(v):
            return 'N/A'
        return f"{v:{fmt}}"

    rows = [
        ('Trades', 'n_trades', 'd'),
        ('Win rate', 'win_rate', '.1%'),
        ('Avg PnL (pts)', 'avg_pnl', '+.2f'),
        ('Total PnL (pts)', 'total_pnl', '+.1f'),
        ('Profit factor', 'profit_factor', '.2f'),
        ('Max drawdown (pts)', 'max_drawdown', '.1f'),
        ('% stopped', 'pct_stop', '.1%'),
        ('% take profit', 'pct_take', '.1%'),
        ('% close (EOD)', 'pct_close', '.1%'),
    ]

    for label, key, fmt in rows:
        parts = [f"  {label:<22}"]
        for src in sources:
            m = source_results[src]['overall']
            parts.append(f"{_fmt(m[key], fmt):>16}")
        print("".join(parts))

    # ── By side ──
    print()
    for side_label, side_key in [('SHORT at pred_high', 'fade_high'),
                                  ('LONG at pred_low', 'fade_low')]:
        print(f"  {side_label}")
        for src in sources:
            m = source_results[src]['by_side'].get(side_key, {})
            n = m.get('n_trades', 0)
            wr = m.get('win_rate', np.nan)
            pnl = m.get('avg_pnl', np.nan)
            pf = m.get('profit_factor', np.nan)
            wr_s = f"{wr:.0%}" if not np.isnan(wr) else "N/A"
            pnl_s = f"{pnl:+.1f}" if not np.isnan(pnl) else "N/A"
            pf_s = f"{pf:.2f}" if not (np.isnan(pf) or np.isinf(pf)) else "N/A"
            print(f"    {src:<16} n={n:<4} WR={wr_s:<5} PnL={pnl_s:<7} PF={pf_s}")

    # ── By regime ──
    print()
    print("BY VOLATILITY REGIME")
    for regime in ['low_vol', 'normal', 'high_vol']:
        parts = [f"  {regime:<12}"]
        for src in sources:
            m = source_results[src]['by_regime'].get(regime, {})
            n = m.get('n_trades', 0)
            wr = m.get('win_rate', np.nan)
            pnl = m.get('avg_pnl', np.nan)
            wr_s = f"{wr:.0%}" if not np.isnan(wr) else "N/A"
            pnl_s = f"{pnl:+.1f}" if not np.isnan(pnl) else "N/A"
            parts.append(f"  {src} n={n} WR={wr_s} PnL={pnl_s}")
        print("".join(parts))


def _print_asymmetry(asym: dict) -> None:
    """Print newsletter asymmetry analysis."""
    print()
    print("=" * 72)
    print("NEWSLETTER RANGE ASYMMETRY ANALYSIS")
    print("-" * 72)

    def _row(label, data):
        up = data['avg_upside']
        down = data['avg_downside']
        bias = data['bias']
        print(f"  {label:<20} Upside={up:+.1f}  Downside={down:+.1f}  "
              f"Bias(down-up)={bias:+.1f}")

    _row("Realized", asym['realized'])
    if 'newsletter' in asym:
        _row(f"Newsletter (n={asym['newsletter']['n']})", asym['newsletter'])

    if 'realized_by_regime' in asym and asym['realized_by_regime']:
        print()
        print("  By volatility regime:")
        for regime in ['low_vol', 'normal', 'high_vol']:
            r_data = asym['realized_by_regime'].get(regime)
            if r_data is None:
                continue
            line = f"    {regime:<12} (n={r_data['n']:>3d})  Realized bias={r_data['bias']:+.1f}"
            nl_data = asym.get('newsletter_by_regime', {}).get(regime)
            if nl_data:
                line += f"  NL bias={nl_data['bias']:+.1f} (n={nl_data['n']})"
            print(line)

    print("=" * 72)


# ── Main entry point ──────────────────────────────────────────────────

def backtest_range_signals(
    df_1min: pd.DataFrame,
    daily: pd.DataFrame,
    newsletter: Optional[pd.DataFrame] = None,
    config: Optional[SignalStudyConfig] = None,
) -> dict:
    """Run intraday range signal backtest comparing 3 sources.

    Args:
        df_1min: 1-min OHLCV with trading_day, dt, ovn columns.
        daily: Daily OHLCV (DatetimeIndex), from aggregate_to_daily.
        newsletter: Optional newsletter CSV with range predictions.
        config: Signal study configuration.

    Returns:
        Dict with all results for further analysis.
    """
    if config is None:
        config = SignalStudyConfig()

    td_to_date, date_to_td = _build_day_maps(df_1min)
    regime_by_td = _compute_regime_by_td(daily, date_to_td)

    source_results = {}

    # ── 1. Model OOS predictions (walk-forward) ──
    print("Generating walk-forward OOS predictions...")
    oos_preds = _generate_oos_predictions(
        daily, 'daily',
        n_folds=config.n_folds,
        min_train_days=config.min_train_days,
    )
    print(f"  OOS predictions: {len(oos_preds)} days")

    oos_by_day = _build_predictions_by_day(oos_preds, date_to_td)
    oos_entries = _find_touch_entries(df_1min, oos_by_day, config)
    oos_trades = _simulate_all_trades(df_1min, oos_entries)
    source_results['Model OOS'] = _build_source_results(oos_trades, regime_by_td)
    print(f"  Model OOS trades: {len(oos_trades)}")

    # ── 2. Newsletter predictions ──
    if newsletter is not None:
        nl = _filter_es_daily_newsletter(newsletter)
        nl = nl.set_index('date').sort_index()
        nl.index = pd.to_datetime(nl.index)
        nl_preds = nl.rename(columns={
            'nl_range_high': 'pred_range_high',
            'nl_range_low': 'pred_range_low',
        })

        nl_by_day = _build_predictions_by_day(nl_preds, date_to_td)
        nl_entries = _find_touch_entries(df_1min, nl_by_day, config)
        nl_trades = _simulate_all_trades(df_1min, nl_entries)
        source_results['Newsletter'] = _build_source_results(nl_trades, regime_by_td)
        print(f"  Newsletter trades: {len(nl_trades)}")

    # ── 3. ATR baseline ──
    atr_preds = _build_atr_predictions(daily, config.atr_mult)
    atr_by_day = _build_predictions_by_day(atr_preds, date_to_td)
    atr_entries = _find_touch_entries(df_1min, atr_by_day, config)
    atr_trades = _simulate_all_trades(df_1min, atr_entries)
    source_results['ATR Baseline'] = _build_source_results(atr_trades, regime_by_td)
    print(f"  ATR Baseline trades: {len(atr_trades)}")

    # ── Print results ──
    _print_backtest_results(source_results, config)

    # ── Asymmetry analysis ──
    asym = analyze_range_asymmetry(daily, newsletter)
    _print_asymmetry(asym)

    return {
        'source_results': source_results,
        'asymmetry': asym,
        'config': config,
    }
