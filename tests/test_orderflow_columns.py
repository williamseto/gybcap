import pandas as pd

from strategies.realtime.orderflow_columns import normalize_orderflow_columns


def test_normalize_maps_legacy_buys_sells_to_canonical() -> None:
    df = pd.DataFrame(
        {
            "buys": [120.0, 40.0],
            "sells": [80.0, 90.0],
        }
    )
    out = normalize_orderflow_columns(df)

    # Canonical mapping contract: buys -> askvolume, sells -> bidvolume.
    assert out["askvolume"].tolist() == [120.0, 40.0]
    assert out["bidvolume"].tolist() == [80.0, 90.0]

    # OFI sign should align with ask-bid difference.
    ofi = out["askvolume"] - out["bidvolume"]
    assert ofi.iloc[0] > 0.0
    assert ofi.iloc[1] < 0.0

    # Default path returns a copy and does not mutate source columns.
    assert "askvolume" not in df.columns
    assert "bidvolume" not in df.columns


def test_normalize_preserves_existing_canonical_columns() -> None:
    df = pd.DataFrame(
        {
            "askvolume": [500.0, 600.0],
            "bidvolume": [450.0, 580.0],
            "buys": [1.0, 2.0],
            "sells": [3.0, 4.0],
        }
    )
    out = normalize_orderflow_columns(df)

    # Existing canonical columns should not be overwritten by legacy aliases.
    assert out["askvolume"].tolist() == [500.0, 600.0]
    assert out["bidvolume"].tolist() == [450.0, 580.0]
