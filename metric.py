"""
Evaluate prediction accuracy of the sentiment fear score against actual S&P 500 movements.

Usage:
    python metric.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from scipy import stats


def load_sentiment() -> pd.DataFrame:
    df = pd.read_csv("data/processed/daily_sentiment_us.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fetch_market(start_date, end_date) -> pd.DataFrame:
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(
        start=str(start_date - timedelta(days=5)),
        end=str(end_date + timedelta(days=5)),
    )
    if hist.empty:
        return pd.DataFrame()
    df = hist[["Close"]].reset_index()
    df.columns = ["date", "sp500_close"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sp500_return"] = df["sp500_close"].pct_change()
    df["sp500_direction"] = (df["sp500_return"] > 0).astype(int)  # 1=up, 0=down
    return df


def merge_data(sentiment_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(sentiment_df, market_df, on="date", how="inner")

    # next-day return: shift market data back so each row has tomorrow's result
    merged["next_day_return"] = merged["sp500_return"].shift(-1)
    merged["next_day_direction"] = merged["sp500_direction"].shift(-1)  # 1=up, 0=down

    # our prediction: fear > 5 = predict down (0), fear < 5 = predict up (1)
    merged["our_prediction"] = (merged["avg_recession_fear"] < 5).astype(float)
    merged.loc[merged["avg_recession_fear"] == 5, "our_prediction"] = np.nan

    return merged.dropna(subset=["next_day_return", "our_prediction"])


def direction_accuracy(df: pd.DataFrame):
    """Simple direction accuracy: did our fear-based call match next-day direction?"""
    print("=" * 60)
    print("  1. DIRECTION ACCURACY")
    print("     Rule: fear > 5 -> predict DOWN | fear < 5 -> predict UP")
    print("=" * 60)

    df["correct"] = (df["our_prediction"] == df["next_day_direction"])

    total = len(df)
    correct = df["correct"].sum()
    accuracy = correct / total if total > 0 else 0

    print(f"\n  Trading days tested:  {total}")
    print(f"  Correct predictions:  {int(correct)}")
    print(f"  Direction accuracy:   {accuracy:.1%}")
    print(f"  Coin flip baseline:   50.0%")
    print(f"  Edge over random:     {(accuracy - 0.5):+.1%}")

    # Day-by-day breakdown
    print(f"\n  {'Date':<14} {'Fear':>5} {'Predict':>8} {'Actual':>8} {'S&P Return':>11} {'Result':>8}")
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        fear = row["avg_recession_fear"]
        pred = "UP" if row["our_prediction"] == 1 else "DOWN"
        actual = "UP" if row["next_day_direction"] == 1 else "DOWN"
        ret = row["next_day_return"]
        result = "HIT" if row["correct"] else "MISS"
        color = result
        print(f"  {row['date']}   {fear:>5.1f}   {pred:>6}   {actual:>6}   {ret:>+9.2%}     {result}")

    return accuracy


def confidence_weighted_accuracy(df: pd.DataFrame):
    """When fear is extreme, are we more accurate?"""
    print(f"\n{'=' * 60}")
    print("  2. CONFIDENCE-WEIGHTED ACCURACY")
    print("     Are extreme fear readings more reliable than mild ones?")
    print("=" * 60)

    # Define confidence bands based on distance from neutral (5)
    df["distance_from_neutral"] = abs(df["avg_recession_fear"] - 5)

    bands = [
        ("High confidence (fear < 2 or > 8)", lambda d: d >= 3),
        ("Medium confidence (fear 2-4 or 6-8)", lambda d: (d >= 1) & (d < 3)),
        ("Low confidence (fear 4-6)", lambda d: d < 1),
    ]

    print(f"\n  {'Band':<42} {'Days':>5} {'Accuracy':>9} {'Avg Return':>11}")
    print(f"  {'-'*67}")

    for label, condition in bands:
        subset = df[condition(df["distance_from_neutral"])]
        if subset.empty:
            print(f"  {label:<42} {'0':>5} {'--':>9} {'--':>11}")
            continue

        n = len(subset)
        acc = subset["correct"].mean()
        avg_ret = subset["next_day_return"].mean()
        print(f"  {label:<42} {n:>5} {acc:>8.1%} {avg_ret:>+10.2%}")

    # Show that extreme readings are more valuable
    if len(df[df["distance_from_neutral"] >= 2]) >= 3:
        strong = df[df["distance_from_neutral"] >= 2]
        weak = df[df["distance_from_neutral"] < 2]
        if not weak.empty:
            strong_acc = strong["correct"].mean()
            weak_acc = weak["correct"].mean()
            print(f"\n  Strong signals (|fear - 5| >= 2): {strong_acc:.1%} accuracy over {len(strong)} days")
            print(f"  Weak signals   (|fear - 5| < 2):  {weak_acc:.1%} accuracy over {len(weak)} days")
            diff = strong_acc - weak_acc
            print(f"  Extreme readings are {diff:+.1%} {'more' if diff > 0 else 'less'} accurate")


def print_correlations(df: pd.DataFrame):
    """Compute and explain lag correlations."""
    print(f"\n{'=' * 60}")
    print("  3. LAG CORRELATIONS")
    print("     Does today's fear predict future returns?")
    print("=" * 60)

    fear = df["avg_recession_fear"].values
    returns = df["sp500_return"].values

    print(f"\n  {'Lag':<6} {'Pearson r':>10} {'p-value':>9} {'Spearman r':>11} {'p-value':>9} {'Interpretation'}")
    print(f"  {'-'*70}")

    for lag in [0, 1, 2, 3]:
        if lag == 0:
            s, m = fear, returns
        else:
            s = fear[:-lag]
            m = returns[lag:]

        mask = ~(np.isnan(s) | np.isnan(m))
        s_clean, m_clean = s[mask], m[mask]

        if len(s_clean) < 3:
            continue

        pr, pp = stats.pearsonr(s_clean, m_clean)
        sr, sp = stats.spearmanr(s_clean, m_clean)

        # Interpret
        if abs(sr) < 0.1:
            interp = "No relationship"
        elif abs(sr) < 0.3:
            sign = "contrarian" if sr > 0 else "predictive"
            interp = f"Weak {sign}"
        elif abs(sr) < 0.5:
            sign = "contrarian" if sr > 0 else "predictive"
            interp = f"Moderate {sign}"
        else:
            sign = "contrarian" if sr > 0 else "predictive"
            interp = f"Strong {sign}"

        sig = "*" if sp < 0.05 else ""
        print(f"  T+{lag:<4} {pr:>+9.3f}  {pp:>8.3f}  {sr:>+10.3f}  {sp:>8.3f}  {interp}{sig}")

    print(f"""
  How to read this:
  - Negative correlation = fear predicts market DROPS (intuitive)
  - Positive correlation = fear predicts market RISES (contrarian/mean-reversion)
  - p-value < 0.05 means statistically significant (marked with *)
  - With ~25 trading days, you need |r| > 0.36 for significance
  - In finance, |r| > 0.2 is a useful signal""")


def main():
    print("\n" + "=" * 60)
    print("  SENTIMENT PREDICTION ACCURACY REPORT")
    print("=" * 60)

    sentiment_df = load_sentiment()
    print(f"\n  Loaded {len(sentiment_df)} days of sentiment data")
    print(f"  Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")

    market_df = fetch_market(sentiment_df["date"].min(), sentiment_df["date"].max())
    if market_df.empty:
        print("  ERROR: Could not fetch S&P 500 data")
        return

    print(f"  Loaded {len(market_df)} days of S&P 500 data")

    merged = merge_data(sentiment_df, market_df)
    print(f"  Matched {len(merged)} trading days with both sentiment and market data\n")

    if merged.empty:
        print("  No overlapping trading days found")
        return

    acc = direction_accuracy(merged)
    confidence_weighted_accuracy(merged)
    print_correlations(merged)

    # Final verdict
    print(f"\n{'=' * 60}")
    print("  VERDICT")
    print("=" * 60)
    if acc > 0.6:
        print(f"  {acc:.0%} direction accuracy -- signal has predictive value")
    elif acc > 0.5:
        print(f"  {acc:.0%} direction accuracy -- slight edge over random")
    else:
        print(f"  {acc:.0%} direction accuracy -- no better than coin flip")
        print("  Note: positive correlations suggest fear is a CONTRARIAN")
        print("  indicator -- high fear may predict recoveries, not drops")

    print()


if __name__ == "__main__":
    main()
