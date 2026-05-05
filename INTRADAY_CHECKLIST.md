# Intraday Checklist

Use this app as an intraday regime filter, not as a price-target tool.

## 20-Second Workflow

1. Start on `15m`.
   Scan the broader intraday backdrop first. Switch to `5m` only if you want finer timing after the higher-level read is clear.

2. Prefer liquid symbols.
   Use names like `SPY`, `QQQ`, `AAPL`, `NVDA`, `GLD`, or `BTC-USD`. The app is most useful when the tape is active and the bar feed is stable.

3. Check the regime first.
   Read `Current regime` and `Posterior weights P(regime | v_t)`.
   A useful read is one where bull or bear clearly dominates, not a nearly split posterior.

4. Check state alignment.
   Confirm that `v(t)` and `drift(v_t)` support the same directional story as the displayed regime.
   If the regime says bull but momentum or drift is fading the other way, treat the read as weak.

5. Check persistence.
   Read `tau_L (clock)` as the approximate memory horizon of the current state.
   If your intended hold time is much longer than that horizon, the signal is less relevant.

6. Compare KF and EKF.
   If `Linear KF` and `EKF` broadly agree on regime and direction, the read is cleaner.
   If they disagree materially, reduce confidence.

7. Use the replay as a sanity check only.
   `Historical one-bar-ahead replay` is not a backtest and not a tradable edge by itself.
   Use it only to see whether the recent behavior has been completely unstable.

8. Translate to action conservatively.
   `BULL` + positive `v(t)` + supportive drift: long bias.
   `BEAR` + negative `v(t)` + supportive drift: reduce long exposure, hedge, or bearish bias if that fits your process.
   `TRANSITION`, low confidence, or filter disagreement: do less or do nothing.

9. Re-run after shocks.
   Recheck after sharp moves, macro headlines, or the open/close transition. Intraday state can flip quickly.

## Good Default Routine

- Check `15m` for market backdrop.
- If the read is clean, switch the same symbol to `5m` for timing.
- Do not use the app outside regular-session assumptions without extra caution.
- Treat `Suggested exposure` as a bias indicator, not an order instruction.

## What Not To Do

- Do not treat the replay hit rate as proof of edge.
- Do not read the exposure ring as literal sizing.
- Do not force trades when the app shows `TRANSITION`.
- Do not rely on one symbol alone; compare it with index context when possible.

## One-Line Rule

Trade only when regime, posterior, momentum, drift, and the KF/EKF comparison all tell the same story.