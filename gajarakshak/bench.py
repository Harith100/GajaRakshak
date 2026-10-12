"""Offline A/B benchmark on synthetic data.

    python -m gajarakshak.bench --hours 24 --seeds 3

Runs the same synthetic world (same seed = same herds, trains, confusers) under each
operating mode and compares safety vs operational cost. Writes reports/bench.{json,md}
and the full event log of the last gajarakshak run to reports/events_seed<N>.csv.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .runtime import Runtime

MODES = ["none", "blanket", "gajarakshak", "gajarakshak+kavach"]
LABEL = {"none": "No system", "blanket": "Blanket 45 km/h at night (today)",
         "gajarakshak": "GajaRakshak (SMS/IVR/VHF)", "gajarakshak+kavach": "GajaRakshak + Kavach TSR"}


def run(hours: float, seeds: list[int], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {m: [] for m in MODES}
    for seed in seeds:
        for mode in MODES:
            t0 = time.time()
            rt = Runtime(seed=seed, mode=mode)
            rt.run_for(hours * 3600)
            s = rt.stats()
            s["wall_s"] = round(time.time() - t0, 1)
            results[mode].append(s)
            print(f"seed={seed:<3} {mode:<20} collisions={s['collisions']:<3} delay={s['train_delay_total_min']:>7} min "
                  f"alerts={s['alerts_total']:<4} false={s['false_actionable_alerts']:<3} ({s['wall_s']}s)")
            if mode == "gajarakshak":
                rt.store.export_csv(out / f"events_seed{seed}.csv")

    def agg(mode, key):
        vals = [r[key] for r in results[mode] if r[key] is not None]
        return round(statistics.mean(vals), 1) if vals else None

    summary = {}
    for m in MODES:
        summary[m] = {k: agg(m, k) for k in ("collisions", "near_misses", "herds_exposed", "herds_missed",
                                            "train_delay_total_min", "train_delay_avg_min", "trains_passed",
                                            "alerts_total", "actionable_alerts", "false_actionable_alerts",
                                            "trains_restricted", "lead_time_p50_s", "ack_p95_s",
                                            "detect_to_alert_p50_s", "uplink_kb", "video_equiv_gb")}
    report = {"hours_per_run": hours, "seeds": seeds, "summary": summary, "runs": results}
    (out / "bench.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    rows = [("Collisions / day", "collisions"), ("Herds reaching track area / day", "herds_exposed"),
            ("Herds missed (no alert)", "herds_missed"), ("Train delay, total min / day", "train_delay_total_min"),
            ("Avg delay per train, min", "train_delay_avg_min"), ("Actionable alerts (CAUTION/STOP)", "actionable_alerts"),
            ("…of which false", "false_actionable_alerts"), ("Herd in camera range → first alert, p50 s", "detect_to_alert_p50_s"),
            ("Warning lead time to train, p50 s", "lead_time_p50_s"), ("Pilot ack latency p95, s", "ack_p95_s"),
            ("Edge→cloud uplink, KB / day", "uplink_kb"), ("Same, if video streamed 24×7, GB / day", "video_equiv_gb")]
    scale = 24 / hours
    md = [f"# Benchmark — {hours:g} h × {len(seeds)} seeds (values normalised per day, mean of seeds)\n",
          "| Metric | " + " | ".join(LABEL[m] for m in MODES) + " |", "|---" * (len(MODES) + 1) + "|"]
    per_day = {"collisions", "herds_exposed", "herds_missed", "train_delay_total_min", "actionable_alerts",
               "false_actionable_alerts", "uplink_kb", "video_equiv_gb"}
    for name, key in rows:
        cells = []
        for m in MODES:
            v = summary[m][key]
            if v is not None and key in per_day:
                v = round(v * scale, 1)
            cells.append("—" if v is None or (m in ("none", "blanket") and key in
                         ("actionable_alerts", "false_actionable_alerts", "lead_time_p50_s", "ack_p95_s",
                          "detect_to_alert_p50_s", "herds_missed", "uplink_kb", "video_equiv_gb")) else str(v))
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "bench.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    run(a.hours, list(range(1, a.seeds + 1)), Path(a.out))


if __name__ == "__main__":
    main()
