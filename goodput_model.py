"""How much of a large training run's time goes into useful work?

A model of a synchronous training job that saves a checkpoint every `tau` hours,
loses the work since its last checkpoint whenever any GPU fails, and pays a fixed
restart time per failure. Failures arrive at random (exponential gaps), at a rate
set by the number of GPUs. The first-order formula is checked by simulation.

Inputs are taken from published figures, and the checkpoint and restart times
are assumptions, so the outputs are estimates, not measurements.
Run: python goodput_model.py   (needs numpy)
     python goodput_model.py --help   for your own failure rate and recovery times
"""
import argparse
import math

import numpy as np

# Llama 3 (arXiv 2407.21783): 419 unexpected interruptions in 54 days on ~16,384 GPUs
GPU_MTBF_H = 54 * 24 / 419 * 16384          # about 50,700 hours per GPU between interruptions

CASES = {  # checkpoint write time (delta) and restart time (R), in hours
    "slow recovery (5 min checkpoint, 30 min restart)": (5 / 60, 30 / 60),
    "fast recovery (30 s checkpoint, 10 min restart)": (0.5 / 60, 10 / 60),
}

def young_interval(delta, mtbf):
    """Young (1974): checkpoint interval that minimizes lost time, sqrt(2 * delta * M)."""
    return math.sqrt(2 * delta * mtbf)

def goodput_formula(tau, delta, restart, mtbf):
    """First-order estimate: time spent checkpointing, plus half an interval of lost work
    and one restart per failure, all as a share of the run."""
    return max(0.0, 1 - delta / tau - (tau / 2 + restart) / mtbf)

def goodput_simulated(tau, delta, restart, mtbf, years=20, seed=0):
    """Monte Carlo check: work only counts once the checkpoint after it is written."""
    rng = np.random.default_rng(seed)
    clock, useful, horizon = 0.0, 0.0, years * 365 * 24
    while clock < horizon:
        time_to_failure = rng.exponential(mtbf)
        saved_segments = int(time_to_failure // (tau + delta))
        useful += saved_segments * tau
        clock += time_to_failure + restart
    return useful / clock

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu-mtbf-hours", type=float, default=GPU_MTBF_H,
                        help="hours of running per GPU between interruptions (default: the Llama 3 rate)")
    parser.add_argument("--checkpoint-min", type=float,
                        help="your checkpoint write time in minutes; with --restart-min, replaces the two default cases")
    parser.add_argument("--restart-min", type=float,
                        help="your restart time in minutes: detection, replacement and reload")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1024, 4096, 16384, 32768, 65536, 100000],
                        help="job sizes to report")
    parser.add_argument("--sweep-gpus", type=int, default=16384,
                        help="job size for the checkpoint-interval sweep")
    args = parser.parse_args()

    cases = CASES
    if args.checkpoint_min is not None or args.restart_min is not None:
        if args.checkpoint_min is None or args.restart_min is None:
            parser.error("give both --checkpoint-min and --restart-min")
        cases = {f"your stack ({args.checkpoint_min:g} min checkpoint, {args.restart_min:g} min restart)":
                 (args.checkpoint_min / 60, args.restart_min / 60)}

    print(f"per-GPU time between interruptions: {args.gpu_mtbf_hours:,.0f} hours\n")
    for gpus in args.gpus:
        mtbf = args.gpu_mtbf_hours / gpus
        for name, (delta, restart) in cases.items():
            tau = young_interval(delta, mtbf)
            print(f"{gpus:>7,} GPUs  job MTBF {mtbf * 60:6.0f} min  {name:<50}"
                  f"best interval {tau * 60:5.0f} min  goodput "
                  f"formula {goodput_formula(tau, delta, restart, mtbf):.3f}  "
                  f"simulated {goodput_simulated(tau, delta, restart, mtbf):.3f}")
    print(f"\n{args.sweep_gpus:,} GPUs, goodput by checkpoint interval (simulated)")
    mtbf = args.gpu_mtbf_hours / args.sweep_gpus
    for minutes in (5, 10, 15, 20, 30, 45, 60, 90, 120, 180):
        row = "  ".join(f"{name.split(' (')[0]} {goodput_simulated(minutes / 60, d, r, mtbf):.3f}"
                        for name, (d, r) in cases.items())
        print(f"  every {minutes:>3} min  {row}")
