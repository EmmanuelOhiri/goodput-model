# Training goodput model

Code and results for the CUDO Compute article [How to enhance hardware reliability for AI acceleration at scale](https://www.cudocompute.com/blog/enhance-hardware-reliability-for-ai-acceleration-at-scale). The script estimates how much of a large training run's time ends in saved progress, given how often the job fails, how long a checkpoint takes to write and how long a restart takes.

## What the script models

A synchronous training job saves a checkpoint every few minutes. When any GPU fails, the job loses the work done since its last checkpoint and pays a fixed restart time, which covers detecting the failure, replacing the server and reloading the checkpoint. Failures arrive at random, with exponential gaps between them, at a rate that grows in step with the number of GPUs.

The script reports goodput, the share of wall-clock time that ends in saved progress, in two ways:

- **First-order formula:** `1 - δ/τ - (τ/2 + R)/M`, where δ is the checkpoint write time, τ the checkpoint interval, R the restart time and M the job's mean time between failures. It is quick but only holds while failures are rare next to the checkpoint interval.
- **Simulation:** 20 years of simulated running per setting, which counts work as saved only once the checkpoint after it has been written and charges a restart for every failure. Where the two disagree, the simulation is the better guide.

The checkpoint interval comes from [Young's 1974 formula](https://graal.ens-lyon.fr/~abenoit/CR02/papers/young.pdf), `τ ≈ √(2δM)`, unless the sweep sets it.

## Inputs

| Input | Default | Where it comes from |
|---|---|---|
| Hours of running per GPU between interruptions | 50,677 | 419 unexpected interruptions in 54 days on about 16,384 GPUs, from the [Llama 3 report](https://arxiv.org/abs/2407.21783) |
| Slow case | 5 minute checkpoint, 30 minute restart | An assumption |
| Fast case | 30 second checkpoint, 10 minute restart | An assumption, in line with the checkpoint and detection times in the [MegaScale paper](https://arxiv.org/abs/2402.15627) |

The checkpoint and restart times are assumptions, so every output is an estimate, not a measurement.

## Results

With the default inputs, each case at its best checkpoint interval:

| GPUs | Job fails every | Slow: interval | Slow: goodput | Fast: interval | Fast: goodput |
|---|---|---|---|---|---|
| 1,024 | 49.5 hours | 172 min | 93.4% | 54 min | 97.9% |
| 4,096 | 12.4 hours | 86 min | 85.3% | 27 min | 95.1% |
| 16,384 | 3.1 hours | 43 min | 67.5% | 14 min | 88.1% |
| 32,768 | 93 min | 30 min | 53.3% | 10 min | 81.2% |
| 65,536 | 46 min | 22 min | 36.5% | 7 min | 70.7% |
| 100,000 | 30 min | 17 min | 26.4% | 6 min | 62.4% |

Goodput figures are simulated. At 65,536 and 100,000 GPUs in the slow case, the first-order formula drops to zero while the simulation still finds a quarter to a third of the time kept, because a restart there takes up a large share of the time between failures. The full output, including the formula's figures and a sweep of checkpoint intervals at 16,384 GPUs, is in `results/default_output.txt`.

## Files

| File | Contents |
|---|---|
| `goodput_model.py` | The model: Young's interval, the first-order formula, the simulation and the report. |
| `results/default_output.txt` | The output of a run with the default inputs, which matches every figure in the article. |
| `requirements.txt` | The one package the script needs. |

## Running it

```bash
pip install -r requirements.txt
python goodput_model.py
```

It runs in about a second on a CPU and needs no GPU.

To use your own numbers, give a failure rate measured on your hardware, or a published figure for similar hardware, and the checkpoint and restart times your stack achieves:

```bash
python goodput_model.py --gpu-mtbf-hours 30000 --checkpoint-min 2 --restart-min 20 --gpus 2048 8192
```

`--checkpoint-min` and `--restart-min` go together and replace the two default cases. `--sweep-gpus` sets the job size for the checkpoint-interval sweep. `python goodput_model.py --help` lists every option.

## Limits

The model assumes failures arrive independently at a steady rate, every restart takes the same time, and no GPU runs slowly. Real clusters break each of those assumptions; failures can cluster, a missing spare server can stretch one restart into hours, and a single slow GPU holds back every step. Treat the outputs as a way to compare the levers, faster checkpoints against faster restarts against fewer failures, not as a prediction for a specific cluster.

The simulation uses seed 0, so repeated runs print the same numbers. The recorded run used Python 3.13 and NumPy.

The code in this repository is released under the [MIT License](LICENSE).
