# Intrusion Detection under Measurement Shift

Reproducible packet-replay experiments for separating collection changes from
changes in network behavior. The study crosses Bernoulli packet retention
probabilities (1, 0.5, 0.1, 0.01) with idle timeouts (5, 30, 60 seconds) and active
timeouts (30, 120, 600 seconds) on eight IoT-23 and four Kitsune captures.

## Contents

- `src/`: acquisition, packet preparation, sampling and metering, six classifier
  baselines, and two drift-monitoring experiments.
- `data/packets/`: compact, payload-free labeled packet-header arrays.
- `configs/`: exact settings, feature definitions, source URLs, and SHA-256 hashes.
- `results/expected/`: numerical reference outputs from the completed experiment.
- `tests/`: conservation, timing, grouping, label, and metric checks.
- `MANIFEST.sha256`: integrity hashes for the packaged data and reference outputs.
- `DATA_LICENSE.md`: source credits, transformation details, and data licenses.

## Environment

Python 3.12 was used on macOS with an Apple M3 Pro (12 CPU cores, 18 GB unified
memory). Create an isolated environment and install the pinned packages:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_integrity.py
python -m unittest discover -s tests -v
```

## Reproduce the numerical experiments

```sh
python run_experiments.py --workers 4 --device cpu
python compare_results.py
```

On Apple Silicon, use `--device mps` to accelerate the random-feature and batch
mean operations on the GPU. Tree training and packet metering use CPU workers.
The CPU option is the portable numerical reference. Sampling, splitting, model,
and resampling seeds are fixed. Serial tree prediction fixes accumulation order
at threshold ties. Small floating-point changes across processor/library builds
can affect decisions exactly at an empirical cutoff; `compare_results.py`
reports both exact integer agreement and maximum numeric differences. Scientific
metrics are compared; runtime measurements vary with hardware.

The default run rebuilds all 1,728 measurement exports from the included header
arrays, evaluates every baseline and setting, and writes CSV results into
`results/`. It reuses existing exports and completed classifier files. To force
an entirely fresh run, use `python run_experiments.py --fresh --workers 4
--device cpu`; this removes generated feature caches and experiment outputs
while retaining the packaged reference outputs and input arrays.

Classifier evaluation uses both leave-one-capture-out transfer and disjoint
five-tuple holdouts within captures. Sampling seeds and grouped split seeds are
repeated measurements, not independent deployment sites. Recall includes attack
windows with no retained packets. A second recall column conditions on visibility.
The drift control uses independent draws from the same finite empirical
population; the reference-mismatch experiment uses disjoint packet-window pools
and independently sampled current traffic. Its unchanged label-mixture setting
is a stress test, not a guaranteed identical-distribution null.

## Rebuild packet arrays from public sources

```sh
python src/download.py
python src/prepare.py
python src/prepare_kitsune.py
```

The downloader retrieves and verifies the pinned public source files and
extracts the Kitsune archives. Preparation checks packet/label alignment and
writes capture audits. Existing prepared arrays are reused; move the corresponding
`data/packets/<capture>.npz` and `.json` aside to regenerate that capture.
Source downloads require several gigabytes of storage and an internet connection.
The included header arrays allow the experiment workflow to run offline.

## Interpretation

The exporter samples packets before applying timeouts. It implements a documented
timeout-only meter with a 600-second observation-window flush. IP addresses and
ports identify communication groups but do not enter the classifier. The code
models Bernoulli selection and does not emulate packet loss caused by congestion.

Configuration-matched replay is compared with a frozen detector, inverse-
probability count correction, threshold-only replay, pooled configuration-aware
training, and unweighted replay. The results expose visibility limits and cases
where representative calibration or transfer between captures fails. Use both
recall and false-positive rates when comparing methods.

## Citation and license

Use `CITATION.cff` for software citation. The original code is MIT licensed.
Derived data retain CC BY 4.0 terms and source attribution in `DATA_LICENSE.md`.
