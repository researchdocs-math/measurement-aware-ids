# Data attribution

The files in `data/packets/` are derived, payload-free packet-header arrays.
They retain relative timestamps, original packet indices, IP packet lengths,
transport flags, transport protocol, anonymous communication-group indices,
and inherited binary labels. Network addresses and port values have been
replaced by local group indices. The transformation and eligibility rules are
implemented in `src/prepare.py` and `src/prepare_kitsune.py`.

## IoT-23

Sebastian Garcia, Agustin Parmisano, and Maria Jose Erquiaga (2020),
*IoT-23: A labeled dataset with malicious and benign IoT network traffic*,
version 1.0.0. DOI: https://doi.org/10.5281/zenodo.4743746.
Dataset: https://www.stratosphereips.org/datasets-iot23.
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/.

The eight selected captures are identified in `configs/sources.json`.
The transformation keeps labeled, unfragmented IPv4 TCP/UDP packets and
excludes communication windows containing unmatched packets.

## Kitsune

Yisroel Mirsky and the Kitsune dataset contributors, *Kitsune Network Attack
Dataset*, UCI Machine Learning Repository (2019).
DOI: https://doi.org/10.24432/C5D90Q.
Official dataset record: https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset.
The author's public packet captures and labels are distributed at
https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune.
The UCI dataset is licensed under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.

The four captures are Mirai Botnet, OS Scan, Fuzzing, and SYN DoS.
Packet labels are aligned against every original packet before filtering to
unfragmented IPv4 TCP/UDP traffic. A window is malicious if any included packet
has an attack label. Source file checksums are in `configs/kitsune_sources.json`.

## Changes and reuse

The derived arrays are distributed under CC BY 4.0 with the above attribution.
The original labels are inherited from their creators. Reusers should retain
these credits and identify further modifications. The original experiment
software is licensed under the MIT License in `LICENSE`.
