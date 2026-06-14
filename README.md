\# CR-PGNN: Camouflage-Resistant Graph Neural Network for Power Grid Anomaly Detection



A PyTorch implementation for the paper below:



\*\*Camouflage-Resistant Graph Neural Networks for Power Grid Anomaly Detection\*\*



Ya Guo, Junyi Wang, Boyu Liu, Dong Li, Zhaotai Meng, Ying Zhu.



\[Paper] \[Code] \[Dataset]



\## Overview



Camouflage-Resistant Graph Neural Network (CR-PGNN) is a GNN-based anomaly detector designed for power grid cyber-physical systems. It incorporates three modules that enhance detection performance against stealthy camouflaged anomalies:



\- \*\*A multi-head physics consistency (MPC) module\*\* which computes physics-aware similarity scores between neighboring nodes by checking AC power flow residuals, exposing feature-camouflaged anomalies that manipulate measurements to mimic normal patterns.



\- \*\*A reinforcement learning (RL) neighbor selector\*\* which dynamically optimizes filtering thresholds to adaptively prune relation-camouflaged connections, preventing anomaly signals from being diluted by dense healthy neighborhoods.



\- \*\*A relation-aware gated aggregator\*\* which fuses information across multiple grid relations (physical, geographic, logical) using attention weights derived from learned thresholds.



\## Advantages



CR-PGNN has the following advantages:



\- \*\*Camouflage resistance\*\*. CR-PGNN detects both feature camouflage (via physics consistency) and relation camouflage (via RL-based pruning), unlike purely data-driven methods.



\- \*\*Domain knowledge integration\*\*. The MPC module incorporates AC power flow equations, providing a physical constraint that pure statistical methods lack.



\- \*\*Adaptability\*\*. The RL neighbor selector adaptively adjusts pruning thresholds per node and per relation without manual tuning.



\- \*\*Lightweight\*\*. The RL policy network has only 1,632 parameters, and the full model size is 0.52 MB in float32.



\## Setup



Download the project and install the required packages:



```bash

git clone https://github.com/123gaoyi/CR-PGNN

cd CR-PGNN

pip install -r requirements.txt

Running

Unzip the IEEE benchmark datasets:



bash

unzip data/IEEE14-bus.zip

unzip data/IEEE57-bus.zip

unzip data/IEEE118-bus.zip

Preprocess the data:



bash

python data\_process.py

Train and evaluate CR-PGNN:



bash

python train.py --dataset IEEE118 --epochs 200

For other dataset and parameter settings, refer to the argument parser in train.py. The model supports both CPU and GPU modes.



Running on Your Datasets

To run CR-PGNN on your own power grid data, prepare the following:



Multi-relational graph: A graph with multiple relation types (physical transmission lines, geographic proximity, logical control dependencies). Each relation is stored as an adjacency list.



Node features: PMU measurement vectors (voltage magnitude, phase angle, active/reactive power) stored in a numpy array or scipy.sparse matrix.



Node labels: Binary labels (0: normal, 1: anomalous) stored in a numpy array.



The feature extraction pipeline for PMU data is provided in utils.py.

Dependencies

Python >= 3.8



PyTorch >= 2.0



DGL (Deep Graph Library) >= 1.0



NumPy, SciPy, scikit-learn

