# BeauPPG - Heart Rate Estimation from PPG signals

This is the official code release for the UAI 2023 conference paper **"BeaU-PPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation"** [Link]

TL; DR

BeauPPG is a probabilistic heart rate inference framework. Taking multi-channel PPG and Accelerometer signals as input, it predicts the instantaneous heart rate and provides an uncertainty estimate for the prediction. This repository contains code to run leave-one-session-out cross-validation experiments on multiple supported datasets.

## Abstract
 We present a novel learning-based method that achieves state-of-the-art performance on several heart rate estimation benchmarks extracted from photoplethysmography signals (PPG). We consider the evolution of the heart rate in the context of a discrete-time stochastic process that we represent as a hidden Markov model. We derive a distribution over possible heart rate values for a given PPG signal window through a trained neural network. Using belief propagation, we incorporate the statistical distribution of heart rate changes to refine these estimates in a temporal context. From this, we obtain a quantized probability distribution over the range of possible heart rate values that captures a meaningful and well-calibrated estimate of the inherent predictive uncertainty. We show the robustness of our method on eight public datasets with three-different cross-validation experiments.*


## Getting Started
To re-train and evaluate the model in leave-one-session-out (LoSo) cross-validation, run the following lines of code in your terminal:

`pip install -r requirements.txt`

 `sh download_data.sh`
 
`python train_eval.py --data_dir ${DATA_PATH} --dataset dalia `

This will trigger the following steps:
1. **Downloads the datasets** DaLiA, WESAD, BAMI-1 and BAMI-2 from their original hosts. *Note that WESAD does not natively include  ground truth HR. Labels can be gererated from the provided ECG measurements instead.  Also note that support for the IEEE datasets is implemented, but the original data format seems to be no longer available. You can download it in the new format under https://zenodo.org/record/3902710#.ZGM9l3ZBy3C and restructure/convert the files or implement your own file reader.*
2. **Runs LoSo cross-validation** on the DaLiA dataset. On a modern GPU, expect one full run to take roughly 10-14 hours.
3. **Saves the results**, that is the MAEs, the predictions and the models. The output directory can be modified with the `--out_dir` argument. Set the `--use_wandb` flag to get additional logging data. *Note that you may have to debug your h5py installation in order for the models to be saved correctly.*

## Weights & Biases

We highly recommend that you use Weights&Biases to monitor model training. Make sure to log into W&B in the console and then simply add the argument `--use_wandb`.


