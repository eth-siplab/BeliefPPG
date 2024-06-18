## BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation (UAI 2023, Official Code)

Valentin Bieri<sup>*2</sup>, [Paul Streli](https://paulstreli.com/)<sup>*1</sup>, Berken Utku Demirel<sup>1</sup>, [Christian Holz](https://www.christianholz.net/)<sup>1</sup>

<sup>1</sup> [Sensing, Interaction & Perception Lab](https://siplab.org/), Department of Computer Science, ETH Zürich, Switzerland
<br>
<sup>2</sup> MSc Student, Department of Computer Science, ETH Zürich, Switzerland
<br>
<sup>*</sup> These authors contributed equally to this work

___________

<p align="center">
<img src="plot.svg" width="350">
</p>

---

> We present a novel learning-based method that achieves state-of-the-art performance on several heart rate estimation benchmarks extracted from photoplethysmography signals (PPG). We consider the evolution of the heart rate in the context of a discrete-time stochastic process that we represent as a hidden Markov model. We derive a distribution over possible heart rate values for a given PPG signal window through a trained neural network. Using belief propagation, we incorporate the statistical distribution of heart rate changes to refine these estimates in a temporal context. From this, we obtain a quantized probability distribution over the range of possible heart rate values that captures a meaningful and well-calibrated estimate of the inherent predictive uncertainty. We show the robustness of our method on eight public datasets with three different cross-validation experiments.

Contents
----------

<b>TL; DR</b>
<br>
This repository contains instructions on how to install BeliefPPG for inference and code to run leave-one-session-out cross-validation experiments on multiple supported datasets. Taking multi-channel PPG and Accelerometer signals as input, BeliefPPG predicts the instantaneous heart rate and provides an uncertainty estimate for the prediction.

- [Install](#install)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Training and Inference](#training-and-inference) 
- [Citation](#citation)
- [License and Acknowledgement](#license-and-acknowledgement)

Install
----------
You can install the pip package using:
```bash
pip install beliefppg
```

Quick Start
----------
To start inferring heart rate from PPG and accelerometer data, you first need to import the `infer_hr` function from the `beliefppg` package.
The function `infer_hr` returns the estimated heart rate and the corresponding midpoint indices of the windows used for HR inference.
```python
from beliefppg import infer_hr

# Example of a simple function call (adjust 'ppg' and 'sampling_rate' as per your data)
hr, idxs = infer_hr(ppg, sampling_rate)
```

The accuracy of BeliefPPG can be enhanced by incorporating accelerometer data alongside the PPG signals.
```python
from beliefppg import infer_hr

ppg_sampling_rate = 128  # Hz (sampling rate of ppg sensor)
acc_sampling_rate = 128 # Hz (sampling rate of accelerometer)

# Load data item containing the PPG, HR, and IMU signals --- challenging custom dataset
data = np.load('Data/example.npy', allow_pickle=True).item()

ppg = data['PPG head'].reshape((-1,1)) # reshape ppg to (n_samples, n_channels)
IMU_X = data['IMU X head']
IMU_Y = data['IMU Y head']
IMU_Z = data['IMU Z head']
acc = np.stack([IMU_X,IMU_X, IMU_Z], axis=-1)

hr, idxs = infer_hr(
    ppg, # PPG signal data with shape (n_samples, n_channels)
    ppg_sampling_rate, # Sampling frequency of the PPG signal in Hz
    acc=acc, # Accelerometer signal data with shape (n_samples, n_channels). BeliefPPG to function without accelerometer signal data, but its accuracy may be reduced.
    acc_freq=acc_sampling_rate, # Sampling frequency of the accelerometer signal in Hz
)
```

The `infer_hr_uncertainty` function from the `beliefppg` package returns the estimated heart rate, uncertainty estimates about the heart rate values, and the bounds [start_ts, end_ts] of the corresponding windows used for HR inference in seconds.
Additionally, users can choose between belief propagation and Viterbi decoding, specify the uncertainty measure, and decide whether to disable the time-domain backbone of the network architecture. Detailed explanations of these features can be found in our [paper](https://static.siplab.org/papers/uai2023-beliefppg.pdf) and [supplementary material](https://static.siplab.org/papers/uai2023-beliefppg-supplementary.pdf).
```python
from beliefppg import infer_hr

ppg_sampling_rate = 128  # Hz (sampling rate of ppg sensor)
acc_sampling_rate = 128 # Hz (sampling rate of accelerometer)

# Load data item containing the PPG, HR, and IMU signals --- challenging custom dataset
data = np.load('Data/example.npy', allow_pickle=True).item()

ppg = data['PPG head'].reshape((-1,1)) # reshape ppg to (n_samples, n_channels)
IMU_X = data['IMU X head']
IMU_Y = data['IMU Y head']
IMU_Z = data['IMU Z head']
acc = np.stack([IMU_X,IMU_X, IMU_Z], axis=-1)

hr, uncertainty, time_intervals = infer_hr_uncertainty(
    ppg, # PPG signal data with shape (n_samples, n_channels)
    ppg_sampling_rate, # Sampling frequency of the PPG signal in Hz
    acc=acc, # Accelerometer signal data with shape (n_samples, n_channels). BeliefPPG to function without accelerometer signal data, but its accuracy may be reduced.
    acc_freq=acc_sampling_rate, # Sampling frequency of the accelerometer signal in Hz
    decoding='sumproduct', # Decoding method to use, either "sumproduct" or "viterbi"
    use_time_backbone=True, # Whether to use the time-domain backbone or not
    uncertainty='std' # Measure for predictive uncertainty, either "entropy" or "std"
)
# The function returns predicted heart rates in BPM, uncertainties (entropy or std), and time intervals in seconds.
```
For a complete example demonstrating how to use BeliefPPG for heart rate inference, see the [tutorial notebook](https://github.com/eth-siplab/BeliefPPG/blob/master/tutorial.ipynb).

Datasets
----------
We provide a shell script which downloads the datasets DaLiA, WESAD, BAMI-1 and BAMI-2 from their original hosts. Run the following line in your terminal:

```
sh download_data.sh
```
- Note that WESAD does not natively include ground-truth HR. Labels can be generated from the provided ECG recordings instead.
- Support for the IEEE datasets is implemented, but the original data format seems to be no longer available. You can download it in the new format under https://zenodo.org/record/3902710#.ZGM9l3ZBy3C and restructure/convert the files or implement your own file reader.

Training and Inference
----------
Run the following in your terminal: 

```
python train_eval.py --data_dir ${DATA_PATH} --dataset dalia 
```

This will run LoSo cross-validation on the DaLiA dataset. On a modern GPU, expect one full run to take roughly 10&ndash;14 hours.
Results, that is the MAEs, predictions and models, are saved in the output directory, which can be specified with the `--out_dir` argument. *Note that you may have to reinstall h5py in order for the models to be saved correctly.*

We highly recommend that you use Weights&Biases to monitor model training. Make sure to log into W&B in the console and then simply add the argument `--use_wandb` to save additional plots and logging information.


Citation
----------
If your find our paper or codes useful, please cite our work:

    @InProceedings{uai2023-beliefppg,
        author={Bieri, Valentin and Streli, Paul and Demirel, Berken Utku and Holz, Christian},
        title = {BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation},
        year = {2023},
        organization={PMLR},
        booktitle = {Conference on Uncertainty in Artificial Intelligence (UAI)}
    }

License and Acknowledgement
----------
This project is released under the MIT license.



