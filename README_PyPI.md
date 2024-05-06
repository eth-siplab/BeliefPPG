## BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation

BeliefPPG is a novel learning-based method that achieves state-of-the-art performance on several heart rate estimation benchmarks extracted from photoplethysmography signals (PPG). It considers the evolution of the heart rate in the context of a discrete-time stochastic process that is represented as a hidden Markov model. It derives a distribution over possible heart rate values for a given PPG signal window through a trained neural network. Using belief propagation, it incorporates the statistical distribution of heart rate changes to refine these estimates in a temporal context. From this, it obtains a quantized probability distribution over the range of possible heart rate values that captures a meaningful and well-calibrated estimate of the inherent predictive uncertainty.

Install
----------
You can install the pip package using:
```bash
pip install beliefppg
```

Quick Start
----------
To start inferring heart rate from PPG and accelerometer data, you first need to import the `infer_hr` function from the `beliefppg` package.

```python
from beliefppg import infer_hr

sampling_rate = 128  # Hz (sampling rate of accelerometer and ppg sensor)

# Load data item containing the PPG, HR, and IMU signals --- challenging custom dataset
data = np.load('Data/example.npy', allow_pickle=True).item()

ppg = data['PPG head'].reshape((-1,1)) # reshape ppg to (n_samples, n_channels)
IMU_X = data['IMU X head']
IMU_Y = data['IMU Y head']
IMU_Z = data['IMU Z head']
acc = np.stack([IMU_X,IMU_X, IMU_Z], axis=-1)

y_pred, y_uncertainty, time_intervals = infer_hr(
    ppg=ppg, # PPG signal data with shape (n_samples, n_channels)
    acc=acc, # Accelerometer signal data with shape (n_samples, n_channels)
    ppg_freq=sampling_rate, # Sampling frequency of the PPG signal in Hz 
    acc_freq=sampling_rate, # Sampling frequency of the accelerometer signal in Hz
    decoding='sumproduct', # Decoding method to use, either "sumproduct" or "viterbi"
    use_time_backbone=True, # Whether to use the time-domain backbone or not
    uncertainty="std" # Metric for predictive uncertainty, either "entropy" or "std"
)
# The function returns predicted heart rates in BPM, uncertainties, and time intervals in seconds.
```
For a complete example demonstrating how to use BeliefPPG for heart rate inference, see the [tutorial notebook](https://github.com/eth-siplab/BeliefPPG/blob/master/tutorial.ipynb).

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



