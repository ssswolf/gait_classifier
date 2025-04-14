import warnings
warnings.filterwarnings("ignore")

import os
import time
import pywt
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from utils.gait import bandpass_filter, bandpass_filter_original


class DataLoader:
    def __init__(self, data_folder, target_activities, window_size=100, stride=1, fs=32, load_if_available=True):
        self.data_folder = data_folder
        self.target_activities = target_activities
        self.selected_activities = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
        sorted_activities = [activity for activity in target_activities if activity in self.selected_activities]
        remaining_activities = [activity for activity in self.selected_activities if activity not in target_activities]
        random.shuffle(remaining_activities)
        self.selected_activities = sorted_activities + remaining_activities
        self.class_labels = {activity: idx for idx, activity in enumerate(self.selected_activities)}

        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.load_if_available = load_if_available

    def load_data(self):
        ft_path = "dataset/processed/dataset_ft.csv"
        ts_path = "dataset/processed/dataset_ts.npy"
        mag_path = "dataset/processed/dataset_mag.npy"
        label_path = "dataset/processed/labels.npy"

        if self.load_if_available and all(os.path.exists(p) for p in [ft_path, ts_path, mag_path, label_path]):
            print('\nLoading Processed Data from disk...')
            dataset_ft = pd.read_csv(ft_path)
            dataset_ts = np.load(ts_path)
            dataset_mag = np.load(mag_path)
            labels_array = np.load(label_path)
            dataset_ft, dataset_ts, dataset_mag, labels_array = self.balance_dataset(dataset_ft, dataset_ts, dataset_mag, labels_array)
            return dataset_ft, dataset_ts, dataset_mag, labels_array

        print('\nLoading Raw Data and Processing...')
        dataset_ft, dataset_ts, dataset_mag, labels = [], [], [], []
        class_map = {activity: i for i, activity in enumerate(self.selected_activities)}

        for activity in self.selected_activities:
            print(f"Loading from {activity}")
            folder_path = os.path.join(self.data_folder, activity)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path, delimiter="\s+", header=None, names=["coded_x", "coded_y", "coded_z"])  
                for i in range(0, len(data) - self.window_size, self.stride):
                    dataset_ft.append(self.extract_features(data[i: i + self.window_size]))
                    mag, ts = self.extract_timeseries(data[i: i + self.window_size])
                    dataset_ts.append(ts)
                    dataset_mag.append(mag)
                    labels.append(class_map[activity])
        
        dataset_ft = pd.DataFrame(dataset_ft)
        dataset_ts = np.array(dataset_ts)
        dataset_mag = np.array(dataset_mag)
        labels_array = np.array(labels)

        print('\nSaving Processed Data...')
        os.makedirs("dataset/processed", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dataset_ft.to_csv(f"dataset/processed/dataset_ft_{timestamp}.csv", index=False)
        np.save(f"dataset/processed/dataset_ts_{timestamp}.npy", dataset_ts)  
        np.save(f"dataset/processed/dataset_mag_{timestamp}.npy", dataset_mag)  
        np.save(f"dataset/processed/labels_{timestamp}.npy", labels_array)  

        dataset_ft, dataset_ts, dataset_mag, labels_array = self.balance_dataset(dataset_ft, dataset_ts, dataset_mag, labels_array)

        return dataset_ft, dataset_ts, dataset_mag, labels_array
    
    def balance_dataset(self, dataset_ft, dataset_ts, dataset_mag, labels_array):
        print('\nBalancing Dataset...')
        X_ds = dataset_ft.copy()  
        X_ds['target'] = labels_array

        target_counts = X_ds['target'].value_counts()
        
        target_labels = [self.class_labels[activity] for activity in self.target_activities]

        min_len = np.inf
        for label in target_labels:
            if target_counts[label] <= min_len:
                min_len = target_counts[label]

        selected_indices = np.concatenate([
            X_ds[X_ds['target'] == label].sample(n=min_len, random_state=42).index
            for label in target_labels
        ])

        remaining_classes = X_ds[~X_ds['target'].isin(target_labels)]
        remaining_sampled_indices = remaining_classes.sample(n=min_len, random_state=42).index

        final_indices = np.concatenate([selected_indices, remaining_sampled_indices]) 
        np.random.shuffle(final_indices)

        dataset_bal = X_ds.loc[final_indices] 
        dataset_ts_bal = dataset_ts[final_indices]
        dataset_mag_bal = dataset_mag[final_indices]
        dataset_bal.loc[~dataset_bal['target'].isin(target_labels), 'target'] = len(self.target_activities)
        dataset_ft_bal = dataset_bal.drop(columns=['target'])
        labels_array_bal = dataset_bal['target']

        return dataset_ft_bal, dataset_ts_bal, dataset_mag_bal, labels_array_bal

    def extract_timeseries(self, window):
        x, y, z = -1.5 + (window['coded_x'] / 63) * 3, \
                -1.5 + (window['coded_y'] / 63) * 3, \
                -1.5 + (window['coded_z'] / 63) * 3
        
        acc_mag = np.sqrt(x**2 + y**2 + z**2)
        acc_mag_filt = bandpass_filter_original(acc_mag, 0.4, self.fs/2 - 1e-4, self.fs)
        
        return acc_mag_filt, np.column_stack((x, y, z))

    def extract_features(self, window):
        features = {}
        x, y, z = -1.5 + (window['coded_x'] / 63) * 3, \
                -1.5 + (window['coded_y'] / 63) * 3, \
                -1.5 + (window['coded_z'] / 63) * 3

        for axis, signal in zip(['x', 'y', 'z'], [x, y, z]):  
            # Apply bandpass filter
            signal_filt, signal_freq = bandpass_filter(signal, 0.4, self.fs/2 - 1e-4, self.fs)

            # Time-Domain Features
            features.update({
                f'mean_{axis}': np.mean(signal_filt),
                f'std_{axis}': np.std(signal_filt),
                f'var_{axis}': np.var(signal_filt),
                f'rms_{axis}': np.sqrt(np.mean(signal_filt**2)),
                f'median_{axis}': np.median(signal_filt),
                f'iqr_{axis}': stats.iqr(signal_filt),
                f'skewness_{axis}': stats.skew(signal_filt),
                f'kurtosis_{axis}': stats.kurtosis(signal_filt),
                f'zcr_{axis}': np.sum(np.diff(np.sign(signal_filt)) != 0) / len(signal),
                f'peak_to_peak_{axis}': np.ptp(signal_filt),
                f'sma_{axis}': np.sum(np.abs(signal)) / len(signal_filt),
            })

            # Frequency-Domain Features (FFT)
            fft_vals = np.fft.rfft(signal_freq)
            fft_freqs = np.fft.rfftfreq(len(signal_freq), d=1/self.fs)
            power_spectrum = np.abs(fft_vals)**2

            features.update({
                f'dominant_freq_{axis}': fft_freqs[np.argmax(power_spectrum)],
                f'spectral_energy_{axis}': np.sum(power_spectrum),
                f'spectral_entropy_{axis}': -np.sum((power_spectrum / np.sum(power_spectrum)) * np.log2(power_spectrum / np.sum(power_spectrum))),
                f'spectral_centroid_{axis}': np.sum(fft_freqs * power_spectrum) / np.sum(power_spectrum),
                f'spectral_flatness_{axis}': np.exp(np.mean(np.log(power_spectrum))) / np.mean(power_spectrum),
                f'peak_freq_{axis}': fft_freqs[np.argmax(power_spectrum)]
            })

            # Wavelet Features
            wavelet = 'db4'
            coeffs = pywt.wavedec(signal_freq, wavelet, level=2)

            for i, c in enumerate(coeffs):
                features.update({
                    f'wavelet_energy_{axis}_{i}': np.sum(c**2),
                    f'wavelet_entropy_{axis}_{i}': -np.sum((c**2 / np.sum(c**2)) * np.log2(c**2 / np.sum(c**2))),
                    f'wavelet_mean_{axis}_{i}': np.mean(c),
                    f'wavelet_std_{axis}_{i}': np.std(c),
                    f'wavelet_skew_{axis}_{i}': stats.skew(c),
                    f'wavelet_kurtosis_{axis}_{i}': stats.kurtosis(c)
                })

        return features