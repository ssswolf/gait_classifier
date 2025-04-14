import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter_original(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def detect_steps(accel_data, fs):
    filtered_signal = accel_data.copy() # bandpass_filter_original(accel_data, 0.25, fs/2-1e-4, fs) 
    peak_indices, _ = find_peaks(filtered_signal, height=0.1, distance=fs//2) 
    step_times = peak_indices / fs
    return step_times, peak_indices, filtered_signal

def locomotion_classification(step_times):
    step_intervals = np.diff(step_times)
    mean_interval = np.mean(step_intervals) if len(step_intervals) > 0 else np.inf
    cadence = 1/mean_interval
    
    if mean_interval < 0.6:  
        return "Running", cadence
    elif mean_interval < 1.2:
        return "Walking", cadence
    else:
        return "Stationary", 0

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data + (np.mean(data) - np.mean(filtered_data)), filtered_data