import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_all_files_paths(path,allFiles):
    folders = os.listdir(path)
    for i in folders:
        if os.path.isdir(path + "\\" + i):
            read_all_files_paths(path + "\\" + i,allFiles)
        else:
            if i.count("Raw") > 0:
                allFiles.append(path + "\\" + i)
    return


sampling_rate = 128


def file_reading_reshaping_with_overlap(path, length, overlap):
    overlap_step = int(length - overlap) - 1
    data = pd.read_csv(path)
    data = np.array(data)
    data = data[:, 0:14]
    data = data - np.mean(data, axis=0)
    samples = []
    for i in range((data.shape[0] // sampling_rate) - 5):
        samples.append(data[(i + overlap_step) * sampling_rate:(i + length) * sampling_rate, :])
    samples = np.array(samples)
    return samples


file_reading_reshaping_with_overlap(
    "C:\\Users\\Kagero\\PycharmProjects\\GAME-Emotion-Classification\\GAMEEMO\\(S01)\\Raw EEG Data\\.csv format\\S01G1AllRawChannels.csv",
    5, 4)


def band_pass(p1, p2, freqs, power_FFT):
    ids = [np.where(freqs >= p1)[0][0], np.where(freqs <= p2)[0][-1]]
    freqs_s = freqs[ids[0]:ids[1]]
    power_FFT_s = power_FFT[ids[0]:ids[1], :]
    return freqs_s, power_FFT_s


def read_plot(path, p1, p2, time=500):
    data = pd.read_csv(path)
    columns = data.columns
    data = np.array(data)
    data = data[:, 0:14]
    data = data - np.mean(data, axis=0)
    for i in range(14):
        plt.plot(data[:500, i] + (100 * i), label=str(columns[i]))
    plt.legend(loc='upper right')
    plt.title(path[68:73] + "  " + path[102:104])
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    power_FFT = (np.abs(np.fft.rfft(data, axis=0)) ** 2) / (data.shape[0] * 128)
    freqs = np.fft.rfftfreq(data.shape[0], 1 / 128)
    if p2==-1:
        p2=freqs[-1]
    freqs, power_FFT = band_pass(p1, p2, freqs, power_FFT)
    power_FFT = np.mean(power_FFT, axis=1)
    plt.bar(freqs, power_FFT)
    plt.title("Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.show()
