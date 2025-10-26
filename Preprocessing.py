import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

def read_all_files_paths(path, allFiles):
    folders = os.listdir(path)
    for i in folders:
        if os.path.isdir(path + "\\" + i):
            read_all_files_paths(path + "\\" + i, allFiles)
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
    Y=np.zeros((samples.shape[0], 4))
    Y[:,int(path[103])-1]=int(path[103])-1
    return samples,Y


file_reading_reshaping_with_overlap(
    "C:\\Users\\Kagero\\PycharmProjects\\GAME-Emotion-Classification\\GAMEEMO\\(S01)\\Raw EEG Data\\.csv format\\S01G1AllRawChannels.csv",
    5, 4)


def band_pass(p1, p2, freqs, power_FFT):
    ids = [np.where(freqs >= p1)[0][0], np.where(freqs <= p2)[0][-1]]
    freqs_s = freqs[ids[0]:ids[1]]
    power_FFT_s = power_FFT[ids[0]:ids[1], :]
    return freqs_s, power_FFT_s


def spectrogram_plot(path, p1, p2, time=500):
    data = pd.read_csv(path)
    columns = data.columns
    data = np.array(data)
    data = data[:, 0:14]
    data = data - np.mean(data, axis=0)
    for i in range(14):
        plt.plot(data[:500, i] + (100 * i), label=str(columns[i]))
    plt.legend(loc='upper right')
    plt.title(path[68:73] + " " + path[102:104])
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    power_FFT = (np.abs(np.fft.rfft(data, axis=0)) ** 2) / (data.shape[0] * 128)
    freqs = np.fft.rfftfreq(data.shape[0], 1 / 128)
    if p2 == -1:
        p2 = freqs[-1]
    freqs, power_FFT = band_pass(p1, p2, freqs, power_FFT)
    power_FFT = np.mean(power_FFT, axis=1)
    plt.bar(freqs, power_FFT)
    plt.title("Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.show()


def K_fold_test(K=5, X=None, Y=None, model=None, metrics=None):
    if X is None or Y is None or model is None or metrics is None:
        raise ValueError('Please provide X, Y and model and metrics')
    folds_results = {}
    fold_size = X.shape[0] // K
    ides = np.arange(X.shape[0])
    np.random.shuffle(ides)
    for i in range(K):
        fold = {"fold number": i + 1}
        mlflow.log_param("Fold number", i + 1)
        test_ides = ides[i * fold_size:(i + 1) * fold_size]
        train_ides = np.concatenate(ides[:i * fold_size], ides[(i + 1) * fold_size:])
        x_train, y_train = X[train_ides], Y[train_ides]
        x_test, y_test = X[test_ides], Y[test_ides]
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)
        if type(metrics) == list:
            for metric in metrics:
                metric_result=metric(y_test, y_prediction)
                fold["Metric " + metric.__name__ + " result "] = metric_result
                mlflow.log_param("Metric " + metric.__name__ + " result ", metric_result)
        folds_results[i + 1] = fold
        return folds_results

def data_preparing_and_reshaping_with_overlap(path, length, overlap):
    all_files = []
    read_all_files_paths(path,all_files)
    X=[]
    Y=[]
    for i in all_files:
        x,y=file_reading_reshaping_with_overlap(i, length, overlap)
        X.append(x)
        Y.append(y)
    X=np.array(X)
    Y=np.array(Y)
    return X,Y

#mlflow ui