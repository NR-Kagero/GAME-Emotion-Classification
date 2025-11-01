import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

sampling_rate = 128


def band_pass(p1, p2, freqs, power_FFT):
    ids = [np.where(freqs >= p1)[0][0], np.where(freqs <= p2)[0][-1]]
    freqs_s = freqs[ids[0]:ids[1]]
    power_FFT_s = power_FFT[ids[0]:ids[1], :]
    return freqs_s, power_FFT_s


def fft_power_freqs(data, p1, p2, return_freqs=True):
    power_FFT = (np.abs(np.fft.rfft(data, axis=0)) ** 2) / (data.shape[0] * sampling_rate)
    freqs = np.fft.rfftfreq(data.shape[0], 1 / sampling_rate)
    if p2 == -1:
        p2 = freqs[-1]
    freqs, power_FFT = band_pass(p1, p2, freqs, power_FFT)
    power_FFT = np.mean(power_FFT, axis=1)
    if return_freqs:
        return freqs, power_FFT
    else:
        return power_FFT


def read_all_files_paths(path, allFiles):
    folders = os.listdir(path)
    for i in folders:
        if os.path.isdir(path + "\\" + i):
            read_all_files_paths(path + "\\" + i, allFiles)
        else:
            if i.count("Raw") > 0:
                allFiles.append(path + "\\" + i)
    return


def file_reading_preprocessing_reshaping_with_overlap(path, length, overlap):
    overlap_step = int(length - overlap) - 1
    data = pd.read_csv(path)
    data = np.array(data)
    data = data[:, 0:14]
    data = data - np.mean(data, axis=0)
    samples = []
    for i in range((data.shape[0] // sampling_rate) - length + 1):
        samples.append(
            fft_power_freqs(data[(i + overlap_step) * sampling_rate:(i + length) * sampling_rate, :], 1.6, -1, False))
    X = np.array(samples)
    Y = np.zeros((X.shape[0], 4))
    Y[np.arange(X.shape[0],dtype=np.int16), np.zeros(X.shape[0],dtype=np.int16)+ (int(path[103]) - 1)] = 1
    return X, Y


def plot(data, freqs, power_FFT, title, columns):
    for i in range(14):
        plt.plot(data[:500, i] + (100 * i), label=str(columns[i]))
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    current_figure = plt.gcf()
    mlflow.log_figure(current_figure, "plots/" + title + ".png")
    plt.bar(freqs, power_FFT)
    plt.title("Frequency " + title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.show()
    current_figure = plt.gcf()
    mlflow.log_figure(current_figure, "plots/Frequency " + title + ".png")
    return


def file_plot(path, p1, p2, time=500):
    data = pd.read_csv(path)
    columns = data.columns
    data = np.array(data)
    data = data[:, 0:14]
    data = data - np.mean(data, axis=0)
    freqs, power_FFT = fft_power_freqs(data, p1, p2)
    plot(data, freqs, power_FFT, path[68:73] + " " + path[102:104], columns)


def k_fold_test(K=5, X=None, Y=None, model=None, metrics=None):
    if X is None or Y is None or model is None or metrics is None:
        raise ValueError('Please provide X, Y and model and metrics')
    folds_results = {}
    fold_size = X.shape[0] // K
    ides = np.arange(X.shape[0],dtype=np.int64)
    np.random.shuffle(ides)
    for i in range(K):
        fold = {}
        mlflow.log_param("Fold number", i + 1)
        test_ides = ides[i * fold_size:(i + 1) * fold_size]
        train_ides = np.concatenate( (ides[:i * fold_size], ides[(i + 1) * fold_size:]) )
        x_train, y_train = X[train_ides], Y[train_ides]
        x_test, y_test = X[test_ides], Y[test_ides]
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)
        y_prediction = np.argmax(y_prediction, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print(y_prediction)
        print(y_test)
        if type(metrics) == list:
            for metric in metrics:
                metric_name = metric.__name__
                if metric_name in ['f1_score', 'recall_score', 'precision_score']:
                    metric_result = metric(y_test, y_prediction, average='macro')
                else:
                    metric_result = metric(y_test, y_prediction)
                print(f"{metric_name}: {metric_result}")
                fold["Metric " + metric_name + " result "] = metric_result
                mlflow.log_param("Metric " + metric_name + " result ", metric_result)
        folds_results[i + 1] = fold
    return folds_results


def data_preparing_preprocessing_reshaping_with_overlap(path, length, overlap):
    all_files = []
    read_all_files_paths(path, all_files)
    X = []
    Y = []
    for i in all_files:
        x, y = file_reading_preprocessing_reshaping_with_overlap(i, length, overlap)
        X.extend(x)
        Y.extend(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def k_fold_results_plots(results):
    folds = results.keys()
    values = list()
    metrics = list(folds[0].keys())
    for fold in folds:
        values.append(list(results[fold].values()))
    values = np.array(values)
    for metric in metrics:
        plt.plot(np.arange(len(folds)) + 1, values[:, metric.index(metric)], label=metric)
        plt.ylabel("Result")
        plt.xlabel("Fold")
        plt.title(metric + " plot")
        plt.text(10, 10, f"Max = {max(values[:, metric.index(metric)])}")
        plt.show()
        current_figure = plt.gcf()
        mlflow.log_figure(current_figure, "plots/" + metric + ".png")
    return values

