#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：zk_server 
@File    ：test.py
@Author  ：gjg
@Date    ：2025/10/14 10:10 
@explain : init
'''
import time, os, mne
import numpy as np
import json
import mne
import pyedflib
# from edflib import edfreader
from scipy.signal import butter, remez, lfilter, filtfilt, kaiser_atten, kaiser_beta, firwin, welch, sosfilt
import matplotlib.pyplot as plt

epoch_length = 10  # s

def sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels):
    train_data = []
    start_time = start_seconds
    # epoch_length = 10
    step_size = epoch_length  # 10
    while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
        # features = []
        start, stop = raw.time_as_index([start_time, start_time + epoch_length])
        # temp = raw[:, start:stop][0]
        temp = raw[[raw.ch_names.index(chn) for chn in channels], start:stop][0]
        train_data.append(temp)
        start_time += step_size

    train_data = np.array(train_data)
    return train_data


def multi_relative_band_energy(eeg_signal, fs, bp_type, bands=None):
    """
    计算EEG信号的相对频带能量

    参数:
    --------
    eeg_signal : 1D numpy array
        EEG 信号
    fs : float
        采样率 (Hz)
    bp_type: int band power type: 1-abp 2-rbp
    bands : dict
        自定义频带范围, 例如:
        {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45)
        }

    返回:
    --------
    rel_power : dict
        各频带相对能量
    abs_power : dict
        各频带绝对能量
    """
    if bands is None:
        bands = {
            "delta": (0.5, 5),  # (0.5, 4),
            "theta": (5, 8),  # (4, 7), # (4, 5.25),
            "alpha": (8, 13),  # (7, 12), # (5.25, 6.25),
            "beta": (13, 30),  # (12, 30), # (6.25, 8)
        }
    order = 4
    sos = butter(order, 0.035, btype='highpass', fs=fs, output='sos')
    y = sosfilt(sos, eeg_signal)

    # 计算功率谱密度 (Welch 方法)
    freqs, psd = welch(y, fs=fs, nperseg=fs*2)  # 2s 窗长   window='hamming', , scaling='spectrum'
    # idx = np.logical_and(freqs >= 0.5, freqs <= 30)
    # total_power = np.trapz(psd[:, idx], freqs[idx])

    for band, (low, high) in bands.items():
        # 找到频段索引
        idx = np.logical_and(freqs >= low, freqs <= high)
        if len(psd.shape) == 1:
            band_power = np.asarray(np.trapz(psd[idx], freqs[idx]))  # 频带能量
            band_power = band_power.reshape((1, 1))
        else:
            band_power = np.trapz(psd[:, idx], freqs[idx])  # 频带能量
            band_power = band_power.reshape((len(band_power), 1))

        if band == "delta":
            abs_power = band_power

        else:
            abs_power = np.hstack([abs_power, band_power])

    if bp_type == 'rbp':
        row_sums = np.sum(np.nan_to_num(abs_power, nan=0.0), axis=1, keepdims=True)
        # Divide each element by its corresponding row sum and multiply by 100
        rel_power = (abs_power / row_sums) * 100
        return rel_power
    else:
        return abs_power


def abp_rbp(train_data, sample_frequency, bp_type, history_monitor):
    band_powers = []
    if history_monitor == 'history':
        unit_coef = 1e6
    else:  # 'monitor'
        unit_coef = 1
    for segment in train_data:
        # Calculate RBP features for the current epoch
        band_power = multi_relative_band_energy(segment * unit_coef, int(sample_frequency), bp_type)
        band_powers.append(band_power)

    band_powers = np.array(band_powers)
    # bp_type: 0-abp and rbp 1-abp 2-rbp
    sec_num = band_powers.shape[1]
    channel_values = []
    if sec_num > 0:  # {通道：{时间位：max-min}}
        for i in range(sec_num):
            abp_power = band_powers[:, i, :]
            abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
                                range(abp_power.shape[0])], 2)
            channel_values.append(abp_tmp.tolist())

    else:
        print("Channels Values is Null")

    return channel_values


def plot_rbp(rbp_features, tmp_data_path, fig_name):
    plt.figure(figsize=(12, 4))
    # 绘制堆叠图
    # plt.stackplot(days, sleep, eat, work, play, colors=['m', 'c', 'r', 'y'])
    plt.stackplot(np.arange(0, len(rbp_features)),
                  rbp_features[:, 0].tolist(),
                  rbp_features[:, 1].tolist(),
                  rbp_features[:, 2].tolist(),
                  rbp_features[:, 3].tolist(), colors=['r', 'y', 'g', 'b'])

    show_step = 5  # min
    x_interval = np.arange(0, int(np.ceil(len(rbp_features) / 6)), show_step)
    # 设置x轴的刻度间隔和格式
    xtics = x_interval * 6
    ax = plt.gca()
    ax.xaxis.set_ticks(xtics, x_interval)

    # 添加注释
    # plt.annotate('Sleep', xy=(1, 7), xytext=(1.5, 8), arrowprops=dict(arrowstyle='->'))

    # 设置坐标轴名称
    plt.xlabel('Time/min')
    plt.ylabel('Band Power')
    plt.title('Relative Band Power')
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(tmp_data_path), fig_name))
    # plt.savefig('/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/aEEG0912.png')
    print("RBP波形保存完成")

if __name__ == '__main__':
    is_history = 5  # 1-history 0-monitor 2-abp 3-rbp 4-env 5-rbp_sim
    if is_history == 5:  # rbp_sim
        start_time1 = time.time()
        # TODO 多个bdf文件读取
        file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20241206184941"  # 20250912110203 20241206184941
        for fname in os.listdir(file_path):  # 寻找.rml文件
            if '.bdf' not in fname:
                continue

            raw = mne.io.read_raw_bdf(os.path.join(file_path, fname), include=['Fp1'])  # , preload=True  , 'T4', 'O1'
            # raw = raw.filter(l_freq=2, h_freq=70)
            sample_frequency = raw.info['sfreq']

            start_seconds = 0
            stop_seconds = round(raw.n_times / sample_frequency, 2)
            signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

            train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, raw.ch_names)

            channel_values = abp_rbp(train_data, sample_frequency, 'rbp', 'history')

            for chn in range(len(channel_values)):
                # RBP波形
                plot_rbp(np.array(channel_values[chn]),
                         '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/',
                         raw.ch_names[chn] + 'rbp.png')

            print("RBP计算完成，共运行：%.8s s" % (time.time() - start_time1))