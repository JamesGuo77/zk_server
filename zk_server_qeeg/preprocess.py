#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：zk_server 
@File    ：preprocess.py
@Author  ：gjg
@Date    ：2025/10/14 20:19 
@explain : init
'''

import numpy as np
from django.http import HttpResponse
import json
import mne
import pyedflib
# from edflib import edfreader
from scipy.signal import butter, remez, lfilter, filtfilt, kaiser_atten, kaiser_beta, firwin, welch, sosfilt
import matplotlib.pyplot as plt
# ssh root@172.24.60.106
# conda activate py39_tf270_torch1101
# cd /disk1/workspace/py39_tf270/zk_server/
# export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
# python manage.py runserver 0.0.0.0:8041 --noreload
# import sys
import os
import argparse
import logging
import time
# import multiprocessing
from joblib import Parallel, delayed
import threading
import queue
import requests

# 2025/6/6 V0.1 init pyedflib读取bdf 并完成aEEG
# 2025/6/8 V0.2 修复2400-3000s运行报错
# 2025/6/9 V0.3 采用mne读取bdf 并行计算效率提升不明显暂放弃
# 2025/6/11 V0.4 读取包含bdf文件的文件夹
# 2025/6/11 V0.5 新增监测端的接口
# 2025/6/21 V0.6 优化PM滤波
# 2025/7/15 V0.7 PM滤波前进行带通[2, 70]滤波,降采样至250Hz，包络参数优化
# 2025/7/16 V0.8 修复4个通道被降采样至1个通道
# view2暂停更新
# 2025/7/23 V0.9 初始滤波；ICA去除眼动、肌电等伪迹;平滑尖峰信号
# view3暂停更新
# 2025/7/23 V1.0 firwin并行滤波，并行数=通道数，最大8个并行
# 2025/8/7 V1.1 rbp初版
# 2025/9/11 V1.2 aEEG滤波器输入信号长度不低于padlen(3*numtaps)优化
# 2025/9/25 V1.3 aEEG滤波器输入信号加入IIR高通滤波
# 2025/9/26 V1.4 aEEG采样IIR高通滤波
# view6暂停更新
# 2025/10/3 V1.5 abp回读测试通过；多线程任务测试
# view7暂停更新（可删）
# 2025/10/9 V1.6 abp监测测试通过；epoch_length全局设置
# view8暂停更新（可删）
# 2025/10/10 V1.7 aeeg、abp、rbp、sef 回读多线程测试通过
# view9暂停更新（可删）
# 2025/10/11 V1.8 TP、ADR、ABR、RAV、Envolpe 回读多线程测试通过（不满足新报文输出要求）
# view10暂停更新
# 2025/10/12 V1.9 aeeg、abp、rbp、sef、TP、ADR、ABR、RAV、Envolpe 回读多线程测试通过，满足新报文输出要求
# 2025/10/12 V2.0 SE、CSA 回读多线程测试通过，满足新报文输出要求
# view11暂停更新

epoch_length = 10  # s
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def segment_extra(aeeg_output, fs, window_length):
    sec_num = int(aeeg_output.shape[1] / fs)  # len(aeeg_output)
    utp, ltp = [], []  # UTP LTP
    for i in range(sec_num - window_length + 1):
        utp.append(np.percentile(aeeg_output[0, i * fs:(window_length + i) * fs], 70))  # 70 80
        ltp.append(np.percentile(aeeg_output[0, i * fs:(window_length + i) * fs], 50))  # 30 50
    return utp, ltp

def segment_extra1(aeeg_output, fs, window_length):  #70/50,  80/50, 85/55, 90/55
    sec_num = int(aeeg_output.shape[1] / fs)

    k = 0
    utp, ltp = np.percentile(aeeg_output[:, k * fs:(int(window_length) + k) * fs], 70, axis=1), np.percentile(
        aeeg_output[:, k * fs:(int(window_length) + k) * fs], 50, axis=1)

    if sec_num > 1:
        for i in range(1, int(sec_num) - window_length + 1):
            utp_tmp = np.percentile(aeeg_output[:, i * fs:(int(window_length) + i) * fs], 70, axis=1)
            utp = np.vstack((utp, utp_tmp))
            ltp_tmp = np.percentile(aeeg_output[:, i * fs:(int(window_length) + i) * fs], 50, axis=1)
            ltp = np.vstack((ltp, ltp_tmp))

        # {通道：{时间位：max-min}}
        res = np.round([[[utp[j, i], ltp[j, i]] for j in range(utp.shape[0])] for i in range(utp.shape[1])], 2)
        return res.tolist()

    else:
        res = np.round([[[utp[j], ltp[j]]] for j in range(utp.shape[0])], 2)  #np.round([[[utp[j], ltp[j]] for j in range(utp.shape[0])]], 2)
        return res.tolist()

def plot_aEEG(utp, ltp, tmp_data_path, fig_name):

    x1 = np.log10(10)
    x2 = np.log10(100)
    dx = (x2 - x1) * 10

    utp = list(map(lambda k: (np.log10(k) - x1) / (x2 - x1) * dx + 10 if k > 10 else k, utp))
    ltp = list(map(lambda k: (np.log10(k) - x1) / (x2 - x1) * dx + 10 if k > 10 else k, ltp))

    t = np.arange(len(utp)) / 60

    ytics = [0, 5, 10, 10 * np.log10(25), 10 * np.log10(50), 10 * np.log10(100)]
    log_ytics_labels = ['0', '5', '10', '25', '50', '100']
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    ax.yaxis.set_ticks(ytics, log_ytics_labels)
    ax.fill_between(t, utp, ltp, alpha=1, lw=1, color="b")
    ax.set_ylim([0, max(ytics)])

    show_step = 5  # min
    # ax.xaxis.set_ticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    ax.xaxis.set_ticks(np.arange(0, (np.ceil(max(t) / show_step) + 1) * show_step, show_step))
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.axhline(y=2, color='red', linestyle='-', linewidth=1)
    plt.axhline(y=3, color='red', linestyle='-', linewidth=1)
    plt.axhline(y=4, color='red', linestyle='-', linewidth=1)
    plt.axhline(y=5, color='red', linestyle='-', linewidth=1)
    plt.axhline(y=10, color='red', linestyle='-', linewidth=1)
    plt.axhline(y=10 * np.log10(25), color='red', linestyle='-', linewidth=1)
    plt.axhline(y=10 * np.log10(50), color='red', linestyle='-', linewidth=1)
    plt.savefig(os.path.join(os.path.dirname(tmp_data_path), fig_name))
    # plt.savefig('/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/aEEG0912.png')
    print("aEEG波形保存完成")


def filter_one_channel(channel_data, fs, numtaps, cutoff):
    bp_fir = firwin(numtaps, cutoff, fs=fs, pass_zero=False)  # 2181 6601  # TODO 2181
    # eeg_filtered = filtfilt(bp_fir, [1.0], raw_bdf[:][0][:]*1e6)  # [:2400000]
    return filtfilt(bp_fir, [1.0], channel_data)

def filter_one_channel1(channel_data, fs, numtaps, cutoff):
    # sos = butter(4, 1, btype='highpass', fs=fs, output='sos')
    # iir1 = sosfilt(sos, channel_data)
    bp_fir = firwin(numtaps, cutoff, fs=fs, pass_zero=False)  # 2181 6601  # TODO 2181
    # eeg_filtered = filtfilt(bp_fir, [1.0], raw_bdf[:][0][:]*1e6)  # [:2400000]
    return filtfilt(bp_fir, [1.0], channel_data)  # iir1

# 多通道滤波函数
def filter_multichannel_eeg(eeg_data, fs, numtaps, cutoff, n_jobs=3):
    filtered = Parallel(n_jobs=n_jobs)(delayed(filter_one_channel1)(ch_data, fs, numtaps, cutoff) for ch_data in eeg_data)
    return np.array(filtered)

def aEEG_compute_h(sigbufs, channels, sample_frequency, band, window_length):
    channel_labels = channels
    if len(channel_labels) > 0:
        source_eeg = sigbufs
        fs = sample_frequency
        trans_width = 0.4  # Width of transition from pass to stop, Hz
        numtaps = 2181  # numtaps = filter orders + 1 301
        asym_filter_freq_hz = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5 * fs]

        source_eeg[np.isnan(source_eeg)] = 0
        n_jobs = len(channel_labels) if len(channel_labels) < 8 else 8
        # taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], type='bandpass', fs=fs)
        # y = lfilter(taps, 1, np.apply_along_axis(remove_spikes, axis=1, arr=eeg_filtered))
        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filter_multichannel_eeg(source_eeg, fs, numtaps, n_jobs)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filter_one_channel(source_eeg_tile[:, :(3 * numtaps + 1)], fs, numtaps)[:, :source_eeg.shape[1]]

        aeeg_output = 1.231 * np.abs(eeg_filtered) + 4

        fs_new = 250
        decimation_factor = int(fs / fs_new)
        # 降采样到250Hz
        aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, fs_new, window_length)

        # 分段提取UTP LTP
        # utp, ltp = segment_extra(aeeg_output, fs_new, window_length)  # fs
        #
        # # aEEG波形
        # plot_aEEG(utp, ltp, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY')

    else:
        channel_values = []
        print("Channels Values is Null")

    return channel_labels, channel_values

def special_node(special_electrodes):
    node_dict = {}
    for spec_nodes in special_electrodes:
        node_dict[spec_nodes["name"]] = spec_nodes["electrodes"]
    return node_dict

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    nor_lowcut = lowcut / nyq
    nor_highcut = highcut / nyq
    b = butter(order, [nor_lowcut, nor_highcut], btype='band', output='sos')
    return b

def aEEG_compute_h1(input_data_dict, band, window_length):
    channles = len([channel for channel in input_data_dict['labels']])  #  if 'EEG' in channel
    data_values = np.array(input_data_dict['values'][:channles])

    channel_labels = input_data_dict['labels'][:channles]
    fs_list = input_data_dict['samples']
    # band = [2, 15]  # Desired pass band, Hz

    if channles > 0:
        # source_eeg = data_values
        fs = int(fs_list[0])
        numtaps = int(60 * fs / (22 * 5) / 2)  # numtaps = filter orders + 1  301
        data_values[np.isnan(data_values)] = 0
        # 非对称滤波 整流 振幅放大
        sos = butter(4, 1, btype='highpass', fs=fs, output='sos')
        source_eeg = sosfilt(sos, data_values)
        bp_fir = firwin(numtaps, cutoff=band, fs=fs, pass_zero=False)  # [2, 15]
        # if source_eeg.shape[1] > 3 * numtaps:
        #     eeg_filtered = filtfilt(bp_fir, [1.0], iir1)
        # else:
        #     source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
        #     eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]
        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]

        aeeg_output = 1.631 * np.abs(eeg_filtered) + 4  # 1.231 * np.abs(eeg_filtered) + 4

        # fs_new = 250  # 新的采样率
        # decimation_factor = int(fs / fs_new)
        # # 降采样到250Hz
        # aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, int(fs), window_length)  # fs_new

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        print("Channels Values is Null")
        channel_values = []

    return channel_labels, channel_values

def aEEG_com1(input_data_dict, band, window_length):
    channles = len([channel for channel in input_data_dict['labels']])  #  if 'EEG' in channel
    data_values = np.array(input_data_dict['values'][:channles])

    channel_labels = input_data_dict['labels'][:channles]
    fs_list = input_data_dict['samples']

    if channles > 0:
        fs = int(fs_list[0])
        numtaps = int(60 * fs / (22 * 5) / 2)  # numtaps = filter orders + 1  301
        data_values[np.isnan(data_values)] = 0
        # 非对称滤波 整流 振幅放大
        sos = butter(4, 1, btype='highpass', fs=fs, output='sos')
        source_eeg = sosfilt(sos, data_values)
        bp_fir = firwin(numtaps, cutoff=band, fs=fs, pass_zero=False)  # [2, 15]

        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]

        aeeg_output = 1.631 * np.abs(eeg_filtered) + 4  # 1.231 * np.abs(eeg_filtered) + 4

        # fs_new = 250  # 新的采样率
        # decimation_factor = int(fs / fs_new)
        # # 降采样到250Hz
        # aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, int(fs), window_length)  # fs_new
        # utp, ltp = segment_extra(aeeg_output, fs, window_length)

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        print("Channels Values is Null")
        channel_values = []
        # utp, ltp = [], []

    return channel_labels, channel_values


def read_bdf(sigbufs, signal_labels, channels, node_dict):
    # 20250609 兼容mne读取bdf文件
    # 读取json数据并处理 Start
    # json_path = '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/inputData2.txt'
    # with open(json_path, 'r') as file:
    #     content = file.read()
    #
    # json_data = json.loads(content)

    # 读取bdf文件
    # file_path = r'/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20241206184941/'
    # f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
    # n = f.signals_in_file
    # signal_labels = f.getSignalLabels()
    # sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致
    # start_inx = int(sample_frequency * int(json_data['startSeconds']))
    # stop_inx = int(sample_frequency * int(json_data['stopSeconds']))
    # col_length = stop_inx - start_inx
    # channels = json_data['channels']
    sigbufs_res = np.zeros((len(channels), sigbufs.shape[1]))
    # node_dict = special_node(json_data['specialElectrodes'])

    # 计算输入信号电位差
    for inx in range(len(channels)):
        chn_split = channels[inx].split('-')
        chn_inx0 = signal_labels.index(chn_split[0])

        if chn_split[1] in ('Ref', 'REF', 'ref'):
            sigbufs_res[inx, :] = sigbufs[chn_inx0, :]

        elif chn_split[1] in signal_labels:
            chn_inx1 = signal_labels.index(chn_split[1])
            sigbufs_res[inx, :] = sigbufs[chn_inx0, :] - sigbufs[chn_inx1, :]

        # elif chn_split[1] in av_dict.keys():
        #     sigbufs[inx, 0:col_length] = f.readSignal(chn_inx0, start_inx, stop_inx) - av_dict[chn_split[1]]
        elif chn_split[1] in node_dict.keys():
            av_value = np.zeros((1, sigbufs.shape[1]))
            spec_nodes = node_dict[chn_split[1]]
            for spec_node in spec_nodes:
                chn_inx = signal_labels.index(spec_node["electrode"])
                av_value = av_value + sigbufs[chn_inx, :] * int(spec_node["weight"])
            av_value = av_value/len(spec_nodes) / 100

            sigbufs_res[inx, :] = sigbufs[chn_inx0, :] - av_value

    return sigbufs_res


def calculate_rbp(epoch_data, fs=500):
    # Define frequency bands
    # import pdb
    # pdb.set_trace()
    frequency_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)] # [(1, 4), (4, 8), (8, 13), (13, 25), (25, 45)]

    rbp_features = []
    abp_features = []
    for channel in epoch_data:
        # Calculate power spectral density (PSD) using Welch method
        freqs, psd = welch(channel, fs=fs, nperseg=fs*4, noverlap=fs*2)
        freq_idx_total = np.where((freqs >= 0.5) & (freqs <= 30))[0]  # (freqs >= 1) & (freqs <= 45)
        total_psd = np.trapz(psd[freq_idx_total], axis=0)  # Sum across frequency bins

        rbp = []
        abp = []
        for band in frequency_bands:
            start_freq, end_freq = band
            freq_idx = np.where((freqs >= start_freq) & (freqs <= end_freq))[0]

            band_psd = np.trapz(psd[freq_idx], axis=0)  # Sum across selected frequency bins
            band_rbp = band_psd / total_psd
            rbp.append(band_rbp)
            abp.append(band_psd)

        rbp_features.append(rbp)
        abp_features.append(abp)

    rbp_features = np.array(rbp_features)
    abp_features = np.array(abp_features)
    # rbp2d = rbp_features.reshape((rbp_features.shape[0], rbp_features.shape[1], 1))

    return rbp_features, abp_features



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
            "delta": (0.5, 4),  # (0.5, 4),
            "theta": (4, 8),  # (4, 7), # (4, 5.25),
            "alpha": (8, 13),  # (7, 12), # (5.25, 6.25),
            "beta": (13, 30),  # (12, 30), # (6.25, 8)
        }
    order = 4
    # sos = butter(order, 0.035, btype='highpass', fs=fs, output='sos')
    sos = butter_bandpass(0.5, 30, fs, order=4)
    y = sosfilt(sos, eeg_signal)

    # 计算功率谱密度 (Welch 方法)
    freqs, psd = welch(y, fs=fs, nperseg=fs*3, scaling='spectrum', noverlap=0.5*fs)  # nperseg=fs*2 window='hamming', , scaling='spectrum'
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

# def abp_rbp_com(start_seconds, stop_seconds, raw, channels, sample_frequency, bp_type):
#     # abp start
#     train_data = []
#     start_time = start_seconds
#     # epoch_length = 10
#     step_size = epoch_length  # 10
#     while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
#         # features = []
#         start, stop = raw.time_as_index([start_time, start_time + epoch_length])
#         temp = raw[:, start:stop][0]
#         train_data.append(temp)
#         start_time += step_size
#
#     train_data = np.array(train_data)
#
#     rbp_powers = []
#     abp_powers = []
#     band_powers = []
#     for segment in train_data:
#         # Calculate RBP features for the current epoch
#         band_power = multi_relative_band_energy(segment * 1e6, int(sample_frequency), bp_type)
#         # rbp_powers.append(rbp_power)
#         # abp_powers.append(abp_power)
#         band_powers.append(band_power)
#
#     # rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
#     # abp_powers = np.array(abp_powers)
#     band_powers = np.array(band_powers)
#
#     # bp_type: 0-abp and rbp 1-abp 2-rbp
#     sec_num = band_powers.shape[1]
#     channel_values = []
#     # channel_values1 = []
#     if sec_num > 0:  # {通道：{时间位：max-min}}
#         # channel_values = np.round(abp_powers, 2)
#         for i in range(sec_num):
#             if bp_type == 1:
#                 abp_power = band_powers[:, i, :]
#                 abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
#                                     range(abp_power.shape[0])], 2)
#                 channel_values.append(abp_tmp.tolist())
#                 # abp_tmp = abp_powers[:, i, :].tolist()
#                 # channel_values.append(abp_tmp)
#
#             if bp_type == 2:
#                 rbp_power = band_powers[:, i, :]
#                 rbp_tmp = np.round([[rbp_power[j, 0], rbp_power[j, 1], rbp_power[j, 2], rbp_power[j, 3]] for j in
#                                     range(rbp_power.shape[0])], 2)
#                 channel_values.append(rbp_tmp.tolist())
#     else:
#         print("Channels Values is Null")
#
#     return channels, channel_values

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

def abp_rbp_com(start_seconds, stop_seconds, raw, channels, sample_frequency, bp_type):
    # abp start
    train_data = []
    start_time = start_seconds
    # epoch_length = 10
    step_size = epoch_length  # 10
    while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
        # features = []
        start, stop = raw.time_as_index([start_time, start_time + epoch_length])
        temp = raw[:, start:stop][0]
        train_data.append(temp)
        start_time += step_size

    train_data = np.array(train_data)
    channel_values = abp_rbp(train_data, sample_frequency, bp_type, 'history')

    return channels, channel_values

def abp_rbp_com1(input_data_dict, bp_type):
    channles = len([channel for channel in input_data_dict['labels']])  # if 'EEG' in channel
    data_values = np.array(input_data_dict['values'][:channles])

    channel_labels = input_data_dict['labels'][:channles]
    fs_list = input_data_dict['samples']

    if channles > 0:
        fs = int(fs_list[0])
        data_values[np.isnan(data_values)] = 0
        stop_seconds = int(data_values.shape[1]/fs)
        inx = 0
        # train_data = []
        start_time = 0  # 10
        # epoch_length = 10
        step_size = epoch_length  # 10
        while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
            # features = []
            start, stop = inx, inx + epoch_length  # start_time, start_time + epoch_length
            temp = data_values[:, int(start*fs):int(stop*fs)]
            temp = temp.reshape((1, temp.shape[0], temp.shape[1]))

            if start == 0:
                train_data = temp
            else:
                train_data = np.vstack([train_data, temp])  # axis = 1 第二维 train_data.(temp)

            start_time += step_size
            inx = inx + step_size

        # rbp_powers = []
        # abp_powers = []
        # for segment in train_data:
        #     # Calculate RBP features for the current epoch
        #     rbp_power, abp_power = multi_relative_band_energy(segment, fs)
        #     rbp_powers.append(rbp_power)
        #     abp_powers.append(abp_power)
        #
        # rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
        # abp_powers = np.array(abp_powers)
        #
        # # bp_type: 0-abp and rbp 1-abp 2-rbp
        # sec_num = abp_powers.shape[1]
        # channel_values = []
        # channel_values1 = []
        # if sec_num > 0:  # {通道：{时间位：max-min}}
        #     # channel_values = np.round(abp_powers, 2)
        #     for i in range(sec_num):
        #         if (bp_type == 0) or (bp_type == 1):
        #             abp_power = abp_powers[:, i, :]
        #             abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
        #                                 range(abp_power.shape[0])], 2)
        #             channel_values.append(abp_tmp.tolist())
        #
        #         if (bp_type == 0) or (bp_type == 2):
        #             rbp_power = rbp_powers[:, i, :]
        #             rbp_tmp = np.round([[rbp_power[j, 0], rbp_power[j, 1], rbp_power[j, 2], rbp_power[j, 3]] for j in
        #                                 range(rbp_power.shape[0])], 2)
        #             channel_values1.append(rbp_tmp.tolist())
        # else:
        #     print("Channels Values is Null")
        channel_values = abp_rbp(train_data, fs, bp_type, 'monitor')

    else:
        print("Channels length is Zero")

    return channel_labels, channel_values  # , channel_values1

# time split
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

def abp_rbp_com_t(start_seconds, stop_seconds, raw, ttype, channels, sample_frequency, bp_type, task_id):
    # abp start
    # train_data = []
    # start_time = start_seconds
    # # epoch_length = 10
    # step_size = epoch_length  # 10
    # while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
    #     # features = []
    #     start, stop = raw.time_as_index([start_time, start_time + epoch_length])
    #     # temp = raw[:, start:stop][0]
    #     temp = raw[[raw.ch_names.index(chn) for chn in channels], start:stop][0]
    #     train_data.append(temp)
    #     start_time += step_size
    #
    # train_data = np.array(train_data)
    train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)
    # rbp_powers = []
    # abp_powers = []
    # for segment in train_data:
    #     # Calculate RBP features for the current epoch
    #     rbp_power, abp_power = multi_relative_band_energy(segment * 1e6, int(sample_frequency))
    #     rbp_powers.append(rbp_power)
    #     abp_powers.append(abp_power)
    #
    # rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
    # abp_powers = np.array(abp_powers)
    #
    # # bp_type: 0-abp and rbp 1-abp 2-rbp
    #
    # sec_num = abp_powers.shape[1]
    # channel_values = []
    # channel_values1 = []
    # if sec_num > 0:  # {通道：{时间位：max-min}}
    #     # channel_values = np.round(abp_powers, 2)
    #     for i in range(sec_num):
    #         if (bp_type == 0) or (bp_type == 1):
    #             abp_power = abp_powers[:, i, :]
    #             abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
    #                                 range(abp_power.shape[0])], 2)
    #             channel_values.append(abp_tmp.tolist())
    #
    #         if (bp_type == 0) or (bp_type == 2):
    #             rbp_power = rbp_powers[:, i, :]
    #             rbp_tmp = np.round([[rbp_power[j, 0], rbp_power[j, 1], rbp_power[j, 2], rbp_power[j, 3]] for j in
    #                                 range(rbp_power.shape[0])], 2)
    #             channel_values1.append(rbp_tmp.tolist())
    #
    #     # if len(channel_values) > 0:
    #     #     channel_values = np.round(np.array(channel_values), 2)
    # else:
    #     print("Channels Values is Null")
    channel_values = abp_rbp(train_data, sample_frequency, bp_type, 'history')

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values: ", channel_values)
    # print("ttype: ", ttype)
    # return channel_values



# def qeeg_history(request):
#     if request.method == 'POST':
#         if request.content_type == 'application/json':
#             jsonStr = request.body.decode('utf-8')
#             json_data = json.loads(jsonStr)
#
#             #### Start
#             trend_channels = json_data['trendChannels']
#             aeeg_chns = []  # labels: aEEG
#             rbp_chns = []  # labels: RBP
#             aeeg_chns_sim, rbp_chns_sim = [], []
#             for tchns in trend_channels:  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR
#                 if tchns["type"] == "aEEG":
#                     aeeg_chns.append(tchns["label"])
#
#                 elif tchns["type"] == "RBP":
#                     rbp_chns.append(tchns["label"])
#
#             # channels = json_data['channels']  # trendChannels
#             if len(aeeg_chns) > 0:
#                 aeeg_chns_split = []
#                 for chns in aeeg_chns:
#                     chn_split = chns.split('-')
#                     aeeg_chns_split.append(chn_split[0])
#                     aeeg_chns_split.append(chn_split[1])
#                 aeeg_chns_split = list(set(aeeg_chns_split))
#                 aeeg_chns_sim = [chn for chn in aeeg_chns_split if chn not in ('AV', 'Ref', 'REF', 'ref')]
#
#             if len(rbp_chns) > 0:
#                 rbp_chns_split = []
#                 for chns in rbp_chns:
#                     chn_split = chns.split('-')
#                     rbp_chns_split.append(chn_split[0])
#                     rbp_chns_split.append(chn_split[1])
#                 rbp_chns_split = list(set(rbp_chns_split))
#                 rbp_chns_sim = [chn for chn in rbp_chns_split if chn not in ('AV', 'Ref', 'REF', 'ref')]
#
#             #### End
#             chn_list_sim = list(set(aeeg_chns_sim + rbp_chns_sim))
#
#             # chn_list = []
#             # channels = json_data['channels']
#             # for chns in channels:
#             #     chn_split = chns.split('-')
#             #     chn_list.append(chn_split[0])
#             #     chn_list.append(chn_split[1])
#             # chn_list = list(set(chn_list))
#             # chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]
#
#             # TODO 多个bdf文件读取
#             for fname in os.listdir(json_data['fileDir']):  # 寻找.bdf文件
#                 if '.bdf' not in fname:
#                     continue
#                 raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True tuple(chn_list_sim)
#                 # raw = raw.filter(l_freq=2, h_freq=70)
#                 # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
#                 sample_frequency = raw.info['sfreq']
#
#                 start_seconds = int(json_data['startSeconds'])  # 4200
#                 stop_seconds = int(json_data['stopSeconds'])  # 4763 (4800)
#
#                 signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750
#
#                 if start_seconds > signal_seconds:  # 不分析
#                     error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
#                     return HttpResponse(json.dumps(error_message))
#
#                 elif start_seconds > stop_seconds:  #
#                     error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
#                     return HttpResponse(json.dumps(error_message))
#
#                 elif stop_seconds > signal_seconds:
#                     stop_seconds = signal_seconds
#
#                 # aEEG_compute_h
#                 # aEEG_labels, aEEG_values = aEEG_compute_h3(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp
#                 aEEG_labels, aEEG_values = aEEG_com(start_seconds, stop_seconds, json_data['specialElectrodes'], raw, aeeg_chns, sample_frequency)
#                 # abp bp_type: 0-abp and rbp 1-abp 2-rbp
#                 bp_labels, abp_values, rbp_values = abp_rbp_com(start_seconds, stop_seconds, raw, rbp_chns, sample_frequency, 2)
#
#                 task_id = json_data["taskID"]
#                 ana_type = "aEEG_rbp"
#                 qeeg_type = "aEEG"
#                 qeeg_type1 = "rbp"
#
#                 tk = str(aEEG_labels)
#                 tk = tk.replace("\'", "\"")
#                 tk1 = str(bp_labels)
#                 tk1 = tk1.replace("\'", "\"")
#                 qeeg_data = "{ " + f" \"type\": \"{qeeg_type}\", \"labels\": {tk}, \"values\": {aEEG_values}" + " }"
#                 qeeg_data1 = "{ " + f" \"type\": \"{qeeg_type1}\", \"labels\": {tk1}, \"values\": {rbp_values}" + " }"
#                 qeeg_data2 = "[" + f"{qeeg_data}"  + ", " + f"{qeeg_data1}" + "]"
#
#                 # result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"
#                 result = "{ " + f"\"taskID\": \"{task_id}\", \"qeeg_data\": {qeeg_data2}" + " }"
#
#                 return HttpResponse(result)
#                 # return HttpResponse(json.dumps(success_message))
#         else:
#             error_message["message"] = "The content type is incorrect. Please input it application/json"
#             return HttpResponse(json.dumps(error_message))
#     else:
#         error_message["message"] = "The request method is incorrect"
#         return HttpResponse(json.dumps(error_message))



# def qteeg_history(request):
#     if request.method == 'POST':
#         if request.content_type == 'application/json':
#             jsonStr = request.body.decode('utf-8')
#             json_data = json.loads(jsonStr)
#
#             chn_list = []
#             channels = json_data['channels']
#             for chns in channels:
#                 chn_split = chns.split('-')
#                 chn_list.append(chn_split[0])
#                 chn_list.append(chn_split[1])
#             chn_list = list(set(chn_list))
#             chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]
#
#             # TODO 多个bdf文件读取
#             for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
#                 if '.bdf' not in fname:
#                     continue
#                 raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True tuple(chn_list_sim)
#                 # raw = raw.filter(l_freq=2, h_freq=70)
#                 # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
#                 sample_frequency = raw.info['sfreq']
#
#                 start_seconds = int(json_data['startSeconds'])  # 4200
#                 stop_seconds = int(json_data['stopSeconds'])  # 4763 (4800)
#
#                 signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750
#
#                 if start_seconds > signal_seconds:  # 不分析
#                     error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
#                     return HttpResponse(json.dumps(error_message))
#
#                 elif start_seconds > stop_seconds:  #
#                     error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
#                     return HttpResponse(json.dumps(error_message))
#
#                 elif stop_seconds > signal_seconds:
#                     stop_seconds = signal_seconds
#
#                 # # abp bp_type: 0-abp and rbp 1-abp 2-rbp
#                 # 创建线程
#                 aEEG_t = threading.Thread(target=thread_workers, args=(aEEG_com_t, (start_seconds, stop_seconds, json_data['specialElectrodes'], raw, channels, sample_frequency, json_data["taskID"]), result_queue))
#                 abp_t = threading.Thread(target=thread_workers, args=(abp_rbp_com_t, (start_seconds, stop_seconds, raw, channels, sample_frequency, 1, json_data["taskID"]), result_queue))
#
#                 # 启动线程
#                 aEEG_t.start()
#                 abp_t.start()
#
#                 # 等待所有线程结束
#                 aEEG_t.join()
#                 abp_t.join()
#                 # 合并结果
#                 results = []
#                 while not result_queue.empty():
#                     results.append(result_queue.get())
#
#                 return HttpResponse(results)
#         else:
#             error_message["message"] = "The content type is incorrect. Please input it application/json"
#             return HttpResponse(json.dumps(error_message))
#     else:
#         error_message["message"] = "The request method is incorrect"
#         return HttpResponse(json.dumps(error_message))

def multi_chns_rav(start_seconds, stop_seconds, raw, ttype, channels, fs, task_id, alpha_band=(6,14), alpha_band1=(1,20), window=2.0, overlap=0, nperseg=None):
    if stop_seconds - start_seconds < 120:  # 2 min
        print('Time length less than 2min')
        channel_values = [0.0, 0.0]

    else:
        train_data = sig_chns_split(start_seconds, stop_seconds, 120, raw, channels)  # 2 min

        # if history_monitor == 'history':
        #     unit_coef = 1e6
        # else:  # 'monitor'
        #     unit_coef = 1

        channel_values1 = []
        step = int(window * fs * (1 - overlap))
        size = int(window * fs)

        if train_data.shape[1] > 1:
            for segment in train_data:
                alpha_powers, alpha_powers1 = [], []
                for start in range(0, segment.shape[1] - size + 1, step):
                    seg = segment[:, start:start + size]
                    freqs, psd = welch(seg, fs=fs, nperseg=size)
                    idx_alpha = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
                    idx_alpha1 = np.logical_and(freqs >= alpha_band1[0], freqs <= alpha_band1[1])

                    alpha_power = np.trapz(psd[:, idx_alpha], freqs[idx_alpha])
                    alpha_power1 = np.trapz(psd[:, idx_alpha1], freqs[idx_alpha1])
                    alpha_powers.append(alpha_power)
                    alpha_powers1.append(alpha_power1)

                alpha_powers = np.array(alpha_powers)
                alpha_powers1 = np.array(alpha_powers1)

                mean_alpha = np.mean(alpha_powers, axis=0)
                mean_alpha1 = np.mean(alpha_powers1, axis=0)
                channel_values1.append(mean_alpha * 100 / mean_alpha1.tolist())

            # channel_values1 = np.array(channel_values1)
            # for i in range(train_data.shape[1]):
            #     rav_tmp = np.round(channel_values1[:, i], 2)
            #     channel_values.append(rav_tmp.tolist())
            channel_values = multi_chns_out(channel_values1)

        else:
            # print(type(train_data), train_data.shape)
            for segment in train_data:
                alpha_powers, alpha_powers1 = [], []
                for start in range(0, segment.shape[1] - size + 1, step):
                    seg = segment[:, start:start + size]
                    freqs, psd = welch(seg, fs=fs, nperseg=size)
                    idx_alpha = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
                    idx_alpha1 = np.logical_and(freqs >= alpha_band1[0], freqs <= alpha_band1[1])

                    alpha_power = np.trapz(psd[:, idx_alpha], freqs[idx_alpha])
                    alpha_power1 = np.trapz(psd[:, idx_alpha1], freqs[idx_alpha1])
                    alpha_powers.append(alpha_power)
                    alpha_powers1.append(alpha_power1)

                alpha_powers = np.array(alpha_powers)
                alpha_powers1 = np.array(alpha_powers1)
                # print(type(alpha_powers), alpha_powers.shape, alpha_powers)
                if len(alpha_powers) < 2:
                    print("Single channel data length less than 4s!")
                    channel_values1.append(0)

                else:
                    mean_alpha = np.mean(alpha_powers)
                    mean_alpha1 = np.mean(alpha_powers1)
                    # print(type(mean_alpha), mean_alpha.shape, mean_alpha)
                    channel_values1.append(round(mean_alpha * 100 / mean_alpha1, 2))

            channel_values = sig_chn_out(channel_values1)

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values: ", channel_values)
    # return channel_values

def multi_chns_csa(start_seconds, stop_seconds, raw, ttype, channels, fs, task_id, band=(0.5, 30), window=2.0, nfft=None, log_power=True):
    train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)

    # if history_monitor == 'history':
    #     unit_coef = 1e6
    # else:  # 'monitor'
    #     unit_coef = 1

    channel_values, channel_values1 = [], []
    if train_data.shape[1] > 1:
        for segment in train_data:
            ## Start
            size = int(window * fs)
            # step = int(size * (1 - overlap))
            if nfft is None:
                nfft = size

            freqs, _ = welch(np.zeros(size), fs=fs, nperseg=size, nfft=nfft)
            idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
            freqs = freqs[idx_band]

            # csa_list = []
            # times = []
            f, psd = welch(segment, fs=fs, nperseg=size, nfft=nfft)
            psd_band = psd[:, idx_band]
            if log_power:
                psd_band = 10 * np.log10(psd_band + 1e-12)  # dB

            # csa_list.append(psd_band)
            # times.append((start + size / 2) / fs)

            # csa_matrix = np.array(csa_list).T  # (freq × time)
            # return np.array(times), freqs, csa_matrix
            channel_values1.append(np.round(psd_band, 2).tolist())


        channel_values1 = np.array(channel_values1).T
        channel_values = multi_chns_out(channel_values1)

    else:
        for segment in train_data:
            size = int(window * fs)
            # step = int(size * (1 - overlap))
            if nfft is None:
                nfft = size

            freqs, _ = welch(np.zeros(size), fs=fs, nperseg=size, nfft=nfft)
            idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
            freqs = freqs[idx_band]

            # csa_list = []
            # times = []
            f, psd = welch(segment, fs=fs, nperseg=size, nfft=nfft)
            psd_band = psd[:, idx_band]
            if log_power:
                psd_band = 10 * np.log10(psd_band + 1e-12)  # dB

            channel_values1.append(np.round(psd_band, 2))

        channel_values1 = np.array(channel_values1).T
        # channel_values = sig_chn_out(channel_values1)

        for tk in channel_values1:
            channel_values.append(tk.tolist())
        channel_values = [channel_values]

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values: ", channel_values)
    # print("ttype ", ttype)
    # return channel_values

def sig_chn_out(channel_values1):
    channel_values = []
    channel_values1 = np.array(channel_values1)
    channel_values1 = channel_values1.reshape((len(channel_values1), 1))

    for tk in channel_values1:
        channel_values.append(tk.tolist())
    channel_values = [channel_values]
    return channel_values

def multi_chns_out(channel_values1):
    channel_values = []
    channel_values1 = np.array(channel_values1)
    for i in range(channel_values1.shape[1]):
        env_tmp = np.round([[channel_values1[j, i]] for j in range(channel_values1.shape[0])], 2)  # np.round(channel_values1[:, i], 2)
        channel_values.append(env_tmp.tolist())
    return channel_values

def multi_chns_env(start_seconds, stop_seconds, special_electrodes, raw, ttype, channels, fs, history_monitor, task_id):
    # train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)
    sigbufs_res = sig_uv_read(start_seconds, stop_seconds, special_electrodes, raw, channels, fs, history_monitor)

    # data split
    train_data = []
    start_time = start_seconds
    # epoch_length = 10
    step_size = epoch_length  # 10
    inx = 0
    while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
        # features = []
        start, stop = inx, inx + epoch_length  # start_time, start_time + epoch_length
        temp = sigbufs_res[:, int(start * fs):int(stop * fs)]
        temp = temp.reshape((1, temp.shape[0], temp.shape[1]))

        if start == 0:
            train_data = temp
        else:
            train_data = np.vstack([train_data, temp])  # axis = 1 第二维 train_data.(temp)

        start_time += step_size
        inx = inx + step_size

    channel_values, channel_values1 = [], []
    numtaps = int(60 * fs / (22 * 5) / 2)  # 2181 1091 273 # numtaps = filter orders + 1 301
    order = 4
    sos = butter(order, 0.5, btype='highpass', fs=fs, output='sos')
    band=[0.5, 30]
    # print("train_data.shape[1]: ", train_data.shape[1])

    if train_data.shape[1] > 1:
        segment = sosfilt(sos, train_data)
        for source_eeg in segment:
            n_jobs = train_data.shape[1] if train_data.shape[1] < 8 else 8
            if source_eeg.shape[1] > 3 * numtaps:
                eeg_filtered = filter_multichannel_eeg(source_eeg, fs, numtaps, band, n_jobs)
            else:
                source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
                eeg_filtered = filter_one_channel(source_eeg_tile[:, :(3 * numtaps + 1)], fs, numtaps, band)[:,
                               :source_eeg.shape[1]]

            aeeg_output = 1.631 * np.abs(eeg_filtered) + 4
            channel_values1.append(np.percentile(aeeg_output, 50, axis=1).tolist())  #

        # channel_values1 = np.array(channel_values1)
        # for i in range(channel_values1.shape[1]):
        #     env_tmp = np.round([[channel_values1[j, i]] for j in range(channel_values1.shape[0])], 2)  # np.round(channel_values1[:, i], 2)
        #     channel_values.append(env_tmp.tolist())
        channel_values = multi_chns_out(channel_values1)

    else:
        segment = sosfilt(sos, train_data)
        for source_eeg in segment:
            if source_eeg.shape[1] > 3 * numtaps:
                eeg_filtered = filter_one_channel(source_eeg, fs, numtaps, band)
            else:
                source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
                eeg_filtered = filter_one_channel(source_eeg_tile[:(3 * numtaps + 1)], fs, numtaps, band)[:source_eeg.shape[1]]

            aeeg_output = 1.631 * np.abs(eeg_filtered) + 4

            channel_values1.append(round(np.percentile(aeeg_output, 50), 2))

        # channel_values1 = np.array(channel_values1)
        # channel_values1 = channel_values1.reshape((len(channel_values1), 1))
        #
        # for tk in channel_values1:
        #     channel_values.append(tk.tolist())
        # channel_values = [channel_values]
        channel_values = sig_chn_out(channel_values1)

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values: ", channel_values)
    # return channel_values

def multi_chns_tp(start_seconds, stop_seconds, raw, ttype, channels, fs, history_monitor, task_id, band=(0.5, 30), nperseg=None):
    train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)

    if history_monitor == 'history':
        unit_coef = 1e6
    else:  # 'monitor'
        unit_coef = 1

    order = 4
    # sos = butter(order, 0.5, btype='highpass', fs=fs, output='sos')
    sos = butter_bandpass(band[0], band[1], fs, order=4)
    train_data = sosfilt(sos, train_data * unit_coef)

    channel_values, channel_values1 = [], []
    if nperseg is None:
        nperseg = fs * 3  # 2秒窗  5
    if train_data.shape[1] > 1:
        for segment in train_data:
            # Calculate RBP features for the current epoch
            # 1. 计算功率谱密度
            freqs, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=0.5*fs, scaling='spectrum')

            # 2. 取分析频带
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            freqs = freqs[idx]
            sef = psd[:, idx]
            # 3. 积分求总功率 (面积)
            tp = np.trapz(sef, freqs)
            channel_values1.append(tp.tolist())

        # channel_values1 = np.array(channel_values1)
        # for i in range(train_data.shape[1]):
        #     tp_tmp = np.round(channel_values1[:, i], 2)
        #     channel_values.append(tp_tmp.tolist())
        channel_values = multi_chns_out(channel_values1)

    else:
        for segment in train_data:
            # Calculate RBP features for the current epoch
            # 1. 计算功率谱密度
            freqs, psd = welch(segment, fs=fs, nperseg=nperseg)

            # 2. 取分析频带
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            freqs = freqs[idx]
            sef = psd[:, idx]

            # 3. 积分求总功率 (面积)
            tp = np.trapz(sef, freqs)
            # print(type(tp), tp.shape, tp)
            channel_values1.append(round(tp[0], 2))

        channel_values = sig_chn_out(channel_values1)

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result

def multi_chns_adr_abr(start_seconds, stop_seconds, raw, ttype, channels, fs, ar_type, task_id, alpha_band=(8, 13), delta_band=(0.5, 4), beta_band=(13,30), nperseg=None):
    train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)

    # if history_monitor == 'history':
    #     unit_coef = 1e6
    # else:  # 'monitor'
    #     unit_coef = 1

    order = 4
    sos = butter(order, 0.035, btype='highpass', fs=fs, output='sos')
    train_data = sosfilt(sos, train_data)

    channel_values, channel_values1 = [], []
    if train_data.shape[1] > 1:
        for segment in train_data:
            # Calculate RBP features for the current epoch
            # 1. 使用Welch方法估计功率谱密度
            freqs, psd = welch(segment, fs=fs, nperseg=fs * 2)

            # 2. 计算各频带功率
            alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
            alpha_power = np.trapz(psd[:, alpha_mask], freqs[alpha_mask])
            if ar_type == 'adr':
                bd_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
                bd_power = np.trapz(psd[:, bd_mask], freqs[bd_mask])
                # ana_type = "ADR"
            elif ar_type == 'abr':
                bd_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
                bd_power = np.trapz(psd[:, bd_mask], freqs[bd_mask])
                # ana_type = "ABR"

            # 3. 防止除零
            adr = np.where(bd_power > 0, alpha_power / bd_power, 0)
            channel_values1.append(adr.tolist())

        # channel_values1 = np.array(channel_values1)
        # for i in range(train_data.shape[1]):
        #     adr_tmp = np.round(channel_values1[:, i], 2)
        #     channel_values.append(adr_tmp.tolist())
        channel_values = multi_chns_out(channel_values1)

    else:
        for segment in train_data:
            # Calculate RBP features for the current epoch
            # 1. 使用Welch方法估计功率谱密度
            freqs, psd = welch(segment, fs=fs, nperseg=fs * 2)

            # 2. 计算各频带功率
            alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
            alpha_power = np.trapz(psd[:, alpha_mask], freqs[alpha_mask])
            if ar_type == 'adr':
                bd_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
                bd_power = np.trapz(psd[:, bd_mask], freqs[bd_mask])
                # ana_type = "ADR"
            elif ar_type == 'abr':
                bd_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
                bd_power = np.trapz(psd[:, bd_mask], freqs[bd_mask])
                # ana_type = "ABR"

            # 3. 防止除零
            adr = alpha_power[0] / bd_power[0] if bd_power[0] > 0 else 0
            # print(type(alpha_power), alpha_power, bd_power)
            channel_values1.append(round(adr, 2))

        channel_values = sig_chn_out(channel_values1)

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values ", channel_values)
    # return channel_values

def multi_chns_sef(start_seconds, stop_seconds, raw, ttype, channels, fs, task_id, band=(0.5, 30), edge=0.95, nperseg=None):
    train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)

    # if history_monitor == 'history':
    #     unit_coef = 1e6
    # else:  # 'monitor'
    #     unit_coef = 1

    channel_values, channel_values1 = [], []
    if train_data.shape[1] > 1:
        for segment in train_data:
            # Calculate RBP features for the current epoch
            # 1. 计算功率谱密度
            freqs, psd = welch(segment, fs=fs, nperseg=2*fs)

            # 2. 取分析频带
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            freqs = freqs[idx]
            sef = psd[:, idx]
            # 3. 累积分布
            cumulative_power = np.cumsum(sef, axis=1)
            total_power = cumulative_power[:, -1]

            target_power = edge * total_power
            target_power = target_power.reshape((len(target_power), 1))
            # 4. 找到累计功率>=目标功率的第一个频率
            mask = cumulative_power >= target_power
            channel_values1.append(freqs[mask.argmax(axis=1)].tolist())

        # channel_values1 = np.array(channel_values1)
        # for i in range(train_data.shape[1]):
        #     sef_tmp = np.round(channel_values1[:, i], 2)
        #     channel_values.append(sef_tmp.tolist())
        channel_values = multi_chns_out(channel_values1)

    else:
        for segment in train_data:
            # Calculate RBP features for the current epoch
            # 1. 计算功率谱密度
            freqs, psd = welch(segment, fs=fs, nperseg=2*fs)

            # 2. 取分析频带
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            freqs = freqs[idx]
            # print(type(psd), psd.shape)
            sef = psd[:, idx]
            # 3. 累积分布
            cumulative_power = np.cumsum(sef)
            total_power = cumulative_power[-1]
            if total_power == 0:
                channel_values1.append(0)

            else:
                target_power = edge * total_power
                # 4. 找到累计功率>=目标功率的第一个频率
                idx_edge = np.where(cumulative_power >= target_power)[0][0]
                sef = freqs[idx_edge]
                channel_values1.append(round(sef, 2))

        # channel_values1 = np.array(channel_values1)
        # channel_values1 = channel_values1.reshape((len(channel_values1), 1))
        #
        # for tk in channel_values1:
        #     channel_values.append(tk.tolist())
        # channel_values = [channel_values]
        channel_values = sig_chn_out(channel_values1)

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values: ", channel_values)
    # return channel_values

def multi_chns_se(start_seconds, stop_seconds, raw, ttype, channels, fs, task_id, band=(0.5, 40), method='welch', nperseg=None, normalize=True):
    train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, channels)

    # if history_monitor == 'history':
    #     unit_coef = 1e6
    # else:  # 'monitor'
    #     unit_coef = 1

    channel_values, channel_values1 = [], []
    if train_data.shape[1] > 1:
        for segment in train_data:
            if method == 'welch':
                freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
            else:
                freqs = np.fft.rfftfreq(len(segment), 1 / fs)
                psd = np.abs(np.fft.rfft(segment)) ** 2
                psd /= len(psd)

            # 限制频带
            if band is not None:
                idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                psd = psd[:, idx_band]
                freqs = freqs[idx_band]

            # 归一化得到概率分布
            psd_sum = np.array(np.sum(psd, axis=1))
            # psd_norm = psd / psd_sum
            psd_norm = np.array([psd[inx, :] / psd_sum[inx] for inx in range(len(psd_sum))])

            # 避免log(0)
            # psd_norm = psd_norm[psd_norm > 0]
            se = -np.sum(psd_norm * np.log2(psd_norm), axis=1)

            if normalize:
                se /= np.log2(psd_norm.shape[1])  # 归一化到 [0,1]

            channel_values1.append(se.tolist())

        channel_values = multi_chns_out(channel_values1)

    else:
        for segment in train_data:
            if method == 'welch':
                freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
            else:
                freqs = np.fft.rfftfreq(len(segment), 1 / fs)
                psd = np.abs(np.fft.rfft(segment)) ** 2
                psd /= len(psd)

            # 限制频带
            if band is not None:
                idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                psd = psd[:, idx_band]
                freqs = freqs[idx_band]

            # 归一化得到概率分布
            psd_sum = np.sum(psd)
            if psd_sum == 0:
                channel_values1.append(0.0)

            else:
                psd_norm = psd / psd_sum

                # 避免log(0)
                psd_norm = psd_norm[psd_norm > 0]
                se = -np.sum(psd_norm * np.log2(psd_norm))

                if normalize:
                    se /= np.log2(len(psd_norm))  # 归一化到 [0,1]

                channel_values1.append(round(se, 2))

        channel_values = sig_chn_out(channel_values1)

    ana_type = ttype
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    return result
    # print("channel_values: ", channel_values)
    # return channel_values


def aEEG_com(start_seconds, stop_seconds, special_electrodes, raw, channels, sample_frequency):
    start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
    stop_inx = int(sample_frequency * stop_seconds)
    t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
    sigbufs = sigbufs * 1e6  # V 放大为uV
    band = [2, 15]
    window_length = 1

    node_dict = special_node(special_electrodes)
    signal_labels = raw.ch_names

    sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
    # aEEG_compute_h
    aEEG_labels, aEEG_values = aEEG_compute_h3(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp

    return aEEG_labels, aEEG_values

def sig_uv_read(start_seconds, stop_seconds, special_electrodes, raw, channels, sample_frequency, history_monitor):
    start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
    stop_inx = int(sample_frequency * stop_seconds)
    t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]

    if history_monitor == 'history':
        unit_coef = 1e6
    else:  # 'monitor'
        unit_coef = 1

    sigbufs = sigbufs * unit_coef  # V 放大为uV

    node_dict = special_node(special_electrodes)
    signal_labels = raw.ch_names
    sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV

    return sigbufs_res

def aEEG_com_t(start_seconds, stop_seconds, special_electrodes, raw, ttype, channels, sample_frequency, history_monitor, task_id):
    # start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
    # stop_inx = int(sample_frequency * stop_seconds)
    # t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)
    # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    # sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
    # sigbufs = sigbufs * 1e6  # V 放大为uV

    # node_dict = special_node(special_electrodes)
    # signal_labels = raw.ch_names
    #
    # sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
    sigbufs_res = sig_uv_read(start_seconds, stop_seconds, special_electrodes, raw, channels, sample_frequency, history_monitor)
    band = [2, 15]
    window_length = 1
    # aEEG_compute_h
    aEEG_labels, aEEG_values = aEEG_compute_h3(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp

    ana_type = ttype
    tk = str(aEEG_labels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {aEEG_values}" + " }"
    # result = "{ " + f"\"taskID\": \"{task_id}\", \"qeeg_data\": {qeeg_data2}" + " }"

    # print(ana_type, aEEG_values[0][:2])
    return result

def aEEG_compute_h3(sigbufs, channels, sample_frequency, band, window_length):
    channel_labels = channels
    if len(channel_labels) > 0:
        # source_eeg = sigbufs
        # source_eeg = sigbufs
        fs = int(sample_frequency)  # 4000
        numtaps = int(60*fs/(22*5)/2)  # 2181 1091 273 # numtaps = filter orders + 1 301
        sigbufs[np.isnan(sigbufs)] = 0
        order = 4
        sos = butter(order, 1, btype='highpass', fs=fs, output='sos')
        source_eeg = sosfilt(sos, sigbufs)
        n_jobs = len(channel_labels) if len(channel_labels) < 8 else 8

        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filter_multichannel_eeg(source_eeg, fs, numtaps, band, n_jobs)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filter_one_channel(source_eeg_tile[:, :(3 * numtaps + 1)], fs, numtaps, band)[:, :source_eeg.shape[1]]

        aeeg_output = 1.631 * np.abs(eeg_filtered) + 4  # 1.231 * np.abs(eeg_filtered) + 4

        # fs_new = 250
        # decimation_factor = int(fs / fs_new)  # fs
        # # 降采样到250Hz
        # aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, fs, window_length)  # fs_new
        # utp, ltp = segment_extra(aeeg_output, fs, window_length)
        # plot_aEEG(utp, ltp, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/',
        #           'aEEG1015' + '.png')

    else:
        channel_values = []
        print("Channels Values is Null")

    return channel_labels, channel_values

def aEEG_compute_h2(input_data_dict, band, window_length):
    channles = len([channel for channel in input_data_dict['labels']])  #  if 'EEG' in channel
    data_values = np.array(input_data_dict['values'][:channles])

    channel_labels = input_data_dict['labels'][:channles]
    fs_list = input_data_dict['samples']
    # band = [2, 15]  # Desired pass band, Hz

    if channles > 0:
        # source_eeg = data_values
        fs = int(fs_list[0])
        numtaps = int(60 * fs / (22 * 5) / 2)  # numtaps = filter orders + 1  301
        data_values[np.isnan(data_values)] = 0
        # 非对称滤波 整流 振幅放大
        sos = butter(4, 1, btype='highpass', fs=fs, output='sos')
        source_eeg = sosfilt(sos, data_values)
        bp_fir = firwin(numtaps, cutoff=band, fs=fs, pass_zero=False)  # [2, 15]
        # if source_eeg.shape[1] > 3 * numtaps:
        #     eeg_filtered = filtfilt(bp_fir, [1.0], iir1)
        # else:
        #     source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
        #     eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]
        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]

        aeeg_output = 1.631 * np.abs(eeg_filtered) + 4  # 1.231 * np.abs(eeg_filtered) + 4

        # fs_new = 250  # 新的采样率
        # decimation_factor = int(fs / fs_new)
        # # 降采样到250Hz
        # aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        # channel_values = segment_extra1(aeeg_output, int(fs), window_length)  # fs_new
        utp, ltp = segment_extra(aeeg_output, fs, window_length)

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        print("Channels Values is Null")
        # channel_values = []
        utp = []
        ltp = []

    return channel_labels, utp, ltp

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

#
# butter_lowpass, segment_extra, segment_extra1, plot_aEEG, filter_one_channel,
# filter_one_channel1, filter_multichannel_eeg, aEEG_compute_h, special_node, butter_bandpass,
# aEEG_compute_h1, aEEG_com1, read_bdf, calculate_rbp, multi_relative_band_energy, abp_rbp,
# abp_rbp_com, abp_rbp_com1, sig_chns_split, abp_rbp_com_t, multi_chns_rav, multi_chns_csa,
# sig_chn_out, multi_chns_out, multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef,
# multi_chns_se, aEEG_com, sig_uv_read, aEEG_com_t, aEEG_compute_h3, aEEG_compute_h2, plot_rbp