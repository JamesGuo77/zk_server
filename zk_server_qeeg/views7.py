import numpy as np
from django.http import HttpResponse
from scipy.signal import butter
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
# 2025/10/3 V1.5 abp回读测试通过；多线程任务测试
# 2025/10/9 V1.6 abp监测测试通过；epoch_length全局设置

epoch_length = 10  # s

success_message = {
    "status": "200",
    "message": "ok"
}

error_message = {
    "status": "500",
    "message": "ok"
}

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

def plot_aEEG(utp, ltp, tmp_data_path):

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
    plt.savefig(os.path.dirname(tmp_data_path) + '/aEEG1010.png')
    # plt.savefig('/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/aEEG0912.png')
    print("aEEG波形保存完成")


def filter_one_channel(channel_data, fs, numtaps):
    bp_fir = firwin(numtaps, cutoff=[2, 15], fs=fs, pass_zero=False)  # 2181 6601  # TODO 2181
    # eeg_filtered = filtfilt(bp_fir, [1.0], raw_bdf[:][0][:]*1e6)  # [:2400000]
    return filtfilt(bp_fir, [1.0], channel_data)

def filter_one_channel1(channel_data, fs, numtaps):
    sos = butter(4, 1, btype='highpass', fs=fs, output='sos')
    iir1 = sosfilt(sos, channel_data)
    bp_fir = firwin(numtaps, cutoff=[2, 15], fs=fs, pass_zero=False)  # 2181 6601  # TODO 2181
    # eeg_filtered = filtfilt(bp_fir, [1.0], raw_bdf[:][0][:]*1e6)  # [:2400000]
    return filtfilt(bp_fir, [1.0], iir1)

# 多通道滤波函数
def filter_multichannel_eeg(eeg_data, fs, numtaps, n_jobs=3):
    filtered = Parallel(n_jobs=n_jobs)(delayed(filter_one_channel1)(ch_data, fs, numtaps) for ch_data in eeg_data)
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
    b, a = butter(order, [nor_lowcut, nor_highcut], btype='band')
    return b, a

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

def aeeg_monitor(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            input_data_dict = json.loads(jsonStr)
            additional_input = json.loads(input_data_dict["additionalInput"])
            band_tmp = additional_input["band"].split(":")
            band = [int(band_tmp[0]), int(band_tmp[1])]
            window_length = int(additional_input["windowLength"])

            channel_labels, channel_values = aEEG_compute_h1(input_data_dict, band, window_length)  # utp, ltp
            task_id = input_data_dict["taskID"]
            ana_type = "aEEG"
            # channel_labels = str(channel_labels).replace("\'", '\"')

            tk = str(channel_labels)
            tk = tk.replace("\'", "\"")

            result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"
            return HttpResponse(result)
            # return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))


def qeeg_monitor(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            input_data_dict = json.loads(jsonStr)
            additional_input = json.loads(input_data_dict["additionalInput"])
            band_tmp = additional_input["band"].split(":")
            band = [int(band_tmp[0]), int(band_tmp[1])]
            window_length = int(additional_input["windowLength"])

            # channel_labels, channel_values = aEEG_compute_h1(input_data_dict, band, window_length)  # utp, ltp
            # aEEG_com1(input_data_dict, band, window_length)
            aEEG_labels, aEEG_values = aEEG_com1(input_data_dict, band, window_length)
            # abp bp_type: 0-abp and rbp 1-abp 2-rbp abp_rbp_com1(input_data_dict, bp_type)
            abp_labels, abp_values, rbp_values = abp_rbp_com1(input_data_dict, 1)

            task_id = input_data_dict["taskID"]
            ana_type = "aEEG_abp"
            qeeg_type = "aEEG"
            qeeg_type1 = "abp"

            tk = str(aEEG_labels)
            tk = tk.replace("\'", "\"")
            tk1 = str(abp_labels)
            tk1 = tk1.replace("\'", "\"")
            qeeg_data = "{ " + f" \"type\": \"{qeeg_type}\", \"labels\": {tk}, \"values\": {aEEG_values}" + " }"
            qeeg_data1 = "{ " + f" \"type\": \"{qeeg_type1}\", \"labels\": {tk1}, \"values\": {abp_values}" + " }"
            qeeg_data2 = "[" + f"{qeeg_data}" + ", " + f"{qeeg_data1}" + "]"

            # result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"
            result = "{ " + f"\"taskID\": \"{task_id}\", \"qeeg_data\": {qeeg_data2}" + " }"

            return HttpResponse(result)
            # return HttpResponse(json.dumps(success_message))

        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

def qteeg_monitor(request):  # 待验证
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            input_data_dict = json.loads(jsonStr)
            additional_input = json.loads(input_data_dict["additionalInput"])
            band_tmp = additional_input["band"].split(":")
            band = [int(band_tmp[0]), int(band_tmp[1])]
            window_length = int(additional_input["windowLength"])
            qeeg_act = additional_input["windowLength"]  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR

            # 创建线程
            aEEG_t = threading.Thread(target=aEEG_com_t, args=(
            start_seconds, stop_seconds, raw, channels, sample_frequency, input_data_dict["taskID"]))  # Lack specialElectronode
            abp_t = threading.Thread(target=abp_rbp_com_t, args=(
            start_seconds, stop_seconds, raw, channels, sample_frequency, 1, input_data_dict["taskID"]))

            # 启动线程
            aEEG_t.start()
            abp_t.start()

            # 等待所有线程结束
            aEEG_t.join()
            abp_t.join()

            return HttpResponse(json.dumps(success_message))

        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

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

def aeeg_history2(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            # f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
            # # n = f.signals_in_file
            # signal_labels = f.getSignalLabels()
            # sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致

            trend_channels = json_data['trendChannels']
            aeeg_chns = []  # labels: aEEG
            rbp_chns = []  # labels: RBP
            for tchns in trend_channels:
                if tchns["type"] == "aEEG":
                    aeeg_chns.append(tchns["label"])

                elif tchns["type"] == "RBP":
                    rbp_chns.append(tchns["label"])

            channels = json_data['channels']  # trendChannels
            for chns in channels:
                chn_split = chns.split('-')
                aeeg_chns.append(chn_split[0])
                aeeg_chns.append(chn_split[1])
            aeeg_chns = list(set(aeeg_chns))
            aeeg_chns_sim = [chn for chn in aeeg_chns if chn not in ('AV', 'Ref', 'REF', 'ref')]

            start_time = time.time()
            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue

                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(aeeg_chns_sim))  # , preload=True
                # raw = raw.filter(l_freq=2, h_freq=70)
                picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                sample_frequency = raw.info['sfreq']

                start_seconds = int(json_data['startSeconds'])
                stop_seconds = int(json_data['stopSeconds'])

                signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长
                if start_seconds > signal_seconds:  # 不分析
                    error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif start_seconds > stop_seconds:  #
                    error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif stop_seconds > signal_seconds:
                    stop_seconds = signal_seconds

                start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
                stop_inx = int(sample_frequency * stop_seconds)
                t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

                sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
                sigbufs = sigbufs * 1e6
                band = [2, 15]
                window_length = 1

                node_dict = special_node(json_data['specialElectrodes'])
                signal_labels = raw.ch_names

                sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
                aeeg_labels, aeeg_values = aEEG_compute_h(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp

                task_id = json_data["taskID"]

                # ana_type = "aEEG"
                # {
                #     "label": "Fp1-REF",
                #     "type": "RBP"
                #     "values": [[82.46, 6.54, 3.44, 7.56], [86.33, 7.57, 3.45, 2.65], [77.9, 12.82, 3.23, 6.06],
                #                        [59.13, 17.86, 9.67, 13.34]]
                # }
                # tk = str(channel_labels)
                # tk = tk.replace("\'", "\"")

                # trendChannels{}
                trend_channels_res = []
                for aeeg_inx in range(len(aeeg_labels)):
                    tchns_res = {}
                    tchns_res['label'] = aeeg_labels[aeeg_inx]
                    tchns_res['type'] = 'aEEG'
                    tchns_res['values'] = aeeg_values[aeeg_inx]
                trend_channels_res.append(str(tchns_res).replace("\'", "\""))

                ######## 加入rbp
                rbp_data = []
                start_time = start_seconds
                # epoch_length = 10
                step_size = epoch_length  # 10
                while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
                    # features = []
                    start, stop = raw.time_as_index([start_time, start_time + epoch_length])
                    temp = raw[:, start:stop][0]
                    rbp_data.append(temp)
                    start_time += step_size

                rbp_data = np.array(rbp_data)

                rbp_features_all_segments = []
                for segment in rbp_data:
                    # Calculate RBP features for the current epoch
                    rbp_features, rbp2d_features = calculate_rbp(segment, int(sample_frequency))
                    rbp_features_all_segments.append(rbp_features)
                    # rbp2d_features_all_segments.append(rbp2d_features)

                rbp_features_all_segments = np.array(rbp_features_all_segments)  # (265, 9, 5)

                shape_size = rbp_features_all_segments.shape
                # rbp_features = rbp_features_all_segments[:, 0, :].reshape((shape_size[0], shape_size[2]))  # 取出对应通道数据

                sec_num = shape_size[1]
                rbp_values = []
                if sec_num > 0:
                    for i in range(sec_num):
                        rbp_tmp = np.multiply(rbp_features_all_segments[:, i, :], 100).tolist(),
                        rbp_values.extend(rbp_tmp)
                    # print(res.shape)
                    rbp_values = np.round(np.array(rbp_values), 2)
                    # return channel_values.tolist()

                else:
                    print("Channels Values is Null")

                for rbp_inx in range(len(rbp_chns)):
                    tchns_res = {}
                    tchns_res['label'] = rbp_chns[rbp_inx]
                    tchns_res['type'] = 'RBP'
                    tchns_res['values'] = rbp_values[rbp_inx]
                trend_channels_res.append(str(tchns_res).replace("\'", "\""))
                ######## 加入rbp End

                # result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

                result = "{ " + f"\"taskID\": \"{task_id}\", \"trendChannels\": {trend_channels_res} " + " }"

                return HttpResponse(result)

                # return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))


def rbp_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            chn_list = []
            channels = json_data['channels']
            for chns in channels:
                chn_split = chns.split('-')
                chn_list.append(chn_split[0])
                chn_list.append(chn_split[1])
            chn_list = list(set(chn_list))
            chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue

                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True
                # raw = raw.filter(l_freq=2, h_freq=70)
                # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                sample_frequency = raw.info['sfreq']

                start_seconds = int(json_data['startSeconds'])
                stop_seconds = int(json_data['stopSeconds'])

                signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长
                if start_seconds > signal_seconds:  # 不分析
                    error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif start_seconds > stop_seconds:  #
                    error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif stop_seconds > signal_seconds:
                    stop_seconds = signal_seconds

                # rbp start
                train_data = []
                start_time = start_seconds
                # epoch_length = 10
                step_size = epoch_length  #10
                while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
                    # features = []
                    start, stop = raw.time_as_index([start_time, start_time + epoch_length])
                    temp = raw[:, start:stop][0]
                    train_data.append(temp)
                    start_time += step_size

                train_data = np.array(train_data)

                rbp_features_all_segments = []
                abp_features_all_segments = []
                for segment in train_data:
                    # Calculate RBP features for the current epoch
                    rbp_features, abp_features = calculate_rbp(segment * 1e6, int(sample_frequency))
                    rbp_features_all_segments.append(rbp_features)
                    abp_features_all_segments.append(abp_features)
                    # rbp2d_features_all_segments.append(rbp2d_features)

                rbp_features_all_segments = np.array(rbp_features_all_segments)  # (265, 9, 5)
                abp_features_all_segments = np.array(abp_features_all_segments)

                shape_size = rbp_features_all_segments.shape
                # rbp_features = rbp_features_all_segments[:, 0, :].reshape((shape_size[0], shape_size[2]))  # 取出对应通道数据

                sec_num = shape_size[1]
                channel_values = []
                if sec_num > 0:
                    for i in range(sec_num):
                        rbp_tmp = np.multiply(rbp_features_all_segments[:, i, :], 100).tolist(),
                        channel_values.append(rbp_tmp)
                    # print(res.shape)
                    # return rbp_fea.tolist()
                    channel_values = np.round(np.array(channel_values), 2)

                else:
                    print("Channels Values is Null")

                # rbp end
                task_id = json_data["taskID"]
                ana_type = "RBP"

                tk = str(channels)  # channel_labels
                tk = tk.replace("\'", "\"")

                result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values.tolist()}" + " }"

                return HttpResponse(result)

                # return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

def multi_relative_band_energy(eeg_signal, fs, bands=None):
    """
    计算EEG信号的相对频带能量

    参数:
    --------
    eeg_signal : 1D numpy array
        EEG 信号
    fs : float
        采样率 (Hz)
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

    row_sums = np.sum(np.nan_to_num(abs_power, nan=0.0), axis=1, keepdims=True)
    # Divide each element by its corresponding row sum and multiply by 100
    rel_power = (abs_power / row_sums) * 100

    return rel_power, abs_power

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

    rbp_powers = []
    abp_powers = []
    for segment in train_data:
        # Calculate RBP features for the current epoch
        rbp_power, abp_power = multi_relative_band_energy(segment * 1e6, int(sample_frequency))
        rbp_powers.append(rbp_power)
        abp_powers.append(abp_power)

    rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
    abp_powers = np.array(abp_powers)

    # bp_type: 0-abp and rbp 1-abp 2-rbp

    sec_num = abp_powers.shape[1]
    channel_values = []
    channel_values1 = []
    if sec_num > 0:  # {通道：{时间位：max-min}}
        # channel_values = np.round(abp_powers, 2)
        for i in range(sec_num):
            if (bp_type == 0) or (bp_type == 1):
                abp_power = abp_powers[:, i, :]
                abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
                                    range(abp_power.shape[0])], 2)
                channel_values.append(abp_tmp.tolist())
                # abp_tmp = abp_powers[:, i, :].tolist()
                # channel_values.append(abp_tmp)

            if (bp_type == 0) or (bp_type == 2):
                rbp_power = rbp_powers[:, i, :]
                rbp_tmp = np.round([[rbp_power[j, 0], rbp_power[j, 1], rbp_power[j, 2], rbp_power[j, 3]] for j in
                                    range(rbp_power.shape[0])], 2)
                channel_values1.append(rbp_tmp.tolist())

        # if len(channel_values) > 0:
        #     channel_values = np.round(np.array(channel_values), 2)
    else:
        print("Channels Values is Null")

    return channels, channel_values, channel_values1

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

        # train_data = np.array(train_data)

        rbp_powers = []
        abp_powers = []
        for segment in train_data:
            # Calculate RBP features for the current epoch
            rbp_power, abp_power = multi_relative_band_energy(segment, fs)
            rbp_powers.append(rbp_power)
            abp_powers.append(abp_power)

        rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
        abp_powers = np.array(abp_powers)

        # bp_type: 0-abp and rbp 1-abp 2-rbp
        sec_num = abp_powers.shape[1]
        channel_values = []
        channel_values1 = []
        if sec_num > 0:  # {通道：{时间位：max-min}}
            # channel_values = np.round(abp_powers, 2)
            for i in range(sec_num):
                if (bp_type == 0) or (bp_type == 1):
                    abp_power = abp_powers[:, i, :]
                    abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
                                        range(abp_power.shape[0])], 2)
                    channel_values.append(abp_tmp.tolist())

                if (bp_type == 0) or (bp_type == 2):
                    rbp_power = rbp_powers[:, i, :]
                    rbp_tmp = np.round([[rbp_power[j, 0], rbp_power[j, 1], rbp_power[j, 2], rbp_power[j, 3]] for j in
                                        range(rbp_power.shape[0])], 2)
                    channel_values1.append(rbp_tmp.tolist())
        else:
            print("Channels length is Zero")

    else:
        print("Channels Values is Null")

    return channel_labels, channel_values, channel_values1

def abp_rbp_com_t(start_seconds, stop_seconds, raw, channels, sample_frequency, bp_type, task_id):
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

    rbp_powers = []
    abp_powers = []
    for segment in train_data:
        # Calculate RBP features for the current epoch
        rbp_power, abp_power = multi_relative_band_energy(segment * 1e6, int(sample_frequency))
        rbp_powers.append(rbp_power)
        abp_powers.append(abp_power)

    rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
    abp_powers = np.array(abp_powers)

    # bp_type: 0-abp and rbp 1-abp 2-rbp

    sec_num = abp_powers.shape[1]
    channel_values = []
    channel_values1 = []
    if sec_num > 0:  # {通道：{时间位：max-min}}
        # channel_values = np.round(abp_powers, 2)
        for i in range(sec_num):
            if (bp_type == 0) or (bp_type == 1):
                abp_power = abp_powers[:, i, :]
                abp_tmp = np.round([[abp_power[j, 0], abp_power[j, 1], abp_power[j, 2], abp_power[j, 3]] for j in
                                    range(abp_power.shape[0])], 2)
                channel_values.append(abp_tmp.tolist())
                # abp_tmp = abp_powers[:, i, :].tolist()
                # channel_values.append(abp_tmp)

            if (bp_type == 0) or (bp_type == 2):
                rbp_power = rbp_powers[:, i, :]
                rbp_tmp = np.round([[rbp_power[j, 0], rbp_power[j, 1], rbp_power[j, 2], rbp_power[j, 3]] for j in
                                    range(rbp_power.shape[0])], 2)
                channel_values1.append(rbp_tmp.tolist())

        # if len(channel_values) > 0:
        #     channel_values = np.round(np.array(channel_values), 2)
    else:
        print("Channels Values is Null")

    ana_type = "abp"
    tk = str(channels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

    qteeg_history_api = "/qteeg_history"
    # response = requests.post(url=get_pc_url(qteeg_history_api), json=result, headers=header)
    print(ana_type, channel_values[0][:2])


def abp_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            chn_list = []
            channels = json_data['channels']
            for chns in channels:
                chn_split = chns.split('-')
                chn_list.append(chn_split[0])
                chn_list.append(chn_split[1])
            chn_list = list(set(chn_list))
            chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue

                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True
                # raw = raw.filter(l_freq=2, h_freq=70)
                # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                sample_frequency = raw.info['sfreq']

                start_seconds = int(json_data['startSeconds'])
                stop_seconds = int(json_data['stopSeconds'])

                signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长
                if start_seconds > signal_seconds:  # 不分析
                    error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif start_seconds > stop_seconds:  #
                    error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif stop_seconds > signal_seconds:
                    stop_seconds = signal_seconds

                # abp start
                # bp_type: 0-abp and rbp 1-abp 2-rbp
                channel_labels, channel_values, channel_values1 = abp_rbp_com(start_seconds, stop_seconds, raw, chn_list_sim, sample_frequency, 1)

                # abp end
                task_id = json_data["taskID"]
                ana_type = "ABP"

                tk = str(channel_labels)  # channel_labels
                tk = tk.replace("\'", "\"")

                result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values.tolist()}" + " }"

                return HttpResponse(result)

                # return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

def aeeg_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            # logging.basicConfig(
            #     level=logging.DEBUG,
            #     format='%(asctime)s | %(levelname)s | %(message)s',
            #     filename='/disk1/workspace/py39_tf270/vehicle_recommend/run_log/aeeg_logs.log',
            #     filemode='a'
            # )

            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            chn_list = []
            channels = json_data['channels']
            for chns in channels:
                chn_split = chns.split('-')
                chn_list.append(chn_split[0])
                chn_list.append(chn_split[1])
            chn_list = list(set(chn_list))
            chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]

            start_time = time.time()
            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue

                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True
                # raw = raw.filter(l_freq=2, h_freq=70)
                picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                sample_frequency = raw.info['sfreq']

                start_seconds = int(json_data['startSeconds'])  # 4200
                stop_seconds = int(json_data['stopSeconds'])  # 4763 (4800)

                signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

                if start_seconds > signal_seconds:  # 不分析
                    error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif start_seconds > stop_seconds:  #
                    error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif stop_seconds > signal_seconds:
                    stop_seconds = signal_seconds

                start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
                stop_inx = int(sample_frequency * stop_seconds)
                t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

                sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
                sigbufs = sigbufs * 1e6  # V 放大为uV
                band = [2, 15]
                window_length = 1

                node_dict = special_node(json_data['specialElectrodes'])
                signal_labels = raw.ch_names

                sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
                # aEEG_compute_h
                channel_labels, channel_values = aEEG_compute_h3(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp

                task_id = json_data["taskID"]
                ana_type = "aEEG"

                tk = str(channel_labels)
                tk = tk.replace("\'", "\"")

                result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"

                return HttpResponse(result)
                # return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

def qeeg_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            chn_list = []
            channels = json_data['channels']
            for chns in channels:
                chn_split = chns.split('-')
                chn_list.append(chn_split[0])
                chn_list.append(chn_split[1])
            chn_list = list(set(chn_list))
            chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]
            qeeg_act = ['aEEG', 'RBP']  # json_data["qeegAct"]  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue
                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True tuple(chn_list_sim)
                # raw = raw.filter(l_freq=2, h_freq=70)
                # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                sample_frequency = raw.info['sfreq']

                start_seconds = int(json_data['startSeconds'])  # 4200
                stop_seconds = int(json_data['stopSeconds'])  # 4763 (4800)

                signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

                if start_seconds > signal_seconds:  # 不分析
                    error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif start_seconds > stop_seconds:  #
                    error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif stop_seconds > signal_seconds:
                    stop_seconds = signal_seconds

                # aEEG_compute_h
                # aEEG_labels, aEEG_values = aEEG_compute_h3(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp
                aEEG_labels, aEEG_values = aEEG_com(start_seconds, stop_seconds, json_data['specialElectrodes'], raw, channels, sample_frequency)
                # abp bp_type: 0-abp and rbp 1-abp 2-rbp
                abp_labels, abp_values, abp_values1 = abp_rbp_com(start_seconds, stop_seconds, raw, channels, sample_frequency, 1)

                task_id = json_data["taskID"]
                ana_type = "aEEG_abp"
                qeeg_type = "aEEG"
                qeeg_type1 = "abp"

                tk = str(aEEG_labels)
                tk = tk.replace("\'", "\"")
                tk1 = str(abp_labels)
                tk1 = tk1.replace("\'", "\"")
                qeeg_data = "{ " + f" \"type\": \"{qeeg_type}\", \"labels\": {tk}, \"values\": {aEEG_values}" + " }"
                qeeg_data1 = "{ " + f" \"type\": \"{qeeg_type1}\", \"labels\": {tk1}, \"values\": {abp_values}" + " }"
                qeeg_data2 = "[" + f"{qeeg_data}"  + ", " + f"{qeeg_data1}" + "]"

                # result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"
                result = "{ " + f"\"taskID\": \"{task_id}\", \"qeeg_data\": {qeeg_data2}" + " }"

                return HttpResponse(result)
                # return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

def qteeg_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            chn_list = []
            channels = json_data['channels']
            for chns in channels:
                chn_split = chns.split('-')
                chn_list.append(chn_split[0])
                chn_list.append(chn_split[1])
            chn_list = list(set(chn_list))
            chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]
            qeeg_act = ['aEEG', 'RBP']  # json_data["qeegAct"]  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue
                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True tuple(chn_list_sim)
                # raw = raw.filter(l_freq=2, h_freq=70)
                # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                sample_frequency = raw.info['sfreq']

                start_seconds = int(json_data['startSeconds'])  # 4200
                stop_seconds = int(json_data['stopSeconds'])  # 4763 (4800)

                signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

                if start_seconds > signal_seconds:  # 不分析
                    error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif start_seconds > stop_seconds:  #
                    error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
                    return HttpResponse(json.dumps(error_message))

                elif stop_seconds > signal_seconds:
                    stop_seconds = signal_seconds

                # # abp bp_type: 0-abp and rbp 1-abp 2-rbp
                # 创建线程
                aEEG_t = threading.Thread(target=aEEG_com_t, args=(start_seconds, stop_seconds, json_data['specialElectrodes'], raw, channels, sample_frequency, json_data["taskID"]))
                abp_t = threading.Thread(target=abp_rbp_com_t, args=(start_seconds, stop_seconds, raw, channels, sample_frequency, 1, json_data["taskID"]))

                # 启动线程
                aEEG_t.start()
                abp_t.start()

                # 等待所有线程结束
                aEEG_t.join()
                abp_t.join()

                return HttpResponse(json.dumps(success_message))
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

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

def aEEG_com_t(start_seconds, stop_seconds, special_electrodes, raw, channels, sample_frequency, task_id):
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

    ana_type = "aEEG"
    tk = str(aEEG_labels)
    tk = tk.replace("\'", "\"")
    result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {aEEG_values}" + " }"
    # result = "{ " + f"\"taskID\": \"{task_id}\", \"qeeg_data\": {qeeg_data2}" + " }"

    qteeg_history_api = "/qteeg_history"

    # requests.post(url=get_pc_url(qteeg_history_api), json=result, headers=header)
    print(ana_type, aEEG_values[0][:2])

    # return HttpResponse(result)

## 获取主机url
pc_host = "172.24.60.106"
pc_port = "8040"
header = {
        "Content-Type": "application/json;charset=UTF-8"
    }

def get_pc_url(api):
    return "http://" + pc_host + ":" + pc_port + "/" + api

def aEEG_compute_h3(sigbufs, channels, sample_frequency, band, window_length):
    channel_labels = channels
    if len(channel_labels) > 0:
        # source_eeg = sigbufs
        fs = int(sample_frequency)  # 4000
        numtaps = int(60*fs/(22*5)/2)  # 2181 1091 273 # numtaps = filter orders + 1 301
        sigbufs[np.isnan(sigbufs)] = 0
        order = 4
        sos = butter(order, 1, btype='highpass', fs=fs, output='sos')
        source_eeg = sosfilt(sos, sigbufs)
        n_jobs = len(channel_labels) if len(channel_labels) < 8 else 8

        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filter_multichannel_eeg(source_eeg, fs, numtaps, n_jobs)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filter_one_channel(source_eeg_tile[:, :(3 * numtaps + 1)], fs, numtaps)[:, :source_eeg.shape[1]]

        aeeg_output = 1.631 * np.abs(eeg_filtered) + 4  # 1.231 * np.abs(eeg_filtered) + 4

        # fs_new = 250
        # decimation_factor = int(fs / fs_new)  # fs
        # # 降采样到250Hz
        # aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, fs, window_length)  # fs_new

        # 分段提取UTP LTP
        # utp, ltp = segment_extra(aeeg_output, fs, window_length)  # fs_new
        #
        # # aEEG波形
        # plot_aEEG(utp, ltp, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/')

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

tkk = 8
if __name__ == '__main__':
    is_history = 0  # 1-history 0-monitor 2-abp 3-rbp
    signal_type = 1  # 1-signal 2-sin
    time_cn = 0

    if is_history == 1:
        chn_list = []
        channels = ["Fp1-REF", "Fp2-REF", "F3-F4", "O1-AV"]
        for chns in channels:
            chn_split = chns.split('-')
            chn_list.append(chn_split[0])
            chn_list.append(chn_split[1])
        chn_list = list(set(chn_list))
        chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]

        start_time1 = time.time()
        # TODO 多个bdf文件读取
        file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20250912110203"  # 20250912110203 20241206184941
        for fname in os.listdir(file_path):  # 寻找.rml文件
            if '.bdf' not in fname:
                continue

            raw = mne.io.read_raw_bdf(os.path.join(file_path, fname), include=tuple(chn_list_sim))  # , preload=True
            # raw = raw.filter(l_freq=2, h_freq=70)
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            sample_frequency = raw.info['sfreq']

            start_seconds = 0
            stop_seconds = round(raw.n_times / sample_frequency, 2)
            signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

            if start_seconds > signal_seconds:  # 不分析
                print("The start senonds is over signal lengths! Read signal failure. ")

            elif start_seconds > stop_seconds:  #
                print("The start senonds is over stop senonds! Read signal failure. ")

            elif stop_seconds > signal_seconds:
                stop_seconds = signal_seconds

            start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
            stop_inx = int(sample_frequency * stop_seconds)  # 19000000
            t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

            sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
            sigbufs = sigbufs * 1e6
            band = [2, 15]
            window_length = 1

            node_dict = special_node([
                {
                    "name": "AV",
                    "electrodes": [
                        {
                            "electrode": "Fp1",
                            "weight": 80
                        },
                        {
                            "electrode": "Fp2",
                            "weight": 100
                        }
                    ]
                }
            ])
            signal_labels = raw.ch_names

            sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
            channel_labels, channel_values = aEEG_compute_h3(sigbufs_res, channels, sample_frequency, band,
                                                            window_length)  # utp, ltp
            print("aEEG计算完成，共运行：%.8s s" % (time.time() - start_time1))
            ana_type = "aEEG"

            result = {
                "type": ana_type,
                "labels": channel_labels,
                "values": len(channel_values)
            }
            print(result)

    elif is_history == 2:  # abp
        chn_list = []
        channels = ["Fp1-REF", "Fp2-REF", "F3-F4", "O1-AV"]
        for chns in channels:
            chn_split = chns.split('-')
            chn_list.append(chn_split[0])
            chn_list.append(chn_split[1])
        chn_list = list(set(chn_list))
        chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]

        start_time1 = time.time()
        # TODO 多个bdf文件读取
        file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20250912110203"  # 20250912110203 20241206184941
        for fname in os.listdir(file_path):  # 寻找.rml文件
            if '.bdf' not in fname:
                continue

            raw = mne.io.read_raw_bdf(os.path.join(file_path, fname), include=["Fp1", "Fp2", "F3", "F4"])  # , preload=True tuple(chn_list_sim)
            # raw = raw.filter(l_freq=2, h_freq=70)
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            sample_frequency = raw.info['sfreq']

            start_seconds = 0
            stop_seconds = round(raw.n_times / sample_frequency, 2)
            signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

            # if start_seconds > signal_seconds:  # 不分析
            #     print("The start senonds is over signal lengths! Read signal failure. ")
            #
            # elif start_seconds > stop_seconds:  #
            #     print("The start senonds is over stop senonds! Read signal failure. ")
            #
            # elif stop_seconds > signal_seconds:
            #     stop_seconds = signal_seconds
            #
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
            print("Data import Finish! ")

            rbp_powers = []
            abp_powers = []
            for segment in train_data:
                # Calculate RBP features for the current epoch
                # segment_ch = segment[:, :] * 1e6
                rbp_power, abp_power = multi_relative_band_energy(segment * 1e6, int(sample_frequency))
                rbp_powers.append(rbp_power)
                abp_powers.append(abp_power)

            rbp_powers = np.array(rbp_powers)  # (265, 9, 5)
            abp_powers = np.array(abp_powers)

            sec_num = abp_powers.shape[1]
            channel_values = []
            if sec_num > 0:
                channel_values = np.round(abp_powers, 2)

            else:
                print("Channels Values is Null")
            print("ABP计算完成，共运行：%.8s s" % (time.time() - start_time1))
            # abp end
            ana_type = "ABP"

            result = {
                "type": ana_type,
                "labels": chn_list_sim,
                "values": len(channel_values)
            }
            print(result)

    elif is_history == 3:  # rbp
        chn_list = []
        channels = ["Fp1-REF", "Fp2-REF", "F3-F4", "O1-AV"]
        for chns in channels:
            chn_split = chns.split('-')
            chn_list.append(chn_split[0])
            chn_list.append(chn_split[1])
        chn_list = list(set(chn_list))
        chn_list_sim = [chn for chn in chn_list if chn not in ('AV', 'Ref', 'REF', 'ref')]

        start_time1 = time.time()
        # TODO 多个bdf文件读取
        file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20250912110203"  # 20250912110203 20241206184941
        for fname in os.listdir(file_path):  # 寻找.rml文件
            if '.bdf' not in fname:
                continue

            raw = mne.io.read_raw_bdf(os.path.join(file_path, fname), include=tuple(chn_list_sim))  # , preload=True
            # raw = raw.filter(l_freq=2, h_freq=70)
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            sample_frequency = raw.info['sfreq']

            start_seconds = 0
            stop_seconds = round(raw.n_times / sample_frequency, 2)
            signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长 4750

            if start_seconds > signal_seconds:  # 不分析
                print("The start senonds is over signal lengths! Read signal failure. ")

            elif start_seconds > stop_seconds:  #
                print("The start senonds is over stop senonds! Read signal failure. ")

            elif stop_seconds > signal_seconds:
                stop_seconds = signal_seconds

            start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
            stop_inx = int(sample_frequency * stop_seconds)  # 19000000
            t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

            sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
            sigbufs = sigbufs * 1e6
            band = [2, 15]
            window_length = 1

            # rbp start
            train_data = []
            start_time = start_seconds
            # epoch_length = 10
            step_size = epoch_length  #10
            while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
                # features = []
                start, stop = raw.time_as_index([start_time, start_time + epoch_length])
                temp = raw[:, start:stop][0]
                train_data.append(temp)
                start_time += step_size

            train_data = np.array(train_data)

            rbp_features_all_segments = []
            for segment in train_data:
                # Calculate RBP features for the current epoch
                rbp_features, rbp2d_features = calculate_rbp(segment, int(sample_frequency))
                rbp_features_all_segments.append(rbp_features)
                # rbp2d_features_all_segments.append(rbp2d_features)

            rbp_features_all_segments = np.array(rbp_features_all_segments)  # (265, 9, 5)

            shape_size = rbp_features_all_segments.shape
            # rbp_features = rbp_features_all_segments[:, 0, :].reshape((shape_size[0], shape_size[2]))  # 取出对应通道数据

            sec_num = shape_size[1]
            channel_values = []
            if sec_num > 0:
                for i in range(sec_num):
                    rbp_tmp = np.multiply(rbp_features_all_segments[:, i, :], 100).tolist(),
                    channel_values.extend(rbp_tmp)
                channel_values = np.round(np.array(channel_values), 2)

            else:
                print("Channels Values is Null")
            print("RBP计算完成，共运行：%.8s s" % (time.time() - start_time1))
            ana_type = "RBP"

            result = {
                "type": ana_type,
                "labels": chn_list_sim,
                "values": len(channel_values)
            }
            print(result)

    elif is_history == 0:
        # TODO 监测
        file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20250912110203"  # 20241206184941
        chn_list_sim = ["Fp1", "T4"]
        for fname in os.listdir(file_path):  # 寻找.rml文件
            if '.bdf' not in fname:
                continue

            input_data_dict = {}
            if signal_type == 1:
                raw = mne.io.read_raw_bdf(os.path.join(file_path, fname), include=tuple(chn_list_sim))  # , preload=True ,
                # raw = raw.filter(l_freq=0.5, h_freq=35)
                picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

                input_data_dict['labels'] = raw.ch_names
                input_data_dict['samples'] = [raw.info['sfreq']] * len(raw.ch_names)
                sample_frequency = raw.info['sfreq']
                signal_seconds = round(raw.n_times / sample_frequency, 2)

            else:
                input_data_dict['labels'] = chn_list_sim
                sampling_rate = 4000  # 采样频率为4000Hz
                input_data_dict['samples'] =[sampling_rate]
                duration = 2000  # 信号持续时间为1秒
                frequency = 10  # 正弦波频率为10Hz
                sample_frequency = sampling_rate


                # 生成时间轴
                t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

                # 生成正弦信号
                signal1 = 50 * np.sin(2 * np.pi * frequency * t)
                signal_seconds = round(len(t)/ sample_frequency, 2)

            utp_t = []
            ltp_t = []

            start_seconds = 0
            stop_seconds = start_seconds + epoch_length  # 10
            while stop_seconds < signal_seconds:
                start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
                stop_inx = int(sample_frequency * stop_seconds)  # 19000000
                if signal_type == 1:
                    t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

                    sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
                    sigbufs = sigbufs * 1e6
                    input_data_dict['values'] = np.array(sigbufs[:len(raw.ch_names), :])  # 10s

                else:
                    input_data_dict['values'] = np.array(signal1[start_inx: stop_inx]).reshape((1, stop_inx - start_inx))

                band = [2, 15]  # [int(band_tmp[0]), int(band_tmp[1])]
                window_length = 1  # int(additional_input["windowLength"])
                # aEEG_labels, aEEG_values = aEEG_com1(input_data_dict, band, window_length)
                abp_labels, abp_values, rbp_values = abp_rbp_com1(input_data_dict, 1)
                channel_labels, utp, ltp = aEEG_com1(input_data_dict, band, window_length)  # utp, ltp
                utp_t.extend(utp)
                ltp_t.extend(ltp)
                start_seconds = stop_seconds
                stop_seconds = stop_seconds + epoch_length  # 10

                # aEEG波形
            plot_aEEG(utp_t, ltp_t, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/')





