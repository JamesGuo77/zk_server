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

def asym_filter_envelope(numtaps, asym_filter_freq_hz, fs, source_eeg):
    taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], type='bandpass', fs=fs)
    y = lfilter(taps, 1, source_eeg)  # 对数据序列进行滤波，是标准差分方程直接形式的转换
    # Y = np.fft.fft(y, fs)

    # Setting standard filter requirements.
    lp_order = 1
    cutoff = asym_filter_freq_hz[3]
    rectify = np.abs(y)
    b, a = butter_lowpass(cutoff, fs, lp_order)
    # print(b)
    # print(a)
    enveloped_eeg = filtfilt(b, a, rectify)

    aeeg_output = 2 * enveloped_eeg + 1
    return aeeg_output

def asym_filter_envelope1(numtaps, asym_filter_freq_hz, fs, source_eeg):
    bp_fir = firwin(numtaps, cutoff=[2, 70], fs=fs, pass_zero=False)
    eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg)
    taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], type='bandpass', fs=fs)
    y = lfilter(taps, 1, eeg_filtered)  # 对数据序列进行滤波，是标准差分方程直接形式的转换

    # Setting standard filter requirements.
    lp_order = 1
    cutoff = asym_filter_freq_hz[3]
    rectify = np.abs(y)
    b, a = butter_lowpass(cutoff, fs, lp_order)
    enveloped_eeg = filtfilt(b, a, rectify)

    aeeg_output = 1.631 * enveloped_eeg + 4  # 1.631 * enveloped_eeg + 4  # 2
    fs_new = 250  # 新的采样率
    decimation_factor = int(fs / fs_new)
    # 降采样到250Hz
    aeeg_output = aeeg_output[:, ::decimation_factor]
    return aeeg_output, fs_new

def asym_filter_envelope2(numtaps, asym_filter_freq_hz, desired, fs, source_eeg):
    taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], desired, type='bandpass', fs=fs)
    y = lfilter(taps, 1, source_eeg)  # 对数据序列进行滤波，是标准差分方程直接形式的转换
    # Y = np.fft.fft(y, fs)

    # Setting standard filter requirements.
    lp_order = 1
    cutoff = asym_filter_freq_hz[3]
    rectify = np.abs(y)
    b, a = butter_lowpass(cutoff, fs, lp_order)
    # print(b)
    # print(a)
    enveloped_eeg = filtfilt(b, a, rectify)

    aeeg_output = 1.631 * enveloped_eeg + 1
    return aeeg_output


def segment_extra(aeeg_output, fs, window_length):
    sec_num = int(aeeg_output.shape[1] / fs)  # len(aeeg_output)
    utp, ltp = [], []  # UTP LTP
    for i in range(sec_num - window_length + 1):
        utp.append(np.percentile(aeeg_output[0, i * fs:(window_length + i) * fs], 70))  # 70 80
        ltp.append(np.percentile(aeeg_output[0, i * fs:(window_length + i) * fs], 50))  # 30 50
    return utp, ltp

def segment_extra1(aeeg_output, fs, window_length):  # 80/50, 70/50, 85/55, 90/55
    # logging.info(aeeg_output.shape)
    sec_num = int(aeeg_output.shape[1] / fs)

    k = 0
    utp, ltp = np.percentile(aeeg_output[:, k * fs:(int(window_length) + k) * fs], 80, axis=1), np.percentile(
        aeeg_output[:, k * fs:(int(window_length) + k) * fs], 50, axis=1)

    if sec_num > 1:
        for i in range(1, int(sec_num) - window_length + 1):
            utp_tmp = np.percentile(aeeg_output[:, i * fs:(int(window_length) + i) * fs], 80, axis=1)
            utp = np.vstack((utp, utp_tmp))
            ltp_tmp = np.percentile(aeeg_output[:, i * fs:(int(window_length) + i) * fs], 50, axis=1)
            ltp = np.vstack((ltp, ltp_tmp))

        # {通道：{时间位：max-min}}
        res = np.round([[[utp[j, i], ltp[j, i]] for j in range(utp.shape[0])] for i in range(utp.shape[1])], 2)
        # print(res.shape)
        return res.tolist()

    else:
        res = np.round([[[utp[j], ltp[j]]] for j in range(utp.shape[0])], 2)  #np.round([[[utp[j], ltp[j]] for j in range(utp.shape[0])]], 2)
        # print(res.shape)
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
    plt.savefig(os.path.dirname(tmp_data_path) + '/aEEG0915.png')
    # plt.savefig('/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/aEEG0912.png')
    print("aEEG波形保存完成")

def argv_parse():
    parser = argparse.ArgumentParser(description="Input Parameter")

    parser.add_argument("-read_type", "--read_type", default="read_type: 1-pyedflib 2-mne")
    parser.add_argument("-input_data_path", "--input_data_path", default="input_data_path")
    args = parser.parse_args()
    return args

def main_fun(read_type, input_data_path, fs, window_length):
    tmp_data_path = input_data_path  # args[2]
    if read_type == 2:  # read_type == 2  args[1] 1-pyedflib 2-mne 3-edflib
        # tmp_data_path = args[2] # r'/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/L20241223_JJY1.edf'
        raw = mne.io.read_raw_edf(tmp_data_path)
        source_eeg = raw[:][0][0] * 1e6

    elif read_type == 1:  #
        # tmp_data_path = '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/L20241223_JJY11_filtered.edf'
        f = pyedflib.EdfReader(tmp_data_path)
        sig_num = f.signals_in_file
        raw = np.zeros((sig_num, f.getNSamples()[0]))
        for i in range(sig_num):
            raw[i, :] = f.readSignal(i)
        source_eeg = raw[sig_num - 1, :]

    # elif read_type == 0:  # json
    #     source_eeg = np.array(input_data_path['values'][0])

    # fs = 500
    asym_filter_freq_hz = [0, 1.6, 2.0, 14.4, 15.36, fs / 2]
    # asym_filter_freq_norm = (np.asarray(asym_filter_freq_hz)*2)/fs
    # asym_filter_amp = [0, 0, 0.73, 2, 0, 0]

    source_eeg[np.isnan(source_eeg)] = 0
    numtaps = 301  # numtaps = filter orders + 1

    # 非对称滤波 整流 振幅放大
    aeeg_output = asym_filter_envelope(numtaps, asym_filter_freq_hz, fs, source_eeg)

    # 分段提取UTP LTP
    utp, ltp = segment_extra(aeeg_output, fs, window_length)

    # aEEG波形
    # plot_aEEG(utp, ltp, tmp_data_path)
    return utp, ltp

def aEEG_compute(sigbufs, channels, sample_frequency, band, window_length):

    channel_labels = channels
    # band = [2, 15]  # Desired pass band, Hz
    trans_width = 0.4    # Width of transition from pass to stop, Hz
    numtaps = 273  # numtaps = filter orders + 1 301
    if len(channel_labels) > 0:
        source_eeg = sigbufs
        fs = sample_frequency
        # asym_filter_freq_hz = [0, 1.6, 2.0, 14.4, 15.36, fs / 2]
        asym_filter_freq_hz = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]

        # asym_filter_freq_norm = (np.asarray(asym_filter_freq_hz)*2)/fs
        # asym_filter_amp = [0, 0, 0.73, 2, 0, 0]

        source_eeg[np.isnan(source_eeg)] = 0
        desired = [3, 6, 18]
        aeeg_output, fs_new = asym_filter_envelope1(numtaps, asym_filter_freq_hz, fs, source_eeg)
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, int(fs_new), window_length)

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        channel_values = []
        print("Channels Values is Null")

    return channel_labels, channel_values

def remove_spikes(signal, threshold=12):  # 13.7
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    mask = np.abs(signal - median) < threshold * mad
    signal_clean = np.copy(signal)
    signal_clean[~mask] = median  # 替换异常值
    return signal_clean

from joblib import Parallel, delayed
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
        utp, ltp = segment_extra(aeeg_output, fs_new, window_length)  # fs

        # aEEG波形
        plot_aEEG(utp, ltp, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY')

    else:
        channel_values = []
        print("Channels Values is Null")

    return channel_labels, channel_values

def butter_filter(fs, cutoff, btype, order=5):
    normal_f = 2 * np.array(cutoff) / fs
    b, a = butter(N=order, Wn=normal_f, btype=btype, analog=False)
    # 根据阶数n和归一化截止频率Wn计算ButterWorth滤波器分子分母系数（b为分子系数的矢量形式，a为分母系数的矢量形式)
    return b, a

def iir(request):

    if request.method == 'POST':
        if request.content_type == 'application/json':

            jsonStr = request.body.decode('utf-8')

            if(len(jsonStr) != 0):
                # 获取请求数据
                try:
                    input_data_dict = json.loads(jsonStr)
                except:
                    error_message["message"] = "input data not json"
                    return HttpResponse(json.dumps(error_message))

                    # 输入校验
            b, a = butter_filter(2000, 35, 'lowpass', 10)

            result = "{ " + f"\"a\": {a.tolist()}, \"b\": {b.tolist()}" + " }"

            return HttpResponse(result)
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))

def special_node(special_electrodes):
    node_dict = {}
    for spec_nodes in special_electrodes:
        node_dict[spec_nodes["name"]] = spec_nodes["electrodes"]
    return node_dict

# def read_bdf(f, signal_labels, sample_frequency, start_inx, stop_inx, channels, node_dict):
#     # 20250609 兼容pyedflib读取bdf文件
#     # 读取json数据并处理 Start
#     # json_path = '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/inputData2.txt'
#     # with open(json_path, 'r') as file:
#     #     content = file.read()
#     #
#     # json_data = json.loads(content)
#
#     # 读取bdf文件
#     # file_path = r'/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20241206184941/'
#     # f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
#     # n = f.signals_in_file
#     # signal_labels = f.getSignalLabels()
#     # sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致
#     # start_inx = int(sample_frequency * int(json_data['startSeconds']))
#     # stop_inx = int(sample_frequency * int(json_data['stopSeconds']))
#     col_length = stop_inx - start_inx
#     # channels = json_data['channels']
#     sigbufs = np.zeros((len(channels), col_length))
#     # node_dict = special_node(json_data['specialElectrodes'])
#
#     # 计算输入信号电位差
#     for inx in range(len(channels)):
#         chn_split = channels[inx].split('-')
#         chn_inx0 = signal_labels.index(chn_split[0])
#
#         if chn_split[1] in ('Ref', 'REF', 'ref'):
#             sigbuf = f.readSignal(chn_inx0, 0, stop_inx)
#             sigbufs[inx, 0:col_length] = sigbuf[start_inx:stop_inx]
#
#         elif chn_split[1] in signal_labels:
#             chn_inx1 = signal_labels.index(chn_split[1])
#             sigbuf0 = f.readSignal(chn_inx0, 0, stop_inx)
#             sigbuf1 = f.readSignal(chn_inx1, 0, stop_inx)
#             sigbufs[inx, 0:col_length] = sigbuf0[start_inx:stop_inx] - sigbuf1[start_inx:stop_inx]
#
#         # elif chn_split[1] in av_dict.keys():
#         #     sigbufs[inx, 0:col_length] = f.readSignal(chn_inx0, start_inx, stop_inx) - av_dict[chn_split[1]]
#         elif chn_split[1] in node_dict.keys():
#             av_value = np.zeros((1, col_length))
#             spec_nodes = node_dict[chn_split[1]]
#             for spec_node in spec_nodes:
#                 chn_inx = signal_labels.index(spec_node["electrode"])
#                 sigbuf = f.readSignal(chn_inx, 0, stop_inx)
#                 av_value = av_value + sigbuf[start_inx:stop_inx] * int(spec_node["weight"])
#             av_value = av_value/len(spec_nodes) / 100
#
#             sigbuf = f.readSignal(chn_inx0, 0, stop_inx)
#             sigbufs[inx, 0:col_length] = sigbuf[start_inx:stop_inx] - av_value
#
#     return sigbufs

# def aeeg(request):
#     if request.method == 'POST':
#         if request.content_type == 'application/json':
#             # logging.basicConfig(
#             #     level=logging.DEBUG,
#             #     format='%(asctime)s | %(levelname)s | %(message)s',
#             #     filename='/disk1/workspace/py39_tf270/vehicle_recommend/run_log/aeeg_logs.log',
#             #     filemode='a'
#             # )
#
#             jsonStr = request.body.decode('utf-8')
#             json_data = json.loads(jsonStr)
#
#             f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
#             # n = f.signals_in_file
#             signal_labels = f.getSignalLabels()
#             sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致
#             start_seconds = float(json_data['startSeconds'])
#             stop_seconds = float(json_data['stopSeconds'])
#
#             signal_seconds = round(f.getNSamples()[0] / f.getSampleFrequency(0), 2)  # 信号总时长
#             if start_seconds > signal_seconds:  # 不分析
#                 error_message["message"] = "The start senonds is over signal lengths! Read signal failure. "
#                 return HttpResponse(json.dumps(error_message))
#
#             elif start_seconds > stop_seconds:  #
#                 error_message["message"] = "The start senonds is over stop senonds! Read signal failure. "
#                 return HttpResponse(json.dumps(error_message))
#
#             elif stop_seconds > signal_seconds:
#                 stop_seconds = signal_seconds
#
#             start_inx = int(sample_frequency * start_seconds)
#             stop_inx = int(sample_frequency * stop_seconds)
#             channels = json_data['channels']
#             node_dict = special_node(json_data['specialElectrodes'])
#
#             sigbufs = read_bdf(f, signal_labels, sample_frequency, start_inx, stop_inx, channels, node_dict)
#             band = [2, 15]
#             window_length = 1
#
#             channel_labels, channel_values = aEEG_compute(sigbufs, channels, sample_frequency, band, window_length)  # utp, ltp
#             task_id = json_data["taskID"]
#             ana_type = "aEEG"
#
#             result = "{ " + f"\"task_id\": {task_id}, \"type\": {ana_type}, \"labels\": {channel_labels}, \"values\": {channel_values}" + " }"
#
#             return HttpResponse(result)
#
#             # return HttpResponse(json.dumps(success_message))
#         else:
#             error_message["message"] = "The content type is incorrect. Please input it application/json"
#             return HttpResponse(json.dumps(error_message))
#     else:
#         error_message["message"] = "The request method is incorrect"
#         return HttpResponse(json.dumps(error_message))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    nor_lowcut = lowcut / nyq
    nor_highcut = highcut / nyq
    b, a = butter(order, [nor_lowcut, nor_highcut], btype='band')
    return b, a

def aEEG_compute1(input_data_dict, band, window_length):
    channles = len([channel for channel in input_data_dict['labels']])  #  if 'EEG' in channel
    data_values = np.array(input_data_dict['values'][:channles])

    channel_labels = input_data_dict['labels'][:channles]
    fs_list = input_data_dict['samples']
    # band = [2, 15]  # Desired pass band, Hz
    trans_width = 0.4    # Width of transition from pass to stop, Hz
    numtaps = 273  # numtaps = filter orders + 1  301
    if channles > 0:
        source_eeg = data_values
        fs = int(fs_list[0])

        # asym_filter_freq_hz = [0, 1.6, 2.0, 14.4, 15.36, fs / 2]
        asym_filter_freq_hz = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]

        # asym_filter_freq_norm = (np.asarray(asym_filter_freq_hz)*2)/fs
        # asym_filter_amp = [0, 0, 0.73, 2, 0, 0]
        # asym_filter_freq_hz = [0, 1, band[0], band[1], 20, 0.5 * fs]

        source_eeg[np.isnan(source_eeg)] = 0
        desired = [3, 6, 18]  # [1, 3, 18]
        # 非对称滤波 整流 振幅放大
        aeeg_output, fs_new = asym_filter_envelope1(numtaps, asym_filter_freq_hz, fs, source_eeg)

        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, int(fs_new), window_length)

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        print("Channels Values is Null")
        channel_values = []

    return channel_labels, channel_values


def aEEG_compute_h1(input_data_dict, band, window_length):
    channles = len([channel for channel in input_data_dict['labels']])  #  if 'EEG' in channel
    data_values = np.array(input_data_dict['values'][:channles])

    channel_labels = input_data_dict['labels'][:channles]
    fs_list = input_data_dict['samples']
    # band = [2, 15]  # Desired pass band, Hz
    trans_width = 0.4    # Width of transition from pass to stop, Hz
    numtaps = 273  # numtaps = filter orders + 1  301
    if channles > 0:
        source_eeg = data_values
        fs = int(fs_list[0])

        # asym_filter_freq_hz = [0, 1.6, 2.0, 14.4, 15.36, fs / 2]
        asym_filter_freq_hz = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]

        # asym_filter_freq_norm = (np.asarray(asym_filter_freq_hz)*2)/fs
        # asym_filter_amp = [0, 0, 0.73, 2, 0, 0]
        # asym_filter_freq_hz = [0, 1, band[0], band[1], 20, 0.5 * fs]

        source_eeg[np.isnan(source_eeg)] = 0
        # 非对称滤波 整流 振幅放大
        # aeeg_output, fs_new = asym_filter_envelope1(numtaps, asym_filter_freq_hz, fs, source_eeg)

        bp_fir = firwin(numtaps, cutoff=[2, 15], fs=fs, pass_zero=False)
        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]

        # n_jobs = len(channel_labels) if len(channel_labels) < 8 else 8
        # eeg_filtered = filter_multichannel_eeg(source_eeg, fs, n_jobs)
        # taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], type='bandpass', fs=fs)
        # y = lfilter(taps, 1, np.apply_along_axis(remove_spikes, axis=1, arr=eeg_filtered))
        aeeg_output = 1.231 * np.abs(eeg_filtered) + 4

        fs_new = 250  # 新的采样率
        decimation_factor = int(fs / fs_new)
        # 降采样到250Hz
        aeeg_output = aeeg_output[:, ::decimation_factor]

        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, fs_new, window_length)

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        print("Channels Values is Null")
        channel_values = []

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
            # data_values = input_data_dict['values']
            # print(data_values[0][:5])
            # b, a = [1, 2, 1], [2, 4, 6]
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

    for channel in epoch_data:
        # Calculate power spectral density (PSD) using Welch method
        freqs, psd = welch(channel, fs=fs, nperseg=fs*4, noverlap=fs*2)
        freq_idx_total = np.where((freqs >= 0.5) & (freqs <= 30))[0]  # (freqs >= 1) & (freqs <= 45)
        total_psd = np.trapz(psd[freq_idx_total], axis=0)  # Sum across frequency bins

        rbp = []
        for band in frequency_bands:
            start_freq, end_freq = band
            freq_idx = np.where((freqs >= start_freq) & (freqs <= end_freq))[0]

            band_psd = np.trapz(psd[freq_idx], axis=0)  # Sum across selected frequency bins
            band_rbp = band_psd / total_psd
            rbp.append(band_rbp)

        rbp_features.append(rbp)

    rbp_features = np.array(rbp_features)
    rbp2d = rbp_features.reshape((rbp_features.shape[0], rbp_features.shape[1], 1))

    return rbp_features, rbp2d

def rbp_com(start_seconds, stop_seconds):
    train_data = []
    start_time = start_seconds
    epoch_length = 10
    step_size = 10
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
        # print(res.shape)
        channel_values = np.round(np.array(channel_values), 2)
        return channel_values.tolist()

    else:
        print("Channels Values is Null")
        return []

# def aeeg_history2(request):
#     if request.method == 'POST':
#         if request.content_type == 'application/json':
#             # logging.basicConfig(
#             #     level=logging.DEBUG,
#             #     format='%(asctime)s | %(levelname)s | %(message)s',
#             #     filename='/disk1/workspace/py39_tf270/vehicle_recommend/run_log/aeeg_logs.log',
#             #     filemode='a'
#             # )
#
#             jsonStr = request.body.decode('utf-8')
#             json_data = json.loads(jsonStr)
#
#             # f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
#             # # n = f.signals_in_file
#             # signal_labels = f.getSignalLabels()
#             # sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致
#
#             trend_channels = json_data['trendChannels']
#             aeeg_chns = []  # labels: aEEG
#             rbp_chns = []  # labels: RBP
#             for tchns in trend_channels:
#                 if tchns["type"] == "aEEG":
#                     aeeg_chns.append(tchns["label"])
#
#                 elif tchns["type"] == "RBP":
#                     rbp_chns.append(tchns["label"])
#
#             channels = json_data['channels']  # trendChannels
#             for chns in channels:
#                 chn_split = chns.split('-')
#                 aeeg_chns.append(chn_split[0])
#                 aeeg_chns.append(chn_split[1])
#             aeeg_chns = list(set(aeeg_chns))
#             aeeg_chns_sim = [chn for chn in aeeg_chns if chn not in ('AV', 'Ref', 'REF', 'ref')]
#
#             start_time = time.time()
#             # TODO 多个bdf文件读取
#             for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
#                 if '.bdf' not in fname:
#                     continue
#
#                 raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(aeeg_chns_sim))  # , preload=True
#                 # raw = raw.filter(l_freq=2, h_freq=70)
#                 picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
#                 sample_frequency = raw.info['sfreq']
#
#                 start_seconds = int(json_data['startSeconds'])
#                 stop_seconds = int(json_data['stopSeconds'])
#
#                 signal_seconds = round(raw.n_times / sample_frequency, 2)  # 信号总时长
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
#                 start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
#                 stop_inx = int(sample_frequency * stop_seconds)
#                 t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)
#
#                 sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
#                 sigbufs = sigbufs * 1e6
#                 band = [2, 15]
#                 window_length = 1
#
#                 node_dict = special_node(json_data['specialElectrodes'])
#                 signal_labels = raw.ch_names
#
#                 sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
#                 aeeg_labels, aeeg_values = aEEG_compute_h(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp
#
#                 task_id = json_data["taskID"]
#
#                 # ana_type = "aEEG"
#                 # {
#                 #     "label": "Fp1-REF",
#                 #     "type": "RBP"
#                 #     "values": [[82.46, 6.54, 3.44, 7.56], [86.33, 7.57, 3.45, 2.65], [77.9, 12.82, 3.23, 6.06],
#                 #                        [59.13, 17.86, 9.67, 13.34]]
#                 # }
#                 # tk = str(channel_labels)
#                 # tk = tk.replace("\'", "\"")
#
#                 # trendChannels{}
#                 trend_channels_res = []
#                 for aeeg_inx in range(len(aeeg_labels)):
#                     tchns_res = {}
#                     tchns_res['label'] = aeeg_labels[aeeg_inx]
#                     tchns_res['type'] = 'aEEG'
#                     tchns_res['values'] = aeeg_values[aeeg_inx]
#                 trend_channels_res.append(str(tchns_res).replace("\'", "\""))
#
#                 ######## 加入rbp
#                 rbp_data = []
#                 start_time = start_seconds
#                 epoch_length = 10
#                 step_size = 10
#                 while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
#                     # features = []
#                     start, stop = raw.time_as_index([start_time, start_time + epoch_length])
#                     temp = raw[:, start:stop][0]
#                     rbp_data.append(temp)
#                     start_time += step_size
#
#                 rbp_data = np.array(rbp_data)
#
#                 rbp_features_all_segments = []
#                 for segment in rbp_data:
#                     # Calculate RBP features for the current epoch
#                     rbp_features, rbp2d_features = calculate_rbp(segment, int(sample_frequency))
#                     rbp_features_all_segments.append(rbp_features)
#                     # rbp2d_features_all_segments.append(rbp2d_features)
#
#                 rbp_features_all_segments = np.array(rbp_features_all_segments)  # (265, 9, 5)
#
#                 shape_size = rbp_features_all_segments.shape
#                 # rbp_features = rbp_features_all_segments[:, 0, :].reshape((shape_size[0], shape_size[2]))  # 取出对应通道数据
#
#                 sec_num = shape_size[1]
#                 rbp_values = []
#                 if sec_num > 0:
#                     for i in range(sec_num):
#                         rbp_tmp = np.multiply(rbp_features_all_segments[:, i, :], 100).tolist(),
#                         rbp_values.extend(rbp_tmp)
#                     # print(res.shape)
#                     rbp_values = np.round(np.array(rbp_values), 2)
#                     # return channel_values.tolist()
#
#                 else:
#                     print("Channels Values is Null")
#
#                 for rbp_inx in range(len(rbp_chns)):
#                     tchns_res = {}
#                     tchns_res['label'] = rbp_chns[rbp_inx]
#                     tchns_res['type'] = 'RBP'
#                     tchns_res['values'] = rbp_values[rbp_inx]
#                 trend_channels_res.append(str(tchns_res).replace("\'", "\""))
#                 ######## 加入rbp End
#
#                 # result = "{ " + f"\"taskID\": \"{task_id}\", \"type\": \"{ana_type}\", \"labels\": {tk}, \"values\": {channel_values}" + " }"
#
#                 result = "{ " + f"\"taskID\": \"{task_id}\", \"trendChannels\": {trend_channels_res} " + " }"
#
#                 return HttpResponse(result)
#
#                 # return HttpResponse(json.dumps(success_message))
#         else:
#             error_message["message"] = "The content type is incorrect. Please input it application/json"
#             return HttpResponse(json.dumps(error_message))
#     else:
#         error_message["message"] = "The request method is incorrect"
#         return HttpResponse(json.dumps(error_message))


def rbp_history(request):
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

            # f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
            # # n = f.signals_in_file
            # signal_labels = f.getSignalLabels()
            # sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致
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

                # start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
                # stop_inx = int(sample_frequency * stop_seconds)
                # t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)
                #
                # sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
                # sigbufs = sigbufs * 1e6
                # band = [2, 15]
                # window_length = 1

                # rbp start
                train_data = []
                start_time = start_seconds
                epoch_length = 10
                step_size = 10
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
                    # print(res.shape)
                    # return rbp_fea.tolist()
                    channel_values = np.round(np.array(channel_values), 2)

                else:
                    print("Channels Values is Null")

                # rbp end

                # node_dict = special_node(json_data['specialElectrodes'])
                # signal_labels = raw.ch_names
                #
                # sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
                # channel_labels, channel_values = aEEG_compute_h(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp

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


def relative_band_energy(eeg_signal, fs, bands=None):
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
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30)
            # "gamma": (30, 45)
        }

    # 计算功率谱密度 (Welch 方法)
    freqs, psd = welch(eeg_signal, fs=fs, nperseg=fs*4)  # 2s 窗长
    # total_power = np.trapz(psd, freqs)  # 总功率

    # freq_idx_total = np.where((freqs >= 0.5) & (freqs <= 30))[0]
    idx = np.logical_and(freqs >= 0.5, freqs <= 30)
    total_power = np.trapz(psd[idx], freqs[idx])

    abs_power = []  # {}
    rel_power = []  # {}

    for band, (low, high) in bands.items():
        # 找到频段索引
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.trapz(psd[idx], freqs[idx])  # 频带能量
        # abs_power[band] = band_power
        # rel_power[band] = band_power / total_power if total_power > 0 else 0
        abs_power.append(band_power)
        rel_power.append(band_power / total_power if total_power > 0 else 0)

    return rel_power, abs_power

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

            start_time = time.time()
            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue

                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True
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

                # abp start
                train_data = []
                start_time = start_seconds
                epoch_length = 10
                step_size = 10
                while start_time <= stop_seconds + 0.01 - epoch_length:  # max(raw.times) = 3600
                    # features = []
                    start, stop = raw.time_as_index([start_time, start_time + epoch_length])
                    temp = raw[:, start:stop][0]
                    train_data.append(temp)
                    start_time += step_size

                train_data = np.array(train_data)


                rbp_features_all_segments = []
                for segment in range(len(train_data)):
                    rel_power, abs_power = relative_band_energy(train_data[segment, 0, :] * 1e6, int(sample_frequency))
                    # Calculate RBP features for the current epoch
                    # rbp_features, rbp2d_features = calculate_rbp(segment, int(sample_frequency))
                    rbp_features_all_segments.append(abs_power)
                    # rbp2d_features_all_segments.append(rbp2d_features)

                rbp_features_all_segments = np.array(rbp_features_all_segments)  # (265, 9, 5)

                shape_size = rbp_features_all_segments.shape
                # rbp_features = rbp_features_all_segments[:, 0, :].reshape((shape_size[0], shape_size[2]))  # 取出对应通道数据

                sec_num = shape_size[0]
                channel_values = []
                if sec_num > 0:
                    for i in range(sec_num):
                        rbp_tmp = rbp_features_all_segments[i, :].tolist(),
                        channel_values.extend(rbp_tmp)
                    # print(res.shape)
                    # return rbp_fea.tolist()
                    channel_values = np.round(np.array(channel_values), 2)

                else:
                    print("Channels Values is Null")

                # abp end
                task_id = json_data["taskID"]
                ana_type = "ABP"

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

            # f = pyedflib.EdfReader(os.path.join(json_data['fileDir'], '20241206184941.bdf'))
            # # n = f.signals_in_file
            # signal_labels = f.getSignalLabels()
            # sample_frequency = f.getSampleFrequency(0)  # !!! EEG 通道采样频率需一致
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
                stop_inx = int(sample_frequency * stop_seconds)  # 19000000
                t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

                sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
                sigbufs = sigbufs * 1e6
                band = [2, 15]
                window_length = 1

                # 输入信号滤波
                # b, a = butter_bandpass(band[0], band[1], sample_frequency, 2)
                # sigbufs = filtfilt(b, a, sigbufs)

                node_dict = special_node(json_data['specialElectrodes'])
                signal_labels = raw.ch_names

                sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV
                channel_labels, channel_values = aEEG_compute_h(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp

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


def aEEG_compute_h3(sigbufs, channels, sample_frequency, band, window_length):
    channel_labels = channels
    if len(channel_labels) > 0:
        source_eeg = sigbufs
        fs = int(sample_frequency)  # 4000
        trans_width = 0.4  # Width of transition from pass to stop, Hz
        numtaps = 1091  # 2181 273 # numtaps = filter orders + 1 301
        asym_filter_freq_hz = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5 * fs]

        source_eeg[np.isnan(source_eeg)] = 0
        n_jobs = len(channel_labels) if len(channel_labels) < 8 else 8
        # taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], type='bandpass', fs=fs)
        # y = lfilter(taps, 1, np.apply_along_axis(remove_spikes, axis=1, arr=eeg_filtered))
        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filter_multichannel_eeg(source_eeg, fs, numtaps, n_jobs)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filter_one_channel(source_eeg_tile[:, :(3 * numtaps + 1)], fs, numtaps)[:,
                           :source_eeg.shape[1]]

        aeeg_output = 1.631 * np.abs(eeg_filtered) + 4  # 1.231 * np.abs(eeg_filtered) + 4

        # fs_new = 250
        # decimation_factor = int(fs / fs_new)  # fs
        # # 降采样到250Hz
        # aeeg_output = aeeg_output[:, ::decimation_factor]
        # 分段提取UTP LTP
        channel_values = segment_extra1(aeeg_output, fs, window_length)  # fs_new

        # 分段提取UTP LTP
        utp, ltp = segment_extra(aeeg_output, fs, window_length)  # fs_new

        # aEEG波形
        plot_aEEG(utp, ltp, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/')

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
    trans_width = 0.4    # Width of transition from pass to stop, Hz
    numtaps = 273  # numtaps = filter orders + 1  301
    if channles > 0:
        source_eeg = data_values
        global time_cn
        time_cn = time_cn + 1
        if time_cn < 5:
            np.save('/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/test_data/source_eeg' + str(time_cn) + '.npy', source_eeg)
            print("fs: ", fs_list[0])
        fs = int(fs_list[0])

        # asym_filter_freq_hz = [0, 1.6, 2.0, 14.4, 15.36, fs / 2]
        asym_filter_freq_hz = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]

        # asym_filter_freq_norm = (np.asarray(asym_filter_freq_hz)*2)/fs
        # asym_filter_amp = [0, 0, 0.73, 2, 0, 0]
        # asym_filter_freq_hz = [0, 1, band[0], band[1], 20, 0.5 * fs]

        source_eeg[np.isnan(source_eeg)] = 0
        # 非对称滤波 整流 振幅放大
        # aeeg_output, fs_new = asym_filter_envelope1(numtaps, asym_filter_freq_hz, fs, source_eeg)

        bp_fir = firwin(numtaps, cutoff=[2, 15], fs=fs, pass_zero=False)
        if source_eeg.shape[1] > 3 * numtaps:
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg)
        else:
            source_eeg_tile = np.tile(source_eeg, int(np.ceil((3 * numtaps + 1) / source_eeg.shape[1])))
            eeg_filtered = filtfilt(bp_fir, [1.0], source_eeg_tile[:, :(3 * numtaps + 1)])[:, :source_eeg.shape[1]]

        # n_jobs = len(channel_labels) if len(channel_labels) < 8 else 8
        # eeg_filtered = filter_multichannel_eeg(source_eeg, fs, n_jobs)
        # taps = remez(numtaps, asym_filter_freq_hz, [0, 1, 0], type='bandpass', fs=fs)
        # y = lfilter(taps, 1, np.apply_along_axis(remove_spikes, axis=1, arr=eeg_filtered))
        aeeg_output = 1.231 * np.abs(eeg_filtered) + 4

        fs_new = 250  # 新的采样率
        decimation_factor = int(fs / fs_new)
        # 降采样到250Hz
        aeeg_output = aeeg_output[:, ::decimation_factor]

        # 分段提取UTP LTP
        # channel_values = segment_extra1(aeeg_output, fs_new, window_length)
        # 分段提取UTP LTP
        utp, ltp = segment_extra(aeeg_output, fs_new, window_length)  # fs

        # aEEG波形
        # plot_aEEG(utp, ltp, tmp_data_path)

    else:
        print("Channels Values is Null")
        channel_values = []
        utp = []
        ltp = []

    return channel_labels, utp, ltp # channel_values


if __name__ == '__main__':
    is_history = 1  # 1-history 0-monitor
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

        start_time = time.time()
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

            ana_type = "aEEG"

            result = {
                "type": ana_type,
                "labels": channel_labels,
                "values": len(channel_values)
            }
            print(result)

    else:
        # TODO 监测
        file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20250912110203"  # 20241206184941
        chn_list_sim = ["Fp1"]
        for fname in os.listdir(file_path):  # 寻找.rml文件
            if '.bdf' not in fname:
                continue

            input_data_dict = {}
            if signal_type == 1:
                raw = mne.io.read_raw_bdf(os.path.join(file_path, fname), include=tuple(chn_list_sim), preload=True)  # , preload=True ,
                raw = raw.filter(l_freq=1, h_freq=35)
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
            stop_seconds = start_seconds + 10
            while stop_seconds < signal_seconds:
                start_inx = int(sample_frequency * start_seconds)  # 可以为float类型
                stop_inx = int(sample_frequency * stop_seconds)  # 19000000
                if signal_type == 1:
                    t_idx = raw.time_as_index([start_inx, stop_inx], use_rounding=True)

                    sigbufs, times = raw[picks, int(t_idx[0] / sample_frequency):int(t_idx[1] / sample_frequency)]
                    sigbufs = sigbufs * 1e6
                    input_data_dict['values'] = np.array(sigbufs[:len(raw.ch_names), :])

                else:
                    input_data_dict['values'] = np.array(signal1[start_inx: stop_inx]).reshape((1, stop_inx - start_inx))

                band = [2, 15]  # [int(band_tmp[0]), int(band_tmp[1])]
                window_length = 1 # int(additional_input["windowLength"])
                channel_labels, utp, ltp = aEEG_compute_h2(input_data_dict, band, window_length)  # utp, ltp
                utp_t.extend(utp)
                ltp_t.extend(ltp)
                start_seconds = stop_seconds
                stop_seconds = stop_seconds + 10

                # aEEG波形
            plot_aEEG(utp_t, ltp_t, '/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/')





