import numpy as np
from django.http import HttpResponse
from scipy.signal import butter
import json
import mne
import pyedflib
# from edflib import edfreader
from scipy.signal import butter, remez, lfilter, filtfilt, kaiser_atten, kaiser_beta, firwin
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
# import multiprocessing

# 2025/6/6 V0.1 init pyedflib读取bdf 并完成aEEG
# 2025/6/8 V0.2 修复2400-3000s运行报错
# 2025/6/9 V0.3 采用mne读取bdf 并行计算效率提升不明显暂放弃
# 2025/6/11 V0.4 读取包含bdf文件的文件夹
# 2025/6/11 V0.5 新增监测端的接口
# 2025/6/21 V0.6 优化PM滤波
# 2025/7/15 V0.7 PM滤波前进行带通[2, 70]滤波,降采样至250Hz，包络参数优化
# 2025/7/16 V0.8 修复4个通道被降采样至1个通道


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
    # print(b)
    # print(a)
    enveloped_eeg = filtfilt(b, a, rectify)

    aeeg_output = 1.631 * enveloped_eeg + 4  # 2
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
    sec_num = int(len(aeeg_output) / fs)
    utp, ltp = [], []  # UTP LTP
    for i in range(sec_num - window_length + 1):
        utp.append(np.percentile(aeeg_output[i * fs:(window_length + i) * fs], 70))
        ltp.append(np.percentile(aeeg_output[i * fs:(window_length + i) * fs], 30))
    return utp, ltp

def segment_extra1(aeeg_output, fs, window_length):
    # logging.info(aeeg_output.shape)
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
    plt.savefig(os.path.dirname(tmp_data_path) + '/aEEG.png')
    # plt.savefig('/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/JJY/aEEG.png')
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
            channel_labels, channel_values = aEEG_compute1(input_data_dict, band, window_length)  # utp, ltp
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

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue

                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))
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

                # 输入信号滤波
                # b, a = butter_bandpass(band[0], band[1], sample_frequency, 2)
                # sigbufs = filtfilt(b, a, sigbufs)

                node_dict = special_node(json_data['specialElectrodes'])
                signal_labels = raw.ch_names

                sigbufs_res = read_bdf(sigbufs, signal_labels, channels, node_dict)  # mne读取信号单位为uV

                channel_labels, channel_values = aEEG_compute(sigbufs_res, channels, sample_frequency, band, window_length)  # utp, ltp
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