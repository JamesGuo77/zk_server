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

from zk_server_qeeg.preprocess import  butter_lowpass, segment_extra, segment_extra1, plot_aEEG, filter_one_channel
from zk_server_qeeg.preprocess import filter_one_channel1, filter_multichannel_eeg, aEEG_compute_h, special_node, butter_bandpass
from zk_server_qeeg.preprocess import aEEG_compute_h1, aEEG_com1, read_bdf, calculate_rbp, multi_relative_band_energy, abp_rbp
from zk_server_qeeg.preprocess import abp_rbp_com, abp_rbp_com1, sig_chns_split, abp_rbp_com_t, multi_chns_rav, multi_chns_csa
from zk_server_qeeg.preprocess import sig_chn_out, multi_chns_out, multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef
from zk_server_qeeg.preprocess import multi_chns_se, aEEG_com, sig_uv_read, aEEG_com_t, aEEG_compute_h3, aEEG_compute_h2, plot_rbp

epoch_length = 10  # s
trend_name = ["aEEG", "ABP", "RBP", "RAV", "SE", "CSA", "Envelope", "TP", "ADR", "ABR", "SEF"]

success_message = {
    "status": "200",
    "message": "ok"
}

error_message = {
    "status": "500",
    "message": "ok"
}


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
            abp_labels, abp_values, rbp_values = abp_rbp_com1(input_data_dict, 'abp')

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
            start_seconds, stop_seconds, raw, trend_name[1], channels, sample_frequency, 'abp', input_data_dict["taskID"]))

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
                channel_labels, channel_values, channel_values1 = abp_rbp_com(start_seconds, stop_seconds, raw, chn_list_sim, sample_frequency, 'abp')

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

            trend_channels = json_data['trendChannels']
            trend_chns_dict, trend_chns_sim_dict = {}, {}  # list(dict)

            for tchns in trend_channels:  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR、SEF
                if tchns["type"] in trend_name:
                    if tchns["type"] in trend_chns_dict.keys():
                        trend_chns_dict[tchns["type"]].append(tchns["label"])
                    else:
                        trend_chns_dict[tchns["type"]] = [tchns["label"]]

            # channels = json_data['channels']  # trendChannels
            for ttype in trend_chns_dict.keys():
                chns_split = []
                for chns in trend_chns_dict[ttype]:
                    chn_split = chns.split('-')
                    chns_split.append(chn_split[0])
                    chns_split.append(chn_split[1])
                chns_split = list(set(chns_split))
                trend_chns_sim_dict[ttype] = [chn for chn in chns_split if chn not in ('AV', 'Ref', 'REF', 'ref')]

            chn_list_sim = list({item for sublist in trend_chns_sim_dict.values() for item in sublist})  # list(set(aeeg_chns_sim + rbp_chns_sim))

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.bdf文件
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
                aEEG_labels, aEEG_values = aEEG_com(start_seconds, stop_seconds, json_data['specialElectrodes'], raw, trend_chns_dict["aEEG"], sample_frequency)
                # abp bp_type: 0-abp and rbp 1-abp 2-rbp
                bp_labels, _, rbp_values = abp_rbp_com(start_seconds, stop_seconds, raw, trend_chns_dict["RBP"], sample_frequency, 'rbp')

                task_id = json_data["taskID"]
                ana_type = "aEEG_rbp"
                qeeg_type = "aEEG"
                qeeg_type1 = "rbp"

                tk = str(aEEG_labels)
                tk = tk.replace("\'", "\"")
                tk1 = str(bp_labels)
                tk1 = tk1.replace("\'", "\"")
                qeeg_data = "{ " + f" \"type\": \"{qeeg_type}\", \"labels\": {tk}, \"values\": {aEEG_values}" + " }"
                qeeg_data1 = "{ " + f" \"type\": \"{qeeg_type1}\", \"labels\": {tk1}, \"values\": {rbp_values}" + " }"
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

def thread_workers(fn, args, q):
    result = fn(*args)
    q.put(result)

# 创建线程安全队列保存结果
result_queue = queue.Queue()


# def qteeg_history(request):
#     if request.method == 'POST':
#         if request.content_type == 'application/json':
#             jsonStr = request.body.decode('utf-8')
#             json_data = json.loads(jsonStr)
#
#             trend_channels = json_data['trendChannels']
#             trend_chns_dict, trend_chns_sim_dict = {}, {}  # list(dict)
#
#             for tchns in trend_channels:  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR
#                 if tchns["type"] in trend_name:
#                     if tchns["type"] in trend_chns_dict.keys():
#                         trend_chns_dict[tchns["type"]].append(tchns["label"])
#                     else:
#                         trend_chns_dict[tchns["type"]] = [tchns["label"]]
#
#             # channels = json_data['channels']  # trendChannels
#             for ttype in trend_chns_dict.keys():
#                 chns_split = []
#                 for chns in trend_chns_dict[ttype]:
#                     chn_split = chns.split('-')
#                     chns_split.append(chn_split[0])
#                     chns_split.append(chn_split[1])
#                 chns_split = list(set(chns_split))
#                 trend_chns_sim_dict[ttype] = [chn for chn in chns_split if chn not in ('AV', 'Ref', 'REF', 'ref')]
#
#             chn_list_sim = list({item for sublist in trend_chns_sim_dict.values() for item in
#                                  sublist})  # list(set(aeeg_chns_sim + rbp_chns_sim))
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
#                 threads = []
#                 for ttype in trend_chns_dict.keys():  # "ABP", "RBP",
#                     if ttype == trend_name[0]:  # ["aEEG", "ABP", "RBP", "RAV", "SE", "CSA", "Envelope", "TP", "ADR", "ABR"]
#                         aEEG_t = threading.Thread(target=thread_workers, args=(aEEG_com_t, (
#                         start_seconds, stop_seconds, json_data['specialElectrodes'], raw, trend_chns_dict["aEEG"],
#                         sample_frequency, json_data["taskID"]), result_queue))
#                         # 启动线程
#                         threads.append(aEEG_t)
#                         aEEG_t.start()
#
#                     elif ttype == trend_name[1]:
#                         # abp bp_type: 0-abp and rbp 1-abp 2-rbp
#                         abp_t = threading.Thread(target=thread_workers, args=(abp_rbp_com_t, (
#                         start_seconds, stop_seconds, raw, trend_chns_sim_dict["ABP"], sample_frequency, 'abp',
#                         json_data["taskID"]), result_queue))
#                         threads.append(abp_t)
#                         abp_t.start()
#
#                     elif ttype == trend_name[2]:
#                         rbp_t = threading.Thread(target=thread_workers, args=(abp_rbp_com_t, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["RBP"], sample_frequency, 'rbp',
#                             json_data["taskID"]), result_queue))
#                         threads.append(rbp_t)
#                         rbp_t.start()
#
#                     elif ttype == trend_name[3]:
#                         rav_t = threading.Thread(target=thread_workers, args=(multi_chns_rav, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["RAV"], sample_frequency,
#                             json_data["taskID"]), result_queue))
#                         threads.append(rav_t)
#                         rav_t.start()
#
#                     elif ttype == trend_name[6]:
#                         env_t = threading.Thread(target=thread_workers, args=(multi_chns_env, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["Envelope"], sample_frequency, 'history',
#                             json_data["taskID"]), result_queue))
#                         threads.append(env_t)
#                         env_t.start()
#
#                     elif ttype == trend_name[7]:
#                         tp_t = threading.Thread(target=thread_workers, args=(multi_chns_tp, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["TP"], sample_frequency, 'history',
#                             json_data["taskID"]), result_queue))
#                         threads.append(tp_t)
#                         tp_t.start()
#
#                     elif ttype == trend_name[8]:
#                         adr_t = threading.Thread(target=thread_workers, args=(multi_chns_adr_abr, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["ADR"], sample_frequency, 'adr',
#                             json_data["taskID"]), result_queue))
#                         threads.append(adr_t)
#                         adr_t.start()
#
#                     elif ttype == trend_name[9]:
#                         abr_t = threading.Thread(target=thread_workers, args=(multi_chns_adr_abr, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["ABR"], sample_frequency, 'abr',
#                             json_data["taskID"]), result_queue))
#                         threads.append(abr_t)
#                         abr_t.start()
#
#                     elif ttype == trend_name[10]:
#                         sef_t = threading.Thread(target=thread_workers, args=(multi_chns_sef, (
#                             start_seconds, stop_seconds, raw, trend_chns_sim_dict["SEF"], sample_frequency,
#                             json_data["taskID"]), result_queue))
#                         threads.append(sef_t)
#                         sef_t.start()
#
#                 # 等待所有线程结束
#                 for tt in threads:
#                     tt.join()
#
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

def qteeg_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)

            trend_channels = json_data['trendChannels']
            trend_chns_dict, trend_chns_sim_dict = {}, {}  # list(dict)

            for tchns in trend_channels:  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR
                if tchns["type"] in trend_name:
                    if tchns["type"] in trend_chns_dict.keys():
                        trend_chns_dict[tchns["type"]].append(tchns["label"])
                    else:
                        trend_chns_dict[tchns["type"]] = [tchns["label"]]

            # channels = json_data['channels']  # trendChannels
            for ttype in trend_chns_dict.keys():
                chns_split = []
                for chns in trend_chns_dict[ttype]:
                    chn_split = chns.split('-')
                    chns_split.append(chn_split[0])
                    chns_split.append(chn_split[1])
                chns_split = list(set(chns_split))
                trend_chns_sim_dict[ttype] = [chn for chn in chns_split if chn not in ('AV', 'Ref', 'REF', 'ref')]

            chn_list_sim = list({item for sublist in trend_chns_sim_dict.values() for item in
                                 sublist})  # list(set(aeeg_chns_sim + rbp_chns_sim))

            # TODO 多个bdf文件读取
            for fname in os.listdir(json_data['fileDir']):  # 寻找.rml文件
                if '.bdf' not in fname:
                    continue
                raw = mne.io.read_raw_bdf(os.path.join(json_data['fileDir'], fname), include=tuple(chn_list_sim))  # , preload=True tuple(chn_list_sim)
                # raw = raw.filter(l_freq=2, h_freq=70)
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

                threads = []
                for ttype in trend_chns_dict.keys():  # "ABP", "RBP",
                    if ttype == trend_name[0]:  # ["aEEG", "ABP", "RBP", "RAV", "SE", "CSA", "Envelope", "TP", "ADR", "ABR"]
                        aEEG_t = threading.Thread(target=thread_workers, args=(aEEG_com_t, (
                        start_seconds, stop_seconds, json_data['specialElectrodes'], raw, ttype, trend_chns_dict["aEEG"],
                        sample_frequency, 'history', json_data["taskID"]), result_queue))
                        # 启动线程
                        threads.append(aEEG_t)
                        aEEG_t.start()

                    elif ttype == trend_name[1]:
                        # abp bp_type: 0-abp and rbp 1-abp 2-rbp
                        abp_t = threading.Thread(target=thread_workers, args=(abp_rbp_com_t, (
                        start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["ABP"], sample_frequency, 'abp',
                        json_data["taskID"]), result_queue))
                        threads.append(abp_t)
                        abp_t.start()

                    elif ttype == trend_name[2]:
                        rbp_t = threading.Thread(target=thread_workers, args=(abp_rbp_com_t, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["RBP"], sample_frequency, 'rbp',
                            json_data["taskID"]), result_queue))
                        threads.append(rbp_t)
                        rbp_t.start()

                    elif ttype == trend_name[3]:
                        rav_t = threading.Thread(target=thread_workers, args=(multi_chns_rav, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["RAV"], sample_frequency,
                            json_data["taskID"]), result_queue))
                        threads.append(rav_t)
                        rav_t.start()

                    elif ttype == trend_name[4]:
                        se_t = threading.Thread(target=thread_workers, args=(multi_chns_se, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["SE"], sample_frequency,
                            json_data["taskID"]), result_queue))
                        threads.append(se_t)
                        se_t.start()

                    elif ttype == trend_name[5]:
                        csa_t = threading.Thread(target=thread_workers, args=(multi_chns_csa, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["CSA"], sample_frequency,
                            json_data["taskID"]), result_queue))
                        threads.append(csa_t)
                        csa_t.start()

                    elif ttype == trend_name[6]:
                        env_t = threading.Thread(target=thread_workers, args=(multi_chns_env, (
                            start_seconds, stop_seconds, json_data['specialElectrodes'], raw, ttype, trend_chns_dict["Envelope"], sample_frequency, 'history',
                            json_data["taskID"]), result_queue))
                        threads.append(env_t)
                        env_t.start()

                    elif ttype == trend_name[7]:
                        tp_t = threading.Thread(target=thread_workers, args=(multi_chns_tp, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["TP"], sample_frequency, 'history',
                            json_data["taskID"]), result_queue))
                        threads.append(tp_t)
                        tp_t.start()

                    elif ttype == trend_name[8]:
                        adr_t = threading.Thread(target=thread_workers, args=(multi_chns_adr_abr, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["ADR"], sample_frequency, 'adr',
                            json_data["taskID"]), result_queue))
                        threads.append(adr_t)
                        adr_t.start()

                    elif ttype == trend_name[9]:
                        abr_t = threading.Thread(target=thread_workers, args=(multi_chns_adr_abr, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["ABR"], sample_frequency, 'abr',
                            json_data["taskID"]), result_queue))
                        threads.append(abr_t)
                        abr_t.start()

                    elif ttype == trend_name[10]:
                        sef_t = threading.Thread(target=thread_workers, args=(multi_chns_sef, (
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["SEF"], sample_frequency,
                            json_data["taskID"]), result_queue))
                        threads.append(sef_t)
                        sef_t.start()

                # 等待所有线程结束
                for tt in threads:
                    tt.join()

                # 合并结果
                results = []
                while not result_queue.empty():
                    results.append(result_queue.get())

                task_id = json_data["taskID"]
                type_values = {}
                channel_values = []
                for js in results:
                    json_data = json.loads(js)
                    if json_data["taskID"] == task_id:
                        chn_l = list(json_data["labels"])
                        chn_values = np.array(json_data["values"])
                        for inx in range(len(chn_l)):
                            type_values[json_data['type'] + chn_l[inx]] = chn_values[inx]

                for tchns in trend_channels:  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR
                    # aEEG and Envelope unit is uV, return type+label, others return type+signal
                    if (tchns["type"] == trend_name[0]) or (tchns["type"] == trend_name[6]):
                        type_chn = tchns["type"] + tchns["label"]
                    else:
                        type_chn = tchns["type"] + tchns["signal"]

                    channel_values.append(type_values[type_chn].tolist())
                    # print(type_chn, type_values[type_chn].shape, len(type_values[type_chn].shape))

                tk = str(trend_channels)
                tk = tk.replace("\'", "\"")
                print("Hello, World!")

                result = "{ " + f"\"taskID\": \"{task_id}\", \"trendChannels\": {tk}, \"values\": {channel_values}" + " }"

                return HttpResponse(result)  # results
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))
