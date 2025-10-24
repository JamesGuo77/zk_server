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

# from zk_server_qeeg.preprocess import  butter_lowpass, segment_extra, segment_extra1, plot_aEEG, filter_one_channel
# from zk_server_qeeg.preprocess import filter_one_channel1, filter_multichannel_eeg, aEEG_compute_h, special_node, butter_bandpass
# from zk_server_qeeg.preprocess import aEEG_compute_h1, aEEG_com1, read_bdf, calculate_rbp, multi_relative_band_energy, abp_rbp
# from zk_server_qeeg.preprocess import abp_rbp_com, abp_rbp_com1, sig_chns_split, abp_rbp_com_t, multi_chns_rav, multi_chns_csa
# from zk_server_qeeg.preprocess import sig_chn_out, multi_chns_out, multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef
# from zk_server_qeeg.preprocess import multi_chns_se, aEEG_com, sig_uv_read, aEEG_com_t, aEEG_compute_h3, aEEG_compute_h2, plot_rbp

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


def qteeg_history(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            jsonStr = request.body.decode('utf-8')
            json_data = json.loads(jsonStr)
            print("ZK")

            trend_channels = json_data['trendChannels']
            trend_chns_dict, trend_chns_sim_dict = {}, {}  # list(dict)

            from .apps import aEEG_com_t, abp_rbp_com_t, multi_chns_se, multi_chns_rav, multi_chns_csa
            from .apps import multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef

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

                results = []
                for ttype in trend_chns_dict.keys():  # "ABP", "RBP",
                    if ttype == trend_name[0]:  # ["aEEG", "ABP", "RBP", "RAV", "SE", "CSA", "Envelope", "TP", "ADR", "ABR"]
                        results.append(aEEG_com_t(
                            start_seconds, stop_seconds, json_data['specialElectrodes'], raw, ttype,
                            trend_chns_dict["aEEG"],
                            sample_frequency, 'history', json_data["taskID"]))

                    elif ttype == trend_name[1]:
                        # abp bp_type: 0-abp and rbp 1-abp 2-rbp
                        results.append(abp_rbp_com_t(
                        start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["ABP"], sample_frequency, 'abp',
                        json_data["taskID"]))

                    elif ttype == trend_name[2]:
                        results.append(abp_rbp_com_t(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["RBP"], sample_frequency, 'rbp',
                            json_data["taskID"]))

                    elif ttype == trend_name[3]:
                        results.append(multi_chns_rav(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["RAV"], sample_frequency,
                            json_data["taskID"]))

                    elif ttype == trend_name[4]:
                        results.append(multi_chns_se(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["SE"], sample_frequency,
                            json_data["taskID"]))

                    elif ttype == trend_name[5]:
                        results.append(multi_chns_csa(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["CSA"], sample_frequency,
                            json_data["taskID"]))

                    elif ttype == trend_name[6]:
                        results.append(multi_chns_env(
                            start_seconds, stop_seconds, json_data['specialElectrodes'], raw, ttype, trend_chns_dict["Envelope"], sample_frequency, 'history',
                            json_data["taskID"]))

                    elif ttype == trend_name[7]:
                        results.append(multi_chns_tp(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["TP"], sample_frequency, 'history',
                            json_data["taskID"]))

                    elif ttype == trend_name[8]:
                        results.append(multi_chns_adr_abr(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["ADR"], sample_frequency, 'adr',
                            json_data["taskID"]))

                    elif ttype == trend_name[9]:
                        results.append(multi_chns_adr_abr(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["ABR"], sample_frequency, 'abr',
                            json_data["taskID"]))

                    elif ttype == trend_name[10]:
                        results.append(multi_chns_sef(
                            start_seconds, stop_seconds, raw, ttype, trend_chns_sim_dict["SEF"], sample_frequency,
                            json_data["taskID"]))

                # 等待所有线程结束
                # for tt in threads:
                #     tt.join()
                #
                # # 合并结果
                # results = []
                # while not result_queue.empty():
                #     results.append(result_queue.get())
                #
                # task_id = json_data["taskID"]
                # type_values = {}
                # channel_values = []
                # for js in results:
                #     json_data = json.loads(js)
                #     if json_data["taskID"] == task_id:
                #         chn_l = list(json_data["labels"])
                #         chn_values = np.array(json_data["values"])
                #         for inx in range(len(chn_l)):
                #             type_values[json_data['type'] + chn_l[inx]] = chn_values[inx]
                #
                # for tchns in trend_channels:  # aEEG、RBP、ABP、RAV、SE（Spectral Edge）、CSA、Envelope、TP、ADR、ABR
                #     # aEEG and Envelope unit is uV, return type+label, others return type+signal
                #     if (tchns["type"] == trend_name[0]) or (tchns["type"] == trend_name[6]):
                #         type_chn = tchns["type"] + tchns["label"]
                #     else:
                #         type_chn = tchns["type"] + tchns["signal"]
                #
                #     channel_values.append(type_values[type_chn].tolist())
                #     # print(type_chn, type_values[type_chn].shape, len(type_values[type_chn].shape))
                #
                # tk = str(trend_channels)
                # tk = tk.replace("\'", "\"")
                #
                # result = "{ " + f"\"taskID\": \"{task_id}\", \"trendChannels\": {tk}, \"values\": {channel_values}" + " }"
                print("Hello, World!")
                results = "China"
                return HttpResponse("China")  # results
        else:
            error_message["message"] = "The content type is incorrect. Please input it application/json"
            return HttpResponse(json.dumps(error_message))
    else:
        error_message["message"] = "The request method is incorrect"
        return HttpResponse(json.dumps(error_message))
