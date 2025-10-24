from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules
from django.core.signals import request_finished, request_started
from django.db.models.signals import post_migrate
from django.dispatch import receiver
import threading
import os, mne

import zk_server_qeeg.preprocess as zkp
# from zk_server_qeeg.preprocess import butter_lowpass, segment_extra, segment_extra1, plot_aEEG, filter_one_channel
# from zk_server_qeeg.preprocess import filter_one_channel1, filter_multichannel_eeg, aEEG_compute_h, special_node, butter_bandpass
# from zk_server_qeeg.preprocess import aEEG_compute_h1, aEEG_com1, read_bdf, calculate_rbp, multi_relative_band_energy, abp_rbp
# from zk_server_qeeg.preprocess import abp_rbp_com, abp_rbp_com1, sig_chns_split, abp_rbp_com_t, multi_chns_rav, multi_chns_csa
# from zk_server_qeeg.preprocess import sig_chn_out, multi_chns_out, multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef
# from zk_server_qeeg.preprocess import multi_chns_se, aEEG_com, sig_uv_read, aEEG_com_t, aEEG_compute_h3, aEEG_compute_h2, plot_rbp

epoch_length = 10  # s
trend_name = ["aEEG", "ABP", "RBP", "RAV", "SE", "CSA", "Envelope", "TP", "ADR", "ABR", "SEF"]


# 这是一个模型加载的函数，它将在应用启动时执行
def load_model_on_startup():
    global butter_lowpass, segment_extra, segment_extra1, plot_aEEG, filter_one_channel
    global filter_one_channel1, filter_multichannel_eeg, aEEG_compute_h, special_node, butter_bandpass
    global aEEG_compute_h1, aEEG_com1, read_bdf, calculate_rbp, multi_relative_band_energy, abp_rbp
    global abp_rbp_com, abp_rbp_com1, sig_chns_split, abp_rbp_com_t, multi_chns_rav, multi_chns_csa
    global sig_chn_out, multi_chns_out, multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef
    global multi_chns_se, aEEG_com, sig_uv_read, aEEG_com_t, aEEG_compute_h3, aEEG_compute_h2, plot_rbp

    # butter_lowpass = zkp.butter_lowpass
    # segment_extra = zkp.segment_extra
    # segment_extra1 = zkp.segment_extra1
    # plot_aEEG = zkp.plot_aEEG
    # filter_one_channel = zkp.filter_one_channel

    aEEG_com_t = zkp.aEEG_com_t
    abp_rbp_com_t = zkp.abp_rbp_com_t
    multi_chns_se = zkp.multi_chns_se
    multi_chns_rav = zkp.multi_chns_rav
    multi_chns_csa = zkp.multi_chns_csa
    multi_chns_env = zkp.multi_chns_env
    multi_chns_tp = zkp.multi_chns_tp
    multi_chns_adr_abr = zkp.multi_chns_adr_abr
    multi_chns_sef = zkp.multi_chns_sef
    multi_chns_se = zkp.multi_chns_se

    # TODO 多个bdf文件读取
    file_path = r"/disk1/workspace/py39_tf270/SleepEpilepsy1/resource/test/sleepstage/20241206184941"  # 20250912110203 20241206184941

    raw = mne.io.read_raw_bdf(os.path.join(file_path, '20241206184941.bdf'), include=['Fp1', 'T4'])  # , preload=True
    # raw = raw.filter(l_freq=2, h_freq=70)
    sample_frequency = raw.info['sfreq']
    # print("aEEG")

    start_seconds = 0
    stop_seconds = 600  # round(raw.n_times / sample_frequency, 2)

    # train_data = sig_chns_split(start_seconds, stop_seconds, epoch_length, raw, raw.ch_names)
    #
    # channel_values = abp_rbp(train_data, sample_frequency, 'rbp', 'history')
    channel_values = aEEG_com_t(start_seconds, stop_seconds, [], raw, 'aEEG', ["Fp1-REF"],
        sample_frequency, 'history', '123456')

    channel_values = multi_chns_adr_abr(start_seconds, stop_seconds, raw, 'ABR', ['Fp1'], sample_frequency, 'abr',
        '123456')
    print("ZK")
    # channel_values = multi_chns_sef(start_seconds, stop_seconds, raw, 'SEF', ['Fp1'], sample_frequency, '123456')


# 这是一个信号接收器，在Django请求开始时执行模型加载
# @receiver(request_started)
# def load_model_on_request_start(sender, **kwargs):
#     global aEEG_com_t, abp_rbp_com_t, multi_chns_se, multi_chns_rav, multi_chns_csa
#     global multi_chns_env, multi_chns_tp, multi_chns_adr_abr, multi_chns_sef
#
#     if (aEEG_com_t is None) | (abp_rbp_com_t is None) | (multi_chns_se is None) | (multi_chns_rav is None) | (multi_chns_csa is None) | \
#         (multi_chns_env is None) | (multi_chns_tp is None) | (multi_chns_adr_abr is None) | (multi_chns_sef is None) :  # | (deep_model_instance is None)
#         thread = threading.Thread(target=load_model_on_startup)
#         thread.start()

class ZkServerQeegConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'zk_server_qeeg'

    # def ready(self):
    #     # 当应用启动时，开始预加载模型
    #
    #     load_model_on_startup()

