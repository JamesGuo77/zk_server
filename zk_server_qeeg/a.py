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