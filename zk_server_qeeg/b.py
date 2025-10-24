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