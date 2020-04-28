
import mne
import numpy as np
from mne.time_frequency import psd_welch
from numpy.lib import stride_tricks
from scipy.signal import filtfilt, lfilter
from timeflux_dsp.utils.filters import construct_iir_filter
from moabb.datasets import SSVEPExo

dataset = SSVEPExo()
event_id = dataset.event_id

def compute_mega_raw(raw, filt_method, frequencies,  order):
    if filt_method == 'lfilter':
        filt_method = lfilter
    elif filt_method == 'filtfilt':
        filt_method = filtfilt
    else:
        raise ValueError(f'Unknown filt_method {filt_method}')
    # read from info
    ch_names = np.array(raw.info['ch_names'])
    info_kind_dict = {2 : 'eeg', 3: 'stim'}
    ch_types = np.array([info_kind_dict.get(info['kind']) for info in raw.info['chs']])
    sfreq = raw.info['sfreq']

    _stim_data = raw._data[ch_types=='stim', :]
    _eeg_data = raw._data[ch_types=='eeg', :]
    _eeg_ch_names = ch_names[ch_types=='eeg']
    n_ch = len(_eeg_ch_names)

    # loop over frrquencies, filter and concatenate channels
    _mega_ch_types = ['stim']
    _mega_ch_names = ['STI']
    _mega_eeg_data = [_stim_data]
    for frequency in frequencies:
        ba, _ = construct_iir_filter(sfreq, [frequency[0], frequency[1]], 'bandpass', order=order, output='ba')
        _eeg_data_bandpassed = filt_method(*ba, _eeg_data)
        _mega_eeg_data.append(_eeg_data_bandpassed)
        _mega_ch_names += [ch + f'_{frequency}Hz' for ch in _eeg_ch_names]
        _mega_ch_types += ['eeg'] * n_ch

    mega_eeg_data = np.vstack(_mega_eeg_data)
    mega_info = mne.create_info(ch_names=_mega_ch_names, sfreq=sfreq, ch_types=_mega_ch_types, verbose=False)
    mega_raw = mne.io.RawArray(mega_eeg_data, mega_info, verbose=False)
    return mega_raw

def compute_mega_epochs(raw, tmin=1, tmax=5, **kwargs):
    mega_raw = compute_mega_raw(raw, **kwargs)
    X, y = extract_epochs(mega_raw, tmin=tmin, tmax=tmax)
    return mega_raw, X, y

def extract_epochs(raw, tmin=1, tmax=5):
    # Extract epochs
    events = mne.find_events(raw, verbose=False)

    picks = mne.pick_types(raw.info, eeg=True)
    epochs_all = mne.epochs.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                         tmax=tmax, baseline=None, verbose=False, picks=picks)
    X = epochs_all.get_data() * 1e6
    y = epochs_all.events[:, -1]
    return X, y

def get_avg_psd(raw):
    events = mne.find_events(raw, verbose=False)

    epochs = mne.epochs.Epochs(raw, events=events, event_id=event_id,
                    tmin=1, tmax=5, baseline=None, preload=True,
                    verbose=False, picks=[0, 1, 2, 3, 4, 5, 6, 7])
    psd1, freq = psd_welch(epochs['13'], n_fft=1028, n_per_seg=256 * 3, verbose=0)
    psd2, _ = psd_welch(epochs['17'], n_fft=1028, n_per_seg=256 * 3, verbose=0)
    psd3, _ = psd_welch(epochs['21'], n_fft=1028, n_per_seg=256 * 3, verbose=0)
    psd4, _ = psd_welch(epochs['rest'], n_fft=1028, n_per_seg=256 * 3, verbose=0)


    psd1_mean = (10 * np.log10(psd1)).mean(0)
    psd2_mean = (10 * np.log10(psd2)).mean(0)
    psd3_mean = (10 * np.log10(psd3)).mean(0)
    psd4_mean = (10 * np.log10(psd4)).mean(0)

    return freq, psd1_mean, psd2_mean, psd3_mean, psd4_mean


def epoch(a, size, interval, axis=-1):
    """ Small proof of concept of an epoching function using NumPy strides
    License: BSD-3-Clause
    Copyright: David Ojeda <david.ojeda@gmail.com>, 2018
    Create a view of `a` as (possibly overlapping) epochs.
    The intended use-case for this function is to epoch an array representing
    a multi-channels signal with shape `(n_samples, n_channels)` in order
    to create several smaller views as arrays of size `(size, n_channels)`,
    without copying the input array.
    This function uses a new stride definition in order to produce a view of
    `a` that has shape `(num_epochs, ..., size, ...)`. Dimensions other than
    the one represented by `axis` do not change.
    Parameters
    ----------
    a: array_like
        Input array
    size: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    axis: int
        Axis of the samples on `a`. For example, if `a` has a shape of
        `(num_observation, num_samples, num_channels)`, then use `axis=1`.
    Returns
    -------
    ndarray
        Epoched view of `a`. Epochs are in the first dimension.
    """
    a = np.asarray(a)
    n_samples = a.shape[axis]
    n_epochs = (n_samples - size) // interval + 1

    new_shape = list(a.shape)
    new_shape[axis] = size
    new_shape = (n_epochs,) + tuple(new_shape)

    new_strides = (a.strides[axis] * interval,) + a.strides

    return stride_tricks.as_strided(a, new_shape, new_strides)