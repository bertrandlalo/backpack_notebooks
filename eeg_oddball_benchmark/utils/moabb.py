import glob
import json
import os
from collections import defaultdict
from glob import glob

import mne
import numpy as np
import pandas as pd

from moabb.datasets.base import BaseDataset

TXODDBALL_URL = 'timeflux/something/'  # ''
from scipy.interpolate import interp1d

import logging

logger = logging.getLogger()


def uniform_sampling(data, sampling_rate, interpolation_kind='zero'):
    """ Resample a signal to a uniform sampling rate
    Uses a zero order interpolator (aka sample-and-hold) to estimate the value
    of the signal on the exact timestamps that correspond to a uniform sampling
    rate.
    Sampling frequencies whose period cannot be represented as a integer
    nanosecond  are not supported. For example, 3Hz is not supported because its
    period is 333333333.333 nanoseconds. Another example: 4096Hz is not
    supported because its period is 244140.625 nanoseconds.
    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe. All columns must be numeric. It must have a
        datetime-like index.
    sampling_rate: float
        Target sampling rate in Hertz.
    interpolation_kind: str | int | None
        If None, no interpolation is performed, ie. timestamps of the input data
        are ignored.
        Else, it specifies the kind of interpolation as a string or an int
        (see documentation of scipy.interpolate.interp1d)
    Returns
    -------
    pd.DataFrame
        A new dataframe with the same columns as the input but the index will
        change to accomodate a uniform sampling.
    Examples
    --------
    >>> import pandas.util.testing as tm
    >>> data =  tm.makeTimeDataFrame(freq='S').head()  # Create a 1Hz dataframe
    >>> data
                         A         B         C         D
    2000-01-01 00:00:00 -0.739572 -0.191162  1.023474  1.663371
    2000-01-01 00:00:01  1.183841 -0.631689  0.412752  1.488323
    2000-01-01 00:00:02  1.683318  2.237185  0.726931 -0.914066
    2000-01-01 00:00:03  0.948706 -1.087019 -0.685658  0.710647
    2000-01-01 00:00:04  1.177724  0.510797 -0.707243 -0.790019
    >>> uniform_sampling(data, 4)  # Resample to 4Hz
                                    A         B         C         D
    2000-01-01 00:00:00.000 -0.739572 -0.191162  1.023474  1.663371
    2000-01-01 00:00:00.250 -0.739572 -0.191162  1.023474  1.663371
    2000-01-01 00:00:00.500 -0.739572 -0.191162  1.023474  1.663371
    2000-01-01 00:00:00.750 -0.739572 -0.191162  1.023474  1.663371
    2000-01-01 00:00:01.000  1.183841 -0.631689  0.412752  1.488323
    2000-01-01 00:00:01.250  1.183841 -0.631689  0.412752  1.488323
    2000-01-01 00:00:01.500  1.183841 -0.631689  0.412752  1.488323
    2000-01-01 00:00:01.750  1.183841 -0.631689  0.412752  1.488323
    2000-01-01 00:00:02.000  1.683318  2.237185  0.726931 -0.914066
    2000-01-01 00:00:02.250  1.683318  2.237185  0.726931 -0.914066
    2000-01-01 00:00:02.500  1.683318  2.237185  0.726931 -0.914066
    2000-01-01 00:00:02.750  1.683318  2.237185  0.726931 -0.914066
    2000-01-01 00:00:03.000  0.948706 -1.087019 -0.685658  0.710647
    2000-01-01 00:00:03.250  0.948706 -1.087019 -0.685658  0.710647
    2000-01-01 00:00:03.500  0.948706 -1.087019 -0.685658  0.710647
    2000-01-01 00:00:03.750  0.948706 -1.087019 -0.685658  0.710647
    2000-01-01 00:00:04.000  1.177724  0.510797 -0.707243 -0.790019
    """

    if data.empty:
        raise ValueError('Cannot resample an empty dataframe')

    period_ns, fract = divmod(1e9, sampling_rate)
    if fract != 0:
        raise ValueError('Refusing to interpolate under nanosecond scale')

    # the new, uniformly sampled index
    index_new = pd.date_range(data.index[0], data.index[-1], freq=f'{period_ns}N')
    data_new = pd.DataFrame(columns=data.columns, index=index_new)

    t_old = (data.index - data.index[0]).total_seconds()
    t_new = (data_new.index - data_new.index[0]).total_seconds()

    values = data.values
    if interpolation_kind is not None:
        f_interp = interp1d(t_old, values.T, kind=interpolation_kind)
        values = f_interp(t_new).T
    else:
        min_length = min(len(index_new), len(values))
        index_new = index_new[:min_length]
        values = values[:min_length, :]
    output_data = pd.DataFrame(values, columns=data.columns, index=index_new)
    return output_data


def estimate_rate(data, ignore_index=True):
    """ Estimate nominal sampling rate of a DataFrame.
    This function checks if the index are correct, that is monotonic and regular
    (the jitter should not exceed twice the nominal timespan)
    Notes
    -----
    This function does not take care of jitters in the Index and consider that the rate as the 1/Ts
    where Ts is the average timespan between samples.
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with index corresponding to timestamp (either DatetimeIndex or floats)
    Returns
    -------
    rate: nominal rate of the DataFrame
    """
    # check that the index is monotonic
    if not ignore_index and not data.index.is_monotonic:
        raise Exception('Data index should be monotonic')
    if data.shape[0] < 2:
        raise Exception('Sampling rate requires at least 2 points')

    if isinstance(data.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
        delta = data.index - data.index[0]
        index_diff = np.diff(delta) / np.timedelta64(1, 's')
    elif np.issubdtype(data.index, np.number):
        index_diff = np.diff(data.index)
    else:
        raise Exception('Dataframe index is not numeric')

    average_timespan = np.median(index_diff)
    if  not ignore_index and  np.any(index_diff >= average_timespan * 2):
        raise Exception('Effective sampling is greater than twice the nominal rate')

    return 1 / average_timespan


class TxOddall(BaseDataset):
    """Dataset acquired with oddball

    Dataset from ..... TODO .....

    **Dataset Description**
    ..... TODO .....

    References
    ----------
    ..... TODO .....

    """

    def __init__(self, datapath):
        # check if has to unzip
        self.datapath = datapath  # todo: unhardcode that, find an appropriate cloud storage

        # store fnames per device
        self.datapath_per_device = defaultdict(list)
        for file_path in glob(os.path.join(datapath, '*.hdf5')):
            with pd.HDFStore(file_path, 'r') as store:
                meta = store.get_node('/eeg')._v_attrs['meta']
            device = meta.get('device')
            if device is None:
                logger.warning(f'Could not find device in file meta. Skipping {os.path.basename(file_path)}.')
            else:
                self.datapath_per_device[device].append(file_path)

        super().__init__(
            subjects=['unicorn_wet', 'unicorn_dry', 'flexicap', 'enobio_dry'],  # devices?
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),  # dict(Deviant=2, Standard=1),
            code='Tx Oddball dataset',
            interval=[0, 1],  # Todo: what's that?
            paradigm= 'p300',  #'oddball',  # oddball?
            doi='')

    def _get_subject_run_data(self, file_path):

        # data from the .hdf5
        with pd.HDFStore(file_path, 'r') as store:
            eeg = pd.read_hdf(store, '/eeg')
            events = pd.read_hdf(store, '/events')
            meta = store.get_node('/eeg')._v_attrs['meta']

        raw = self._make_raw(eeg, events, meta)

        return raw

    def _get_single_subject_data(self, device):
        """return data for a single subject"""

        file_path_list = self.data_path(device)
        sessions = defaultdict()

        for file_path in sorted(file_path_list):
            session_name = os.path.basename(file_path).split('.hdf5')[0]
            sessions[session_name] = defaultdict()
            # if session_name not in sessions.keys():
            #     sessions[session_name] = {}
            # run_name = 'run_' + str(len(sessions[session_name]) + 1)
            run_name = '0' #  todo: troncate a session into run !
            sessions[session_name][run_name] = self._get_subject_run_data(
                file_path)

        return sessions

    def data_path(
            self,
            device,  # instead of subject
            path=None,  # todo
            force_update=False,  # todo
            update_path=None,  # todo
            verbose=None):

        if device not in self.datapath_per_device.keys():
            raise (ValueError(f"Invalid device {device}."))

        # todo: Quetzal storage, or??
        # check if has the .zip
        # url = '{:s}subject{:d}.zip'.format(EPFLP300_URL, subject)
        # path_zip = dl.data_path(url, 'EPFLP300')
        # path_folder = path_zip.strip('subject{:d}.zip'.format(subject))

        return self.datapath_per_device[device]

    def _make_raw(self, eeg, events, meta):
        # Extract attributes from meta or infer them
        # -------------------------------------------
        rate = meta.get('rate')
        if rate is None:
            rate = estimate_rate(eeg)
        ch_types = meta.get('types', ['eeg'] * eeg.shape[1]) # todo: infer type from column names
        unit_factor = meta.get('unit_factor', 1e-6)
        montage_kind = meta.get('montage_kind', 'standard_1005')

        # Prepare events to extract only stim
        # -----------------------------------
        # copy the metadata on the stim conetxt into events with stim onset
        events.loc[events.label == 'stim_on', 'data'] = events.loc[
            events.label == 'stim_begins', 'data'].values
        # select stim events only
        df_stims = events[events.label == 'stim_on']
        # group stim by their class, ie. Deviant or Standard
        df_stims.loc[:, 'label'] = df_stims.data.apply(lambda d: 'Target' if json.loads(d)['deviant'] else 'NonTarget') # Deviant, Standard instead?

        # Dejitter the data
        # ------------------
        # eeg = uniform_sampling(eeg, rate, 'linear') # todo: investigate interpolation influence on psd :/
        # Todo/question: do we poly-sampling to uniformize all datasets to same rate (eg. 250Hz?)?

        # Convert pandas Dataframe into mne raw object
        # --------------------------------------------
        X = eeg.values
        X *= unit_factor

        fix_ch_names = meta.get('columns')
        if fix_ch_names is not None:
            eeg = eeg.rename(columns=fix_ch_names)

        # headset montage
        ch_names = list(eeg.columns)
        montage = mne.channels.make_standard_montage(montage_kind) if montage_kind is not None else None

        # Append stim channel to the data
        ch_names += ['stim']
        ch_types += ['stim']
        times = eeg.index
        trig = np.zeros((len(X), 1))
        for stim_onset, stim in df_stims.iterrows():
            ix_stim = np.argmin(abs(times - stim_onset))
            label = stim.label
            trig[ix_stim] = self.event_id[label]
        X = np.c_[X, trig]
        # create info instance
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=rate, montage=montage)
        # remove NaN
        X[np.isnan(X)] = np.zeros_like(X)[np.isnan(X)]  # todo: remove this dangereous line?
        # create mne object from numpy array and info
        raw = mne.io.RawArray(data=X.T, info=info, verbose=False)
        return raw
