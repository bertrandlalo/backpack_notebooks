# -*- coding: utf-8 -*-
"""EEG utilities
    Created on december 2018
    @authors: Raphaëlle Bertrand-Lalo, David Ojeda

    Module containing utils functions to load, convert, process and plot EEG data.
"""

import logging

import mne
import numpy as np
import pandas as pd
import yaml
from numpy.lib import stride_tricks
from scipy.interpolate import interp1d
from sklearn.model_selection import cross_validate, StratifiedKFold

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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

def estimate_rate(data):
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
    # if not data.index.is_monotonic:
    # raise Exception('Data index should be monotonic')
    if data.shape[0] < 2:
        raise Exception('Sampling rate requires at least 2 points')

    if isinstance(data.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
        delta = data.index - data.index[0]
        index_diff = np.diff(delta) / np.timedelta64(1, 's')
    elif np.issubdtype(data.index, np.number):
        index_diff = np.diff(data.index)
    else:
        raise Exception('Dataframe index is not numeric')

    average_timespan = np.mean(index_diff)
    # if np.any(index_diff >= average_timespan * 2):
    #    raise Exception('Effective sampling is greater than twice the nominal rate')

    return 1 / average_timespan


def load_standalone_graph(path):
    # Load a graph
    with open(path, 'r') as stream:
        try:
            graph = yaml.safe_load(stream)['graphs'][0]
        except yaml.YAMLError as exc:
            print(exc)
    graph_standalone = graph.copy()
    # Get rid of online specific nodes and edges (zmq, lsl, safeguards)
    graph_standalone['nodes'] = [node for node in graph['nodes'] if
                                 node['module'] not in ['timeflux.nodes.lsl', 'timeflux.nodes.zmq']]
    nodes_id = [node['id'] for node in graph_standalone['nodes']]

    def keep_edge(edge):
        return (edge['source'].split(':')[0] in nodes_id) and (edge['target'].split(':')[0] in nodes_id)

    graph_standalone['edges'] = [edge for edge in graph_standalone['edges'] if keep_edge(edge)]
    return graph_standalone


def pandas_to_mne(data, events=None, montage_kind='standard_1005', unit_factor=1e-6, bad_ch=None):
    ''' Convert a pandas Dataframe into mne raw object 
    Parameters
    ----------
    data: Dataframe with index=timestamps, columns=eeg channels
    events: array, shape = (n_events, 3) with labels on the third axis. 
    unit_factor: unit factor to apply to get Voltage
    bad_ch: list of channels to reject

    Returns
    -------
    raw: raw object
    '''
    bad_ch = bad_ch or []
    n_chan = len(data.columns)

    X = data.copy().values
    times = data.index

    ch_names = list(data.columns)
    ch_types = ['eeg'] * n_chan
    montage = mne.channels.read_montage(montage_kind) if montage_kind is not None else None
    sfreq = estimate_rate(data)
    X *= unit_factor

    if events is not None:
        events_onsets = events.index
        events_labels = events.label.values
        event_id = {mk: (ii + 1) for ii, mk in enumerate(np.unique(events_labels))}
        ch_names += ['stim']
        ch_types += ['stim']

        trig = np.zeros((len(X), 1))
        for ii, m in enumerate(events_onsets):
            ix_tr = np.argmin(np.abs(times - m))
            trig[ix_tr] = event_id[events_labels[ii]]

        X = np.c_[X, trig]
    else:
        event_id = None

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq, montage=montage)
    info["bads"] = bad_ch
    raw = mne.io.RawArray(data=X.T, info=info, verbose=False)
    picks = mne.pick_channels(raw.ch_names, include=[], exclude=["stim"] + bad_ch)
    return raw, event_id, picks


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

def get_scores(clf, X, y, **kwargs):
    ''' Get scores from a model classification using cross validation.

    Parameters
    ----------
    clf: sklearn model

    X: iterable, training/testing data

    y: iterable, training/testing labels

    Returns
    -------
    ax: dataframe
        with scores of the cross validation estimated with metrics given in "scoring"
    '''

    cv = kwargs.get('cv', StratifiedKFold(5, shuffle=True, random_state=42) )

    scoring = kwargs.get('scoring', {'AUC': 'roc_auc', 'Accuracy': 'accuracy'} )

    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, verbose=False)

    n_splits = len(scores["test_AUC"])

    scores_df = pd.DataFrame([np.hstack([scores["test_{metrics}".format(metrics=metrics)] for metrics in scoring.keys()]),
                              np.hstack( [[metrics]*n_splits for metrics in scoring.keys()])],
                    index = ["values", "metrics"]).T
    return scores_df