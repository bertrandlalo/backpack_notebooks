from moabb.datasets import SSVEPExo
from moabb.paradigms import SSVEP
import mne
import numpy as np
import pandas as pd
import os
import json

paradigm = SSVEP(n_classes=4)
dataset = SSVEPExo()
data = dataset.get_data()

onset_label = 'stim_begins'
data_key = 'target'
stim_to_event = {0: {'label': np.NaN, 'data': np.NaN},
                 1: {'label': onset_label, 'data': json.dumps({data_key: '13Hz'})},
                 2: {'label': onset_label, 'data': json.dumps({data_key: '17Hz'})},
                 3: {'label': onset_label, 'data': json.dumps({data_key: '21Hz'})},
                 4: {'label': onset_label, 'data': json.dumps({data_key: 'rest'})},
                 }


def raw_to_frames(raw, stim_to_event, onset='2020-01-01'):
    df_eeg = raw.to_data_frame(picks=mne.pick_types(raw.info, eeg=True, stim=False))
    df_stim = raw.to_data_frame(picks=mne.pick_types(raw.info, eeg=False, stim=True))
    events_values = list(map(lambda stim: stim_to_event.get(stim), list(df_stim.values[:, 0])))
    df_events = pd.DataFrame(events_values, index=df_stim.index)

    sampling_rate = raw.info['sfreq']
    period_ns, fract = divmod(1e9, sampling_rate)
    times = pd.date_range(onset, periods=len(df_events), freq=f'{period_ns}N')
    df_events.index = times
    df_eeg.index = times
    return df_eeg, df_events


def make_subject_hdf5(session_data: dict, output: str, train: str = 'run_0', test: str = 'run_1',
                     onset: str = '2020-01-01', force: bool = False):
    if not force and os.path.exists(output):
        return
    raw_train = session_data.get(train)
    raw_test = session_data.get(test)
    if None in [raw_train, raw_test]:
        raise ValueError(f'Could not find runs {train}, {test} in session data.')

    df_train_eeg, df_train_events = raw_to_frames(raw_train, stim_to_event, onset=onset)
    df_train_events.iloc[0, :] = ['train_begins', '{}']
    df_train_events.iloc[-2, :] = ['train_ends', '{}']
    test_onset = df_train_events.index[-1]
    df_test_eeg, df_test_events = raw_to_frames(raw_test, stim_to_event, onset=test_onset)
    df_eeg = pd.concat([df_train_eeg.iloc[:-1, :], df_test_eeg], axis=0)
    df_events = pd.concat([df_train_events.iloc[:-1, :], df_test_events], axis=0)
    df_events.dropna(inplace=True)
    # save to hdf5 stores
    df_eeg.to_hdf(output, '/eeg', format='table')
    df_events.to_hdf(output, '/events', format='table')

def make_subjects_hdf5():
    for subject in data:
        session_data = data[subject]['session_0']
        make_subject_hdf5(session_data, output=f'data/{subject}.hdf5')


if __name__ == "__main__":
    # execute only if run as a script
    make_subjects_hdf5()