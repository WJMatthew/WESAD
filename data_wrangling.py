import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# E4 (wrist) Sampling Frequencies
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
WINDOW_IN_SECONDS = 15
label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
feat_names = []
savePath = 'data'

if not os.path.exists(savePath):
    os.makedirs(savePath)


class SubjectData:

    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_data(self):
        return self.data['signal']['wrist']

    def get_chest_data(self):
        return self.data['signal']['chest']

    def extract_features(self):  # only wrist
        results = \
            {
                key: get_statistics(self.get_wrist_data()[key].flatten(), self.labels, key)
                for key in self.wrist_keys
            }
        return results


def get_window_stats(data, label=-1):
    mean_features = np.mean(data)
    std_features = np.std(data)
    min_features = np.amin(data)
    max_features = np.amax(data)

    features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                'label': label}
    return features


def get_net_accel(data):
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))


def compute_features(e4_data_dict, norm_type=None):
    # Dataframes for each sensor type
    eda_df = pd.DataFrame(e4_data_dict['EDA'], columns=['EDA'])
    bvp_df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    acc_df = pd.DataFrame(e4_data_dict['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
    temp_df = pd.DataFrame(e4_data_dict['TEMP'], columns=['TEMP'])
    label_df = pd.DataFrame(subject.labels, columns=['label'])

    # Adding indices for combination due to differing sampling frequencies
    eda_df.index = [(1 / fs_dict['EDA']) * i for i in range(len(eda_df))]
    bvp_df.index = [(1 / fs_dict['BVP']) * i for i in range(len(bvp_df))]
    acc_df.index = [(1 / fs_dict['ACC']) * i for i in range(len(acc_df))]
    temp_df.index = [(1 / fs_dict['TEMP']) * i for i in range(len(temp_df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]

    # Change indices to datetime
    eda_df.index = pd.to_datetime(eda_df.index, unit='s')
    bvp_df.index = pd.to_datetime(bvp_df.index, unit='s')
    temp_df.index = pd.to_datetime(temp_df.index, unit='s')
    acc_df.index = pd.to_datetime(acc_df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')

    # Combined dataframe - not used yet
    df = eda_df.join(bvp_df, how='outer')
    df = df.join(temp_df, how='outer')
    df = df.join(acc_df, how='outer')
    df = df.join(label_df, how='outer')
    df['label'] = df['label'].fillna(method='bfill')

    if norm_type is 'std':
        # std norm
        df = (df - df.mean()) / df.std()
    elif norm_type is 'minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)
    return grouped, baseline, stress, amusement


def get_samples(data, n_windows, label):
    global feat_names

    samples = []
    window_len = 700 * WINDOW_IN_SECONDS

    for i in range(n_windows):
        # Get window of data
        w = data[window_len * i: window_len * (i + 1)]

        # Add/Calc rms acc
        w['net_acc'] = get_net_accel(w)

        # Calculate stats for window
        wstats = get_window_stats(data=w, label=label)

        # Seperating sample and label
        x = pd.DataFrame(wstats).drop('label', axis=0)
        y = x['label'][0]
        x.drop('label', axis=1, inplace=True)

        if len(feat_names) == 0:
            for row in x.index:
                for col in x.columns:
                    feat_names.append('_'.join([row, col]))

        # sample df
        wdf = pd.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names
        wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)
        samples.append(wdf)

    return pd.concat(samples)


def make_patient_data(subject_id):
    global savePath

    # Make subject data object for Sx
    subject = SubjectData(main_path='data/WESAD', subject_number=subject_id)

    # Empatica E4 data
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = None

    # The 3 classes we are classifying
    grouped, baseline, stress, amusement = compute_features(e4_data_dict, norm_type)

    print(f'Available windows for {subject.name}:')
    n_baseline_wdws = int(len(baseline) / (700 * WINDOW_IN_SECONDS))
    n_stress_wdws = int(len(stress) / (700 * WINDOW_IN_SECONDS))
    n_amusement_wdws = int(len(amusement) / (700 * WINDOW_IN_SECONDS))
    print(f'Baseline: {n_baseline_wdws}\nStress: {n_stress_wdws}\nAmusement: {n_amusement_wdws}\n')

    #
    baseline_samples = get_samples(baseline, n_baseline_wdws, 1)
    stress_samples = get_samples(stress, n_stress_wdws, 2)
    amusement_samples = get_samples(amusement, n_amusement_wdws, 0)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1)
    # Selected Features
    all_samples = all_samples[['EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max',
                               'BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
                               'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max',
                               'net_acc_mean', 'net_acc_std', 'net_acc_min', 'net_acc_max',
                               0, 1, 2]]
    # Save file as csv (for now)
    all_samples.to_csv(f'{savePath}/S{subject_id}_feats_{norm_type}.csv')


def combine_files(subjects):
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'S{s}_feats.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)

    df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
    df.drop(['0', '1', '2', 'subject'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv('may12_feats.csv')


if __name__ == '__main__':

    subject_ids = [3, 4, 5, 6, 7, 8, 9, 10]

    for patient in subject_ids:
        make_patient_data(patient)

    combine_files(subject_ids)