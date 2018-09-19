import pandas as pd
import numpy as np
import functools

from os import listdir
from os.path import isfile, join
from scipy import stats, signal, fftpack


###################
# Constants
###################


SAMPLING_FREQ = 50
WINDOW_SIZE = 128
OVERLAP_SIZE = WINDOW_SIZE // 2
MEASURES = [np.mean, np.max, np.min, np.std, np.var]
COL_NAMES = ['Acc-x-Mean', 'Acc-y-Mean', 'Acc-z-Mean',
             'Grav-x-Mean', 'Grav-y-Mean', 'Grav-z-Mean',
             'Gyro-x-Mean', 'Gyro-y-Mean', 'Gyro-z-Mean',
             'Acc-x-Max', 'Acc-y-Max', 'Acc-z-Max',
             'Grav-x-Max', 'Grav-y-Max', 'Grav-z-Max',
             'Gyro-x-Max', 'Gyro-y-Max', 'Gyro-z-Max',
             'Acc-x-Min', 'Acc-y-Min', 'Acc-z-Min',
             'Grav-x-Min', 'Grav-y-Min', 'Grav-z-Min',
             'Gyro-x-Min', 'Gyro-y-Min', 'Gyro-z-Min',
             'Acc-x-Std', 'Acc-y-Std', 'Acc-z-Std',
             'Grav-x-Std', 'Grav-y-Std', 'Grav-z-Std',
             'Gyro-x-Std', 'Gyro-y-Std', 'Gyro-z-Std',
             'Acc-x-Var', 'Acc-y-Var', 'Acc-z-Var',
             'Grav-x-Var', 'Grav-y-Var', 'Grav-z-Var',
             'Gyro-x-Var', 'Gyro-y-Var', 'Gyro-z-Var',
             'User ID', 'Move ID']


###################
# Data Extraction
###################


def combine_raw_data(acc_filenames, acc_file_location_prefix, gyro_filenames, gyro_file_location_prefix):
    """ Combines and returns all raw data .txt files as a list of dataframes, given a list of raw data file names """

    combined_data = []
    combined_acc = []
    combined_gyro = []

    # Acc data
    for file_name in acc_filenames:
        # Read in values twice for segmentation into acc and gravity components later
        file_df_acc = pd.read_csv(acc_file_location_prefix +
                                  file_name, sep=" ", header=None)
        file_df_gravity = pd.read_csv(acc_file_location_prefix +
                                      file_name, sep=" ", header=None)
        file_df = pd.concat([file_df_acc, file_df_gravity], axis=1)
        file_df.columns = ['Acc-x', 'Acc-y',
                           'Acc-z', 'Grav-x', 'Grav-y', 'Grav-z']
        combined_acc.append(file_df)

    # Gyro Data
    for file_name in gyro_filenames:
        file_df = pd.read_csv(gyro_file_location_prefix +
                              file_name, sep=" ", header=None)
        file_df.columns = ['Gyro-x', 'Gyro-y',
                           'Gyro-z', ]
        combined_gyro.append(file_df)

    # Concat acc and gyro data
    for i in range(len(combined_acc)):
        combined_data.append(
            pd.concat([combined_acc[i], combined_gyro[i]], axis=1))

    return combined_data


def get_move_info(num_experiments):
    """ 
    Return list of dataframes, where each dataframe is the feature info for an experiment 

    Feature info refers to expr/user/move IDs, Label start and end
    """

    move_info = []

    labels_df = pd.read_csv(
        './dataset/RawData/labels.txt', sep=" ", header=None)
    labels_df.columns = ['Experiment ID', 'User ID',
                         'Move ID', 'Move Start', 'Move End (Inc)']

    for i in range(num_experiments):
        expr_id = i + 1
        move_info.append(labels_df.loc[
            labels_df['Experiment ID'] == expr_id
        ])

    return move_info


###################
# Preprocessing
###################


# Note for butterworth filter:
# sos performs better over ba for higher order digital filters and when cutoff freq is relatively low compared to nyq freq


def medfilt(axis):
    return signal.medfilt(axis)


def butter_noise(axis):
    nyq_freq = SAMPLING_FREQ * 0.5
    cutoff_freq = 20.0 / nyq_freq
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='lowpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_gravity(axis):
    nyq_freq = SAMPLING_FREQ * 0.5
    cutoff_freq = 0.3 / nyq_freq
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='lowpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_body(axis):
    nyq_freq = SAMPLING_FREQ * 0.5
    cutoff_freq = 0.3 / nyq_freq
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='highpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


####################
# Feature Extraction
####################


def extract_features(measures, window, user_id, move_id):
    """ Applies list of given measures to given window to extract features """

    features = pd.DataFrame()

    for measure in measures:
        features = pd.concat(
            [features, window.apply(measure).to_frame().T], axis=1)

    features['User ID'] = user_id
    features['Move ID'] = move_id
    features.columns = range(len(features.columns))

    return features


###################
# Helper Functions
###################


def compose(functions):
    """Composes a bunch of functions"""
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)


###################
# Main
###################


if __name__ == '__main__':
    # Get list of all raw data file names
    acc_filenames = [f for f in listdir(
        './dataset/RawData/Acc') if isfile(join('./dataset/RawData/Acc', f))]
    gyro_filenames = [f for f in listdir(
        './dataset/RawData/Gyro') if isfile(join('./dataset/RawData/Gyro', f))]

    # Combine all raw acc and gyro data into list of dataframes
    combined_data = combine_raw_data(
        acc_filenames, './dataset/RawData/Acc/', gyro_filenames, './dataset/RawData/Gyro/')

    # Read move info into list of dataframes
    combined_move_info = get_move_info(len(combined_data))

    # Pre-process each experiment data with medfilt and butter, split acc signals in bodyAcc and gravAcc
    for i in range(len(combined_data)):
        combined_data[i].iloc[:, 0:3] = combined_data[i].iloc[:, 0:3].apply(  # bodyAcc
            compose([butter_body, butter_noise, medfilt]))
        combined_data[i].iloc[:, 3:6] = combined_data[i].iloc[:, 3:6].apply(  # gravAcc
            compose([butter_gravity, butter_noise, medfilt]))
        combined_data[i].iloc[:, 6:9] = combined_data[i].iloc[:, 6:9].apply(  # Gyro
            compose([butter_noise, medfilt]))

    # Extract features from fixed-width sliding windows (128 cycles with 50% overlap)
    # and save into single dataframe
    combined_extracted_features = pd.DataFrame()

    # Iterate over each experiment for feature extraction
    for i in range(len(combined_data)):
        curr_experiment = combined_data[i]
        # Iterate over each move in the current experiment
        for j in range(len(combined_move_info[i])):
            move_info = combined_move_info[i].iloc[j]
            move_start = int(move_info['Move Start'])
            move_end = int(move_info['Move End (Inc)'])
            user_id = move_info['User ID']
            move_id = move_info['Move ID']
            experiment_end = len(curr_experiment)

            window_start = move_start
            window_end = window_start + WINDOW_SIZE

            # Slide window until it's crossed either end of move or experiment
            while (window_end <= experiment_end and window_end <= move_end):
                window = curr_experiment.iloc[window_start:window_end]
                window_start += OVERLAP_SIZE
                window_end += OVERLAP_SIZE

                combined_extracted_features = pd.concat(
                    [combined_extracted_features, extract_features(MEASURES, window, user_id, move_id)])

    # Rename columns
    combined_extracted_features.columns = COL_NAMES

    # Save to csv
    combined_extracted_features.to_csv('./extracted_features.csv', sep='\t',
                                       encoding='utf-8', index=False)
