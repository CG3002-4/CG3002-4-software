import random
import data
import segmenting
import preprocess
import feature_extraction
import numpy as np

PREPROCESS_FUNCS = [preprocess.hann, preprocess.medfilt]

FEATURES = [feature_extraction.energy,
            feature_extraction.entropy,
            feature_extraction.mean,
            feature_extraction.stdev
            ]

def preprocess_segment(segment):
    segment['acc'] = preprocess.preprocess(PREPROCESS_FUNCS, segment['acc'])
    segment['gyro'] = preprocess.preprocess(PREPROCESS_FUNCS, segment['gyro'])

    return segment

def extract_features(segments):
    segments = np.array([preprocess_segment(segment) for segment in segments])
    return feature_extraction.extract_features(segments, FEATURES)

def split_dict(dictionary, test_function):
    dict_true, dict_false = {}, {}

    for key, value in dictionary.iteritems():
        if test_function(key):
            dict_true[key] = value
        else:
            dict_false[key] = value

    return dict_true, dict_false

def generate_train_test_segments():
    labels_per_file = data.get_labels_per_file()

    users = list(range(1, 31))
    random.shuffle(users)
    train_users, test_users = users[:21], users[21:]
    train_labels, test_labels = split_dict(labels_per_file, lambda (exp, user): user in train_users)

    segmenting.save_segments(segmenting.get_raw_segments(train_labels), 'train_segments.txt')
    segmenting.save_segments(segmenting.get_raw_segments(test_labels), 'test_segments.txt')

def load_train_test_segments():
    return (segmenting.load_segments('train_segments.txt'),
            segmenting.load_segments('test_segments.txt'))

if __name__ == '__main__':
    # generate_train_test_segments()

    train_segments, test_segments = load_train_test_segments()

    print train_segments[0].shape

    # features = extract_features(segments[:, 0])
    # labels = segments[:, 1]
    #
    # print features.shape
