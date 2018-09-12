import random
import data
import segmenting
import preprocess
import feature_extraction
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

PREPROCESS_FUNCS = [preprocess.hann, preprocess.medfilt]

FEATURES = [
    feature_extraction.energy,
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

    for key, value in dictionary.items():
        if test_function(key):
            dict_true[key] = value
        else:
            dict_false[key] = value

    return dict_true, dict_false


def generate_train_test_segments():
    labels_per_file = data.get_labels()

    users = list(range(1, 31))
    random.shuffle(users)
    train_users = users[:21]
    train_labels, test_labels = split_dict(labels_per_file, lambda exp_user: exp_user[1] in train_users)

    segmenting.save_segments(segmenting.segment_activities(train_labels), 'train_segments.txt')
    segmenting.save_segments(segmenting.segment_activities(test_labels), 'test_segments.txt')


def train_model(model):
    train_segments = segmenting.load_segments('train_segments.txt')
    train_features = extract_features(train_segments[:, 0])
    train_labels = list(train_segments[:, 1])

    model.fit(train_features, train_labels)

    return model


def test_model(model):
    test_segments = segmenting.load_segments('test_segments.txt')
    test_features = extract_features(test_segments[:, 0])
    test_labels = list(test_segments[:, 1])

    test_predictions = model.predict(test_features)

    print(classification_report(test_labels, test_predictions))


if __name__ == '__main__':
    # generate_train_test_segments()

    model = OneVsRestClassifier(LinearSVC())
    model = train_model(model)
    test_model(model)
