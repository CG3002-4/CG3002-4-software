import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ESTIMATORS = 39

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Read in features
    combined_extracted_features = pd.read_csv(
        './extracted_features.csv', sep="\t", encoding='utf-8')

    X = combined_extracted_features.iloc[:, :-2]
    y = combined_extracted_features.iloc[:, -1]
    MAX_FEATURES = (np.sqrt(len(X.columns)) + 1) / len(X.columns)

#    # k-fold validation
#    accuracy = []
#    kf = KFold(n_splits=10, shuffle=True, random_state=0)
#    kf.get_n_splits(X)
#
#    for train_index, test_index in kf.split(X):
#       train_X, test_X = X.iloc[train_index], X.iloc[test_index]
#       train_y, test_y = y.iloc[train_index], y.iloc[test_index]
#
#       clf = RandomForestClassifier(random_state=0)
#       clf.fit(train_X, train_y.values.ravel())
#
#       predictions = clf.predict(test_X)
#       accuracy.append(accuracy_score(test_y, predictions))
#
#    print(np.mean(accuracy))

    # Stratified k-fold validation
    accuracy = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf.get_n_splits(X)
    confusion_matrices = []

    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        # std_scale = StandardScaler().fit(train_X)
        # train_X = std_scale.transform(train_X)
        # test_X = std_scale.transform(test_X)

        # clf = OneVsRestClassifier(LinearSVC(random_state=0))
        clf = ExtraTreesClassifier(random_state=0, max_features=MAX_FEATURES,
                                   n_estimators=ESTIMATORS, max_depth=None, min_samples_split=2, bootstrap=False)
#        clf = RandomForestClassifier(random_state=0, max_features='sqrt',
#                                     n_estimators=ESTIMATORS, max_depth=None, min_samples_split=2, bootstrap=True)
        clf.fit(train_X, train_y)

        predictions = clf.predict(test_X)
        accuracy.append(accuracy_score(test_y, predictions))
        confusion_matrices.append(confusion_matrix(test_y, predictions))

    # View a list of the features and their importance scores
    feature_importance = list(zip(train_X, clf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1])
    print(feature_importance)

    print("Accuracy: " + str(np.mean(accuracy)))

    confusion_avg = np.mean(confusion_matrices, axis=0)
    np.set_printoptions(suppress=True)
    print("Confusion matrix:")
    print(confusion_avg)

    confusion_norm = confusion_avg / np.sum(confusion_avg, axis=1)

    plt.figure()
    plt.imshow(confusion_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(confusion_norm))
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

    fmt = '.2f'
    thresh = confusion_norm.max() / 2.
    for i, j in itertools.product(range(confusion_norm.shape[0]), range(confusion_norm.shape[1])):
        plt.text(j, i, format(confusion_norm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_norm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
