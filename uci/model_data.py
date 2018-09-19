import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


if __name__ == '__main__':
    # Read in features
    combined_extracted_features = pd.read_csv(
        './extracted_features.csv', sep="\t", encoding='utf-8')
    X = combined_extracted_features.iloc[:, :-2]
    y = combined_extracted_features.iloc[:, -1]
    
    # Hyperparameters
    MAX_FEATURES = (np.sqrt(len(X.columns)) + 1) / len(X.columns)
    ESTIMATORS = 100

# =============================================================================
#     # k-fold validation
#     accuracy = []
#     kf = KFold(n_splits=10, shuffle=True, random_state=0)
#     kf.get_n_splits(X)
# 
#     for train_index, test_index in kf.split(X):
#        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
#        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
# 
#        clf = RandomForestClassifier(random_state=0)
#        clf.fit(train_X, train_y.values.ravel())
# 
#        predictions = clf.predict(test_X)
#        accuracy.append(accuracy_score(test_y, predictions))
# 
#     print(np.mean(accuracy))
# =============================================================================

    # Stratified k-fold validation
    accuracy = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf.get_n_splits(X)

    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = ExtraTreesClassifier(random_state=0, max_features=MAX_FEATURES,
                                   n_estimators=ESTIMATORS, max_depth=None, min_samples_split=2, bootstrap=False)
#        clf = RandomForestClassifier(random_state=0, max_features='sqrt',
#                                     n_estimators=ESTIMATORS, max_depth=None, min_samples_split=2, bootstrap=True)
        clf.fit(train_X, train_y.values.ravel())

        predictions = clf.predict(test_X)
        accuracy.append(accuracy_score(test_y, predictions))

#    print(confusion_matrix(test_y, predictions))
    print(np.mean(accuracy))

    # View a list of the features and their importance scores
    feature_importance = list(zip(train_X, clf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1])
