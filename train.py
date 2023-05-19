import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split

from models import random_forest_classifier, logistic_regression, decision_tree, naive_bayes
from utils import null_value_rows_remove, random_over_sampling, label_encoder, find_best_features, save_sklearn_model, \
    save_list_to_file


def train(subject_name):
    df = pd.read_csv('student_acadimic_resutl.csv')

    df = null_value_rows_remove(df)

    df = random_over_sampling(df, subject_name)

    df = label_encoder(df)

    features = df.drop(['29. Results (IS3440)', '29. Results(IS3420)'], axis=1)
    targets = df[[subject_name]]

    best_features = find_best_features(features, targets, 10)
    save_list_to_file(best_features.to_list(), 'features_columns.txt')

    X = features[best_features.to_list()]
    y = targets

    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.20)

    print("Random Forest classifier result")
    print("=====================================")
    randon_clf = random_forest_classifier(X_train, x_test, Y_train, y_test)
    save_sklearn_model(randon_clf, 'random_forest')

    print("\nLogistic Regression Result")
    print("=====================================")
    logistic_clf = logistic_regression(X_train, x_test, Y_train, y_test)
    save_sklearn_model(logistic_clf, 'logistic_regression')

    print("\nDecision Tree (CART) Algorithm classifier Result")
    print("=====================================")
    decision_clf = decision_tree(X_train, x_test, Y_train, y_test)
    save_sklearn_model(decision_clf, 'decision_tree')

    print("\nNaive bayes classifier Result")
    print("=====================================")
    naive_clf = naive_bayes(X_train, x_test, Y_train, y_test)
    save_sklearn_model(naive_clf, 'naive_bayes')


if __name__ == '__main__':
    train('29. Results(IS3420)')
