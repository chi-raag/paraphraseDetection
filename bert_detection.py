import json
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from baseline import create_training_data
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
import sys
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default='para_train.jsonl',
                        help='path to output of BERT extract features script with MSRP training data')
    parser.add_argument('--test',
                        default='para_test.jsonl',
                        help='path to output of BERT extract features script with MSRP test data')
    return parser.parse_args()


def get_bert_features(path):

    data = []
    tokens = []
    with open(path) as f:
        for line in f:
            sent = []
            tokes = []
            t = json.loads(line)
            for token in t['features']:
                tokes.append(token['token'])
                sent.append(np.mean(token['layers'][0]['values']))
            data.append(sent)
            tokens.append(tokes)

    return data, tokens


def get_training_features(vecs):
    x_values = []

    i = 0
    while i < len(vecs) - 1:
        temp = np.outer(vecs[i], vecs[i + 1]).flatten()
        x_values.append(temp)
        i += 2

    return x_values


def get_handmade_features(tokens):
    features = []

    i = 0
    while i < len(tokens) - 1:
        temp = []
        overlap = set(tokens[i]) & set(tokens[i + 1])
        temp.append(len(overlap))
        temp.append(abs(len(tokens[i]) - len(tokens[i + 1])))
        features.append(temp)
        i += 2

    return features


def get_labels(path):
    #return np.array(pd.read_csv(path, header=None, sep=","))
    return pd.read_csv(path, header=None, sep=",")


def pad_to_dense(M, maxlen):
    """Appends the minimal required amount of zeroes at the end of each
     array in the jagged array `M`, such that `M` looses its jagedness."""
    temp = []

    for r in M:
        #np.append(r, np.zeros(maxlen - len(r))).flatten()
        if len(r) != maxlen:
            temp1 = r
            #mean = np.mean(r)
            [temp1.append(0) for _ in range(maxlen - len(r))]
            temp.append(temp1)
        else:
            temp.append(r)

    maxtemp = max(len(r) for r in temp)

    mintemp = min(len(r) for r in temp)

    if maxtemp != mintemp:
        print('diff lengths')
        print('exiting')
        sys.exit()

    return temp


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self


def main():
    args = parse_args()

    print("reading data")
    bert_train, train_tokenized = get_bert_features(args.train)
    bert_test, test_tokenized = get_bert_features(args.test)

    print("handmade features")
    train_handmade = get_handmade_features(train_tokenized)
    test_handmade = get_handmade_features(test_tokenized)

    print("getting max len")
    maxlen = max(max(len(r) for r in bert_train), max(len(r) for r in bert_test))

    print("padding data")
    train_features = pad_to_dense(bert_train, maxlen)
    test_features = pad_to_dense(bert_test, maxlen)

    print("getting features")
    train_features = get_training_features(train_features)
    test_features = get_training_features(test_features)

    train_features = np.concatenate((train_features, np.array(train_handmade)), axis=1)
    test_features = np.concatenate((test_features, np.array(test_handmade)), axis=1)

    print("reading labels")
    train_labels = get_labels("train_labels.txt").as_matrix().flatten()
    test_labels = get_labels("test_labels.txt").as_matrix().flatten()

    print(np.array(train_features).shape)
    print(np.array(test_features).shape)
    print(np.array(train_labels).shape)
    print(np.array(test_labels).shape)

    maxlen1 = max(len(r) for r in train_features)

    minlen = min(len(r) for r in train_features)

    if maxlen1 != minlen:
        print('diff lengths - pre model')
        print('exiting')
        sys.exit()

    ### Logistic Regression ###
    print("logistic")

    model = Pipeline([('scale', StandardScaler()),
                        ('tsvd', TruncatedSVD(100)),
                        ('log', LogisticRegressionCV(cv=5, verbose=0, refit=True, max_iter=1000))])

    model.fit(train_features, train_labels)

    pred = model.predict(test_features)

    accuracy = accuracy_score(test_labels, pred)
    precision = precision_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    ### Random Forest ###
    print("rf")

    model = Pipeline([('scale', StandardScaler()),
                        ('tsvd', TruncatedSVD(100)),
                        ('tress', RandomForestClassifier(n_estimators=150, max_depth=None,
                                 min_samples_split=2))])

    model.fit(train_features, train_labels)

    pred = model.predict(test_features)

    accuracy = accuracy_score(test_labels, pred)
    precision = precision_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    ### ENSEMBLE RF ###
    print("ensemble rf")

    pipe1 = Pipeline([
        ('col_extract', ColumnExtractor(cols=range(3481, 3483))),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    pipe2 = Pipeline([
        ('col_extract', ColumnExtractor(cols=range(0, 3481))),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=None,
                                 min_samples_split=2))
    ])

    eclf = VotingClassifier(estimators=[('df1-clf1', pipe1), ('df2-clf2', pipe2)], voting='soft', weights=[1, 0.5])

    eclf.fit(train_features, train_labels)

    pred = eclf.predict(test_features)

    accuracy = accuracy_score(test_labels, pred)
    precision = precision_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    ### ENSEMBLE ADA ###
    print("ensemble ada")

    pipe1 = Pipeline([
        ('col_extract', ColumnExtractor(cols=range(3481, 3483))),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    pipe2 = Pipeline([
        ('col_extract', ColumnExtractor(cols=range(0, 3481))),
        ('clf', AdaBoostClassifier(n_estimators=200))
    ])

    eclf = VotingClassifier(estimators=[('df1-clf1', pipe1), ('df2-clf2', pipe2)], voting='soft', weights=[1, 0.5])

    eclf.fit(train_features, train_labels)

    pred = eclf.predict(test_features)

    accuracy = accuracy_score(test_labels, pred)
    precision = precision_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    ### ENSEMBLE GRADIENT ###
    print("ensemble gradient")

    pipe1 = Pipeline([
        ('col_extract', ColumnExtractor(cols=range(3481, 3483))),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    pipe2 = Pipeline([
        ('col_extract', ColumnExtractor(cols=range(0, 3481))),
        ('clf', GradientBoostingClassifier(n_estimators=200))
    ])

    eclf = VotingClassifier(estimators=[('df1-clf1', pipe1), ('df2-clf2', pipe2)], voting='soft', weights=[1, 0.5])

    eclf.fit(train_features, train_labels)

    pred = eclf.predict(test_features)

    accuracy = accuracy_score(test_labels, pred)
    precision = precision_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    ### END ###

    print("done")

    sys.exit()


if __name__ == '__main__':
    main()
