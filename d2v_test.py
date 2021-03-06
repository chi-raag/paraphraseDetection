from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from baseline import create_training_data
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import sys
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_data(train):
    data_array = []
    data = create_training_data(train)
    tags = []

    for i in range(1, len(data) + 1):
        data_array.append(data[i]['first'])
        tags.append(data[i]['quality'])
        data_array.append(data[i]['second'])
        tags.append(data[i]['quality'])

    tagged_data = [TaggedDocument(words=word_tokenize(d.lower()),
                                       tags=[tags[i], i]) for i, d in enumerate(data_array)]
    return data, data_array, tagged_data


def create_gensim_model(tagged, size=20,
                        alpha=.025, min_alpha=.00025,
                        min_count=1, dm=1):

    model = Doc2Vec(vector_size=300, alpha=alpha, min_alpha=min_alpha,
                    min_count=10, dm=0, sample=0, negative=5, hs=0)

    model.build_vocab(tagged)
    model.train(tagged, total_examples=len(tagged), epochs=100)

    model.save("d2v.model")
    print("Model Saved")


def create_features(docvecs, m):
    x_values = []

    i = 0
    while i < len(docvecs) - 1:
        temp = np.outer(docvecs[i], docvecs[i + 1]).flatten()
        #np.append(cosine_similarity(m[1], m[i+1]).flatten(), temp)
        x_values.append(temp)
        i += 2

    return x_values


def get_labels(data):
    y_values = []

    for i in range(1, len(data) + 1):
        y_values.append(data[i]['quality'])

    return np.array(y_values)


def vec_for_learning(model, docs):
    regressors = []

    for i in range(len(docs)):
        regressors.append(model.infer_vector(docs[i].words))

    return regressors


def main():
    #Get Data
    data, data_array, tagged = prepare_data("data/msr_paraphrase_train.txt")

    vec = TfidfVectorizer(max_df=.3, min_df=7, norm="l2", max_features=5000)
    m = vec.fit_transform(data_array)

    create_gensim_model(tagged)
    model = Doc2Vec.load("d2v.model")
    x_values = create_features(model.docvecs, m)
    y_values = get_labels(data)

    x_values = np.vstack(x_values)

    #Get vecs based on d2v model and predict w model
    test, test_array, yes = prepare_data("data/msr_paraphrase_test.txt")

    docvecs_test = []

    for s in test_array:
        docvecs_test.append(model.infer_vector(word_tokenize(s.lower())))

    print(len(docvecs_test))

    x_test = create_features(docvecs_test, m)
    true = get_labels(test)
    x_test = np.vstack(x_test)

    # Set up model
    parameters = {
        #"kernel": ["rbf"],
        #"C": [1, 10, 10, 100, 1000],
        #"SVC__kernel": ["linear", "rbf"],
        "SVC__C": np.exp(np.arange(-8, 3))
        #"SVC__gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        #"C": [0.0001],
        #"gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        #"gamma": [1e-8]
    }

    SVCpipe = Pipeline([('scale', StandardScaler()),
                        ('tsvd', TruncatedSVD(100)),
                        ('SVC', LinearSVC(max_iter=2000))])
    grid = GridSearchCV(SVCpipe, parameters, cv=2, verbose=2)

    # Fit model
    grid.fit(x_values, y_values)

    pred = grid.predict(x_test)

    #Print Acc etc
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)

    print("SVC ------------------------------------------------------")

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    #Fit logistic Reg model
    log_parameters = {
        "log__penalty": ['l1', 'l2'],
        "log__C": np.logspace(0, 4, 10)
    }

    #LOGpipe
    model = Pipeline([('scale', StandardScaler()),
                        ('tsvd', TruncatedSVD(100)),
                        ('log', LogisticRegressionCV(cv=5, verbose=0, refit=True, max_iter=1000))])

    #logit = GridSearchCV(LOGpipe, log_parameters, cv=2, verbose=0)
    model.fit(x_values, y_values)
        #LogisticRegressionCV(cv=5, max_iter=2000).fit(x_values, y_values)
    pred = model.predict(x_test)

    #Print out testing acc etc
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)

    print("LOGISTIC ---------------------------------------------------")

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    #Extra Trees
    model = ExtraTreesClassifier(n_estimators=300, max_depth=None,
                                 min_samples_split=2, random_state=0)
    model.fit(x_values, y_values)
    pred = model.predict(x_test)

    # Print out testing acc etc
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)

    print("EXTRA TRESS --------------------------------------------------")

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    # RANDOM FOREST

    rf_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [100, 200, 400]}

    rf_grid = {'bootstrap': [True, False]}

    model = RandomizedSearchCV(RandomForestClassifier(), rf_grid,
                               n_iter=5, cv=2,
                               verbose=2, random_state=42)

    model = RandomForestClassifier(n_estimators=300)
    model.fit(x_values, y_values)
    pred = model.predict(x_test)

    # Print out testing acc etc
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)

    print("RANDOM FOREST ----------------------------------------------")
    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

    sys.exit()


if __name__ == '__main__':
    main()
