from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from baseline import create_training_data
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.spatial.distance import cdist, pdist


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
    model = Doc2Vec(vector_size=size, alpha=alpha, min_alpha=min_alpha,
                    min_count=min_count, dm=dm)

    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("d2v.model")
    print("Model Saved")


def create_features(docvecs):
    x_values = []

    i = 0
    while i < len(docvecs) - 1:
        temp = []
        #temp.append(np.dot(docvecs[str(i)], docvecs[str(i + 1)]))
        temp = np.multiply(docvecs[i], docvecs[i + 1])
        #temp.append(np.sqrt(np.sum(np.power(np.subtract(docvecs[str(i)], docvecs[str(i + 1)]), 2))))
        #temp.append(cdist(docvecs[str(i)], docvecs[str(i + 1)], 'cosine'))
        #temp.append(pdist(docvecs[str(i)], docvecs[str(i + 1)], 'seuclidean', V=None))
        x_values.append(temp)
        i += 2

    return x_values


def get_labels(data):
    y_values = []

    for i in range(1, len(data) + 1):
        y_values.append(data[i]['quality'])

    return y_values


def main():
    data, data_array, tagged = prepare_data("data/msr_paraphrase_train.txt")
    create_gensim_model(tagged)
    model = Doc2Vec.load("d2v.model")
    x_values = create_features(model.docvecs)
    y_values = get_labels(data)
    parameters = {
        "kernel": ["rbf"],
        "C": [1, 10, 10, 100, 1000],
        "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
    grid = GridSearchCV(SVC(), parameters, cv=5, verbose=2)
    x_values = np.vstack(x_values)
    grid.fit(x_values, y_values)
    test, test_array, yes = prepare_data("data/msr_paraphrase_test.txt")

    docvecs_test = []

    for s in test_array:
        docvecs_test.append(model.infer_vector([s]))

    print(len(docvecs_test))

    x_test = create_features(docvecs_test)
    true = get_labels(test)
    x_test = np.vstack(x_test)
    pred = grid.predict(x_test)
    print(len(true))
    print(len(pred))
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})
    exit()


if __name__ == '__main__':
    main()