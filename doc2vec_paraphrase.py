from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from baseline import create_training_data
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np


def prepare_data(train):
    data_array = []
    data = create_training_data(train)

    for i in range(1, len(data) + 1):
        data_array.append(data[i]['first'])
        data_array.append(data[i]['second'])

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                                  tags=[str(i)]) for i, _d in enumerate(data_array)]
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


def dot_prod_features(docvecs, data):
    x_values = []
    y_values = []
    i = 0
    while i < len(docvecs) - 1:
        x_values.append(np.dot(docvecs[str(i)], docvecs[str(i + 1)]))
        i += 2

    for i in range(1, len(data) + 1):
        y_values.append(data[i]['quality'])

    return x_values, y_values

def dot_prod_features_test(docvecs, data):
    x_values = []
    y_values = []
    i = 0
    while i < len(docvecs) - 1:
        x_values.append(np.dot(docvecs[(i)], docvecs[(i + 1)]))
        i += 2

    for i in range(1, len(data) + 1):
        y_values.append(data[i]['quality'])

    return x_values, y_values

def main():
    data, data_array, tagged = prepare_data("data/msr_paraphrase_train.txt")
    create_gensim_model(tagged)
    model = Doc2Vec.load("d2v.model")
    x_values, y_values = dot_prod_features(model.docvecs, data)
    glm = LogisticRegressionCV(cv=5).fit(np.reshape(x_values, (-1, 1)), y_values)

    test, test_array, yes = prepare_data("data/msr_paraphrase_test.txt")

    docvecs_test = []

    for s in test_array:
        docvecs_test.append(model.infer_vector([s]))

    print(len(docvecs_test))

    x_test, true = dot_prod_features_test(docvecs_test, test)
    pred = glm.predict(np.reshape(x_test, (-1, 1)))

    print(len(true))
    print(len(pred))
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)

    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})


if __name__ == '__main__':
    main()
