import argparse
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def get_indices(s):
    return [i for i, c in enumerate(s) if c == "\t"]


def get_n_gram_overlap(first, second):
    overlap = set(first) & set(second)
    return [overlap, len(overlap)]


def set_n_grams(first_tokens, second_tokens, n):
    first_n_grams = list(nltk.ngrams(first_tokens, n))
    second_n_grams = list(nltk.ngrams(second_tokens, n))

    return first_n_grams, second_n_grams


def create_training_data(s, n=2):
    train_file = open(s)
    train = train_file.read().split('\n')
    train = train[1:]
    print(train[len(train) - 1])

    train_dict = {}

    for s, i in zip(train, range(1, len(train))):

        tabs = get_indices(s)
        quality = int(s[0])
        first = s[tabs[2] + 1:tabs[3]]
        second = s[tabs[3] + 1:]

        tokenizer = RegexpTokenizer(r'\w+')
        first_tokens = tokenizer.tokenize(first)
        second_tokens = tokenizer.tokenize(second)

        first_n_grams, second_n_grams = set_n_grams(first_tokens,
                                                    second_tokens, n)

        overlap = get_n_gram_overlap(first_n_grams, second_n_grams)

        train_dict[i] = {}
        train_dict[i]['quality'] = quality
        train_dict[i]['first'] = first
        train_dict[i]['second'] = second
        train_dict[i]['ngram overlap'] = overlap[0]
        train_dict[i]['overlap count'] = overlap[1]

    return train_dict

def create_doc2vec_data(s):
    data = create_training_data(s)

def get_average_overlap_proportion(train):

    prop_1 = 0
    prop_0 = 0
    num_1 = 0
    num_0 = 0

    for i in range(1, len(train)):
        if train[i]['quality'] == 1:
            denom = (len(train[i]['first']) + len(train[i]['second'])) / 2
            prop_1 += train[i]['overlap count'] / denom
            num_1 += 1
        else:
            denom = (len(train[i]['first']) + len(train[i]['second'])) / 2
            prop_0 += train[i]['overlap count'] / denom
            num_0 += 1

    return (prop_1 / num_1 + prop_0 / num_0) / 2


def set_new_quality(test, threshold):

    for i in range(1, len(test)):
        denom = (len(test[i]['first']) + len(test[i]['second'])) / 2
        prop = test[i]['overlap count'] / denom

        if prop >= threshold:
            test[i]['model quality'] = 1
        else:
            test[i]['model quality'] = 0

    return test


def get_true_pred(new_test):
    true = []
    pred = []

    for i in range(1, len(new_test)):
        true.append(new_test[i]['quality'])
        pred.append(new_test[i]['model quality'])

    return true, pred


def main():

    accuracy_a = []
    precision_a = []
    recall_a = []
    f1_a = []

    for i in range(1, 6):
        train = create_training_data("data/msr_paraphrase_train.txt", i)
        test = create_training_data("data/msr_paraphrase_test.txt", i)
        threshold = get_average_overlap_proportion(train)
        new_test = set_new_quality(test, threshold)
        true, pred = get_true_pred(new_test)
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)

        print("For ", i, "-gram overlap ", {"Accuracy": accuracy,
                                            "Precision": precision,
                                            "Recall": recall,
                                            "F1 Score": f1}, sep="")

        accuracy_a.append(accuracy)
        precision_a.append(precision)
        recall_a.append(recall)
        f1_a.append(f1)

    plt.figure()
    plt.plot(accuracy_a)
    plt.plot(precision_a)
    plt.plot(recall_a)
    plt.plot(f1_a)
    plt.xlabel("N-gram")
    plt.ylabel("Score")
    plt.show()


if __name__ == '__main__':
    main()
