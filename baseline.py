import argparse
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer

def get_indices(s):
    return [i for i, c in enumerate(s) if c == "\t"]

def get_bigram_overlap(first, second):
    overlap = set(first) & set(second)
    return([overlap, len(overlap)])

def create_training_data(s):
    train_file = open(s)
    train = train_file.read().split('\n')
    train = train[1:]
    print(train[len(train)-1])

    train_dict = {}

    for s, i in zip(train, range(1, len(train))):

        tabs = get_indices(s)
        quality = int(s[0])
        first = s[tabs[2]+1:tabs[3]]
        second = s[tabs[3]+1:]

        tokenizer = RegexpTokenizer(r'\w+')
        first_tokens = tokenizer.tokenize(first)
        second_tokens = tokenizer.tokenize(second)

        first_bigrams = list(nltk.bigrams(first_tokens))
        second_bigrams = list(nltk.bigrams(second_tokens))

        overlap = get_bigram_overlap(first_bigrams, second_bigrams)

        train_dict[i] = {}
        train_dict[i]['quality'] = quality
        train_dict[i]['first'] = first
        train_dict[i]['second'] = second
        train_dict[i]['bigram overlap'] = overlap[0]
        train_dict[i]['overlap count'] = overlap[1]

    return train_dict

def get_average_overlap_proportion(train):

    prop_1 = 0
    prop_0 = 0
    num_1 = 0
    num_0 = 0

    for i in range(1, len(train)):
        if (train[i]['quality'] == 1):
            denom = (len(train[i]['first']) + len(train[i]['second']))/2
            prop_1 += train[i]['overlap count']/denom
            num_1 += 1
        else:
            denom = (len(train[i]['first']) + len(train[i]['second']))/2
            prop_0 += train[i]['overlap count']/denom
            num_0 += 1

    return (prop_1/num_1 + prop_0/num_0)/2

def set_new_quality(test, threshold):

    for i in range(1, len(test)):
        denom = (len(test[i]['first']) + len(test[i]['second']))/2
        prop = test[i]['overlap count']/denom

        if (prop >= threshold):
            test[i]['model quality'] = 1
        else:
            test[i]['model quality'] = 0

def main():
    train = create_training_data("data/msr_paraphrase_train.txt")
    test= create_training_data("data/msr_paraphrase_test.txt")
    threshold = get_average_overlap_proportion(train)
    set_new_quality(test, threshold)


if __name__ == '__main__':
    main()
