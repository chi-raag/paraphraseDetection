import argparse
import numpy as np


def get_indices_period(s):
    return [i for i, c in enumerate(s) if c == "."]

def getindices(s):
    return [i for i, c in enumerate(s) if c.isupper()]

def main():
    train_file = open("data/msr_paraphrase_train.txt")
    train = train_file.read().split('\n')
    train = train[1:]

    train_dict = {}

    for s, i in zip(train, range(1, len(train))):
        upper = getindices(s)
        period = get_indices_period(s)
        quality = s[0]
        first = s[15:period[0]]
        second = s[period[0]+1:]
        train_dict[i] = {}
        train_dict[i]['quality'] = quality
        train_dict[i]['first'] = first
        train_dict[i]['second'] = second

if __name__ == '__main__':
    main()
