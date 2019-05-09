import argparse
import numpy as np

def getindices(s):
    return [i for i, c in enumerate(s) if c == "\t"]

def create_training_data(s):
    train_file = open("data/msr_paraphrase_train.txt")
    train = train_file.read().split('\n')
    train = train[1:]
    print(train[len(train)-1])

    train_dict = {}

    for s, i in zip(train, range(1, len(train))):

        tabs = getindices(s)
        quality = s[0]
        first = s[tabs[2]+1:tabs[3]]
        second = s[tabs[3]+1:]
        train_dict[i] = {}
        train_dict[i]['quality'] = quality
        train_dict[i]['first'] = first
        train_dict[i]['second'] = second

    return train_dict
    
def main():
    train = create_training_data("data/msr_paraphrase_train.txt")


if __name__ == '__main__':
    main()
