# ParaphraseDetection

CS: 322, Natural Language Processing, *Professor Jack Hessel*

All classifiers train and test on the MSRP training and test sets, respectively found in the 'data' directory. 

Run the baseline (classifies data using n-gram overlap) with:

```python
python baseline.py
```

Running this command will print and plot the accuracy, precision, recall, and f1 score of the baseline model.

Run the doc2vec classifiers with:

```python
python d2v_test.py
```

Running this command will print the accuracy, precision, and f1 score of the doc2vec based models.

To run the BERT-embeddings based classifier, it is necessary to download a pre-trained BERT model eg. BERT-Large, Uncased), and then use BERT to extract fixed feature vectors for the sentences in MSRP training and test data. More information available from the [BERT github page](https://github.com/google-research/bert). Then run the classifier using:

```python
bert_detection.py [-h] [--train TRAIN] [--test TEST]
```

optional arguments:
  -h, --help     show this help message and exit
  --train TRAIN  path to output of BERT extract features script with MSRP
                 training data
  --test TEST    path to output of BERT extract features script with MSRP test
                 data

