## SM model 

#### References:
1. Aliaksei _S_everyn and Alessandro _M_oschitti. 2015. Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '15). ACM, New York, NY, USA, 373-382. DOI: http://dx.doi.org/10.1145/2766462.2767738


#### TODOs:
1. figure out if the L2 regularization is correct
2. Batch size of 50 (current batch_size = 1)


#### Getting the data

git clone [lintool/Castor-data](https://github.com/lintool/Castor-data)

Castor-data contains:

```word2vec-models/aquaint*```: word embeddings.
Note that a memory mapped cache will be created on first use on your disk.

```TrecQA/```: the directory with the input data for training the model.


#### Running it

``1.`` Make TrecEval:
```
$ cd trec_eval-8.0
$ make clean
$ make
```

``2.`` Get the Overlapping features for Q and A:
```
$ python overlap_features.py ../../Castor-data/TrecQA
$ python overlap_features.py ../../Castor-data/TrecQA --train_all
```

``3.`` To run the S&M model on TrecQA, please follow the same parameter setting:
```
$ python main.py ../../Castor-data/word2vec-models/aquaint+wiki.txt.gz.ndim\=50.bin ../../Castor-data/TrecQA/ ../../Castor-data/TrecQA/sm.model
```

Run ```python main.py -h``` for more options.

