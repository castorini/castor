# IDF scorer 

Implements a IDF baselines for QA datasets.

### Getting the data

Git clone [castorini/data](https://github.com/castorini/data) to get TrecQA and WikiQA datasets.

Follow instructions in ``TrecQA/README.txt`` and ``WikiQA/README.txt`` to process the data into a _standard_ format.

After running the respectve scripts, you should have the following directories structure in ``castorini/data/TrecQA``
```
├── raw-dev
├── raw-test
├── train
└── train-all
```

and, the following directories in ``castorini/data/WikiQA``.
```
├── dev
├── test
├── train
```

Each directory will have the following files:
``├── a.toks``: question[i]
``├── b.toks``: answer[i]
``├── id.txt``: question_id[i]
``└── sim.txt``: label[i]
where 1 <= i <= (number of QA pairs in respective splits of the data)
 

### Creating indexes for source corpora

We need to index the source corpus from which the question-answer pairs are derived in order to get the IDF weights of the terms.


#### 1. Clone and compile[Anserini](https://github.com/castorini/Anserini.git)

```
git clone https://github.com/castorini/Anserini.git
cd Anserini
mvn clean package appassembler:assemble
``` 
 
#### 2. Indexing WikiQA collection

First, download the Wikipedia dump by running the following command:

```
mkdir WikiQACollection
for line in $(cat idf_baseline/src/main/resources/WikiQA/wikidump-list.txt); do wget $line -P WikiQACollection; done
```

To index the collection:
```
cd Anserini
nohup sh target/appassembler/bin/IndexCollection -collection WikipediaCollection -input ../WikiQACollection
-generator JsoupGenerator -index lucene.index.wikipedia.pos.docvectors -threads 32 -storePositions 
-storeDocvectors -optimize > log.wikipedia.pos.docvectors & 
```

#### 3. Indexing TrecQA collection

Create a new directories called TrecQACollection
```
mkdir TrecQACollection
```

Copy the contents of disk1, disk2, disk3, disk4, and AQUAINT to TrecQACollection

To index the collection:

```
cd Anserini
nohup sh target/appassembler/bin/IndexCollection -collection TrecCollection -input [path of TrecQACollection]
-generator JsoupGenerator -index lucene.index.trecQA.pos.docvectors -threads 32 -storePositions 
-storeDocvectors -optimize > log.trecQA.pos.docvectors & 
```

### Computing the IDF overlap baseline

Build the IDF scorer
```
cd castorini/Castor/idf_baseline
mvn clean package appassembler:assemble
```

Run the following command to score each answer with an IDF value:

```
sh target/appassembler/bin/GetIDF -index ~/large-local-work/indices/index.wikipedia.pos.docvectors -config ../../data/WikiQA/test -output WikiQA.test.scores
```
The above command will create a run file in the `trec_eval` format and a qrel file
at a location specified by `-output`.



Possible parameters are:

```
-index (required)
```

Path of the index

```
-config (required)
```
Configuration of this experiment i.e., dev, train, train-all, test etc.

```
-output (required)
```
Path of the run file to be created

```
-analyze 
```
If specified, the scorer uses  `EnglishAnalyzer` for removing stopwords and performing stemming. In addition to 
the default list, the analyzer uses NLTK's stopword list obtained 
from[here](https://gist.github.com/sebleier/554280)


### Evaluating the system:

To calculate MAP/MRR for the above run file:

- Download and install `trec_eval` from[here](https://github.com/castorini/Anserini/blob/master/eval/trec_eval.9.0.tar.gz)

```
eval/trec_eval.9.0/trec_eval -m map -m recip_rank <qrel-file> <run-file>
```