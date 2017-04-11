# IDF scorer 

[comment]: <> (Update this to a new location once the data is uploaded)
Download the TrecQA, WikiQA, and SQuAD data from 
[here](https://github.com/gauravbaruah/finding-nugget-representations/tree/master/data/TrecQA)

After downloading the data you should have the following directory 
structure:

```
├── clean-dev
├── clean-test
├── raw-dev
├── raw-test
├── train
└── train-all
```
and each directory should have the following files:
```
├── a.toks
├── b.toks
├── boundary.txt
├── id.txt
├── numrels.txt
└── sim.txt
```

Run the following command to score each answer with an IDF value:

```
sh target/appassembler/bin/GetIDF
```

Possible parameters are:

```
-index (required)
```

Path of the index

```
-config (requiered)
```
Configuration of this experiment i.e., dev, train, train-all, test etc.

```
-output (optional: file path)
```

Path of the run file to be created

```
-analyze 
```
If specified, the scorer uses  `EnglishAnalyzer` for removing stopwords and stemming. In addtion to 
the default list, the analyzer uses NLTK's stopword list obtained 
from[here](https://gist.github.com/sebleier/554280)

The above command will create a run file in the `trec_eval` format and a qrel file
at a location specified by `-output`.

### Evaluating the system:

To calculate MAP/MRR for the above run file:

- Download and install `trec_eval` from[here](https://github.com/castorini/Anserini/blob/master/eval/trec_eval.9.0.tar.gz)

```
eval/trec_eval.9.0/trec_eval -m map -m recip_rank <qrel-file> <run-file>
```