import sys
import os
import glob

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

try:
    data_dir = sys.argv[1]
except:
    print("ERROR: the command line arguments passed in were not valid.\n");
    print("USAGE: python scripts/build_vocab.py [data_dir]");
    print("EXAMPLE: python scripts/build_vocab.py data/SimpleQuestions_v2/");
    sys.exit(1);

data_dir = 'data/SimpleQuestions_v2/'
build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab.txt'), False)
