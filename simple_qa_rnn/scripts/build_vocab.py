import string
import os
import glob
import nltk

def build_vocab_SQ(data_dir, lowercase=True):
    filepaths = glob.glob(os.path.join(data_dir, 'annotated_fb_data_.*.txt'))
    dst_path = os.path.join(data_dir, 'vocab.txt')
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                qText = line.split("\t")[3]
                # process text: remove punctuations, lowercase
                punc_remover = str.maketrans('', '', string.punctuation)
                processed_text = qText.lower().translate(punc_remover)
                tokens = nltk.word_tokenize(processed_text)
                vocab |= set(tokens)
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


print("WARNING: This script is dataset specific. Please change it to fit your own dataset.")
data_dir = 'data/SimpleQuestions_v2/'
build_vocab_SQ(data_dir)
