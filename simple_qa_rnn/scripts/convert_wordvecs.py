import sys
import os
from tqdm import tqdm
import torch
import array
import six

try:
    path = sys.argv[1]
    vocabpath = sys.argv[2]
    outpath = sys.argv[3]
except:
    print("ERROR: the command line arguments passed in were not valid.\n");
    print("USAGE: python scripts/convert_wordvecs.py [input_file] [output_file]");
    print("EXAMPLE: python scripts/convert_wordvecs.py glove_300d.txt glove_300d.pt");
    sys.exit(1);

prefix_toks = path.split(".")
print('Converting ' + path + ' to PyTorch serialized format...')

lines = [line.rstrip('\n') for line in open(path)]
print("number of lines: {}".format(len(lines)))

wv_tokens = []
wv_arr = array.array('d')
wv_size = None # dimension of the word vectors
vocab_size = 0 # counts the number of words saved
vocabfile = open(vocabpath, 'w')
if lines is not None:
    for i in tqdm(range(len(lines)), desc="loading word vectors from {}".format(path)):
        entries = lines[i].strip().split()
        word, entries = entries[0], entries[1:]
        if wv_size is None:
            wv_size = len(entries)
        else:
            # safety check that the dimension is the same
            if len(entries) != wv_size:
                continue
        try:
            if isinstance(word, six.binary_type):
                word = word.decode('utf-8')
        except:
            print('non-UTF8 token', repr(word), 'ignored')
            continue
        wv_arr.extend(float(x) for x in entries)
        wv_tokens.append(word)
        vocabfile.write(word + "\n")
        vocab_size += 1

vocabfile.close()
print("vocab size: {}".format(vocab_size))
print("dim: {}".format(wv_size))

wv_dict = {word: i for i, word in enumerate(wv_tokens)} # word to index dictionary
wv_arr = torch.Tensor(wv_arr).view(vocab_size, wv_size) # word embeddings in Tensor of shape (|V|, |D|)
ret = (wv_dict, wv_arr, wv_size) # save all three info in a tuple

print("saving word vectors to {}".format(outpath))
torch.save(ret, outpath)

