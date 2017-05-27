import sys
import os
from tqdm import tqdm
import torch
import array
import six

try:
    path = sys.argv[1]
    outpath = sys.argv[2]
except:
    print("ERROR: the command line arguments passed in were not valid.\n");
    print("USAGE: python scripts/convert_wordvecs.py [input_file] [output_file]");
    print("EXAMPLE: python scripts/convert_wordvecs.py glove_300d.txt glove_300d.pt");
    sys.exit(1);

prefix_toks = path.split(".")
print('Converting ' + path + ' to PyTorch serialized format...')

lines = [line.rstrip('\n') for line in open(path)]
wv_tokens = []
wv_arr = array.array('d')
wv_size = None # dimension of the word vectors
if lines is not None:
    for i in tqdm(range(len(lines)), desc="loading word vectors from {}".format(path)):
        entries = lines[i].strip().split(" ")
        word, entries = entries[0], entries[1:]
        if wv_size is None:
            wv_size = len(entries)
        try:
            if isinstance(word, six.binary_type):
                word = word.decode('utf-8')
        except:
            print('non-UTF8 token', repr(word), 'ignored')
            continue
        wv_arr.extend(float(x) for x in entries)
        wv_tokens.append(word)

wv_dict = {word: i for i, word in enumerate(wv_tokens)} # word to index dictionary
wv_arr = torch.Tensor(wv_arr).view(-1, wv_size) # word embeddings in Tensor of shape (|V|, |D|)
ret = (wv_dict, wv_arr, wv_size) # save all three info in a tuple
print(ret)
print("saving word vectors to {}".format(outpath))
torch.save(ret, outpath + '.pt')

