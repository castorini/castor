import torch
import torch.nn as nn
import time
import os
import numpy as np

from torchtext import data
from args import get_args
from model import SmPlusPlus

import random

from trec_dataset import TrecDataset
from evaluate import evaluate

def set_vectors(field, vector_path):
    if os.path.isfile(vector_path):
        stoi, vectors, dim = torch.load(vector_path)
        field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

        for i, token in enumerate(field.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                field.vocab.vectors[i] = vectors[wv_index]
            else:
                # np.random.uniform(-0.25, 0.25, vec_dim)
                field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
    else:
        print("Error: Need word embedding pt file")
        exit(1)
    return field

# Set default configuration in : args.py
args = get_args()
config = args

# Set random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

QID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
EXTERNAL = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
train, dev, test = TrecDataset.splits(QID, QUESTION, ANSWER, EXTERNAL, LABEL)

QUESTION.build_vocab(train, min_freq=2)
ANSWER.build_vocab(train, min_freq=2)
EXTERNAL.build_vocab(train, min_freq=2)
LABEL.build_vocab(train)


QUESTION = set_vectors(QUESTION, args.vector_cache)
ANSWER = set_vectors(ANSWER, args.vector_cache)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)

#print(config)
print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num",len(QUESTION.vocab))
print("LABEL.target_class:", len(LABEL.vocab))
print("LABELS:",LABEL.vocab.itos)
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))


if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = SmPlusPlus(config)
    print(QUESTION.vocab.vectors.size())
    model.static_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.nonstatic_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.static_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)
    model.nonstatic_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)

    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
early_stop = False
best_dev_map = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)

index2label = np.array(LABEL.vocab.itos)
index2qid = np.array(QID.vocab.itos)

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_map))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    instance = []

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        model.train(); optimizer.zero_grad()
        scores = model(batch)
        n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        # shouldn't these be after the early stop?
        loss = criterion(scores, batch.label)
        loss.backward()

        optimizer.step()


        # Evaluate performance on validation set
        if iterations % args.dev_every == 1:
            # switch model into evalutaion mode
            model.eval(); dev_iter.init_epoch()
            n_dev_correct = 0
            dev_losses = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                qid_array = index2qid[np.transpose(batch.qid.cpu().data.numpy())]
                true_label_array = index2label[np.transpose(batch.label.cpu().data.numpy())]
                scores = model(dev_batch)
                n_dev_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
                dev_loss = criterion(scores, dev_batch.label)

                dev_losses.append(dev_loss.data[0])
                index_label = np.transpose(torch.max(scores, 1)[1].view(batch.label.size()).cpu().data.numpy())
                label_array = index2label[index_label]

                # print and write the result
                for i in range(batch.batch_size):
                    this_qid, predicted_label, gold_label = qid_array[i], label_array[i], true_label_array[i]
                    instance.append((this_qid, predicted_label, gold_label))


            dev_map, dev_mrr = evaluate(instance)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          sum(dev_losses) / len(dev_losses), train_acc, dev_map))

            # Update validation results
            if dev_map > best_dev_map:
                iters_not_improved = 0
                best_dev_map = dev_map
                snapshot_path = os.path.join(args.save_path, args.dataset, args.mode+'_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))
