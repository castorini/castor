# kim_cnn

Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch and Torchtext.

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.# text-classification-cnn
Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch.

## Quick Start

To run the model on SST-1 dataset on multichannel, just run the following code.

```
python train.py --mode multichannel
```

The file will be saved in 

```
saves/best_model.pt
```

To test the model, you can use the following command.

```
python main.py --trained_model saves/best_model.pt --mode multichannel
```

## Dataset

We experiment the model on the following three datasets.

- SST-1: Keep the original splits and train with phrase level dataset and test on sentence level dataset.

## Settings

Adadelta is used for training. 

## Training Time

For training time, when

```
torch.backends.cudnn.deterministic = True
```

is specified, the training will be ~3h because deterministic cnn algorithm is used (accuracy v.s. speed).

Other option is that

```
torch.backends.cudnn.enabled = False
```
but this will take ~6-7x training time.

## Results

Deterministic Algorithm for CNN.  

| Dev Accuracy on SST-1 |     rand      |    static    |   non-static  |  multichannel | 
|:--------------------------:|:-----------:|:-----------:|:-------------:|:---------------:| 
| My-Implementation      | 42.597639| 48.773842| 48.864668   | 49.046322  |  

| Test Accuracy on SST-1|      rand      |    static    |    non-static |  multichannel | 
|:--------------------------:|:-----------:|:-----------:|:-------------:|:---------------:| 
| Kim-Implementation    | 45.0            | 45.5        | 48.0             | 47.4                 | 
| My- Implementation    | 39.683258  | 45.972851| 48.914027|  47.330317       |

## TODO

- More experiments on SST-2 and subjectivity
- Parameters tuning

