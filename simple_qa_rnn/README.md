Setup:
1. Create 3 directories under "simple_qa_rnn" - "resources", "datasets", "saved_checkpoints" 
2. Download the SimpleQA dataset from here and put it under the "datasets" directory
3. Download these files from this [Dropbox link](https://www.dropbox.com/sh/e5g12v7zu7sgzf7/AACW272AqPZJIUC7-A40LAsNa?dl=0) and paste them in the "resources" directory
4. Run this command to train the model. Make sure you have PyTorch and other Python dependencies installed.
```
python train.py 
```
5. Please take a look at the arguments in utils.py and set them accordingly.