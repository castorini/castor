#!/bin/bash

# download dataset and put it in data directory
mkdir data
pushd data

echo "Downloading SimpleQuestions dataset...\n"
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz

echo "\n\nUnzipping SimpleQuestions dataset...\n"
tar -xvzf SimpleQuestions_v2.tgz

echo "\n\nDownloading the names file...\n"
wget https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt

popd

echo "\n\nTrimming the names file for subset 2M...\n"
python scripts/trim_names.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt -n data/FB5M.name.txt -o data/names.trimmed.2M.txt

echo "\n\nTrimming the names file for subset 5M...\n"
python scripts/trim_names.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt -n data/FB5M.name.txt -o data/names.trimmed.5M.txt

echo "\n\nCreate modified dataset...\n"
python scripts/modify_dataset.py -d data/SimpleQuestions_v2 -o data/SimpleQuestions_v2_modified

echo "\n\nDONE!"