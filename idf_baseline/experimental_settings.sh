#!/bin/bash
# The following 5 conditions vary
# idf_source, stopwords_and_stemming, punctuation, words-with-hyphens

echo "uncomment cmd that you want to run"

# setup experimental settings ---------
# import itertools
# idf_sets = ['idf_source:qa-data', 'idf_source:corpus-index']
# stop_stem_sets = ['stop_stem:yes', 'stop_stem:no']
# punc_sets = ['punctuation:keep', 'punctuation:remove']
# dash_sets = ['dash_words:keep', 'dash_words:split']

# for c in itertools.product(idf_sets, stop_stem_sets, punc_sets, dash_sets):    
#     print(c)

runeval()
{
    echo "$1"
    eval "$1"
    for split in train-all raw-dev raw-test; 
    do 
        ../sm_model/trec_eval-8.0/trec_eval ../../data/TrecQA/$split.qrel run.$split.idfsim | grep map; 
    done
}

# ('idf_source:qa-data', 'stop_stem:yes', 'punctuation:keep', 'dash_words:keep')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --stop-and-stem"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:yes', 'punctuation:keep', 'dash_words:split')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --stop-and-stem --dash-split"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:yes', 'punctuation:remove', 'dash_words:keep')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --stop-and-stem --stop-punct"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:yes', 'punctuation:remove', 'dash_words:split')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --stop-and-stem --stop-punct --dash-split"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:no', 'punctuation:keep', 'dash_words:keep')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:no', 'punctuation:keep', 'dash_words:split')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --dash-split"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:no', 'punctuation:remove', 'dash_words:keep')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --stop-punct"
#runeval "$cmd"

# ('idf_source:qa-data', 'stop_stem:no', 'punctuation:remove', 'dash_words:split')
#cmd="python qa-data-only-idf.py ../../data/TrecQA run --stop-punct --dash-split"
#runeval "$cmd"

# # ('idf_source:corpus-index', 'stop_stem:yes', 'punctuation:keep', 'dash_words:keep')
# cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --stop-and-stem"
# runeval "$cmd"

# # ('idf_source:corpus-index', 'stop_stem:yes', 'punctuation:keep', 'dash_words:split')
# cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --stop-and-stem --dash-split"
# runeval "$cmd"

# # ('idf_source:corpus-index', 'stop_stem:yes', 'punctuation:remove', 'dash_words:keep')
# cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --stop-and-stem --stop-punct"
# runeval "$cmd"

# # ('idf_source:corpus-index', 'stop_stem:yes', 'punctuation:remove', 'dash_words:split')
# cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --stop-and-stem --stop-punct --dash-split"
# runeval "$cmd"

# # ('idf_source:corpus-index', 'stop_stem:no', 'punctuation:keep', 'dash_words:keep')
# cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/"
# runeval "$cmd"

# # ('idf_source:corpus-index', 'stop_stem:no', 'punctuation:keep', 'dash_words:split')
# cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --dash-split"
# runeval "$cmd"

# ('idf_source:corpus-index', 'stop_stem:no', 'punctuation:remove', 'dash_words:keep')
cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --stop-punct"
runeval "$cmd"

# ('idf_source:corpus-index', 'stop_stem:no', 'punctuation:remove', 'dash_words:split')
cmd="python qa-data-only-idf.py ../../data/TrecQA run --index ../../data/indices/index.qadata.pos.docvectors.keepstopwords/ --stop-punct --dash-split"
runeval "$cmd"