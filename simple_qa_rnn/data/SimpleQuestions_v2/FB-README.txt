The SimpleQuestions Dataset
--------------------------------------------------------
In this directory is the SimpleQuestions dataset collected for
research in automatic question answering.

** DATA **
SimpleQuestions is a dataset for simple QA, which consists
of a total of 108,442 questions written in natural language by human
English-speaking annotators each paired with a corresponding fact,
formatted as (subject, relationship, object), that provides the answer
but also a complete explanation.  Fast have been extracted from the
Knowledge Base Freebase (freebase.com).  We randomly shuffle these
questions and use 70\% of them (75910) as training set, 10\% as
validation set (10845), and the remaining 20\% as test set.

** FORMAT **
Data is organized in 3 files: annotated_fb_data_{train, valid, test}.txt .
Each file contains one example per line with the following format:
"Subject-entity [tab] relationship [tab] Object-entity [tab] question",
with Subject-entity, relationship and Object-entity being www links
pointing to the actual Freebase entities.

** DATA COLLECTION**
We collected SimpleQuestions in two phases.  The first phase consisted
of shortlisting the set of facts from Freebase to be annotated with
questions.  We used Freebase as background KB and removed all facts
with undefined relationship type i.e. containing the word
"freebase". We also removed all facts for which the (subject,
relationship) pair had more than a threshold number of objects. This
filtering step is crucial to remove facts which would result in
trivial uninformative questions, such as, "Name a person who is an
actor?". The threshold was set to 10.

In the second phase, these selected facts were sampled and delivered
to human annotators to generate questions from them. For the sampling,
each fact was associated with a probability which defined as a
function of its relationship frequency in the KB: to favor
variability, facts with relationship appearing more
frequently were given lower probabilities.  For each sampled facts,
annotators were shown the facts along with hyperlinks to
www.freebase.com to provide some context while framing the
question. Given this information, annotators were asked to phrase a
question involving the subject and the relationship
of the fact, with the answer being the object.  The
annotators were explicitly instructed to phrase the question
differently as much as possible, if they encounter multiple facts with
similar relationship.  They were also given the option of
skipping facts if they wish to do so.  This was very important to
avoid the annotators to write a boiler plate questions when they had
no background knowledge about some facts.

** LICENSE **
This data set is released under a Creative Commons v3.0 license. A
version of this license is included with the data set.

** CITING **
If you use this data set please cite the paper:
@article{BordesUCW15,
  author    = {Antoine Bordes and
               Nicolas Usunier and
               Sumit Chopra and
               Jason Weston},
  title     = {Large-scale Simple Question Answering with Memory Networks},
  journal   = {CoRR},
  volume    = {abs/1506.02075},
  year      = {2015},
  url       = {http://arxiv.org/abs/1506.02075}
}


** UPDATES **

- v2 of the data has been released in December 2015. It contains the
   subsets of Freebase used in conjunctions with SimpleQuestions in
   the paper "Large-scale Simple Question Answering with Memory
   Networks" (http://arxiv.org/abs/1506.02075).  There are 2 subsets
   (FB2M and FB5M) whose statistics are given in the paper. Each file
   is a text file with one fact per line. A fact if made of a
   Subject-entity, a relationship and a list of Object-entities
   connected to this subject by this relation type (the "grouped"
   setting of the paper). Members of triples are tab-separated,
   objects are space separated. As for SimpleQuestions,
   Subject-entities, relationships and Object-entities are www links
   pointing to the actual Freebase entities.
