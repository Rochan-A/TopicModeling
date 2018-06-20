# TopicModel

This repository contains scripts to train Topic Models on [Hotel Reviews](https://github.com/kavgan/data-science-tutorials/blob/master/word2vec/reviews_data.txt.gz).  
This work is done under the guidance of [@Anupam Mediratta](https://github.com/anupamme) as an Intern.  

## Repo Structure

Each folder is named according to the training method or library used.  

|Folder| Description|
|---|---|
|[VanillaLDA](https://radimrehurek.com/gensim/models/ldamodel.html)| Gensim LDA implementation|
|[Mallet](http://mallet.cs.umass.edu/index.php)| MAchine Learning for LanguagE Toolkit|
|[lda2vec Original](https://github.com/cemoody/lda2vec/), [lda2vec used](https://github.com/Rochan-A/lda2vec)| Mix the best parts of word2vec and LDA into a single framework|
|depriciated| Gensim LDA and LSI|
|lda&w2vec| Incomplete custom training algorithm|