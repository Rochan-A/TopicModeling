#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to train LDA and LSA models
"""

#######################################

from nltk.stem.wordnet import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import gensim

# Enable logging for gensim - optional
import logging

# NLTK Stop words
from nltk.corpus import stopwords

from argparse import ArgumentParser
import os, codecs, re

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
stop_words = stopwords.words('english')
stop_words.extend(['hi', 'hello', 'hey', 'thanks', 'thank', 'you', 'regards'])

"""
	Loads all the reviews into a list
"""
def loadReviews(inp_path):
	# Get the list of files
	os.chdir(inp_path)
	currFileList = [f for f in os.listdir('.') if os.path.isfile(f)]

	# Load the lines
	doc = []
	for i in range(len(currFileList)):
		with codecs.open(currFileList[i], "r", encoding='utf8') as f:
			for line in f:
				doc.append(line.encode("ascii","ignore"))
	return doc

"""
	Tokenizes the sentences/reviews
"""
def sent_to_words(sentences):
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

"""
	Define functions for stopwords, bigrams, trigrams and lemmatization
"""
def remove_stopwords(texts):
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

"""
	Bigrams are two words frequently occurring together in the document.
	Gensim’s Phrases model can build and implement the bigrams.
"""
def make_bigrams(texts):
	return [bigram_mod[doc] for doc in texts]

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to parsed reviews", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination of to save model", type=str)
	args = parser.parse_args()

	# Load reviews into a list
	doc_set = loadReviews(args.input_path)

	data_words = list(sent_to_words(doc_set))

	# Build the bigram models
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

	# Faster way to get a sentence clubbed as a bigram
	bigram_mod = gensim.models.phrases.Phraser(bigram)

	# Remove Stop Words
	data_words_nostops = remove_stopwords(data_words)

	# Form Bigrams
	data_words_bigrams = make_bigrams(data_words_nostops)

	# Make sure words are in ascii form
	data_w_bigrams_unicode = []
	for i in range(len(data_words_bigrams)):
		r =[]
		for j in range(len(data_words_bigrams[i])):
			r.append(unicode(data_words_bigrams[i][j].encode('ascii','ignore')))
		data_w_bigrams_unicode.append(r)

	# Initialize word Lemmatizer
	lmtzr = WordNetLemmatizer()

	# Iterate over the data and Lemmatize each word
	lemm = []
	for i in range(len(data_w_bigrams_unicode)):
		k = []
		for j in range(len(data_w_bigrams_unicode[i])):
			k.append(lmtzr.lemmatize(data_w_bigrams_unicode[i][j]))

		lemm.append(k)

	# Create Dictionary
	id2word = corpora.Dictionary(lemm)

	# Create Corpus
	texts = lemm

	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in texts]

	# Build LDA model
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
						id2word=id2word,
						num_topics=20,
						random_state=100,
						update_every=1,
						chunksize=100,
						passes=10,
						alpha='auto',
						per_word_topics=True)

	# Print the Keyword in the 10 topics
	print(lda_model.print_topics())