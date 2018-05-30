#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to train LDA and LSA models
"""

#######################################

from nltk.stem.wordnet import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
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

def loadReviews(inp_path):
	"""
	Loads all the reviews into a list

	Arguments:
		inp_path -> location of parsed data

	Returns list of strings
	"""
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

def sent_to_words(sentences):
	"""
	Tokenizes the sentences/reviews

	Arguments:
		sentences -> list of strings in sentence structure

	Returns list of words
	"""
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
	"""
	Tokenizes, removes punctuation etc

	Arguments:
		texts -> 2 dim list of words

	Returns 2 dim list of words
	"""
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
	"""
	Bigrams are two words frequently occurring together in the document.
	Gensim’s Phrases model can build and implement the bigrams.
	"""
	return [bigram_mod[doc] for doc in texts]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=6):
	"""
	Compute c_v coherence for various number of topics

	Arguments:
		dictionary -> Gensim dictionary
		corpus -> Gensim corpus
		texts -> List of input texts
		limit -> Max num of topics

	Returns:
		model_list -> List of LDA topic models
		coherence_values -> Coherence values corresponding to the LDA model with respective number of topics
	"""
	coherence_values = []
	model_list = []
	for num_topics in range(start, limit, step):
		model = gensim.models.ldamodel.LdaModel(corpus=corpus,
						id2word=id2word,
						num_topics=num_topics,
						random_state=100,
						update_every=1,
						chunksize=100,
						passes=10,
						alpha='auto',
						per_word_topics=True)
		model_list.append(model)
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
		coherence_values.append(coherencemodel.get_coherence())

	return model_list, coherence_values

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
	data_lemmatized = []
	for i in range(len(data_w_bigrams_unicode)):
		k = []
		for j in range(len(data_w_bigrams_unicode[i])):
			k.append(lmtzr.lemmatize(data_w_bigrams_unicode[i][j]))

		data_lemmatized.append(k)

	# Create Dictionary
	id2word = corpora.Dictionary(data_lemmatized)

	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in data_lemmatized]

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

	# Compute Coherence Score
	coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)

	model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=3)

	# Print the coherence scores
	for m, cv in zip(x, coherence_values):
		print("Num Topics =", m, " has Coherence Value of", round(cv, 4))