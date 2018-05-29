#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to train LDA and LSA models
"""

#######################################
from argparse import ArgumentParser
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim, os, codecs

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

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
	with codecs.open(currFileList[i], "r", encoding='utf-8') as f:
		for line in f:
			doc.append(line)
	return doc

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

	tokenizer = RegexpTokenizer(r'\w+')

	# create English stop words list
	en_stop = get_stop_words('en')

	# Create p_stemmer of class PorterStemmer
	p_stemmer = PorterStemmer()

	# list for tokenized documents in loop
	texts = []

	# loop through document list
	for i in doc_set:

		# clean and tokenize document string
		raw = i.lower()
		tokens = tokenizer.tokenize(raw)

		# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in en_stop]

		# stem tokens
		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

		# add tokens to list
		texts.append(stemmed_tokens)

	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)

	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]

	# generate LDA model
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=20)

	# generate LSA model
	lsamodel = gensim.models.lsimodel.LsiModel(corpus, num_topics=100, id2word = dictionary)

	# Save the models
	ldamodel.save(args.output_path + "LDA")
	lsamodel.save(args.output_path + "LSA")