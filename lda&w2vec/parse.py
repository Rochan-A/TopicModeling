#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This script is to be used to preprocess all hotel review data before
	passing on to lda model. Trains word2vec model.
"""

#######################################

from argparse import ArgumentParser
import spacy, re, logging, gensim

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nlp = spacy.load('en_core_web_sm')

def checkNotDuplicate(reviewBuf, inLine):
	"""
	Check if the review is an duplicate or not

	Arguments
	---------
	reviewBuf: list of strings already tested
	inLine: string to test

	Return
	---------
	False: Not duplicate
	True: duplicate
	"""
	for i in range(len(reviewBuf)):
		if reviewBuf[i] == inLine:
			return False
			break
	return True

def splitsentence(reviewBuf):
	"""
	Splits each review into its individual sentences.

	Arguments
	---------
	reviewBuf: List of reviews (strings)

	Return
	---------
	List of sentences.
	"""

	new = []
	# Iterate over every unique review
	for i in range(len(reviewBuf)):
		# Split the sentences
		sentences = reviewBuf[i].split('.')

		# Append the other sentences to the end of the reviewBuf
		for j in range(len(sentences)):
			# Make sure the sentence has more than two words
			if len(sentences[j]) > 2:
				new.append(sentences[j])

	return new

def readReviews(path):
	"""
	Function to store reviews in a list buffer.

	Arguments
	---------
	path: location of reviews

	Return
	---------
	list of reviews
	"""

	# Create an empty buffer to hold all the reviews
	reviewBuf = []

	# Open the file
	with open(path,'r', encoding='utf8') as raw:
		raw_review = [x.strip().split('\t') for x in raw]

	# Select full review only
	for i in range(len(raw_review)):
		raw_review[i] = raw_review[i][-1]
		raw_review[i] = re.sub(r"(\\u[0-z][0-z][0-z])\w", " ", raw_review[i])

	return raw_review

def writeProcessed(reviewBuf, path):
	"""
	Function to save reviews in a list buffer.

	Arguments
	---------
	reviewBuf: parsed, preprocessed reviews
	path: location of reviews

	Return
	---------
	None
	"""

	# Write the preprocessed reviews to a SINGLE file unlike VanillaLDA/parse.py
	with io.open(outFile + "out.txt", "a", encoding='utf8') as outfile:
		for i in range(len(reviewBuf)):
			outfile.write(reviewBuf[i] + "\n")

def preprocess(sentReview):
	"""
	Processes sentences before passing on to train models

	Arguments
	---------
	sentReview: List of reviews split into sentences

	Returns
	---------
	tokens: tokenized, de-accent and lowercased word list
	filtered: filtered numbers, symbols, stopwords etc, list of words
	"""
	# Simple tokens, de-accent and lowercase processor
	tokens = []
	for i in range(len(sentReview)):
		tokens.append(gensim.utils.simple_preprocess(sentReview[i], deacc=True, min_len=3))

	filtered = tokens

	# POS Tagging and filtering sentences
	for i in range(len(sentReview)):
		doc = nlp(unicode(sentReview[i]))
		for tok in doc:
			if tok.is_stop == True or tok.pos_ == 'SYM' or tok.pos_ == 'NUM':
				if filtered[i].__contains__(tok.text):
					filtered[i].remove(tok.text)

	return tokens, filtered

def trainW2V(tokens, outPath):
	"""
	Train word2vec model

	Arguments
	---------
	tokens: list of word tokens
	outPath: Location to save model

	Returns
	---------
	None
	"""
	# build vocabulary
	model = gensim.models.Word2Vec(
		tokens,
		size=150,
		window=10,
		min_count=2,
		workers=10)

	# Train the model
	model.train(tokens, total_examples=len(tokens), epochs=10)

	# Save the model
	model.save(fname=outPath + "w2v")

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to reviews", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination for parsed and preprocessed output", type=str)
	args = parser.parse_args()

	raw_reviews = readReviews(args.input_path)

	sentence = splitsentence(raw_reviews)

	tokens, filtered = preprocess(sentence)

	writeProcessed(tokens, args.output_path)