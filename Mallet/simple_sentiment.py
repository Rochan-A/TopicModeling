#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Run a simple inference on the Mallet topic model trained
"""

#######################################

from argparse import ArgumentParser
import logging, sys, codecs, csv, os, io, re
from encoder import Model
import spacy, gensim

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

nlp = spacy.load('en_core_web_sm')

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

	filtered = []

	# POS Tagging and filtering sentences
	for i in range(len(sentReview)):
		doc = nlp(force_unicode(sentReview[i]))
		b = []
		for tok in doc:
			if tok.is_stop != True and tok.pos_ != 'SYM' and tok.tag_ != 'PRP' and tok.tag_ != 'PRP$' and tok.pos_ != 'NUM' and tok.dep_ != 'aux' and tok.dep_ != 'prep' and tok.dep_ != 'det' and tok.dep_ != 'cc' and len(tok) != 1:
				b.append(tok.lemma_)
		filtered.append(b)

	return tokens, filtered

def force_unicode(s, encoding='utf-8', errors='ignore'):
	"""
	Returns a unicode object representing 's'. Treats bytestrings using the
	'encoding' codec.

	Arguments
	---------
	s: string to be encoded
	encoding: encoding type, defaults to `utf-8`
	errors: whether or not to ignore errors, defaults to `ignore`

	Returns
	---------
	unicode string
	"""

	if s is None:
		return ''

	try:
		if not isinstance(s, str,):
			if hasattr(s, '__unicode__'):
				s = unicode(s)
			else:
				try:
					s = unicode(str(s), encoding, errors)
				except UnicodeEncodeError:
					if not isinstance(s, Exception):
						raise
					# If we get to here, the caller has passed in an Exception
					# subclass populated with non-ASCII data without special
					# handling to display as a string. We need to handle this
					# without raising a further exception. We do an
					# approximation to what the Exception's standard str()
					# output should be.
					s = ' '.join([force_unicode(arg, encoding, errors) for arg in s])
		elif not isinstance(s, str):
			# Note: We use .decode() here, instead of unicode(s, encoding,
			# errors), so that if s is a SafeString, it ends up being a
			# SafeUnicode at the end.
			s = s.decode(encoding, errors)
	except UnicodeDecodeError as e:
		if not isinstance(s, Exception):
			raise UnicodeDecodeError (s, *e.args)
		else:
			# If we get to here, the caller has passed in an Exception
			# subclass populated with non-ASCII bytestring data without a
			# working unicode method. Try to handle this without raising a
			# further exception by individually forcing the exception args
			# to unicode.
			s = ' '.join([force_unicode(arg, encoding, errors) for arg in s])
	return s

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
	with io.open(path,'r') as raw:
		for line in raw:
			reviewBuf.append(line)

	return reviewBuf

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to reviews", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination of inferred topics", type=str)
	args = parser.parse_args()

	# Take reviews and split into sentences
	reviews = readReviews(args.input_path)
	sentences = splitsentence(reviews)
	tok, sent = preprocess(sentences)

	for i in range(len(sentences)):
		model = Model()
		vec = 0
		for k in range(len(tok[i])):
			text_features = model.transform(tok[i][k])
			vec = vec + text_features[0][2388]
		print(vec, sentences[i])
		del model