#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Run a simple inference on the Mallet topic model trained
"""

#######################################

from argparse import ArgumentParser
import logging, sys, codecs, csv, os, io, re

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

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
		if not isinstance(s, basestring,):
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
		elif not isinstance(s, unicode):
			# Note: We use .decode() here, instead of unicode(s, encoding,
			# errors), so that if s is a SafeString, it ends up being a
			# SafeUnicode at the end.
			s = s.decode(encoding, errors)
	except UnicodeDecodeError, e:
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
	with io.open(path,'rb') as raw:
		raw_review = [x.strip().split('\t') for x in raw]

	# Select full review only
	for i in range(len(raw_review)):
		raw_review[i] = raw_review[i][-1]
		raw_review[i] = re.sub(r"(\\u[0-z][0-z][0-z])\w", " ", raw_review[i])

	return raw_review

def writeSentence(sentenceBuf, path, name):
	"""
	Function to save sentence in a file buffer.

	Arguments
	---------
	sentenceBuf: sentence
	path: location of reviews
	name: Name of the file

	Return
	---------
	None
	"""

	# Write the preprocessed reviews to a SINGLE file unlike VanillaLDA/parse.py
	with io.open(path + name + ".txt", "a", encoding='utf8') as outfile:
		for i in range(len(sentenceBuf)):
			if sentenceBuf[i][0] == ' ':
				sentenceBuf[i] = sentenceBuf[i][1:]
			outfile.write(unicode(sentenceBuf[i] + "\n"))

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-m", "--model-path",
			help="Path to Mallet inferencer", type=str)
	parser.add_argument("-M", "--mallet-path",
			help="Path to mallet binary", type=str)
	parser.add_argument("-t", "--train-path",
			help="Path to mallet file used to originally train the model", type=str)
	parser.add_argument("-i", "--input-path",
			help="Path to reviews", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination of inferred topics", type=str)
	args = parser.parse_args()

	# Take reviews and split into sentences
	reviews = readReviews(args.input_path)
	sentences = splitsentence(reviews)
	writeSentence(sentences, "./", "sentence")

	os.system(args.mallet_path + " import-file --input sentence.txt --output new.mallet --remove-stopwords TRUE --keep-sequence TRUE --use-pipe-from " + args.train_path)

	os.system(args.mallet_path + " infer-topics --input new.mallet --inferencer " + args.model_path + " --output-doc-topics new-topic-composition.txt")

