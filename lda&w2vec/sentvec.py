#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This script is to be used to create document bins for all sentences
"""

#######################################

from argparse import ArgumentParser
import spacy, re, logging, gensim, io, sys, codecs
from gensim.models import Doc2Vec
from collections import namedtuple
from sklearn.metrics import pairwise
from scipy import spatial

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
reload(sys)
sys.setdefaultencoding('utf8')

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

def readSentences(path):
	"""
	Function to store sentences in a list buffer.

	Arguments
	---------
	path: location of sentence data file

	Return
	---------
	list of sentences
	"""

	# Create an empty buffer to hold all the sentences
	sentenceBuf = []

	# Open the file
	with codecs.open(path, 'r') as data:
		for line in data:
			sentenceBuf.append(line[:-1])

	return sentenceBuf

def readTokens(path):
	"""
	Function to store tokens.txt in a list buffer.

	Arguments
	---------
	path: location of tokens.txt

	Return
	---------
	list of tokens
	"""

	# Create an empty buffer to hold all the sentences
	tokensBuf = []

	# Open the file
	with codecs.open(path, 'r') as data:
		for line in data:
			tokensBuf.append(line)

	return tokensBuf

def writeProcessed(reviewBuf, path, name):
	"""
	Function to save reviews in a file buffer.

	Arguments
	---------
	reviewBuf: parsed, preprocessed reviews
	path: location of reviews
	name: Name of the file

	Return
	---------
	None
	"""

	# Write the preprocessed reviews to a SINGLE file unlike VanillaLDA/parse.py
	with io.open(path + name + ".txt", "a", encoding='utf8') as outfile:
		for i in range(len(reviewBuf)):
			outfile.write(unicode(','.join(reviewBuf[i]) + "\n"))

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to reviews", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination for parsed and preprocessed output", type=str)
	parser.add_argument("-t", "--type",
			help="File type:\n1.\tsentences.txt\n2.\ttokens.txt\n3.\tfiltered.txt", type=int)
	args = parser.parse_args()

	if args.type == 1:
		# Open sentences.txt data file
		doc = readSentences(args.input_path)
	elif args.type == 2 or args.type == 3:
		# Open tokens.txt data file
		doc = readTokens(args.input_path)
	else:
		print("Missing argument")
		quit()

	# Transform data (you can add more data preprocessing steps)
	docs = []

	analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
	for i, text in enumerate(doc):
		if args.type == 1
			words = text.lower().split()
		else:
			words = text.split(',')
		tags = [i]
		docs.append(analyzedDocument(words, tags))

	# Train model (set min_count = 1, if you want the model to work with the provided example data set)
	model = Doc2Vec(docs, vector_size = 200, window = 300, min_count = 1, workers = 3)

	model.save(args.output_path + "doc2vecmodel")

	# Get the vectors
	for i in range(len(doc)):
		sen1 = model.docvecs[i].reshape(-1, 1)
		sen2 = model.docvecs[i].reshape(-1, 1)
		#print(model.docvecs[1])
		print(1 - spatial.distance.cosine(sen1, sen2))

	#writeProcessed(filtered, args.output_path, "filtered")
	#writeSentence(sentence, args.output_path, "sentences")