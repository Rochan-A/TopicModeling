#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to calcute the term/word weights per topic from an MALLET model
"""

#######################################

import codecs
from argparse import ArgumentParser
import operator
import csv

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "1.0"
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

def open_count(name):
	"""
	Open the word count file

	Arguments
	---------
	name: full path to the word count file

	Returns
	---------
	list of word frequencies
	"""

	"""
	Each line looks like something below:

	<word-number> <word> <topic-number>:<number-of-word-occurrence's> ....
	3 half-hour 41:11 55:10 98:9 68:8 16:8 69:6 56:6 61:5 0:5 46:2
	"""

	h = []
	with codecs.open(name, 'r', encoding='utf8') as F:
		for line in F:
			# Split each line at the spaces
			space_split = line.split(' ')
			# Remove <word-number>
			word_topic_number = space_split[1:]
			h.append(word_topic_number)

	return h

def parseTopicwise(word_topic_number, num):
	"""
	Parse the list

	Arguments
	---------
	word_topic_number: list of list of word and its frequency in a topic
	num: number of topics

	Return
	---------
	list of dictionary for each topic with frequency of each word.
	Output in this format: [{'word':count, ...}, ...]
	"""

	# Create empty dictionary to store the word counts
	topics = []
	for i in range(num):
		topics.append(dict())

	# Iterate over every word
	for i in range(len(word_topic_number)):
		for j in range(1, len(word_topic_number[i])):
			topic_count = word_topic_number[i][j].split(':')
			topics[int(topic_count[0])][word_topic_number[i][0]] = topic_count[1]

	return topics

def compute_weight(parsed, num, alpha):
	"""
	Compute the weights of the words

	Arguments
	---------
	parsed: output of parseTopicwise
	num: number of topics
	alpha: alpha value

	Returns
	---------
	List of words with weights that are not normalized

	"""
	"""
	Use the below formula to calculate the word weight:

	p(word|topic) = (count[topic, word] + alpha / num_word_types) / (sum(count[topic, w] for w in words) + alpha)
	"""

	num_word_types = []
	# Total number of words in the topic
	for i in range(num):
		topic_count = 0
		for key in parsed[i]:
			topic_count += int(parsed[i][key])
			num_word_types.append(len(parsed[i]))

	prob = []
	for i in range(num):
		h = {}
		for key in parsed[i]:
			h[key] = (float(parsed[i][key]) + alpha / num_word_types[i]) / (topic_count + alpha)
		prob.append(h)

	return prob

def sortNormalize(c_weight, num, top_words):
	"""
	Sort the words terms in a topic in decending order and normalize the
	weights

	Arguments
	---------
	c_weight: output from compute_weight
	num: number of topics
	top_words: number of top words per topic to calculate for

	Returns
	---------
	list of dictionaries of top <top_words> per topic
	"""
	# Sort it in decending order
	sorted_weights = []
	for i in range(num):
		sorted_weights.append(sorted(weights[i].items(), key=operator.itemgetter(1), reverse=True)[:top_words])

	# Normalize the weights
	norm = []
	for i in range(args.topic_number):
		print([flo for (word, flo) in sorted_weights[i]])
		normalize = sum([flo for (word, flo) in sorted_weights[i]])
		temp = []
		for j in range(10):
			word_weight = {sorted_weights[i][j][0] : float(sorted_weights[i][j][1])/normalize}
			temp.append(word_weight)
		norm.append(temp)

	return norm

def saveCSV(normalized, path):
	"""
	Save the normalized word/term weights in a CSV file

	Arguments
	---------
	normalized: Output of sortNormalize
	path: path to save the output with filename

	Returns
	---------
	None
	"""
	with open(path, 'wb') as myfile:
		wr = csv.writer(myfile, delimiter=',',quoting=csv.QUOTE_ALL)
		for i in range(len(normalized)):
			wr.writerow(normalized[i])

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to word counts", type=str)
	parser.add_argument("-n", "--topic-number",
			help="number of topics", type=int, default=100)
	parser.add_argument("-t", "--top-words",
			help="number of top terms/words per topic to calculate for", type=int, default=10)
	parser.add_argument("-a", "--alpha",
			help="Alpha value", type=float, default=5.0)
	parser.add_argument("-o", "--output-path",
			help="Destination to topic word weights", type=str)
	args = parser.parse_args()

	# Open the word count file
	w_t_num = open_count(args.input_path)

	# Parse the information
	parsed = parseTopicwise(w_t_num, args.topic_number)

	# Default alpha value in MALLET is 5
	# Default beta value in MALLET is 0.01
	weights = compute_weight(parsed, args.topic_number, args.alpha)

	# Sort in decending order and Normalize the weights
	nomalized = sortNormalize(weights, args.topic_number, args.top_words)

	# Save the word/terms weights
	saveCSV(normalize, args.output_path)