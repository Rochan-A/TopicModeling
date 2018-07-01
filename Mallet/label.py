#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Cross reference the inferred topics with the topic labels
"""

#######################################

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import codecs

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--topics",
			help="Path to inference output", type=str)
	parser.add_argument("-l", "--labels",
			help="Path to labels", type=str)
	parser.add_argument("-q", "--query",
			help="Path to original query file (sentence form)", type=str)
	parser.add_argument("-t", "--ptopics",
			help="Number of topics to print for each sentence", type=int, default=5)
	parser.add_argument("-T", "--inf-topics",
			help="Number of topics whose probabilities are given in the inference output", type=int, default=100)
	parser.add_argument("-o", "--output-path",
			help="Destination of sampled topics", type=str)
	args = parser.parse_args()

	# Number of topics
	num_topics = args.inf_topics

	# Read the file
	topics = []
	count = 0
	with codecs.open(args.topics, 'r') as F:
		for line in F:
			count = count + 1
			# Skip header of the inference output
			if count != 1:
				row = line.split('\t')
				row[num_topics + 1] = row[num_topics + 1][:-1]
				topics.append(row)

	# Ignore the headers of the topic output
	for i in range(0, len(topics)):
		topics[i] = topics[i][2:]

	# Read the labels file
	labels = []
	with codecs.open(args.labels, 'r', encoding='utf8') as F:
		for line in F:
			labels.append(line)

	# Read the sentence file
	sent = []
	with codecs.open(args.query, 'r', encoding='utf8') as F:
		for line in F:
			sent.append(line)

	# Convert the topic probabilities list into a dictionary
	h = []
	for i in range(len(topics)):
		topic = dict((j, topics[i][j]) for j in range(0, num_topics))
		h.append(topic)

	# Identify the top args.ptopics topics
	for i in range(0, len(topics)):
		val = sorted(h[i].values(), reverse=False)[0:args.ptopics]
		index = []
		for k in range(args.ptopics):
			v = h[i].keys()[h[i].values().index(val[k])]
			index.append(labels[v][:-1])
		# Print them
		print(sent[i], index)