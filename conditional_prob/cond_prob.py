#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Calculate the conditional probabilities
"""

#######################################

import csv, io, operator, ast
from argparse import ArgumentParser
import numpy as np

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

def open_doc_file(file_path):
	"""
	Open the document topic file

	Arguments
	---------
	file_path: Location of the document topic file

	Returns
	--------
	Matrix of topic probabilities for each document
	"""

	matrix = []
	with open(file_path, 'r') as F:
		reader = csv.reader(F, delimiter='\t')
		for row in reader:
			# Delete the first two columns in the file
			matrix.append(row[2:])

	return matrix

def open_filtered(file_path):
	"""
	Get the review number for every sentence used

	Arguments
	---------
	file_path: Path to filtered.txt

	Returns
	-------
	List of review numbers
	"""

	matrix = []
	with open(file_path, 'r') as F:
		reader = csv.reader(F, delimiter=',')
		for row in reader:
			matrix.append(row[0])

	return matrix

def construct_matrix(doc_matrix, reviews, num_sent, num_topics):
	"""
	Construct a simple frequency matrix of each topic (current sentence)
	vs topic (next sentence)
	Identify the topics with high probabilities for each sentence

	Arguments
	---------
	doc_matrix: Matrix of topic probabilities for each sentence
	reviews: list of number of lines per review
	num_topics: number of topics
	num_sent: number of sentences

	Returns
	-------
	A numpy array of size (num_topics, num_topics), dtype = int
	topic_freq: total number of topics chosen
	"""

	# Create a dictionary of all the topic frequencies
	topic_freq = dict()
	for i in range(0, num_topics):
		topic_freq[i] = 0

	# Construct an empty matrix
	freq_matrix = np.zeros((num_topics, num_topics), dtype=float)

	# Append values to the matrix
	line_no = 0

	# Iterate over all the sentences - 1
	for i in range(num_sent - 1):

		if reviews[i] == reviews[i + 1]:

			# Create a dictionary of all the probabilities for the two sentences
			topic_dict = [dict(), dict()]
			for j in range(0, num_topics):
				topic_dict[0][j] = ast.literal_eval(doc_matrix[i][j])
				topic_dict[1][j] = ast.literal_eval(doc_matrix[i + 1][j])

			# Sort the topics versus the probabilities
			sorted_topic_dict = [sorted(topic_dict[0].items(), \
				key=operator.itemgetter(1), reverse=True), \
					sorted(topic_dict[1].items(), \
					key=operator.itemgetter(1), reverse=True)]

			# Choose the top few topics (ie. till their probabilities add up to one-third)
			prob_uptill = [0.0, 0.0]
			tmp = [[], []]
			for j in range(num_topics):

				# Threshold
				if prob_uptill[0] < 0.9:

					# Keep track of total probability
					prob_uptill[0] += sorted_topic_dict[0][j][1]

					# Save topic chosen
					tmp[0] += [sorted_topic_dict[0][j][0]]

				# Threshold
				if prob_uptill[1] < 0.9:

					# Keep track of total probability
					prob_uptill[1] += sorted_topic_dict[1][j][1]

					# Save topic chosen
					tmp[1] += [sorted_topic_dict[1][j][0]]

			for x in tmp[0]:
				for y in tmp[1]:
					freq_matrix[x][y] += 1.0
					topic_freq[x] += 1.0

	return freq_matrix, topic_freq

def compute_cond(freq_matrix, topic_freq, num_topics):
	"""
	Compute the conditional probabilities

	Arguments
	---------
	freq_matrix: A numpy array of size (num_topics, num_topics)
	topic_freq: Dictionary of topics versus their frequencies
	num_topics: number of topics

	Returns
	-------
	freq_matrix: A numpy array of size (num_topics, num_topics)
	"""

	# Conditional probability
	for i in range(num_topics):
		for j in range(num_topics):
			if float(topic_freq[i]) != 0:
				freq_matrix[i][j] = float(freq_matrix[i][j])/float(topic_freq[i])
	return freq_matrix

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path", \
			help="Path to document probabilities", type=str)
	parser.add_argument("-f", "--filtered", \
			help="Path to filtered file", type=str)
	parser.add_argument("-o", "--output-path", \
			help="Path to output conditional probabilities matrix", type=str)
	ARGS = parser.parse_args()

	# Open the doc - topics file
	doc_matrix = open_doc_file(ARGS.input_path)

	# Open the filtered file
	reviews = open_filtered(ARGS.filtered)

	# Number of sentences
	NUM_SENT = len(doc_matrix)

	# Number of topics
	NUM_TOPICS = len(doc_matrix[0])

	# Construct the frequency matrix
	freq_matrix, topic_freq = construct_matrix(doc_matrix, reviews, NUM_SENT, NUM_TOPICS)

	# Compute conditional probabilities
	cond_prob = compute_cond(freq_matrix, topic_freq, NUM_TOPICS)

	# Save the conditional probabilities matrix
	with open(ARGS.output_path + "cond_prob.csv", 'w') as out:
		writer = csv.writer(out, delimiter=',')
		for line in cond_prob:
			writer.writerow(line)
