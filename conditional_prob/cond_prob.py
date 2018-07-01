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
			del row[1]
			matrix.append(row)

	return matrix

def open_rev_file(file_path):
	"""
	Open the reviews file and save the number of sentences in each review

	Arguments
	---------
	file_path: Location of the reviews file

	Returns
	--------
	Number of sentences in each review
	"""

	# Open the file
	with io.open(file_path, 'rb') as raw:
		raw_review = [x.strip().split('\t') for x in raw]

	# Select full review only
	for i in range(len(raw_review)):
		raw_review[i] = len(raw_review[i][-1].split('.'))

	return raw_review

def filter_topic(doc_matrix, num_sent, num_topics):
	"""
	Identify the topics with high probabilities for each sentence

	Arguments
	---------
	doc_matrix: Matrix of topic probabilities for each sentence
	num_sent: Number of sentences
	num_topics: Number of topics

	Returns
	---------
	top: List of highest probabilities
	topic_freq: total number of topics chosen
	"""

	# Create a dictionary of all the topic frequencies
	topic_freq = dict()
	for i in range(0, num_topics - 1):
		topic_freq[i] = 0

	# Iterate over all the sentences
	top = []
	for i in range(num_sent):

		# Create a dictionary of all the probabilities for a sentence
		topic_dict = dict()
		for j in range(1, num_topics):
			topic_dict[j] = ast.literal_eval(doc_matrix[i][j])

		# Sort the topics versus the probabilities
		sorted_topic_dict = sorted(topic_dict.items(), key=operator.itemgetter(1), reverse=True)

		# Choose the top few topics (ie. till their probabilities add up to one-third)
		prob_uptill = 0.0
		tmp = []
		for j in range(num_topics):

			# Threshold
			if prob_uptill < 0.9:

				# Keep track of total probability
				prob_uptill += sorted_topic_dict[j][1]

				# Save topic chosen
				tmp += [sorted_topic_dict[j][0]]

				# Keep track of topic freq
				topic_freq[sorted_topic_dict[j][0]] += 1
		top.append(tmp)

	return top, topic_freq

def construct_matrix(selected_topic, reviews, num_topics):
	"""
	Construct a simple frequency matrix of each topic (current sentence)
	vs topic (next sentence)

	Arguments
	---------
	selected_topics: Output of filter_topic
	reviews: list of number of lines per review
	num_topics: number of topics

	Returns
	-------
	A numpy array of size (num_topics, num_topics), dtype = int
	"""

	# Construct an empty matrix
	freq_matrix = np.zeros((num_topics, num_topics), dtype=float)

	# Append values to the matrix
	line_no = 0

	# Iterate over all the reviews
	for k in reviews:

		# If the review has more than one line
		if reviews[k] > 1:

			# Iterate over all the sentences in the review
			for i in range(line_no, reviews[k] - 1):
				curr_topics = selected_topic[i]
				next_topics = selected_topic[i+1]

				for x in curr_topics:
					for y in next_topics:
						freq_matrix[x][y] += 1

		# Keep track of the next line_no to read
		line_no += reviews[k]

	return freq_matrix

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
	for i in range(num_topics - 1):
		for j in range(num_topics - 1):
			try:
				freq_matrix[i][j] = float(freq_matrix[i][j])/float(topic_freq[i])
			except ZeroDivisionError:
				freq_matrix[i][j] = 0.0

	return freq_matrix

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path", \
			help="Path to document probabilities", type=str)
	parser.add_argument("-r", "--ori-reviews", \
			help="Path to document with original reviews file", type=str)
	parser.add_argument("-o", "--output-path", \
			help="Path to output conditional probabilities matrix", type=str)
	ARGS = parser.parse_args()

	# Open the doc - topics file
	doc_matrix = open_doc_file(ARGS.input_path)

	# Open the original reviews file
	reviews = open_rev_file(ARGS.ori_reviews)

	# Number of sentences
	NUM_SENT = len(doc_matrix)

	# Number of topics
	NUM_TOPICS = len(doc_matrix[0])

	# Select the topic with highest probability
	selected_topic, topic_freq = filter_topic(doc_matrix, NUM_SENT, NUM_TOPICS)

	# Construct the frequency matrix
	freq_matrix = construct_matrix(selected_topic, reviews, NUM_TOPICS)

	# Compute conditional probabilities
	cond_prob = compute_cond(freq_matrix, topic_freq, NUM_TOPICS)

	# Save the conditional probabilities matrix
	with open(ARGS.output_path + "cond_prob.csv", 'w') as out:
		writer = csv.writer(out, delimiter=',')
		for line in cond_prob:
			writer.writerow(line)
