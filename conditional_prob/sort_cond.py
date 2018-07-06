#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Sort the conditional probabilities, beautify it
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

def open_file(file_path):
	"""
	Open a csv file

	Arguments
	---------
	file_path: Location of the file

	Returns
	--------
	list
	"""

	matrix = []
	with open(file_path, 'r') as F:
		reader = csv.reader(F, delimiter=',')
		for row in reader:
			matrix.append(row)

	return matrix

def sort_label(matrix, labels):
	"""
	Sort and label each value in the matrix

	Arguments
	---------
	matrix: Conditional Probability matrix
	labels: labels for each topic

	Returns
	-------
	Matrix of type dict()
	"""

	num_topics = len(matrix[0])

	new = []
	for i in range(0, num_topics):
		new.append({})
		new[i]['label'] = labels[i][0]
		for j in range(0, num_topics):
			new[i][labels[j][0]] = matrix[i][j]

	out = []
	for i in range(0, num_topics):
		sort_row = sorted(new[i].items(), key=operator.itemgetter(1), reverse=True)
		out.append(sort_row)

	return out

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path", \
			help="Path to conditional probabilities", type=str)
	parser.add_argument("-l", "--labels", \
			help="Path to labels", type=str)
	parser.add_argument("-o", "--output-path", \
			help="Path to labeled matrix", type=str)
	ARGS = parser.parse_args()

	# Open the conditional probability file
	prob_matrix = open_file(ARGS.input_path)

	# Open the labels file
	labels = open_file(ARGS.labels)

	# Sort and label
	dict_cond = sort_label(prob_matrix, labels)

	with open(ARGS.output_path + 'sorted.csv', 'w') as F:
		writer = csv.writer(F, delimiter=',')
		for r in dict_cond:
			writer.writerow(r)
