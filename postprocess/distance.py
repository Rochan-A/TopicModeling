#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to graph the Hellingers distance between topic vectors
"""

#######################################

import csv, gensim
import numpy as np
import operator
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn import manifold

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path", \
			help="Path to LDA Mallet", type=str)
	parser.add_argument("-l", "--label", \
			help="Path to label file", type=str)
	ARGS = parser.parse_args()

	# Load the LDA mallet model
	lda_mallet = gensim.utils.SaveLoad.load(ARGS.input_path)

	# Construct an empty matrix
	matrix = np.zeros((len(lda_mallet.get_topics()), len(lda_mallet.get_topics()) + 1), dtype=float)

	# Calculate the Hellinger Distance between all pariwise topic vectors
	for i in range(0, len(lda_mallet.get_topics())):
		for j in range(1, len(lda_mallet.get_topics()) + 1):
			matrix[i][j] = (gensim.matutils.hellinger(lda_mallet.get_topics()[i], lda_mallet.get_topics()[j-1]))

	# Add labels to each row
	for i in range(len(lda_mallet.get_topics())):
		matrix[i][0] = i

	# Print the matrix
	print(matrix)

	# Save the matrix
	with open('distance.csv', 'w') as F:
		spamwriter = csv.writer(F, delimiter=',')
		for row in matrix:
			spamwriter.writerow(row)

	# Read the labels file if present
	cities = []
	if ARGS.label != None:
		with open(ARGS.label, 'r') as F:
			spamwriter = csv.reader(F)
			for row in spamwriter:
				cities.append(row)

	data = matrix

	# Separate the distance values from the labels
	dists = []
	for d in data:
		dists.append(map(float, d[1:]))
		if len(cities) == 0:
			cities.append(d[0])

	adist = np.array(dists)
	amax = np.amax(adist)
	adist /= amax

	mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
	results = mds.fit(adist)

	coords = results.embedding_

	plt.subplots_adjust(bottom=0.1)
	plt.scatter(coords[:, 0], coords[:, 1], marker='o')
	for label, x, y in zip(cities, coords[:, 0], coords[:, 1]):
		plt.annotate( \
			label, \
			xy=(x, y), xytext=(-20, 20), \
			textcoords='offset points', ha='right', va='bottom', \
			bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), \
			arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

	plt.show()
