#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This script is to be used to concatenate all the review data that has
	been webscraped from the hotel sites.

	Note: This only works when the collected data is in either for the
		following template:

	filename: #.txt

		Reviewer Name:
		Place:
		Badges:

		title:
		rating:
		Date:
		review:
		room tip:
		management response:
		traveled as:

	OR

	filename: #.txt

		{
			...
			"review": ,
			...
		}

"""

#######################################

import os, sys, json, io

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

"""
	Function to concatenate all Hotel reviews that are present in the
	current directory.
"""
def concateReviews(path, outFile):

	print("Changing working directory to: " + path)
	if os.path.isdir(path):
		os.chdir(path)

	currFileList = [f for f in os.listdir('.') if os.path.isfile(f)]

	# Create an empty buffer to hold all the reviews
	lineBuf = []

	# Iterate over all the files inside the folder
	for i in range(0, len(currFileList)):
		# Open the file
		with io.open(currFileList[i], encoding='utf8') as f:

			# Check the number of line in the file
			num_lines = sum(1 for line in open(currFileList[i]))

			# Data is encoded in JSON format (Template 2)
			if num_lines == 1:
				d = json.load(f)

				# Check if the review is an duplicate or not
				if checkNotDuplicate(lineBuf, d["review"]):
					lineBuf.append(d["review"])

			# Data is encoded in template 1
			elif num_lines == 12:
				lines = f.readlines()
				newLine = lines[7].replace("review: ", "")

				# Check if the review is an duplicate or not
				if checkNotDuplicate(lineBuf, newLine):
					lineBuf.append(newLine)
			else:
				print("None")

	# Write the filtered reviews to a file
	with io.open(outFile + path + ".txt", "w", encoding='utf8') as myfile:
		for i in range(len(lineBuf)):
			myfile.write(lineBuf[i])

	os.chdir("..")

"""
	Check if the review is an duplicate or not
"""
def checkNotDuplicate(lineBuf, inLine):
	for i in range(len(lineBuf)):
		if lineBuf[i] == inLine:
			return False
			break
	return True

if __name__ == '__main__':

	os.chdir(sys.argv[1])

	# Iterate over all the folders in the current working directory
	currFolList = [f for f in os.listdir('.') if os.path.isdir(f)]
	for i in range(0, len(currFolList)):
		concateReviews(currFolList[i], sys.argv[2])