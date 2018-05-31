#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to infer topic from LDAmodels
"""

#######################################

from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
import gensim

# Enable logging for gensim - optional
import logging

# NLTK Stop words
from nltk.corpus import stopwords

from argparse import ArgumentParser
import codecs
from operator import itemgetter

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
stop_words = stopwords.words('english')
stop_words.extend(['hi', 'hello', 'hey', 'thanks', 'thank', 'you', 'regards'])

def sent_to_words(sentences):
	"""
	Tokenizes the sentences/reviews

	Arguments:
		sentences -> list of strings in sentence structure

	Returns list of words
	"""
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
	"""
	Tokenizes, removes punctuation etc

	Arguments:
		texts -> 2 dim list of words

	Returns 2 dim list of words
	"""
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def processText(doc_set):
	"""
	Process each review. Tokenize, stem and lemmatize.

	Arguments:
		doc_set -> list of sentences/reviews

	Returns list of words
	"""

	# Tokenize the sentence
	data_words = list(sent_to_words(doc_set))

	# Build the bigram models
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

	# Faster way to get a sentence clubbed as a bigram
	bigram_mod = gensim.models.phrases.Phraser(bigram)

	# Remove Stop Words
	data_words_nostops = remove_stopwords(data_words)

	# Form Bigrams
	data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]

	# Make sure words are in ascii form
	data_w_bigrams_unicode = []
	for i in range(len(data_words_bigrams)):
		r =[]
		for j in range(len(data_words_bigrams[i])):
			r.append(unicode(data_words_bigrams[i][j].encode('ascii','ignore')))
		data_w_bigrams_unicode.append(r)

	# Initialize word Lemmatizer
	lmtzr = WordNetLemmatizer()

	# Iterate over the data and Lemmatize each word
	data_lemmatized = []
	for i in range(len(data_w_bigrams_unicode)):
		k = []
		for j in range(len(data_w_bigrams_unicode[i])):
			k.append(lmtzr.lemmatize(data_w_bigrams_unicode[i][j]))

		data_lemmatized.append(k)

	return data_lemmatized

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Destination to save model", type=str)
	parser.add_argument("-t", "--test-path",
			help="Path to file with test string", type=str)
	args = parser.parse_args()

	# Define number of topics
	numTopics=65

	# Load reviews to test
	test_set = []
	with codecs.open(args.test_path, "r", encoding='utf8') as f:
			for line in f:
				test_set.append(line.encode("ascii","ignore"))

	# Process test review/sentence
	test_lemmatized = processText(test_set)

	# Temporary loading function
	lda_model = gensim.models.ldamodel.LdaModel.load(args.input_path + "LDA", mmap='r')

	# Get 10 most influencing words of the topic
	topic = []
	for i in range(numTopics):
		topic.append(lda_model.show_topic(i, topn=10))
		#print(lda_model.show_topic(i, topn=10))

	# Compute the score of the test sentence versus every topic
	scoreList = []
	score = 0
	for i in range(len(topic)):
		for j in range(10):
			for u in range(len(test_lemmatized)):
				if test_lemmatized[0][u] == topic[i][j][0]:
					score = topic[i][j][1] + score
		scoreList.append((score, i))
		score = 0

	# Sort the score in decending order
	scoreList.sort(key=itemgetter(0), reverse = True)

	# Print the topic corresponding to the highest scores
	for i in range(numTopics):
		if scoreList[i][0] == 0:
			break
		else:
			print(topic[scoreList[i][1]])