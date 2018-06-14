from lda2vec import preprocess, Corpus
import numpy as np
import logging, gensim, csv, codecs
import cPickle as pickle
from gensim.corpora import Dictionary
from argparse import ArgumentParser

logging.basicConfig()

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

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to tokenized sentences", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination for parsed and preprocessed output", type=str)
	args = parser.parse_args()

	# Limit tokens per sentence
	max_length = 100

	readToken= []
	lineno = 0
	# Open the tokenized file
	with codecs.open(args.input_path, 'r', encoding='utf8') as File:
		for row in File:
			token_in_row = row.split(",")
			lineno = lineno + 1
			for i in range(len(token_in_row)):
				token_in_row[i] = force_unicode(token_in_row[i])
			readToken.append(token_in_row)

	dic = Dictionary(readToken)
	vocab = dict(dic)

	array = np.zeros((lineno, max_length), int)

	for i in range(lineno):
		for j in range(len(readToken[i])):
			array[i][j] = np.int64(vocab.keys()[vocab.values().index(readToken[i][j])])
		if len(readToken[i]) < max_length:
			for k in range(len(readToken[i]), max_length):
				array[i][k] = -2
	tokens = array

	"""
	    arr : 2D array of ints
	        Has shape (len(texts), max_length). Each value represents
	        the word index.
	    vocab : dict
	        Keys are the word index, and values are the string. The pad index gets
	        mapped to None
	"""

	# Make a ranked list of rare vs frequent words
	corpus = Corpus()
	corpus.update_word_count(tokens)
	corpus.finalize()

	# The tokenization uses spaCy indices, and so may have gaps
	# between indices for words that aren't present in our dataset.
	# This builds a new compact index
	compact = corpus.to_compact(tokens)
	# Remove extremely rare words
	pruned = corpus.filter_count(compact, min_count=10)
	# Words tend to have power law frequency, so selectively
	# downsample the most prevalent words
	clean = corpus.subsample_frequent(pruned)
	print "n_words", np.unique(clean).max()

	story_id = np.array(list(range(0, lineno)), int)

	flattened, features_flat = corpus.compact_to_flat(pruned, story_id)

	story_id_f = features_flat

	data = dict(flattened=flattened, story_id=story_id_f)

	pickle.dump(corpus, open('corpus', 'w'), protocol=2)
	pickle.dump(vocab, open('vocab', 'w'), protocol=2)
	np.savez('data', **data)
	np.save(open('tokens', 'w'), tokens)