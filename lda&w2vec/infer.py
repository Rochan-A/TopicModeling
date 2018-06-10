#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
	This script is to be used to segregate sentences into document bins
"""

#######################################

from argparse import ArgumentParser
import logging, gensim, io, sys, codecs, csv
from gensim.models import Doc2Vec
from collections import namedtuple
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

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to Doc2Vec model", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination of sentence matrix", type=str)
	parser.add_argument("-t", "--type",
			help="File type:\n1.\tsentences.txt\n2.\ttokens.txt\n3.\tfiltered.txt", type=int)
	args = parser.parse_args()

	model = Doc2Vec.load(args.input_path)

	# Get the vectors
	out = []
	for i in range(len(model.docvecs)):
		sen1 = model.docvecs[i].reshape(-1, 1)
		o = []
		for j in range(len(model.docvecs)):
			sen2 = model.docvecs[j].reshape(-1, 1)
			o.append(1 - spatial.distance.cosine(sen1, sen2))
		out.append(o)

	with open(args.output_path + "output.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(out)
