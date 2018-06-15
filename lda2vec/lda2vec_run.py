# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import pickle
import time

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic
from lda2vec_model import LDA2Vec

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

# You must run preprocess.py before this data becomes available
vocab = pickle.load(open('vocab', 'r'))
corpus = pickle.load(open('corpus', 'r'))
data = np.load(open('data.npz', 'r'))
flattened = data['flattened']
story_id = data['story_id']

# Model Parameters
# Number of documents
n_stories = story_id.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# Number of dimensions in a single word vector
n_units = 256
# Number of topics to fit
n_story_topics = 10
batchsize = 4096
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

# How many tokens are in each story
sty_idx, lengths = np.unique(story_id, return_counts=True)
sty_len = np.zeros(sty_idx.max() + 1, dtype='int32')
sty_len[sty_idx] = lengths

# Count all token frequencies
tok_idx, freq = np.unique(flattened, return_counts=True)
term_frequency = np.zeros(n_vocab, dtype='int32')
term_frequency[tok_idx] = freq

model = LDA2Vec(n_stories=n_stories, n_story_topics=n_story_topics,
                n_units=n_units, n_vocab=n_vocab, counts=term_frequency,
                n_samples=15)
if os.path.exists('lda2vec.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("lda2vec.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)
clip = chainer.optimizer.GradientClipping(5.0)
optimizer.add_hook(clip)

author_id = story_id
j = 0
epoch = 0
fraction = batchsize * 1.0 / flattened.shape[0]
for epoch in range(5000):
    ts = prepare_topics(cuda.to_cpu(model.mixture_sty.weights.W.data).copy(),
                        cuda.to_cpu(model.mixture_sty.factors.W.data).copy(),
                        cuda.to_cpu(model.sampler.W.data).copy(),
                        words)
    print_top_words_per_topic(ts)
    ts['doc_lengths'] = sty_len
    ts['term_frequency'] = term_frequency
    np.savez('topics.story.pyldavis', **ts)
    optimizer.use_cleargrads(use=True)
    for s, a, f in utils.chunks(batchsize, story_id, author_id, flattened):
        print(s, a, f)
        t0 = time.time()
        optimizer.update()
        l = model.fit_partial(s.copy(), a.copy(), f.copy())
        prior = model.prior()
        loss = prior * fraction
        loss.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
               "P:{prior:1.3e} R:{rate:1.3e}")
        prior.to_cpu()
        loss.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(loss=float(l), epoch=epoch, j=j,
                    prior=float(prior.data), rate=rate)
        print msg.format(**logs)
        j += 1
    serializers.save_hdf5("lda2vec.hdf5", model)
