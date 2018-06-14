import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import codecs

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/2"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def plot_similarity(labels, features, rotation):
	corr = np.inner(features, features)
	np.savetxt("output.csv", corr, delimiter=",")

def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
	message_embeddings_ = session_.run(
		encoding_tensor, feed_dict={input_tensor_: messages_})
	#print(messages_, message_embeddings_)
	plot_similarity(messages_, message_embeddings_, 90)

messages = []

with codecs.open(args.file_path, 'r', encoding = 'utf8') as data:
	for line in data:
		messages.append(line)

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	session.run(tf.tables_initializer())
	run_and_plot(session, similarity_input_placeholder, messages,
				similarity_message_encodings)
