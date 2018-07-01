# Topic Modeling using LDA with Mallet

Use the following scripts for Topic Modeling using [Mallet Topic Modeling](http://mallet.cs.umass.edu/topics.php).  

Refer [Topic Modelling using LDA with MALLET](https://diliprajbaral.com/2017/06/04/topic-modelling-lda-mallet/) for full explanation.

## Usage

Simple inference script for MALLET:

`$ python simple_infer.py -h`
```
usage: simple_infer.py [-h] [-m MODEL_PATH] [-M MALLET_PATH] [-t TRAIN_PATH] [-i INPUT_PATH] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -m MODEL_PATH, --model-path MODEL_PATH	Path to Mallet inferencer
  -M MALLET_PATH, --mallet-path MALLET_PATH	Path to mallet binary
  -t TRAIN_PATH, --train-path TRAIN_PATH	Path to mallet file used to originally train the model
  -i INPUT_PATH, --input-path INPUT_PATH	Path to reviews
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Destination of inferred topics
```
Simple sentiment analysis using [OpenAI's Sentiment Model](https://github.com/openai/generating-reviews-discovering-sentiment):

`$ python simple_sentiment.py -h`
```
usage: simple_sentiment.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to reviews
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Destination of inferred topics
```

Label document topic:

`$ python label.py -h`
```
usage: clustering.py [-h] [-i TOPICS] [-l LABELS] [-q QUERY] [-t PTOPICS] [-T INF_TOPICS] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i TOPICS, --topics TOPICS			Path to inference output
  -l LABELS, --labels LABELS			Path to labels
  -q QUERY, --query QUERY			Path to original query file (sentence form)
  -t PTOPICS, --ptopics PTOPICS			Number of topics to print for each sentence
  -T INF_TOPICS, --inf-topics INF_TOPICS	Number of topics whose probabilities are given in the inference output
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Destination of sampled topics
```

## NOTE

The input to Mallet used in the testing was unprocessed. No tokenization, stemming etc was done.
The input was a file contained one sentences per line.
