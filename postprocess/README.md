# Postprocessing

Scipts to help in computing the word/term weights and compare topic vectors from MALLET LDA output.

## Usage

Compute the word/term weights:

`$ python word_weights.py -h`

```
usage: word_weights.py [-h] [-i INPUT_PATH] [-n TOPIC_NUMBER] [-t TOP_WORDS]
                       [-a ALPHA] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to word counts
  -n TOPIC_NUMBER, --topic-number TOPIC_NUMBER	number of topics
  -t TOP_WORDS, --top-words TOP_WORDS		number of top terms/words per topic to calculate for
  -a ALPHA, --alpha ALPHA			Alpha value
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Destination to topic word weights
```

Compute the Hellsinger distance between topic vectors:

`$ python distance.py -h`

```
usage: distance.py [-h] [-i INPUT_PATH] [-l LABEL]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to LDA Mallet
  -l LABEL, --label LABEL			Path to label file
```