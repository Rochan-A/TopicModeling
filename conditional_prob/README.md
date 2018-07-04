# Conditional Probability Matrix

Construct a topic Conditional Probability Matrix.

## Usage

Parse the reviews ( Similar to [this](../VanillaLDA/parse.py) but filtered.txt will have an extra column on review number):

`$ python parse.py -h`

```
usage: parse.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to reviews
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Destination for parsed and preprocessed output
```

Compute the conditional probability matrix:

`$ python cond_prob.py -h`

```
usage: cond_prob.py [-h] [-i INPUT_PATH] [-r ORI_REVIEWS] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to document probabilities
  -r ORI_REVIEWS, --ori-reviews ORI_REVIEWS	Path to document with original reviews file
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Path to output conditional probabilities matrix
```

Display the matrix with labels etc:

`$ python sort_prob.py -h`

```
usage: sort_cond.py [-h] [-i INPUT_PATH] [-r LABELS] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to conditional probabilities
  -r LABELS, --labels LABELS			Path to labels
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Path to Path to labeled matrix
```

## Required Files

* [original reviews](../sample/review_data.txt)
* [lables](../sample/labels.csv)
