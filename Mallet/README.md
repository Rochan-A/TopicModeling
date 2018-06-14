# Topic Modeling using LDA with Mallet

Use the following scripts for Topic Modeling using [Mallet Topic Modeling](http://mallet.cs.umass.edu/topics.php).  

Refer [Topic Modelling using LDA with MALLET](https://diliprajbaral.com/2017/06/04/topic-modelling-lda-mallet/) for full explanation.  

## Usage

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

## Output

The final inferred topics can be found in `new-topic-composition.txt`.

## NOTE

The input to Mallet used in the testing was unprocessed. No tokenization, stemming etc was done.
The input was a file which contained one sentences per line.
