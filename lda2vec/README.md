# Topic Modeling with lda2vec

### Note: Use [this](https://github.com/Rochan-A/lda2vec) forked version of lda2vec. Original version can be found [here](https://github.com/cemoody/lda2vec)

## Files

* `preprocess.py`		Use the tokenized output from `parse.py` that can be found [here](https://github.com/Rochan-A/TopicModeling/blob/master/lda%26w2vec/parse.py).
	* Use either `filtered.txt` or `tokens.txt` as the input
* `lda2vec_run.py`		Executes model. Requires CUDA
* `lda2vec_model.py`		lda2vec (class) model

## Usage

Generate the `corpus`, `data.npz`, `tokens` and `vocab` files.

`$ python preprocess.py -h`

```
usage: preprocess.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH]

optional arguments:
  -h, --help					show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH	Path to tokenized sentences
  -o OUTPUT_PATH, --output-path OUTPUT_PATH	Destination for parsed and preprocessed output
```