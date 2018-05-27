# TopicModel
Generate topics from data using *[Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)* (LDA) and *[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent\_semantic\_analysis)* (LSA).

## Files
`alpha.py` - Parses data from files in specific format
`lda.py` - Generates models using *LSA* and *LDA*

### Language
`python2.7`
### Libraries
`gensim` , `nptk` and `stop-word`

## Usage
`$ alpha.py [path_to_data] [path_to_output]`
`$ lda.py [path_to_alpha.py_output]`
