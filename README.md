# TopicModel
Generate topics from data using *[Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)* (LDA). Computes coherence to arrive at suitable topic number. Contains simple script to infer topic distribution of new string/sentence.  

## Files
`parse.py` - Parses data from files in specific format  
`model.py` - Generates models using *LDA*  
`inference.py` - Infer topic distribution of new string/sentence

### Language
`python2.7`

### Libraries
`gensim` , `nptk`, `nptk wordnet` and gensims `LDAMallet`

## Usage
`$ parse.py --help`  
`$ model.py --help`  
`$ inference.py --help`  
for options.
