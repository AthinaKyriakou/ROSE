
pip install -r requirements.txt

# For fm_py
pip install git+https://github.com/coreylynch/pyFM

# For Lexicon Construction (sentiment generation)
pip install spacy
python -m spacy download en_core_web_sm
pip install nltk
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon
