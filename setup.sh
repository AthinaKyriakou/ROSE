# Bash script for the needed installations
pip install Cython
pip install numpy
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install git+https://github.com/coreylynch/pyFM
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon