# Bash script for the needed installations

pip install Cython==0.29.36 numpy==1.23.5 scipy mlxtend
python setup.py instal

pip install -r requirements.txt

# For test 
pip install python-box pyyaml

# For ALS 
pip install implicit

# For PHI Explainer
pip install mlxtend

# For fm_py
pip install git+https://github.com/coreylynch/pyFM

# For lime
pip install lime

# For Lexicon Construction (sentiment generation)
pip install spacy
python -m spacy download en_core_web_sm
pip install nltk
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon