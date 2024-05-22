import pandas as pd
import spacy
import csv
import os
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from tqdm import tqdm
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

class SentimentAnalysis:
    """
    Process raw data, like text reviews, to generate lexicons in form of (feature:opinion:+/-1).
    
    Parameters
    ----------
    input: string/dataframe, required
        csv/txt file path. Expected format: the first line in file should be the column names, at least include ['user_id', 'book_id', 'rating', 'review_text'], which are consistent with the usecols parameter.
        or a Dataframe with columns' names specified by usecols
    sep: string, optional, default '\t'
        separator of the file, default is '\t'
    usecols: list, required
        must specific the column names within the file, order matters, [name of user id, name of item id, name of rating, name of review]
    min_frequency: int, optional, default 1
        drop users who have less than min_frequency reviews
    """
    def __init__(self, input, sep='\t', usecols = ['user_id', 'book_id', 'rating', 'review_text'], min_frequency=1):
        
        self.input = input
        self.sep = sep
        self.usecols = usecols
        self.min_frequency = min_frequency
        self.data = pd.DataFrame()
        
    def _get_relations(self, sentence):
        """
        sentence: tokenized sentence

        Returns
        --------
           relations: dict, key: modifier, value: (word, relation)
        """
        relations = {}
        for token in sentence:
            if token.pos_ in ['NOUN'] and token.lower_ not in stop_words:
                    for child in token.children:
                        if child.pos_ in ['ADJ', 'NOUN'] and child.lower_ not in stop_words:
                            relation = child.dep_
                            if relation in ['amod', 'appos', 'nsubj', 'attr']:
                                relations[child.lower_] = (token.lower_, relation)
                            elif relation == 'conj' and token.lower_ in relations:
                                relations[child.lower_] = (relations[token.lower_][0], relation)
        return relations
    
    def _get_reverse_sent(self, sentence):
        """
        
        sentence: tokenized sentence
        Returns:
            reverse_sent: list, all words that are reversed by negation
        """
        reverse_sent = []
        for token in sentence:
            if token.pos_ in ['PART'] and token.dep_ == 'neg':
                if token.head.pos_ in ['ADJ', 'NOUN']:
                    reverse_sent.append(token.head.lower_)
                for child in token.head.children:
                    if child.pos_ in ['ADJ', 'NOUN']:
                        reverse_sent.append(child.lower_)
        return reverse_sent
    
    def _get_polarity(self, word):
        """
        word: string, required
        Returns:
            score: int, 1 if positive, -1 if negative
        """
        score = 1 if sid.polarity_scores(word)['pos'] >= sid.polarity_scores(word)['neg'] else -1
        return score
    
    def _detect_outlier_char(self, word, pattern = r"[^\w\s]"):
        """
        Parameters:
            word: string, required
            pattern: string, optional, default r"[^\w\s]"
        Returns:
            bool: True if word contains outlier characters, False otherwise
        """
        return re.findall(pattern, word)
    
    
    def _analysis_relations(self, relations, reverse_sent):
        """
        Parameters:
            relations: dict, key: modifier, value: (word, relation)
            reverse_sent: list, all words that are reversed by negation
        
        Returns:
            lexicons: list, lexicons in one sentence composed according to relations and reverse_sent
        """
        lexicons = []
        for modifier, (word, relation) in relations.items():
            if self._detect_outlier_char(word) or self._detect_outlier_char(modifier):
                continue
            sentiment_score = self._get_polarity(modifier)
            if modifier in reverse_sent or word in reverse_sent:
                #print(f"reversed modifier: {modifier} word: {word}")
                lexicons.append(f'{word}:{modifier}:{-1 * sentiment_score}')
            else:
                lexicons.append(f'{word}:{modifier}:{sentiment_score}')
        return lexicons
    
    def _transform_format(self, lexicons):
        """
        This function is not useless for now.
        transform list to the format of "aspect:opinion:score1,aspect:opinion:score2,..."
        Parameters:
            lexicons: list, lexicons in one sentence composed according to relations and reverse_sent
        Returns:
            lexicon: "aspect:opinion:score1,aspect:opinion:score2,..."
        """
        tuples = [f'{tup[0]}:{tup[1]}:{tup[2]}' for tup in lexicons]
        # Join the tuples into a comma-separated string
        return ','.join(tuples) if len(tuples)>0 else np.NaN
        
    
    def _build_lexicons_one_text(self, text):
        """
        Parameters:
            text: string, required
                a review text
        Returns:
            lexicons: list, all lexicons detected in the text
        """
        if isinstance(text, str) == False:
            print(f"Error: {text} is not a string")
            return np.NaN
        doc = nlp(text)
        lexicons = []
        for sentence in doc.sents:
            relations = self._get_relations(sentence)
            reverse_sent = self._get_reverse_sent(sentence)
            l = self._analysis_relations(relations, reverse_sent)
            if len(l) == 0:
                continue
            lexicons.extend(l)
        return ','.join(lexicons) if (len(lexicons) > 0) else np.NaN
    
    def build_lexicons(self):
        """ Build the lexicons
        
        Returns
        -------
        df: dataframe
            ['user_id', 'item_id', 'rating, 'lexicon']
        
        """
        self.data = self._read_raw_data()
        self.data['lexicon'] = np.NaN
        df = self.data.__deepcopy__()
        text_name = self.usecols[-1]
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            df.at[i, 'lexicon'] = self._build_lexicons_one_text(row[text_name]) if row[text_name] is not np.NaN else np.NaN
    
        print(f'number of users: {df[self.usecols[0]].nunique()}')
        print(f'number of items: {df[self.usecols[1]].nunique()}')
        #df['lexicon'] = df['lexicon'].apply(self.transform_format)
        #print(f'total{len(df)}')
        print(f'{df["lexicon"].isna().sum()} rows have no lexicon')
        df = df.dropna(axis=0, subset=['lexicon'])
        if self.min_frequency > 1:
            df = self._prune_dataset(df)
        print(f'{len(df)} rows after dropping users having less than {self.min_frequency} reviews')
        self.data = df
        return self.data
    
    def _prune_dataset(self, df):
        """
        Parameters:
            df: dataframe, ['user_id', 'item_id', 'rating, 'lexicon']

        Returns:
            df: dataframe, ['user_id', 'item_id', 'rating, 'lexicon'], pruned dataset, drop out users that have less than [min_frequency] reviews
        """
        user_counts = df[self.usecols[0]].value_counts()  # Count occurrences of each user
        # Get a list of user IDs that appeared more than 10 times
        users_to_keep = user_counts[user_counts >= self.min_frequency].index.tolist()
        # Create a pruned DataFrame containing only users that appeared more than 10 times
        pruned_df = df[df[self.usecols[0]].isin(users_to_keep)]
        return pruned_df
    
    def _read_raw_data(self):
        """
        Returns:
            df: dataframe, ['user_id', 'item_id', 'rating, 'review_text']
        """
        if isinstance(self.input, pd.DataFrame) == True:
            if self.input.columns.tolist() == self.usecols:
                self.data = self.input
            else:
                raise ValueError("Columns are not consistent with usecols")
        
        else:
            try:
                with open(self.input, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=self.sep)
                    header = next(reader)
                    if len(header) > 1 and all(isinstance(col, str) for col in header):
                        pass
                    else:
                        print("File is not in right format")
                self.data = pd.read_csv(self.input, sep=self.sep, usecols=self.usecols)
            except IOError:
                print("File not found or could not be opened")

        return self.data
    
    def save_to_file(self, lexicon_path, rating_path):
        """ save the processed data to two files, one for lexicons, one for ratings
        
        parameters
        ----------
        lexicon_path: string, required 
            path to save the lexicons, including [user_id, item_id, lexicons]
        rating_path: string, required
            path to save the ratings, including [user_id, item_id, rating]

        """
        #Note: 
        #    tear one dataframe to two files, [user_id, item_id] are exactly the same, to ensure the consistency
        columns_sentiment = [self.usecols[0], self.usecols[1], 'lexicon']
        columns_rating = self.usecols[:3]
        # write to output files
        try:
            output_sentiment_dir = os.path.dirname(lexicon_path)
            # create output directory if not exists
            if len(output_sentiment_dir)>0 and not os.path.exists(output_sentiment_dir):
                os.makedirs(output_sentiment_dir)
                
            output_rating_dir = os.path.dirname(rating_path)
            if len(output_rating_dir)>0 and not os.path.exists(output_rating_dir):
                os.makedirs(output_rating_dir)
        except Exception as e:
            print(f"Error creating output directories: {e}")

        try:
            self.data.to_csv(lexicon_path, sep=',', index=False, header=False, columns=columns_sentiment, quoting=csv.QUOTE_NONE, escapechar=' ')
        except Exception as e:
            print(f"Output sentiment.txt Error: {e}")
        try:
            self.data.to_csv(rating_path, sep=',', index=False, header=False, columns=columns_rating, quoting=csv.QUOTE_NONE, escapechar=' ')
        except Exception as e:
            print(f"Output rating.txt Error: {e}")
    
    
### The following functions are used to analyze the lexicon file when needed
class LexiconsStatistics:
    def __init__(self, lexicon_path, sep='\t', columns=['user_id', 'item_id', 'lexicon']):
        """
        Parameters:
            lexicon_path: string, required
                path to the lexicon file
            sep: string, optional, default '\t'
            columns: list, optional, default ['user_id', 'item_id', 'lexicon']
        """
        self.lexicon_path = lexicon_path
        self.sep = sep
        self.data = pd.DataFrame()
        self.usecols = columns
        
    def read_lexicon(self):
        """
        Parameters:
            lexicon_path: string, required
            path to the lexicon file
        Returns:
            data: dataframe, including [user_id, item_id, lexicons]
        """
        data = []
        with open(self.lexicon_path, encoding="utf-8") as f:
            for line in f:
                tup = line.strip().split(',')
                data.append([tup[0], tup[1], ','.join(tup[2:])])
        self.data = pd.DataFrame(data, columns=self.usecols)
        return self.data
    
    def statistics(self):
        """
        Returns:
            unique_aspect: list, all unique aspects detected in the data
            uid_aspect_frequency_dict: dict, {user_id: {aspect1: count1, aspect2: count2, ...}}
                counting the frequency of each aspect mentioned by each user
    
        """
        self.total_number_lexicons = 0
        self.data = self.read_lexicon()
        unique_aspect = set()
        unique_opinion = set()
        uid_aspect_frequency_dict = {}
        for i, row in self.data.iterrows():
            u_id = row[self.usecols[0]]
            lexicons = row['lexicon'].split(',')
            uid_aspect_frequency_dict[u_id] = {}
            self.total_number_lexicons += len(lexicons)
            for lexicon in lexicons:
                aspect = lexicon.split(':')[0]
                opinion = lexicon.split(':')[1]
                unique_aspect.add(aspect)
                unique_opinion.add(opinion)
                if aspect not in uid_aspect_frequency_dict[u_id].keys():
                    uid_aspect_frequency_dict[u_id][aspect] = 1
                else:
                    uid_aspect_frequency_dict[u_id][aspect] += 1
        self.number_users = self.data[self.usecols[0]].nunique()
        self.number_items = self.data[self.usecols[1]].nunique()
        self.unique_aspects = list(unique_aspect)
        self.unique_opinions = list(unique_opinion)
        self.uid_aspect_frequency_dict = uid_aspect_frequency_dict
        
        print(f"number of users: {self.number_users}")
        print(f"number of items: {self.number_items}")
        print(f"number of unique aspects: {len(self.unique_aspects)}")
        print(f"number of unique opinions: {len(self.unique_opinions)}")
        
        return self.uid_aspect_frequency_dict
    