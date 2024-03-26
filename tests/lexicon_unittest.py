import unittest
import os
import pandas as pd
from cornac.data import SentimentAnalysis

class TestProcessToSentimentTxt(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.input_file = './dataset/input/data_for_demo.csv'
        cls.output_sentiment_file = './dataset/output/test_output_sentiment.txt'
        cls.output_rating_file = './dataset/output/test_output_rating.txt'
        # if True, remove the generated output files
        cls.if_remove = True
        # Create a test input file if it doesn't exist
        if not os.path.exists(cls.input_file):
            cls.if_remove = True
            with open(cls.input_file, 'w') as f:
                f.write('user_id\tbook_id\trating\treview_text\n')
                f.write('1\t1\t1.0\tIt\'s a great book. That is a really beautiful cover\n')
                f.write('2\t2\t4.0\tThis is a terrible book.\n')
                f.write('3\t3\t5.0\tI think it\'s a good book\n')
    @classmethod  
    def tup_parser(cls, tokens):
        return [
            (
                tokens[0],
                tokens[1],
                [tuple(tup.split(':')) for tup in tokens[2:]],
            )
        ]
        
    @classmethod    
    def test_process_to_sentiment_txt(cls):
        """
            Test the class SentimentAnalysis
                Expected Input file with user, item, rating, review
                Expected Output file: 
                    sentiment file with columns: user, item, lexicon
                    rating file with columns: user, item, rating
                
        """
        #initiate the class
        SA = SentimentAnalysis(input_path=cls.input_file, sep='\t', usecols=['user_id', 'book_id', 'rating', 'review_text'], min_frequency=1)
        # build the lexicons
        SA.build_lexicons()
        # save lexicons to file
        SA.save_to_file(cls.output_sentiment_file, cls.output_rating_file)
        
        # read constructed data from the output files
        with open(cls.output_sentiment_file, encoding="utf-8") as f:
            output_sent = [
                tup
                for idx, line in enumerate(f)
                for tup in cls.tup_parser(line.strip().split(','))
            ]
        with open(cls.output_rating_file, encoding="utf-8") as f:
            output_rating = [
                tuple(line.split(','))
                    for idx, line in enumerate(f)
                ]

        # Check that the elements in a lexicon is type of [string, string, float]
        for idx, (uid, iid, sentiment_tuples) in enumerate(output_sent):
            #print(sentiment_tuples)
            for tup in sentiment_tuples:
                try:
                    aspect, opinion, polarity = tup[0], tup[1], float(tup[2])
                    assert isinstance(aspect, str) # "aspect should be of type str"
                    assert isinstance(opinion, str) # "opinion should be of type str"
                    assert isinstance(polarity, float) # "polarity should be of type float"
                except:
                    print(f'Error happened in data: {idx}, {uid}, {iid}, {tup}')
                    continue
        # Check that the elements in a rating is type of [string, string, float]
        for idx, (uid, iid, rating) in enumerate(output_rating):
            rating = float(rating)
            assert isinstance(rating, float)
            
        # The sentiment files and ratings should be same length
        assert len(output_sent) == len(output_rating)
        
    @classmethod
    def tearDown(cls):
        # Remove the test input and output files
        if cls.if_remove:
            os.remove(cls.input_file) 
            os.remove(cls.output_sentiment_file)
            os.remove(cls.output_rating_file)
            print("Test complete")
            print("Remove the generated output files Successfully")

if __name__ == '__main__':
    unittest.main()