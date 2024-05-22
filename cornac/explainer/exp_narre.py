import numpy as np
from .explainer import Explainer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
import tensorflow as tf

class Exp_NARRE(Explainer):
    """Explainer for Neural Attentional Rating Regression with Review-level Explanations (NARRE).
    Explains the recommendation by the attention scores of the reviews.
    
    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain.
    name: string, optional, default: 'Exp_NARRE'
    
    References
    ----------
    [1] hen, C., Zhang, M., Liu, Y., & Ma, S. (2018, April). Neural attentional rating regression with review-level explanations. In Proceedings of the 2018 World Wide Web Conference (pp. 1583-1592).
    """

    def __init__(self, rec_model, dataset, name="Exp_NARRE"):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset)
        
        self.all_item_attention = []
        self.all_item_review_ids = []

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """Provide explanation for one user and one item

        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.
        feature_k: int, optional, default:10
            Number of features in explanations created by explainer.

        Returns
        -------
        explanations: list
            List of tuples (attention score, review text) for the item.
        
        """
        
        explanation = {}
        
        if self.model is None:
            raise NotImplementedError("The model is None.")
            
        if not hasattr(self.model, "model"):
            raise AttributeError("The explainer does not support this recommender.")
        
        if self.model.model.graph is None:
            raise NotImplementedError("The model is not trained yet.")
        
        feature_k = kwargs.get('feature_k', 10)

        narre = self.model.model
        train_set = self.dataset
        if hasattr(train_set, 'train_set'):
            train_set = train_set.train_set
        

        if len(self.all_item_attention) == 0:
            all_item_review_ids = []
            all_item_attention = []
            for batch_items in train_set.item_iter(64):
                i_item_review, i_item_uid_review, i_item_num_reviews, i_item_review_ids = _get_data(batch_items, train_set, narre.max_text_length, by='item', max_num_review=narre.max_num_review)
                all_item_review_ids.extend(i_item_review_ids)
                item_review_embedding = narre.graph.l_item_review_embedding(i_item_review)
                item_review_h = narre.graph.item_text_processor(item_review_embedding, training=False)
                a_item = narre.graph.a_item(tf.concat([item_review_h, narre.graph.l_item_uid_embedding(i_item_uid_review)], axis=-1))
                a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=i_item_review.shape[1]), -1)
                item_attention = narre.graph.item_attention(a_item, a_item_masking)
                item_attention = item_attention.numpy()
                all_item_attention.extend(item_attention)

            all_item_attention = np.array(all_item_attention)
            all_item_review_ids = np.array(all_item_review_ids)
            all_item_attention = np.squeeze(all_item_attention, axis=-1)
            self.all_item_attention = all_item_attention
            self.all_item_review_ids = all_item_review_ids
            
        
        item_idx = self.dataset.iid_map[item_id]
        if self.model.is_unknown_item(item_idx):
            return explanation
        attention = self.all_item_attention[item_idx]
        review_ids = self.all_item_review_ids[item_idx]
        top_k = np.argsort(attention)[::-1][:feature_k]
        
        for idx in top_k:
            # explanation.append((attention[idx], train_set.review_text.reviews[review_ids[idx]]))
            text = train_set.review_text.reviews[review_ids[idx]]
            explanation[text] = attention[idx]
        
        return explanation
        

def _get_data(batch_ids, train_set, max_text_length, by='user', max_num_review=None):
    batch_reviews, batch_id_reviews, batch_num_reviews = [], [], []
    batch_reviews_ids = []
    review_group = train_set.review_text.user_review if by == 'user' else train_set.review_text.item_review
    for idx in batch_ids:
        ids, review_ids = [], []
        for inc, (jdx, review_idx) in enumerate(review_group[idx].items()):
            if max_num_review is not None and inc == max_num_review:
                break
            ids.append(jdx)
            review_ids.append(review_idx)
        batch_id_reviews.append(ids)
        batch_reviews_ids.append(review_ids)
        reviews = train_set.review_text.batch_seq(review_ids, max_length=max_text_length)
        batch_reviews.append(reviews)
        batch_num_reviews.append(len(reviews))
    batch_reviews = pad_sequences(batch_reviews, maxlen=max_num_review, padding="post")
    batch_id_reviews = pad_sequences(batch_id_reviews, maxlen=max_num_review, padding="post")
    batch_num_reviews = np.array(batch_num_reviews)
    batch_reviews_ids = pad_sequences(batch_reviews_ids, maxlen=max_num_review, padding="post")
    return batch_reviews, batch_id_reviews, batch_num_reviews, batch_reviews_ids