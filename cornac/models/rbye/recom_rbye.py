from ..recommender import Recommender
from .standard_params import Std_params

import numpy as np
import pandas as pd
import tqdm
import time

"""
Ideas for speed-up:
Somehow generate chains for all items at the same time
Maybe use multi threading for acceleration
"""

class RecByExp(Recommender):
    """
    Recommendation by Explanation
    This class implements the Rec-by-E recommender model as proposed in the below mentioned paper, generally, we generate
    explanation chains for each unknown item for each user. These chains help explaining the unknown item. Within the chain
    all items exceed a certain similarity threshold (theta) to its neighbours and are explaining the target item in some way.
    After the generation process, this algorithm chooses the best n explanation chains and returns their target items as
    recommendations. The algorithm hereby takes diversity into account and aims at returning recommendations with a wide variety.

    Parameters
    ----------
    n: int, optional, default: according to predefined standard parameters in the standard_params file
        The number of recommendations that should be returned per user

    theta: float, optional, default: according to predefined standard parameters in the standard_params file
        Similarity hreshold for the minimally accepted similarity to the predecessor when adding a new member to an explanation chain

    epsilon: float, optional, default: according to predefined standard parameters in the standard_params file
        Reward threshold for the minimally accepted reward when adding a new member to an explanation chain

    It should be noted that the additional parameter MAX_LEN can be adjusted in the Std_params file. This parameter makes sure the
    generation process is limited to a max of 4 explaining items, if sufficient computational resources are available, it can be omitted

    References
    ----------
    * Rana, A. and Bridge, D.,
      2018, July. Explanations that are intrinsic to recommendations. In Proceedings of the
      26th Conference on User Modeling, Adaptation and Personalization (pp. 187-195).
    """

    def __init__(self, n=Std_params.N_REC, theta=Std_params.THETA, epsilon=Std_params.EPSILON, name="RbyE", trainable=True, verbose=False):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.n = n
        self.theta = theta
        self.epsilon = epsilon
    
    def fit(self, train_set, val_set=None):
        """
        Precomputes similarity scores between all items to avoid the repeated computation for each user

        Parameters
        ----------
        train_set: required
            The training set used for this model

        val_set: int, optional, default: None
            The validation set used for this model
        """        
        # Reset all info about the current status such as best value, best epoche etc.
        self.reset_info()
        # Set this recommenders train and validation datasets to the given datasets (reset the random number generator for reproducibility)
        self.train_set = train_set.reset()
        self.val_set = None if val_set is None else val_set.reset()

        # Adjust the text data in order to work with it more easily
        self.text_data = self.train_set.item_text.batch_bow(np.arange(len(self.train_set.item_ids)))

        # Precompute similarities between items to speed up the computation
        start = time.time()
        item_ids = list(self.train_set.item_data.keys())
        num_items = len(item_ids)
        self.similarities = np.zeros((num_items, num_items))
        for item_id1 in item_ids:
            for item_id2 in item_ids:
                self.similarities[item_id1, item_id2] = self.get_similarity(item_id1, item_id2)
        print(f'Time needed to compute similarities: {time.time()-start}')

        return self
    
    # As our model is not being trained on any parameters, we can directly rank items based on the user 
    def rank(self, user_id, include_expl=False):
        """
        Predicts the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.
        include_expl: boolean, optional, default: False
            If True, the function additionally returns the list of explanation chains
            
        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        self.user_id = user_id

        # Extract all the indices of interactions of this user
        ind_user = np.where(self.train_set.uir_tuple[0] == self.user_id)[0]
        # Now get all the items ID's in these indices
        user_profile = self.train_set.uir_tuple[1][ind_user]

        # STEP 1: Create explanation chains for the given user
        explanation_chains, rewards = self.generate_chains(user_profile)

        # STEP 2: Sort the chains according to different factors and take the n highest chains and return them as recommendation
        ranked_recommended_chains, their_scores = self.rank_chains(explanation_chains, rewards)
        recommendations_ranked = [chain[0] for chain in ranked_recommended_chains]
        if include_expl:
            their_explanations = [chain[1:] for chain in ranked_recommended_chains]
            return recommendations_ranked, their_scores, their_explanations
        return recommendations_ranked, their_scores


    def generate_chains(self, user_profile):
        """
        This function generates explanation chains for all items for the current user

        Parameters
        ----------
        user_profile: list
            A list of all items the user has interacted with

        Returns
        -------
        explanation_chains: list
            A list with explanation chains for all unkown items
        rewards: list
            A list containing the rewards for the corresponding explanation chains
        """

        # For each item in the data, generate a chain
        explanation_chains = []
        rewards = []
        # Iterate over all items and create explanation chains for the ones that do not appear in a users profile
        for candidate in tqdm.tqdm(self.train_set.item_data.keys()):
            # If the user has already interacted with this item, do not create a chain
            if candidate in user_profile:
                continue
            chain_reward = 0
            chain = [candidate]

            # Save the possible predecessors in a separate list, each time an item is added to the chain, we delete it from the predecessor candidates
            pred_candidates = set(user_profile)
            # If the current item is known to the user, remove it from the possible predecessors as an item should not be explained by itself
            # This step is only needed if explanation chains are created for known items as well -> ideal for larger dataset
            if candidate in pred_candidates:
                pred_candidates.remove(candidate)
            while True:
                # Stop the loop after the chain contains reaches the MAX_LEN in order to limit the computation time
                if len(chain) >= Std_params.MAX_LEN:
                    break

                # Check for all predecessor candidates that fulfill the requirements
                pred_candidates_array = np.array(list(pred_candidates))

                # Calculate mask for valid predecessor candidates using numpy operations
                valid_pred_candidates_mask_sim = (self.similarities[pred_candidates_array, chain[-1]] > self.theta)
                valid_pred_candidates_mask_rew = (np.vectorize(lambda item: self.get_reward(item, candidate, chain))(pred_candidates_array) > self.epsilon)
                valid_pred_candidates_mask = valid_pred_candidates_mask_sim & valid_pred_candidates_mask_rew

                # Extract valid predecessor candidates using the mask
                valid_pred_candidates = pred_candidates_array[valid_pred_candidates_mask]
                if len(valid_pred_candidates)==0:
                    break

                # Of all valid predecessor candidates, select the one which yields the highest reward
                best_item_id = max(valid_pred_candidates, key=lambda x: self.get_reward(x, candidate, chain))

                # Add the additional reward to this explanation chains reward
                chain_reward += self.get_reward(best_item_id, candidate, chain)

                # Append this item to the explanation chain and delete it from the potential predecessors
                chain.append(best_item_id)
                pred_candidates.remove(best_item_id)

            # If this candidate has at least one other item in its explanation chain, add it to the list of explanation chains
            if len(chain) > 1:
                explanation_chains.append(chain)
                rewards.append(chain_reward)
        return explanation_chains, rewards

    def rank_chains(self, explanation_chains, rewards):
        """
        This function takes a list of explanation chains and their rewards and orders them
        according to how suited they are for a recommendation.
        Based on this order, we can choose the first n chains as recommendations
        to a user.

        Parameters
        ----------
        explanation_chains: list
            A list of explanation chains that needs to be ordered
        rewards: list
            A list with the according rewards for each chain

        Returns
        -------
        ranked_recommended_chains: list
            A ranked list of the given explanation chains
        ranked_scores: list
            A list containing the scores in the same order as the chains
        """

        # Order the chains, consider diversity of chains by computing the score
        ranked_recommended_chains = []
        ranked_scores = []
        while len(explanation_chains) != 0:
            # Determine the best chain by computing the scores if a particular chain was added
            best_chain, best_score = max(zip(explanation_chains, rewards), key=lambda x: get_score(x[0], x[1], ranked_recommended_chains))

            # Add the best chain to the recommended chains and remove it from the search space
            ranked_recommended_chains.append(best_chain)
            ranked_scores.append(best_score)
            explanation_chains.remove(best_chain)
        return ranked_recommended_chains, ranked_scores


    def get_similarity(self, poss_pred, current_item):
        """
        This function computes the Jaccard similarity of two items, in our case we compute the similarity to
        be able to select the most similar item as next predecessor

        Parameters
        ----------
        poss_pred: int
            The ID of the possible predecessor
        current_item: int
            The ID of the current 

        Returns
        -------
        sim_score: int
            A similarity measure stating how similar the two given items are
        """
        shared_features = self.text_data[poss_pred] * self.text_data[current_item]
        total_features = self.text_data[poss_pred] + self.text_data[current_item]
        sim_scor = len(shared_features[shared_features!=0]) / len(total_features[total_features!=0])
        return sim_scor


    def get_reward(self, poss_pred, candidate, chain):
        """
        This function computes the reward that would be added to the chains reward, if this predecessor is added

        Parameters
        ----------
        poss_pred: int
            The ID of the possible predecessor
        candidate: int
            The ID of the current candidate for which an explanation chain is being created
        chain: numpy array
            An array of the items that are currently part of this chain 

        Returns
        -------
        reward: float
            The reward that is being generated if this poss_pred is being attached to the chain
        """
        pred_features = self.text_data[poss_pred]
        cand_features = self.text_data[candidate]
        # Sum up the text vectors, all non-zero values indicate that the chain contains this word
        features_covered_by_chain = sum([self.text_data[elem] for elem in chain])
        # Make sure NOT to consider the first element as this is the candidate itself
        features_covered_by_chain = features_covered_by_chain - cand_features

        # Convert all counts to either 1 or 0 which indicates that this word is either present or not
        features_covered_by_chain[features_covered_by_chain != 0] = 1
        cand_features[cand_features != 0] = 1
        # Create a new array that only shows the features of the candidate that are uncovered by the current chain
        uncovered_features = cand_features - features_covered_by_chain
        uncovered_features[uncovered_features <= 0] = 0

        # Now multiply the uncovered features with the ones present in the possible predecessor to extract how many can be covered
        shared_features = uncovered_features * pred_features
        reward = len(shared_features[shared_features!=0]) / len(cand_features[cand_features!=0]) + len(shared_features[shared_features!=0]) / len(pred_features[pred_features!=0])
        return reward

    def recommend(self, user_ids, n=10, filter_history=True):
        """
        Provide recommendations for a list of users, as this recommender includes the explanation in the
        recommendation process we additionally return the explanation for each recommendation

        Parameters
        ----------
        user_ids: list
            list of users
        n: int
            number of recommendations that should be made per user
        filter history: boolean
            do not recommend items from users history

        Returns
        -------
        recommendations: Dataframe
            Dataframe holding the top n predictions per user together with an explanation in form of a chain if items
        """

        recommendation = []
        uir_df = pd.DataFrame(np.array(self.train_set.uir_tuple).T, columns=['user', 'item', 'rating'])
        uir_df['user'] = uir_df['user'].astype(int)
        uir_df['item'] = uir_df['item'].astype(int)

        for uid in tqdm.tqdm(user_ids):
            if uid not in self.train_set.uid_map:
                continue
            user_idx = self.train_set.uid_map[uid]
            item_rank, _, item_explanation = self.rank(user_idx, include_expl=True)
            recommendation_one_user = []
            if filter_history:
                user_rated_items = uir_df[uir_df['user'] == user_idx]['item']
                # remove user rated items from item_rank
                recommendation_one_user = [[uid, item_rank[i], item_explanation[i]] for i in range(len(item_rank)) if item_rank[i] not in user_rated_items][:n]
            else:
                recommendation_one_user = [[uid, item_rank[i], item_explanation[i]] for i in range(len(item_rank))][:n]
            recommendation.extend(recommendation_one_user)
        recommendations = pd.DataFrame(recommendation, columns=['user_id', 'item_id', 'explanations'])
        return recommendations
        

def get_score(chain_candidate, chain_candidates_reward, recommended_chains):
    """
    This function computes the score of an explanation chain given the already selected explanation chains

    Parameters
    ----------
    chain_candidate: list
        The explanation chain for which we compute the score
    chain_candidates_reward: float
        The computed reward for this explanation chain for which a score is computed
    recommended_chains: list
        A list of all explanation chains that have already been chosen as recommendations

    Returns
    -------
    score: float
        The score that this explanation chain would return if it was added to the recommendation list
    """
    # Detect how many items in the chain candidate (excluding the recommended item itself) are new to the recommended chains
    items_in_chain = set(sum(recommended_chains[1:], []))
    # Detect those items that the new explanation chain covers that have not yet been covered
    new_items = [item for item in chain_candidate[1:] if item not in items_in_chain]
    score = chain_candidates_reward / len(chain_candidate)  +  len(new_items) / len(chain_candidate)
    return score



