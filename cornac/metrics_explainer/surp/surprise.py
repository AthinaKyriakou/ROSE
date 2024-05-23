import numpy as np
from ..metrics import Metrics
class SURP(Metrics):
    """
    Surprise: how surprising is an item to a user
    -> The higher the better
    !!! This is NOT an explanation metric, however, we add it in this folder to jointly evaluate all
        metrics for Rec-by-E and avoid unnecessary extra computation times to regenerate chains
    """

    def __init__(self,name = "surprise"):
        super().__init__(name=name)

    def compute(self, recommender, explanations):
        """
        Search the item in the users profile that is closest to the users profile measured in
        the complement of Jaccard similarity, the distance to the closest item can be interpreted
        as lower bound of how surprising this recommendation is

        Parameters
        ----------
        recommender: Recommender
            The recommender at hand that is being utilized

        explanations: Dataframe
            Dataframe holding the user, its recommendation and their explanations for each recommendation

        Returns
        -------
        surprise_avg: float
            The average surprise for this users recommendations
        surprise_list: list
            A list of surprise values for each recommended item (interesting for plotting)

        References
        -------
        Kaminskas, M. and Bridge, D., 2016.
        Diversity, serendipity, novelty, and coverage: a survey and empirical analysis of beyond-accuracy objectives
        in recommender systems. ACM Transactions on Interactive Intelligent Systems (TiiS), 7(1), pp.1-42.
        """
        surprise_list = []
        for i in range(len(explanations)):

            # Obtain the users profile
            user_id = int(explanations[i][0])
            # Extract all the indices of interactions of this user
            ind_user = np.where(recommender.train_set.uir_tuple[0] == user_id)[0]
            # Now get all the items ID's in these indices
            user_profile = recommender.train_set.uir_tuple[1][ind_user]

            def dist (item, rec):
                # Obtain the keywords for the recommendation and the item
                rec_keywords = recommender.text_data[rec]
                item_keywords = recommender.text_data[item]

                # Compute shared and total features
                shared_features = rec_keywords * item_keywords
                total_features  = rec_keywords + item_keywords

                # Return the complement of the Jaccard similarity
                return 1 - len(shared_features[shared_features!=0]) / len(total_features[total_features!=0])
            
            min_dist = min([dist(item, explanations[i][1]) for item in user_profile])

            # Append the surprise to the list
            surprise_list.append(min_dist)
        surprise_avg = sum(surprise_list) / len(explanations)
        return surprise_avg, surprise_list
