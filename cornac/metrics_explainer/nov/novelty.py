import math
from ..metrics import Metrics
class NOV(Metrics):
    """
    Novelty: How novel is the recommended item in this dataset
    -> The higher the better
    !!! This is NOT an explanation metric, however, we add it in this folder to jointly evaluate all
        metrics for Rec-by-E and avoid unnecessary extra computation times to regenerate chains
    """

    def __init__(self,name = "novelty"):
        super().__init__(name=name)

    def compute(self, recommender, explanations):
        """
        Compute the fraction of users who have not interacted with this item yet
        The logarithm is utilized to emphasize novelty of rare items as explained in the below paper

        Parameters
        ----------
        recommender: Recommender
            The recommender at hand that is being utilized
        explanations: Dataframe
            Dataframe holding the user, its recommendation and their explanations for each recommendation

        Returns
        -------
        novelty_avg: float
            The average novelty for this users recommendations
        novelty_list: list
            A list of novelty values for each recommended item (interesting for plotting)

        References
        -------
        Kaminskas, M. and Bridge, D., 2016.
        Diversity, serendipity, novelty, and coverage: a survey and empirical analysis of beyond-accuracy objectives
        in recommender systems. ACM Transactions on Interactive Intelligent Systems (TiiS), 7(1), pp.1-42.
        """
        novelty_list = []
        for i in range(len(explanations)):

            recommended_item = int(explanations[i][1])

            # Compute how many users have interacted with this item
            n_interactions = sum(recommender.train_set.uir_tuple[1] == recommended_item)

            # Compute novelty according to formula from above mentioned paper
            novelty = - math.log2(n_interactions / recommender.train_set.num_users)

            # Append the novelty value to the list
            novelty_list.append(novelty)
        novelty_avg = sum(novelty_list) / len(explanations)
        return novelty_avg, novelty_list
