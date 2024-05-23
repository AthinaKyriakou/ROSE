from ..metrics import Metrics
class DIV_K(Metrics):
    """
    Diversity in keywords: Evaluate the diversity of an explanation using the items keywords
    """

    def __init__(self,name = "diversity in keywords"):
        super().__init__(name=name)

    def compute(self, recommender, explanations):
        """
        Compute the complement of the Jaccard similarity to obtain the distance between items within
        the explanation chain. 

        Parameters
        ----------
        recommender: Recommender
            The recommender at hand that is being utilized
        explanations: Dataframe
            Dataframe holding the user, its recommendation and their explanations for each recommendation

        Returns
        -------
        diversity_avg: float
            The average coverage for this users recommendations
        diversity_list: list
            A list of coverages for each recommended item (interesting for plotting)
            
        References
        -------
        Kaminskas, M. and Bridge, D., 2016.
        Diversity, serendipity, novelty, and coverage: a survey and empirical analysis of beyond-accuracy objectives
        in recommender systems. ACM Transactions on Interactive Intelligent Systems (TiiS), 7(1), pp.1-42.            
        """
        diversity_list = []

        def dist (item, rec):
            # Obtain the keywords for the recommendation and the item
            rec_keywords = recommender.text_data[rec]
            item_keywords = recommender.text_data[item]

            # Compute shared and total features
            shared_features = rec_keywords * item_keywords
            total_features  = rec_keywords + item_keywords

            # Return the complement of the Jaccard similarity
            return 1 - len(shared_features[shared_features!=0]) / len(total_features[total_features!=0])
        

        for i in range(len(explanations)):

            # Compute the pairwise distance between the item within this explanation chain
            diversity = 0
            counter = 0
            for exp1 in explanations[i][2]:
                for exp2 in explanations[i][2]:
                    if exp1 == exp2:
                        continue
                    diversity += dist(exp1, exp2)
                    counter +=1
            if diversity != 0:
                diversity /= counter

            # Append the coverage to the list
            diversity_list.append(diversity)
        diversity_avg = sum(diversity_list) / len(explanations)
        return diversity_avg, diversity_list
