from ..metrics import Metrics
class COV(Metrics):
    """
    Coverage: How many keywords of the recommendation are truly covered by its explanation
    -> The higher the better
    """

    def __init__(self,name = "coverage"):
        super().__init__(name=name)

    def compute(self, recommender, explanations):
        """
        Compute the percentage of coverage of the recommendation by its explanation

        Parameters
        ----------
        recommender: Recommender
            The recommender at hand that is being utilized
        explanations: Dataframe
            Dataframe holding the user, its recommendation and their explanations for each recommendation

        Returns
        -------
        coverage_avg: float
            The average coverage for this users recommendations
        coverage_list: list
            A list of coverages for each recommended item (interesting for plotting)
        """
        coverage_list = []
        for i in range(len(explanations)):
            # Obtain the keywords for the recommendations and all keywords contained in the explanation
            rec_keywords = recommender.text_data[explanations[i][1]]
            exp_keywords = sum([recommender.text_data[elem] for elem in explanations[i][2]])

            # Set all values to either 0 or 1 (higher values possible if keywords appear more than once or through the summation)
            rec_keywords[rec_keywords != 0] = 1
            exp_keywords[exp_keywords != 0] = 1

            # Compute the keywords that the recommendation shares with its explanation
            covered_keywords = rec_keywords * exp_keywords

            # Append the coverage to the list
            coverage_list.append(sum(covered_keywords) / sum(rec_keywords))
        coverage_avg = sum(coverage_list) / len(explanations)
        return coverage_avg, coverage_list
