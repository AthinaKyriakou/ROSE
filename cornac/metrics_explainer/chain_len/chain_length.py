from ..metrics import Metrics
class C_LEN(Metrics):
    """
    Chain Length: How many items do explanations contain
    -> The lower the better (based on the fact that users tend to prefer a straightforward explanation without too much complexity)
    """

    def __init__(self, name = "chain length"):
        super().__init__(name=name)

    def compute(self, explanations):
        """
        Compute the average length of the explanation list

        Parameters
        ----------
        explanations: Dataframe
            Dataframe holding the user, its recommendation and their explanations for each recommendation

        Returns
        -------
        len_avg: float
            The average length for this users recommendations
        len_list: list
            A list of lengths for each recommended item (interesting for plotting)
        """
        len_list = []
        for i in range(len(explanations)):
            # Extract the number of items present in the explanations
            len_list.append(len(explanations[i][2]))

        len_avg = sum(len_list) / len(explanations)
        return len_avg, len_list
