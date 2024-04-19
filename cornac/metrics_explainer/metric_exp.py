class Metric_Exp:
    """ Metric_Exp class is the base class for all metrics for explainers.
    
    Parameters
    ----------
    name: str
        The name of the metric.
    rec_k: int, optional, default:10
        The number of recommendations.
    feature_k: int, optional, default:10
        The number of features in explanations created by explainer.   
    """
    def __init__(self, name, rec_k=10, feature_k=10):
        self.name = name
        self.rec_k = rec_k
        self.feature_k = feature_k
    
    def compute(self, **kwargs):
        """Compute the metric. This method should be implemented in the child class.
        """
        raise NotImplementedError()
