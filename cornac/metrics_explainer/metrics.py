class Metrics:
    def __init__(self, name, rec_k=10, feature_k=10):
        self.name = name
        self.rec_k = rec_k
        self.feature_k = feature_k
    
    def compute(self, **kwargs):
        raise NotImplementedError()
