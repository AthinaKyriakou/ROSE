from .standard_params import Std_params

import bottleneck as bn
import numpy as np

def Hit_Ratio_at_k_batch(logits: np.ndarray, k=10, sum=True):
    """
    Hit Ratio. It expects the positive logit in the first position of the vector.
    :param logits: Logits. Shape is (batch_size, n_neg + 1).
    :param k: threshold
    :param sum: if we have to sum the values over the batch_size. Default to true.
    :return: HR@K. Shape is (batch_size,) if sum=False, otherwise returns a scalar.
    """

    assert logits.shape[1] >= k, 'k value is too high!'

    # Partition the logits so that the first k elements belong to the k highest logits
    # bn.argpartition stores the indices that belong to the k highets logits, when
    # cutting it at k, we end up with exactly the 10 highest logits
    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    # Lastly we check, if the first element with index 0 (our only positive item)
    # is contained in the top k items, if so, we store it as 1
    hrs = np.any(idx_topk_part[:] == 0, axis=1).astype(int)

    # Sum up the results, hence we end up with the number of elements in a batch
    # that predicts the positive item in the top k items
    if sum:
        return np.sum(hrs)
    else:
        return hrs


def NDCG_at_k_batch(logits: np.ndarray, k=10, sum=True):
    """
    Normalized Discount Cumulative Gain. It expects the positive logit in the first position of the vector.
    :param logits: Logits. Shape is (batch_size, n_neg + 1).
    :param k: threshold
    :param sum: if we have to sum the values over the batch_size. Default to true.
    :return: NDCG@K. Shape is (batch_size,) if sum=False, otherwise returns a scalar.
    """
    assert logits.shape[1] >= k, 'k value is too high!'
    n = logits.shape[0]
    dummy_column = np.arange(n).reshape(n, 1)

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    topk_part = logits[dummy_column, idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[dummy_column, idx_part]

    rows, cols = np.where(idx_topk == 0)
    ndcgs = np.zeros(n)

    if rows.size > 0:
        ndcgs[rows] = 1. / np.log2((cols + 1) + 1)

    if sum:
        return np.sum(ndcgs)
    else:
        return ndcgs


class Evaluator:
    """
    Helper class for the evaluation. When called with eval_batch, it updates the internal results. After the last batch,
    get_results will return the aggregated information for all users.
    """

    def __init__(self, n_users: int):
        self.n_users = n_users

        self.metrics_values = {}

    def eval_batch(self, out: np.ndarray, sum: bool = True):
        """
        Evaluates the given batch, important to say that this evaluator is specifically build for ProtoMF's sampling
        # strategy, meaning we know, the logit in the first position of out belongs to a positively rated item, the
        # remaining ones to negatively rated ones, this is the reason we don't need any additional "true labels"
        :param out: Values after last layer. Shape is (batch_size, n_neg + 1).
        """
        k = Std_params.K
        for metric_name, metric in zip(['ndcg@{}', 'hit_ratio@{}'], [NDCG_at_k_batch, Hit_Ratio_at_k_batch]):
            if sum:
                self.metrics_values[metric_name.format(k)] = self.metrics_values.get(metric_name.format(k), 0) + metric(out, k)
            else:
                self.metrics_values[metric_name.format(k)] = self.metrics_values.get(metric_name.format(k),[]) + list(metric(out, k, False))

    def get_results(self, aggregated=True):
        """
        Returns the aggregated results (avg).
        """
        if aggregated:
            for metric_name in self.metrics_values:
                self.metrics_values[metric_name] /= self.n_users

        # Return the metrics in a dictionary and clears it for the next evaluation
        metrics_dict = self.metrics_values
        self.metrics_values = {}

        return metrics_dict
