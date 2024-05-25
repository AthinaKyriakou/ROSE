from .explainer import Explainer

import networkx as nx
import heapq


class Exp_PRINCE(Explainer):
    """PRINCE: Counterfactual Explanations for Personalized PageRank-based Recommendations.

    Parameters
    ----------
    rec_model: object, recommender model
        The recommender model to be explained.
    dataset: object, dataset
        The dataset object that is used to explain, should have item_graph.
    rec_k: int, optional, default: 10
        Number of top-ranked recommendations to consider.
    name: string, optional, default: 'Exp_PRINCE'
    
    References
    ----------
    [1] Azin Ghazimatin, Oana Balalau, Rishiraj Saha Roy, and Gerhard Weikum. 2020.
    PRINCE: Provider-side Interpretability with Counterfactual Explanations in Recommender Systems.
    In The Thirteenth ACM International Conference on Web Search and Data Mining (WSDM 20),
    February 3-7, 2020, Houston, TX, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3336191.3371824
    """

    def __init__(self, 
                 rec_model, 
                 dataset,
                 rec_k=10,
                 name="Exp_PRINCE"
                 ):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset) 
        self.rec_k = rec_k

        if self.model is None:
            raise NotImplementedError("The model is None.")
        if self.dataset is None:
            raise NotImplementedError("The dataset is None.")
        if hasattr(self.dataset, "train_set"):
            self.dataset = self.dataset.train_set

        self.G = nx.DiGraph()
        self._build_graph()
        

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
        """Provide explanation for one user and one item

        Parameters
        ----------
        user_id: str
            One user's id.
        item_id: str
            One item's id.

        Returns
        -------
        explanations: dict
            A dictionary containing the item as key and a list of items as value. 
            If user did not interact with the itmes in value, then the recommendation will be change to the key item.
        """
        if user_id not in self.dataset.uid_map:
            print(f"User {user_id} not in dataset")
            return {}
        user_idx = self.dataset.uid_map[user_id]
        if user_idx < 0 or user_idx >= self.dataset.num_users:
            return {}

        if item_id not in self.dataset.iid_map:
            print(f"Item {item_id} not in dataset")
            return {}
        item_idx = self.dataset.iid_map[item_id]
        if item_idx < 0 or item_idx >= self.dataset.num_items:
            return {}

        user_id_graph = self._get_user_id_for_graph(user_idx)
        A_star = self.G.out_edges(user_id_graph)
        if len(A_star) == 0:
            return {}
        rec_star = self._get_item_id_for_graph(item_idx)

        item_id_graph = self._get_item_id_for_graph(item_idx)
        user_id_graph = self._get_user_id_for_graph(user_idx)

        candidate_items = self._get_candidate_items(user_idx) - set([rec_star])
        for item in candidate_items:
            A_i = self._swap_order(user_id_graph, rec_star, item)
            if len(A_i) == 0:
                continue
            if len(A_i) < len(A_star):
                A_star = A_i
                rec_star = item
            elif len(A_i) == len(A_star):
                if self._PPR(user_id_graph, item, A_i) > self._PPR(
                    user_id_graph, rec_star, A_i
                ):
                    A_star = A_i
                    rec_star = item

        if rec_star == self._get_item_id_for_graph(item_idx):
            return {}
        
        A_star = [self._get_item_id_from_graph_id(item_id) for _, item_id in A_star]
        rec_star = self._get_item_id_from_graph_id(rec_star)
        
        return {rec_star: A_star}
    
    def _get_candidate_items(self, user_idx):
        """
        Recommend k items for a user_idx.
        """
        top_k, _ = self.model.rank(user_idx, k=self.rec_k)
        top_k = top_k[:self.rec_k]
        candidate_items = set()
        for item_idx in top_k:
            candidate_items.add(self._get_item_id_for_graph(item_idx))
        return candidate_items

    def _swap_order(self, user_id, rec, rec_star):
        A = {(user_id, ni) for ni in self.G.successors(user_id) if ni != user_id}
        A_star = set()
        H = self.MaxHeap()
        sum_diff = 0

        for user, ni in A:
            diff = self.G[user][ni]["weight"] * (
                self._PPR(ni, rec, A) - self._PPR(ni, rec_star, A)
            )
            sum_diff += diff
            H.insert(ni=ni, diff=diff)

        while sum_diff > 0 and not H.isEmpty():
            (diff, ni) = H.delete_max()
            sum_diff -= diff
            A_star.add((user_id, ni))

        if sum_diff > 0:
            A_star = A
        return A_star

    def _PPR(self, user_id, item_id, A):
        G_modified = self.G.copy()
        G_modified.remove_edges_from(A)

        npr = nx.pagerank(G_modified, personalization={user_id: 1})
        return npr[item_id]

    
    def _get_item_id_from_graph_id(self, item_id):
        item_idx = self._get_item_idx_from_graph_id(item_id)
        return self.dataset.item_ids[item_idx]
    
    def _get_item_id_for_graph(self, item_idx):
        # if item_idx is str and starts with 'I', then return it as it is
        if isinstance(item_idx, str) and item_idx.startswith("I"):
            return item_idx
        return "I" + str(item_idx)

    def _get_item_idx_from_graph_id(self, item_id):
        # if item_id is already an integer, then return it as it is
        if isinstance(item_id, int):
            return item_id
        return int(item_id[1:])

    def _get_user_id_for_graph(self, user_idx):
        if isinstance(user_idx, str) and user_idx.startswith("U"):
            return user_idx
        return "U" + str(user_idx)

    def _get_user_idx_from_graph_id(self, user_id):
        if isinstance(user_id, int):
            return user_id
        return int(user_id[1:])
    
    def _get_user_id_from_graph_id(self, user_id):
        user_idx = self._get_user_idx_from_graph_id(user_id)
        return self.dataset.user_ids[user_idx]

    def _build_graph(self):
        """
        Build Graph from the dataset.
        """
        item_graph = self.dataset.item_graph
        train_item_indices = range(
            self.dataset.num_items
        )  # num_items is the number of items in the training set
        for i in train_item_indices:
            self.G.add_node(self._get_item_id_for_graph(i), type="item")
        for u in range(self.dataset.num_users):
            self.G.add_node(self._get_user_id_for_graph(u), type="user")

        (rid, cid, val) = item_graph.get_train_triplet(
            train_item_indices, train_item_indices
        )

        for i in range(len(rid)):
            src = self._get_item_id_for_graph(rid[i])
            dst = self._get_item_id_for_graph(cid[i])
            self.G.add_edge(src, dst, weight=val[i])

        uir_tuple = self.dataset.uir_tuple
        for i in range(len(uir_tuple)):
            src = self._get_user_id_for_graph(uir_tuple[i][0])
            dst = self._get_item_id_for_graph(uir_tuple[i][1])
            # weight should not be rating score
            self.G.add_edge(src, dst, weight=1)

        return self.G

    class MaxHeap:
        def __init__(self):
            self.heap = []

        def insert(self, ni, diff):
            heapq.heappush(self.heap, (-diff, ni))

        def delete_max(self):
            diff, ni = heapq.heappop(self.heap)
            return -diff, ni

        def isEmpty(self):
            return len(self.heap) == 0
