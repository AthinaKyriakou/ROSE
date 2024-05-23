import networkx as nx

from ..recommender import Recommender


class PPRBased(Recommender):

    def __init__(self, name="PPR-based", trainable=True, verbose=False, ignore_neighbors=True, graph=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.ignore_neighbors = ignore_neighbors
        self.graph = graph

    def score(self, user_idx, item_idx=None, ignore_users_neighbors=True):
        assert self.graph is not None, "Graph is None"
        assert user_idx in self.graph.nodes, "User not in the graph"
        assert item_idx is not None, "Item is None"
        assert item_idx in self.graph.nodes, "Item not in the graph"
        assert item_idx != user_idx, "Item is the user"

        if ignore_users_neighbors:
            assert item_idx not in self.graph.neighbors(user_idx), "Item is a neighbor of the user"

        ppr_scores = nx.pagerank(self.graph, personalization={user_idx: 1})
        if item_idx is None:
            return ppr_scores

        if ignore_users_neighbors:
            neighbors = self.graph.neighbors(user_idx)
            for neighbor in neighbors:
                ppr_scores.pop(neighbor, None)

        return ppr_scores.get(item_idx, 0)

    def recommend(self, user):
        assert self.graph is not None, "Graph is None"
        assert user in self.graph.nodes, "User not in the graph"

        # Compute the Personalized PageRank scores for all nodes in the graph
        ppr_scores = nx.pagerank(self.graph, personalization={user: 1}, max_iter=1000)

        # Exclude the user and the user's neighbors from the recommendations
        neighbors = self.graph.neighbors(user)
        for neighbor in neighbors:
            ppr_scores.pop(neighbor, None)
        ppr_scores.pop(user, None)

        if ppr_scores == {}:
            return None

        # Find the node with the highest PPR score
        top_recommendation = max(ppr_scores, key=ppr_scores.get)

        return top_recommendation
