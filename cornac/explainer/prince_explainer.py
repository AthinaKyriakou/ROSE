import heapq
from typing import Tuple, List, Optional

import networkx as nx

from cornac.explainer import Explainer


class Prince(Explainer):
    """
    Provider-side Interpretability with Counterfactual Explanations

    Parameters
    ----------
    graph (nx.DiGraph): The graph representing the data.
    items (List[str]): List of item identifiers.
    rec_model (Optional[object]): Trained recommendation model (default: None). Not used.
    dataset (Optional[object]): Dataset used for explanation (default: None). Not used.
    name (str): Name of the explainer (default: "PRINCE"). Not used.

    References
    ----------
    Azin Ghazimatin, Oana Balalau, Rishiraj Saha Roy, and Gerhard Weikum. 2020.
    PRINCE: Provider-side Interpretability with Counterfactual Explanations in Recommender Systems.
    In The Thirteenth ACM International Conference on Web Search and Data Mining (WSDM ’20),
    February 3–7, 2020, Houston, TX, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10. 1145/3336191.3371824

    Code Reference
    ----------
    https://github.com/azinmatin/prince/
    """

    def __init__(self, graph: nx.DiGraph, items: List[str], rec_model: Optional[object] = None, dataset: Optional[object] = None, name: str = "PRINCE"):
        """
        Initialize the PRINCE explainer.

        Parameters
        ----------
            graph (nx.DiGraph): The graph representing the data.
            items (List[str]): List of item identifiers.
            rec_model (Optional[object]): Trained recommendation model (default: None). Not used.
            dataset (Optional[object]): Dataset used for explanation (default: None). Not used.
            name (str): Name of the explainer (default: "PRINCE"). Not used.
        """
        self.graph = graph
        self.items = items

    def explain_one_recommendation_to_user(self, user_id: str, item_id: str, num_features: int = 10) -> Tuple[
        List[Tuple[str, str]], str]:
        """
        Provide an explanation for a recommendation given to a specific user.

        Parameters
        ----------
            user_id (str): User identifier.
            item_id (str): Recommended item identifier.
            num_features (int): Irrelevant for PRINCE but inherited from Base Explainer.

        Returns:
            Tuple[List[Tuple[str, str]], str]: A list of actions to remove that change the recommendation and the new recommendation.
        """
        return self._prince(self.graph, self.items, user_id, item_id)

    def _prince(self, graph: nx.DiGraph, items: List[str], user: str, recommendation: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        PRINCE algorithm for generating counterfactual explanations. Algorithm 1 from the PRINCE paper.
        """
        self._validate_prince_preconditions(graph, items, recommendation, user)
        optimal_actions = [(user, neighbor) for neighbor in graph.neighbors(user)]
        new_recommendation = recommendation

        for item in items:
            swapping_actions = self._swap_order(graph, user, recommendation, item)
            # Actions that swap orders of recommendation and item
            if len(swapping_actions) < len(optimal_actions):
                optimal_actions = swapping_actions
                new_recommendation = item
            elif len(swapping_actions) == len(optimal_actions):
                if self._is_better_recommendation(user, item, new_recommendation):
                    optimal_actions = swapping_actions
                    new_recommendation = item

        return optimal_actions, new_recommendation

    def _validate_prince_preconditions(self, graph, items, recommendation, user):
        assert graph is not None, "Graph is None"
        assert items is not None, "Items is None"
        assert items, "Items is empty"
        assert user is not None, "User is None"
        assert recommendation is not None, "Recommendation is None"
        assert user in graph.nodes, "User not in the graph"
        assert recommendation in graph.nodes, "Recommendation not in the graph"
        assert recommendation != user, "Recommendation is the user"
        assert recommendation in items, "Recommendation not in items"
        assert user not in items, "User in items"
        assert len(items) > 1, "No other items to recommend"

    def _swap_order(self, graph: nx.DiGraph, user: str, recommendation: str, item: str) -> List[Tuple[str, str]]:
        actions = [(user, neighbor) for neighbor in graph.neighbors(user) if neighbor != user]
        new_actions = []
        min_heap = []
        total = 0

        for (user, neighbor) in actions:
            weight = graph[user][neighbor]['weight']
            diff = weight * (
                    nx.pagerank(graph, personalization={neighbor: 1}, max_iter=1000)[recommendation] -
                    nx.pagerank(graph, personalization={neighbor: 1}, max_iter=1000)[item]
            )
            heapq.heappush(min_heap, (-diff, (user, neighbor)))
            total += diff

        while total >= 0 and min_heap:
            diff, (user, neighbor) = heapq.heappop(min_heap)
            total += diff
            new_actions.append((user, neighbor))

        if total > 0:
            new_actions = actions

        return new_actions

    def _is_better_recommendation(self, user: str, candidate: str, current_best: str) -> bool:
        return nx.pagerank(self.graph, personalization={user: 1}, max_iter=1000)[candidate] > \
               nx.pagerank(self.graph, personalization={user: 1}, max_iter=1000)[current_best]
