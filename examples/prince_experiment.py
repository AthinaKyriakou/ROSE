import networkx as nx
from tqdm import tqdm

from cornac.datasets import amazon_clothing
from cornac.explainer.prince_explainer import Prince
from cornac.models import PPRBased

amazon_clothing_reviews = amazon_clothing.load_feedback()
clothing_items = amazon_clothing.load_graph()
for i, clothing_item in enumerate(clothing_items):
    clothing_items[i] = (f"ITEM-{clothing_item[0]}", "ITEM-"+clothing_item[1], clothing_item[2])
unique_users = {"USER-"+review[0] for review in amazon_clothing_reviews}
no_new_recommendation_possible = 0
plausible_recommendation = 0
implausible_recommendation_all_actions_removed = 0
implausible_recommendation_no_actions_removed = 0
optimal_actions = 0

for user in tqdm(unique_users):
    random_users_reviews = []
    for clothing_review in amazon_clothing_reviews:
        if "USER-"+clothing_review[0] == user:
            clothing_review = ("USER-"+str(clothing_review[0]), "ITEM-"+str(clothing_review[1]), clothing_review[2])
            random_users_reviews.append(clothing_review)

    for user_review in random_users_reviews:
        clothing_items.append(user_review)
    G = nx.DiGraph()
    G.add_weighted_edges_from(clothing_items)

    n_steps = 2
    nodes_within_n_steps = nx.single_source_shortest_path_length(G, user, cutoff=n_steps)
    subgraph_view = G.subgraph(nodes_within_n_steps)
    editable_subgraph = nx.DiGraph(subgraph_view)

    for (u, v) in editable_subgraph.edges():
        if 'weight' not in editable_subgraph[u][v]:
            editable_subgraph[u][v]['weight'] = 1.0

    items = editable_subgraph.copy()
    items.remove_node(user)

    items.remove_nodes_from(editable_subgraph.successors(user))

    ppr_based_rec = PPRBased(ignore_neighbors=True, graph=editable_subgraph)
    recommendation = ppr_based_rec.recommend(user)

    if recommendation is None:
        no_new_recommendation_possible += 1
        continue

    if len(items) == 1:
        no_new_recommendation_possible += 1
        continue

    prince_recommender = Prince(dataset=clothing_items, rec_model=ppr_based_rec, graph=editable_subgraph, items=items)
    actions, new_rec = prince_recommender.explain_one_recommendation_to_user(user, recommendation)

    if len(actions) == len(list(editable_subgraph.successors(user))):
        implausible_recommendation_all_actions_removed += 1
    elif len(actions) == 0:
        implausible_recommendation_no_actions_removed += 1
    elif 3 > len(actions) > 0:
        optimal_actions += 1
    else:
        plausible_recommendation += 1


print(f"No new recommendation possible: {no_new_recommendation_possible}")
print(f"Optimal actions: {optimal_actions}")
print(f"Plausible recommendation: {plausible_recommendation}")
print(f"Implausible recommendation (all actions removed): {implausible_recommendation_all_actions_removed}")
print(f"Implausible recommendation (no actions removed): {implausible_recommendation_no_actions_removed}")

print(f"Strict Coverage: {optimal_actions / (len(unique_users) - no_new_recommendation_possible)}")
print(f"Coverage: {(plausible_recommendation + optimal_actions) / (len(unique_users) - no_new_recommendation_possible)}")


# No new recommendation possible: 1069
# Optimal actions: 2234
# Plausible recommendation: 22
# Implausible recommendation (all actions removed): 1715
# Implausible recommendation (no actions removed): 337
# Strict Coverage: 0.5185701021355618
# Coverage: 0.5236768802228412
