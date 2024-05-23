import numpy as np
from tqdm import tqdm

from cornac.datasets import amazon_toy
from cornac.explainer.counter_explainer import CounterExplainer
from cornac.models.dot_product.recom_dot_product import DotProduct

sentiment_dataset = amazon_toy.load_sentiment()
ranking_N = 5
users = set()
items = set()
aspects = set()
for user, item, aspect_tuple_list in sentiment_dataset:
    users.add(user)
    items.add(item)
    for aspect_tuple in aspect_tuple_list:
        aspects.add(aspect_tuple[0])
items = sorted(list(items))
users = sorted(list(users))
aspects = sorted(list(aspects))
excluded_aspects = [
    '1', '100', '11', '25', '80', 'abc', 'n', 'r', 'u',
    'absolutely', 'also', 'certainly', 'extremely', 'nearly', 'really', 'simply', 'totally',
    'amount', 'back', 'cause', 'fact', 'thing', 'way',
    'im', 'us', 'oh', 'sure', 'item', 'amazon', 'soon', 'later', 'almost', 'yr', 'making'
                                                                                 'ask', 'say', 'tell', 'think',
    'feel', 'seem'
]
aspects = [aspect for aspect in aspects if aspect not in excluded_aspects]
user_aspect_frequency = {}
for user, _, aspect_tuple_list in sentiment_dataset:
    if user not in user_aspect_frequency:
        user_aspect_frequency[user] = {}
    for aspect, _, _ in aspect_tuple_list:
        if aspect not in user_aspect_frequency[user]:
            user_aspect_frequency[user][aspect] = 0
        user_aspect_frequency[user][aspect] += 1
user_aspect_preference_matrix = np.zeros((len(users), len(aspects)))
for i, user in enumerate(tqdm(users)):
    for k, aspect in enumerate(aspects):
        if aspect in user_aspect_frequency[user]:
            frequency = user_aspect_frequency[user][aspect]
            user_aspect_preference_matrix[i, k] = 1 + (ranking_N - 1) * (2 / (1 + np.exp(-frequency)) - 1)
        else:
            user_aspect_preference_matrix[i, k] = 0
item_aspect_frequency_and_sentiment = {}
for _, item, aspect_tuple_list in sentiment_dataset:
    if item not in item_aspect_frequency_and_sentiment:
        item_aspect_frequency_and_sentiment[item] = {}
    for aspect, _, sentiment in aspect_tuple_list:
        if aspect not in item_aspect_frequency_and_sentiment[item]:
            item_aspect_frequency_and_sentiment[item][aspect] = {'frequency': 0, 'total_sentiment': 0}
        item_aspect_frequency_and_sentiment[item][aspect]['frequency'] += 1
        item_aspect_frequency_and_sentiment[item][aspect]['total_sentiment'] += int(sentiment)
# Calculating average sentiment
for item in item_aspect_frequency_and_sentiment:
    for aspect in item_aspect_frequency_and_sentiment[item]:
        frequency = item_aspect_frequency_and_sentiment[item][aspect]['frequency']
        total_sentiment = item_aspect_frequency_and_sentiment[item][aspect]['total_sentiment']
        item_aspect_frequency_and_sentiment[item][aspect]['sentiment'] = total_sentiment / frequency
        del item_aspect_frequency_and_sentiment[item][aspect][
            'total_sentiment']  # Remove total to clean data structure
item_aspect_quality_matrix = np.zeros((len(items), len(aspects)))
preprocessed_data = {
    item: {
        aspect: {
            'frequency': data[aspect]['frequency'],
            'sentiment': data[aspect]['sentiment']
        }
        for aspect in aspects if aspect in data
    }
    for item, data in item_aspect_frequency_and_sentiment.items()
}
for i, item in enumerate(tqdm(items)):
    if item in preprocessed_data:
        for k, aspect in enumerate(aspects):
            if aspect in preprocessed_data[item]:
                freq = preprocessed_data[item][aspect]['frequency']
                sent = preprocessed_data[item][aspect]['sentiment']
                item_aspect_quality_matrix[i, k] = 1 + (ranking_N - 1) / (1 + np.exp(-freq * sent))
np.save('user_aspect_preference_matrix.npy', user_aspect_preference_matrix)
np.save('item_aspect_quality_matrix.npy', item_aspect_quality_matrix)

users = set()
items = set()
aspects = set()
sentiment_dataset = amazon_toy.load_sentiment()

for user, item, aspect_tuple_list in tqdm(sentiment_dataset):
    users.add(user)
    items.add(item)
    for aspect_tuple in aspect_tuple_list:
        aspects.add(aspect_tuple[0])

items = sorted(list(items))
users = sorted(list(users))
aspects = sorted(list(aspects))
excluded_aspects = [
    '1', '100', '11', '25', '80', 'abc', 'n', 'r', 'u',
    'absolutely', 'also', 'certainly', 'extremely', 'nearly', 'really', 'simply', 'totally',
    'amount', 'back', 'cause', 'fact', 'thing', 'way',
    'im', 'us', 'oh', 'sure', 'item', 'amazon', 'soon', 'later', 'almost', 'yr', 'making', 'ask', 'say', 'tell',
    'think', 'feel',
    'seem'
]
aspects = [aspect for aspect in aspects if aspect not in excluded_aspects]

user_to_matrix_row = {user: user_aspect_preference_matrix[i] for i, user in enumerate(users)}
item_to_matrix_row = {item: item_aspect_quality_matrix[i] for i, item in enumerate(items)}

recommender = DotProduct(user_to_matrix_row, item_aspect_quality_matrix, items)

ranking_N = 5

no_recommendation_count = 0
no_aspect_count = 0
optimal_aspect_count = 0
looser_optimal_aspect_count = 0
too_many_aspects_count = 0
all_aspects_count = 0
invalid_counterfactual = 0

lambda_param = 100
alpha = 0.2
gamma = 200.0

for user in tqdm(users):
    recommendation = recommender.recommend(user, 3)

    if recommendation[0][1] == 0:
        no_recommendation_count += 1
        continue

    best_item = recommendation[0][0]
    second_best_item = recommendation[1][0]

    explainer = CounterExplainer(recommender, user_to_matrix_row, item_to_matrix_row, lambda_param=lambda_param,
                                 alpha=alpha, gamma=gamma)
    delta = explainer.explain_one_recommendation_to_user(user, best_item, second_best_item)

    non_zero_numbers = [x for x in delta if x != 0]
    new_score = recommender.score(user_to_matrix_row[user], item_to_matrix_row[best_item] + delta)
    old_score = recommender.score(user_to_matrix_row[user], item_to_matrix_row[best_item])
    score_to_beat = recommender.score(user_to_matrix_row[user], item_to_matrix_row[second_best_item])

    new_delta = np.zeros(user_to_matrix_row[user].shape)
    new_delta[2] = 1
    new_delta[3] = 1
    non_zero_numbers = [x for x in delta if x != 0]
    non_zero_aspects = [x for x in item_to_matrix_row[best_item] if x != 0]

    if new_score > score_to_beat:
        invalid_counterfactual += 1
        continue

    if len(non_zero_numbers) == 0:
        no_aspect_count += 1
    elif 3 > len(non_zero_numbers) > 0:
        optimal_aspect_count += 1
    elif 5 > len(non_zero_numbers) >= 3:
        looser_optimal_aspect_count += 1
    else:
        too_many_aspects_count += 1

print(
    f"Experiment results: {no_recommendation_count} no recommendation, {no_aspect_count} no aspect, {optimal_aspect_count} optimal aspect, {too_many_aspects_count} too many aspects, {all_aspects_count} all aspects, suboptimal aspect count: {looser_optimal_aspect_count}")
print(f"Invalid counterfactuals: {invalid_counterfactual}")
print(f"Coverage (strict): {optimal_aspect_count / (len(users) - no_recommendation_count)}")
print(
    f"Coverage (loose):{(optimal_aspect_count + looser_optimal_aspect_count) / (len(users) - no_recommendation_count)}")
print(
    f"Coverage: {(optimal_aspect_count + too_many_aspects_count + looser_optimal_aspect_count) / (len(users) - no_recommendation_count)}")

# Experiment results: 33 no recommendation, 0 no aspect, 13980 optimal aspect, 2536 too many aspects, 0 all aspects, suboptimal aspect count: 2835
# Invalid counterfactuals: 0
# Coverage (strict): 0.722443284584776
# Coverage (loose):0.8689473412226758
# Coverage: 1.0
