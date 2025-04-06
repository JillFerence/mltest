import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns (commented out as it is not used)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

def top_n_products(final_rating, n, min_interaction):
        recommendations = final_rating[final_rating['rating_count'] > min_interaction]
        recommendations = recommendations.sort_values('avg_rating', ascending=False)
        return recommendations.index[:n]

def similar_users(user_index, interactions_matrix):
        similarity = []
        for user in range(0, interactions_matrix.shape[0]):
            sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
            similarity.append((user, sim))
        similarity.sort(key=lambda x: x[1], reverse=True)
        most_similar_users = [tup[0] for tup in similarity]
        similarity_score = [tup[1] for tup in similarity]
        most_similar_users.remove(user_index)
        similarity_score.remove(similarity_score[0])
        return most_similar_users, similarity_score

def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    prod_ids = set(list(interactions_matrix.columns[np.nonzero(interactions_matrix.loc[user_index] > 0)]))
    recommendations = []
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(list(interactions_matrix.columns[np.nonzero(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    return recommendations[:num_of_products]


def recommend_items(user_index, interactions_matrix, preds_matrix, num_recommendations):
        user_ratings = interactions_matrix[user_index, :].toarray().reshape(-1)
        user_predictions = preds_matrix[user_index, :].toarray().reshape(-1)
        temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
        temp['Recommended Products'] = np.arange(len(user_ratings))
        temp = temp.set_index('Recommended Products')
        temp = temp.loc[temp.user_ratings == 0]
        temp = temp.sort_values('user_predictions', ascending=False)
        print('\nBelow are the recommended products for user(user_id = {}):\n'.format(user_index))
        print(temp['user_predictions'].head(num_recommendations))