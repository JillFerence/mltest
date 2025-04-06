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

from utils import top_n_products, similar_users, recommendations, recommend_items

def run_ml_pipeline(df):
    # Importing Dataset
    df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
    df = df.drop('timestamp', axis=1)
    df_copy = df.copy(deep=True)

    # EDA
    rows, columns = df.shape
    print("No of rows = ", rows)
    print("No of columns = ", columns)

    df.info()
    print(df.isna().sum())
    print(df['rating'].describe())

    plt.figure(figsize=(12, 6))
    df['rating'].value_counts(1).plot(kind='bar')
    plt.show()

    print('Number of unique USERS in Raw data = ', df['user_id'].nunique())
    print('Number of unique ITEMS in Raw data = ', df['prod_id'].nunique())

    most_rated = df.groupby('user_id').size().sort_values(ascending=False)[:10]
    print(most_rated)

    # Pre-Processing
    counts = df['user_id'].value_counts()
    df_final = df[df['user_id'].isin(counts[counts >= 50].index)]
    print('The number of observations in the final data =', len(df_final))
    print('Number of unique USERS in the final data = ', df_final['user_id'].nunique())
    print('Number of unique PRODUCTS in the final data = ', df_final['prod_id'].nunique())

    final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
    print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

    given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
    possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
    density = (given_num_of_ratings / possible_num_of_ratings) * 100
    print('density: {:4.2f}%'.format(density))

    # Rank Based Recommendation System
    average_rating = df_final.groupby('prod_id')['rating'].mean()
    count_rating = df_final.groupby('prod_id').count()['rating']
    final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating})
    final_rating = final_rating.sort_values(by='avg_rating', ascending=False)

    print(list(top_n_products(final_rating, 5, 50)))
    print(list(top_n_products(final_rating, 5, 100)))

    # Collaborative Filtering
    final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])
    final_ratings_matrix.set_index(['user_index'], inplace=True)

    print(recommendations(3, 5, final_ratings_matrix))
    print(recommendations(1521, 5, final_ratings_matrix))

    # Model-based Collaborative Filtering: SVD
    final_ratings_sparse = csr_matrix(final_ratings_matrix.values)
    U, s, Vt = svds(final_ratings_sparse, k=50)
    sigma = np.diag(s)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=final_ratings_matrix.columns)
    preds_matrix = csr_matrix(preds_df.to_numpy())

    recommend_items(121, final_ratings_sparse, preds_matrix, 5)
    recommend_items(100, final_ratings_sparse, preds_matrix, 10)

    # Evaluating the model
    average_rating = final_ratings_matrix.mean()
    avg_preds = preds_df.mean()
    rmse_df = pd.concat([average_rating, avg_preds], axis=1)
    rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
    from sklearn.metrics import mean_squared_error, root_mean_squared_error
    RMSE = root_mean_squared_error(rmse_df['Avg_actual_ratings'], rmse_df['Avg_predicted_ratings'])
    print(f'RMSE SVD Model = {RMSE} \n')