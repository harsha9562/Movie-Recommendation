''' COLLABRATIVE FILTERING BASED ON K-NERAREST NEIGHBOURS ALGORITHM'''

''' importing libraries '''
import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import difflib
from difflib import SequenceMatcher
import scipy

# import ...helpers.poster_fetch as poster_fetch

# reading the dataset
df_movies  = pd.read_csv('Datasets/moviesss.csv')
tmdb = pd.read_csv("Datasets/tmdb_5000_movies.csv")



movie_user_mat_sparse = scipy.sparse.load_npz('Datasets/movie_user_mat_sparse.npz')
with open('Datasets/movie_to_idx.json') as json_file:
    movie_to_idx = json.load(json_file)


''' Making the model '''
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(movie_user_mat_sparse)


def recommend(model_knn=model_knn,data=movie_user_mat_sparse,fav_movie=" ",mapper=movie_to_idx,n_recommendations=10):
    model_knn.fit(data)
    # getting index of the movie
    idx = movie_to_idx.get(fav_movie)
    if idx!=None:
        distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        # get reverse mapper
        reverse_mapper = {v: k for k, v in mapper.items()}
        movie_ls = []

        for i, (idx, dist) in enumerate(raw_recommends):
            movie_ls.append( reverse_mapper[idx])
            
            ## EXECUTE TO SEE THE DISTANCE , ELSE UNNECSSARY
            #print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))
        
        # PRINTING THE RECOMMENDED MOVIES
        return movie_ls 
    else:
        print("Could'nt find movie in our database")
        

     
''' DRIVER FUNCTION TO MAKE RECOMMENDATIONS '''

def KNN_recommend(user_fav_movie):
    # searching for the closest match
    close_match = difflib.get_close_matches(user_fav_movie.title(), list(df_movies['title']))
    # seelcting the most closest one
    if bool(close_match):
        user_fav_moviee = close_match[0]
        # print(user_fav_movie)
        s = SequenceMatcher(None,user_fav_movie.title() , user_fav_moviee)
        if  s.ratio()>0.4:
            movie_ls=recommend(fav_movie=user_fav_moviee)
            if bool(movie_ls):
                titles=[]
                for movie in movie_ls:
                    close_match = difflib.get_close_matches(movie, list(tmdb['title']))
                    if bool(close_match):
                        titles.append(close_match[0])
                    else:
                        titles.append(-1)
                movie_id_ls=[]
                
                for title in titles:
                    if title!=-1:
                        x = tmdb.loc[tmdb['original_title'] == title]
                        if x.size!=0:
                            movie_id = x['id'].values[0]
                            movie_id_ls.append(int(movie_id))
                        else:
                            movie_id_ls.append(-1)
                    else:
                        movie_id_ls.append(-1)
                # movie_dict = dict(zip(movie_ls, movie_id_ls))
                movie_dict = {
                    'movie': movie_ls,
                    'title': movie_id_ls
                }
                return json.dumps(movie_dict)
            else:
                # free all memory
                del movie_ls
                del s
                del user_fav_moviee
                del close_match
                del user_fav_movie
                return json.dumps({'movie': [], 'title': []})
        else:
            del s
            del user_fav_moviee
            del close_match
            del user_fav_movie
            return json.dumps({'movie': [], 'title': []})
        
    else:
        del close_match
        del user_fav_movie
        print("Could'nt find movie in our database")
        return json.dumps({'movie': [], 'title': []})
        

