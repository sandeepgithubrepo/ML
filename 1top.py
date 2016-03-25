from __future__ import division
import sys, ast
import loadmovielens as reader
import numpy as np
from scipy.stats.stats import pearsonr
from operator import itemgetter
import itertools
from itertools import imap
from operator import itemgetter


"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = ''


"load the data into python"
ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()
r,c = ratings.shape

md = {}
for i in xrange(0,r):
    md[(ratings[i][0],ratings[i][1])] = ratings[i][2]



def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID




def Jaccard_Coefficient(movie_id_1, movie_id_2):
    """
    :param movie_id_1: (integer) id regarding the first movie
    :param movie_id2: (integer) id regarding the second movie
    :return: (float) Jaccard_Coefficient regarding these movies based on the given movie IDs
            ROUND OFF TO THREE DECIMAL DIGITS
    """

    # TRUE/FALSE array for the movie_id_1
    movie_1_voters = ratings[:,1] == movie_id_1

    # get all users,movie_id,ratings,timestamp for all movie_id_1 voters 
    movie_1_users_ratings = ratings[movie_1_voters]

    #get users who voted for movie_id_1
    users_movie1 = set(movie_1_users_ratings[:,0])

    # TRUE/FALSE array for the movie_id_2
    movie_2_voters = ratings[:,1] == movie_id_2

    # get all users,movie_id,ratings,timestamp for all movie_id_2 voters
    movie_2_users_ratings = ratings[movie_2_voters]

    #get users who voted for movie_id_2
    users_movie2 = set(movie_2_users_ratings[:,0])

    #find intersection of users who voted for both the movies
    movie_1_2_users = users_movie1 & users_movie2    
    
    #calculated Jaccard Coefficient and return zero if denominator is zero
    try:
     Jaccard_Coefficient = len(movie_1_2_users) / (len(users_movie1) + len(users_movie2)- len(movie_1_2_users))
    except ZeroDivisionError:
           Jaccard_Coefficient = 0.0 

    # round result to 3 digits and return
    return round(Jaccard_Coefficient,3)

    


def Correlation_Coefficient(movie_id_1, movie_id_2):
    """
    :param movie_id_1: (integer) id regarding the first movie
    :param movie_id2: (integer) id regarding the second movie
     :return: (float) Correlation_Coefficient regarding these movies based on the given movie IDs.
            ROUND OFF TO THREE DECIMAL DIGITS
    """
    
    # TRUE/FALSE array for the movie_id_1
    movie_1_voters = ratings[:,1] == movie_id_1

    # get all users,movie_id,ratings,timestamp for all movie_id_1 voters 
    movie_1_users_ratings = ratings[movie_1_voters]

    # get all users who voted for movie_id1
    users_movie1 = set(movie_1_users_ratings[:,0])
    
    # TRUE/FALSE array for the movie_id_2
    movie_2_voters = ratings[:,1] == movie_id_2

    # get all users,movie_id,ratings,timestamp for all movie_id_2 voters 
    movie_2_users_ratings = ratings[movie_2_voters]

    # get all users who voted for movie_id_2
    users_movie2 = set(movie_2_users_ratings[:,0])
    
    # find all users who voted for both the movies
    movie_1_2_users = users_movie1 & users_movie2
    
    #total number of users who voted for both the movies
    movie_1_2_users_count = len(movie_1_2_users)
    
    # avoids  Nan and also 
    #high score of the movies voted by very less number of users
    # 10 is used as threshold  
    if movie_1_2_users_count <= 10:
       return 0

   
    #sort the user_ids
    sorted_user_ids = np.sort(list(movie_1_2_users))
 
    #get ratings of users who rated movie_id_1
    ratings_movies_id_1  = []
    #use dictionary md to  get ratings for user_id,movie_id pair
    ratings_movies_id_1  = [ md[(j,movie_id_1)]  for j in sorted_user_ids ]
    
    #get ratings of users who rated movie_id_1        
    ratings_movies_id_2  = []
    #use dictionary md to  get ratings for user_id,movie_id pair
    ratings_movies_id_2  = [ md[(j,movie_id_2)]  for j in sorted_user_ids]
   
    #calculate Pearson coefficient using ratings of users
    #who rated movie_id_1 and movie_id_2
    coeff = pearson_coff(ratings_movies_id_1 ,ratings_movies_id_2)
    
    #round result to 3 digists and return the result     
    return round(coeff,3)



def pearson_coff(x, y):
  # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  sum_y_sq = sum(map(lambda x: pow(x, 2), y))
  psum = sum(imap(lambda x, y: x * y, x, y))
  num = psum - (sum_x * sum_y/n)
  den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if den == 0: return 0
  return num / den


##calculate pearson_coeff for 
## between given movie and all other movies
def pearson_coeff_all(movie_id_1):
        score_dict = {}
        print "computing pearson coefficient"
	m_list = movie_dictionary.keys()
        m_list = np.sort(m_list)
        #dictionary of movie_id and score
	for i in m_list:
                if i != movie_id_1:
        	  result = Correlation_Coefficient(movie_id_1,i)
                  score_dict[i] = result

        ##sort and pick top-5
	all_movies_score = []
        all_movies_score = sorted(score_dict.items(), key=itemgetter(1),reverse = True)
	score_list = all_movies_score[:5]
        #print movie name and score for the top5
        for m_id,score in score_list:
            print movie_dictionary.get(m_id),",",score

##calculate jaccard_coeff for 
## between given movie and all other movies
def jaccard_coeff_all(movie_id_1):
        score_dict = {}
        print "computing Jaccard coefficient"
	m_list = movie_dictionary.keys()
        m_list = np.sort(m_list)
        #dictionary of movie_id,score
	for i in m_list:
                if i != movie_id_1:
        	  result = Jaccard_Coefficient(movie_id_1,i)
                  score_dict[i] = result

        #sort and pick top 5
        all_movies_score = []
        all_movies_score = sorted(score_dict.items(), key=itemgetter(1),reverse = True)
	score_list = all_movies_score[:5]
        #print movie_name and score for top-5
        for m_id,score in score_list:
            print movie_dictionary.get(m_id),",",score

#wrapper method for computing correlation
def find_similarity_movies_all(movie_id,coff_type):
     if coff_type== "JC":
           jaccard_coeff_all(movie_id)
     elif coff_type== "PC":
           pearson_coeff_all(movie_id)
     else:
         print "Method not supported"


def main():
        #get movie_id for movie
	movie_tuple = reader.give_me_movie_id('GoldenEye', movie_dictionary)          
        movie_id= movie_tuple[0][0];
	coff_type = "JC" #  JC for Jaccard coeff and PC for Pearson Coeff
        print "computing",coff_type,"for movie",movie_tuple[0][1]
        find_similarity_movies_all(movie_id,coff_type)     
	
if __name__ == '__main__':
    main()


