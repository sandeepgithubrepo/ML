from __future__ import division
import sys, ast
import loadmovielens as reader
import numpy as np
from itertools import imap

"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = ''


"load the data into python"
ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()

# get dimesions of the the ratings
r,c = ratings.shape

#create dictionary of (user_id,movie_id)=> rating mapping
# allows faster access to ratings by mapping user_id,movies_id as key
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

#This method calculates the pearson coeff 
#for given x,y arrays
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


def main():
    """
#    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
#    """
    test_cases = ast.literal_eval(sys.argv[1])
    results = str(my_info()) + '\t\t'
    for test_case in test_cases:
        mode = test_case[0]
        id_1 = int(test_case[1])
        id_2 = int(test_case[2])
        if mode == 'jc':
            results += str(Jaccard_Coefficient(id_1, id_2)) + '\t\t'
        elif mode == 'cc':
            results += str(Correlation_Coefficient(id_1, id_2)) + '\t\t'
        else:
            exit('bad command')
    print results + '\n'

if __name__ == '__main__':
    main()


