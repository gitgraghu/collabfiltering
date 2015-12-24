import numpy as np
import sys

def K_nearest_neighbors(user1, k):
    u1id        = userlist.index(user1)
    u1ratings   = ratingsmatrix[u1id]
    u1avgrating = np.sum(u1ratings)/np.count_nonzero(u1ratings)
    u1normrating= np.copy(u1ratings)
    u1normrating[u1normrating!=0]-=u1avgrating

    indices = np.arange(numusers)
    allratings = ratingsmatrix[indices!=u1id,:]
    allratingsums = np.sum(allratings, axis=1)
    allratingcounts = (allratings!=0).sum(axis=1)
    allratingavgs = allratingsums/allratingcounts
    allratingavgs = np.tile(allratingavgs, (nummovies, 1)).T
    allratingnorms= np.copy(allratings)
    allratingnorms= allratingnorms - np.where(allratings >0, allratingavgs, 0)

    numerators = allratingnorms.dot(u1normrating)

    u1ratingssuper = np.tile(u1normrating, (numusers-1, 1))
    u1ratingssuper = np.where(allratings!=0, u1ratingssuper, 0)
    u1denoms = np.linalg.norm(u1ratingssuper, axis=1)

    user1mask = u1ratings!=0
    user1mask = np.tile(user1mask, ((numusers-1), 1))
    allratingnorms = np.where(user1mask, allratingnorms, 0)
    alldenoms = np.linalg.norm(allratingnorms, axis=1)
    denominators = u1denoms*alldenoms
    result = np.divide(numerators, denominators)

    sortedK = np.argsort(result)[::-1][:k]
    Kneighbors = []
    for i in range(k):
        indx = sortedK[i]
        if(indx >= u1id):
            indx+=1
        Kneighbors.append((indx, userlist[indx], result[sortedK[i]]))

    return Kneighbors

def Predict(user1, item, k_nearest_neighbors):
    movieindex = movielist.index(item)
    indices = [neighbor[0] for neighbor in k_nearest_neighbors]
    weights = np.array([neighbor[2] for neighbor in k_nearest_neighbors])
    allratings = ratingsmatrix[indices]
    movieratings = allratings[:, movieindex]
    numerator = (weights).dot(movieratings)
    denominator = np.sum(weights[movieratings>0])

    if denominator == 0:
        return 0

    prediction = (numerator/denominator)
    return prediction


def pearson_correlation(user1, user2):
    user1id = userlist.index(user1)
    user2id = userlist.index(user2)
    user1ratings = ratingsmatrix[user1id]
    user2ratings = ratingsmatrix[user2id]
    user1avg = np.sum(user1ratings)/np.count_nonzero(user1ratings)
    user2avg = np.sum(user2ratings)/np.count_nonzero(user2ratings)
    user1ratingnorms= np.copy(user1ratings)
    user1ratingnorms= user1ratingnorms - np.where(user1ratings > 0, user1avg, 0)
    user2ratingnorms= np.copy(user2ratings)
    user2ratingnorms= user2ratingnorms - np.where(user2ratings > 0, user2avg, 0)
    mask1 = user1ratings!=0
    mask2 = user2ratings!=0
    numerator = user1ratingnorms.dot(user2ratingnorms)
    denominator = np.linalg.norm(user1ratingnorms[mask2])*np.linalg.norm(user2ratingnorms[mask1])
    pc = numerator/denominator
    return pc

if __name__ == '__main__':
    if(len(sys.argv)!=5):
        print 'Format: python <script.py> ratings-dataset.tsv "<user_name>" "<movie_name>" <K>'
        exit(0)

    inputfilename = sys.argv[1]
    ratingsdataset = open(inputfilename)

    # Create user and movies list
    users  = set()
    movies = set()

    for line in ratingsdataset:
        fields = line.rstrip('\n').split('\t')
        username = fields[0]
        movierating = fields[1]
        moviename = fields[2]
        users.add(username)
        movies.add(moviename)

    userlist  = sorted(list(users), reverse=True)
    movielist = list(movies)
    numusers  = len(userlist)
    nummovies = len(movielist)

    # Construct rating matrix
    ratingsmatrix = np.zeros((numusers, nummovies))
    ratingsdataset.seek(0)
    for line in ratingsdataset:
        fields = line.rstrip('\n').split('\t')
        username = fields[0]
        userid  = userlist.index(username)

        moviename = fields[2]
        movieid   = movielist.index(moviename)

        movierating = fields[1]
        ratingsmatrix[userid, movieid] = float(movierating)

    testuser = sys.argv[2]
    testmovie = sys.argv[3]
    K = int(sys.argv[4])

    if testuser not in userlist:
        print 'User ' + testuser + ' does not exist in datafile.'
        exit(0)
    if testmovie not in movielist:
        print 'Movie ' + testmovie + ' does not exist in datafile.'
        exit(0)

    if K >= len(userlist):
        print 'K cannot be larger than total number of users - ' + str(len(userlist))
        exit(0)

    neighbors = K_nearest_neighbors(testuser, K)
    for neighbor in neighbors:
        print neighbor[1], neighbor[2]
    print '\n'

    prediction = Predict(testuser, testmovie, neighbors)
    print prediction
