import torch.nn as nn
from statistic import userEmbeddingLayer,moviesEmbeddingLayer


loss_fn=nn.MSELoss(size_average=None, reduce=None, reduction='mean') 


ratings=[]
movies=[]
users=[]
with open("./datasets/ml-1m/ratings.dat",'r',errors='ignore') as rf:
    ratings=rf.readlines()
with open("./datasets/ml-1m/movies.new.dat",'r',errors='ignore') as mf:
    movies=mf.readlines()
with open("./datasets/ml-1m/users.dat",'r',errors='ignore') as uf:
    users=uf.readlines()

uel=userEmbeddingLayer()
mel=moviesEmbeddingLayer()

for idx in range(len(ratings)):
    userid=int(ratings[idx].split("::")[0])
    movieid=int(ratings[idx].split("::")[1])

    grade=ratings[idx].split("::")[2]

    userstr=users[userid-1]
    moviestr=movies[movieid-1]




    print(uel(userstr))


