import numpy as np
import torch
import torch.nn as nn
from model import recommendMaster
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_sample=200
train_turns=100

loss_fn=nn.MSELoss(size_average=None, reduce=None, reduction='mean') 


# ratings=[]
# movies=[]
# users=[]
with open("./datasets/ml-1m/ratings.dat",'r',errors='ignore') as rf:
    ratings=rf.readlines()
with open("./datasets/ml-1m/movies.new.dat",'r',errors='ignore') as mf:
    movies=mf.readlines()
with open("./datasets/ml-1m/users.dat",'r',errors='ignore') as uf:
    users=uf.readlines()

# uel=userEmbeddingLayer()
# mel=moviesEmbeddingLayer()





recommend_model=recommendMaster()

optimizer = optim.AdamW(recommend_model.parameters(), lr=0.00001)


for ts in range(train_turns):
    train_list = np.floor(np.random.rand(train_sample) * len(ratings)).astype(int)
    likelihoods=[]
    ground_truth_likelihoods=[]
    total_loss=0
    for item in train_list:

        likelihood=recommend_model(users[int(ratings[item].split("::")[0])-1],movies[int(ratings[item].split("::")[1])-1])
       
        likelihoods.append(likelihood.unsqueeze(0))
        
    
        
        ground_truth_likelihoods.append(float(ratings[item].split("::")[2])/5)

    # print(likelihoods.shape)
    likelihoods=torch.cat(likelihoods)
    ground_truth_likelihoods=torch.tensor(ground_truth_likelihoods).to(device)
    
    total_loss+=loss_fn(likelihoods,ground_truth_likelihoods)
    print("Loss Value: ",total_loss.item())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if ts%10==0:
        torch.save(recommend_model.state_dict(), 'recommendMaster.pth')

