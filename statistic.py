import torch
import torch.nn as nn

# 这是对数据进行预处理

class userEmbeddingLayer(nn.modules):
    # 从数据集中提取用户信息
    def __init__(self, record:str):
        super(userEmbeddingLayer, self).__init__()
        self.hidden_embedding_size=16
        info_list=record.split("::")
        self.userid=int(info_list[0])
        self.gender=1 if info_list[1]=='M' else 0
        self.age=int(info_list[2])
        self.occupation=int(info_list[3])
        
        self.userid_fc = nn.Linear(1, self.hidden_embedding_size)
        self.gender_fc = nn.Linear(1,self.hidden_embedding_size/2)
        self.age_fc = nn.Linear(1,self.hidden_embedding_size)
        self.occupation_fc = nn.Linear(1,self.hidden_embedding_size)
        
    def embed_forward(self):
        
        vectorz_userid = self.userid_fc(self.userid)
        vectorz_gender = self.gender_fc(self.gender)
        vectorz_age = self.age_fc(self.age)
        vectorz_occupation = self.occupation_fc(self.occupation)
        vectorz=torch.cat((vectorz_userid, vectorz_gender, vectorz_age, vectorz_occupation), dim=1)
        
        return vectorz



class movies_embedding_layer(nn.modules):
    # 从数据集中提取电影信息
    def __init__(self, record:str):
        # 25::Leaving Las Vegas (1995)::Drama|Romance
        super(movies_embedding_layer, self).__init__()
        self.hidden_embedding_size=16
        
        info_list=record.split("::")
        
        self.movieid=int(info_list[0])
        self.title=info_list[1]
        self.genres=info_list[2].split("|")
        
        self.movieid_fc = nn.Linear(1, self.hidden_embedding_size)
        
        
        
    def embed_forward(self):
        
        pass