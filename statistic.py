import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

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
        self.ll1=nn.Linear(56, 128)
        self.softmax1=nn.Softmax(dim=1)
        self.ll2=nn.Linear(128, 128)
        self.softmax2=nn.Softmax(dim=1)
        
        
    def embed_forward(self):
        
        vectorz_userid = self.userid_fc(self.userid)
        vectorz_gender = self.gender_fc(self.gender)
        vectorz_age = self.age_fc(self.age)
        vectorz_occupation = self.occupation_fc(self.occupation)
        vectorz=torch.cat((vectorz_userid, vectorz_gender, vectorz_age, vectorz_occupation), dim=1)
        
        vectorz=self.softmax1(self.ll1(vectorz))
        vectorz=self.softmax2(self.ll2(vectorz))
        
        return vectorz



class movies_embedding_layer(nn.modules):
    # 从数据集中提取电影信息
    def __init__(self, record:str):
        # 25::Leaving Las Vegas (1995)::Drama|Romance
        super(movies_embedding_layer, self).__init__()
        self.hidden_embedding_size=64
        
        info_list=record.split("::")
        
        self.movieid=int(info_list[0])
        self.title=info_list[1]
        self.genres = info_list[2].replace("|", " ")
        
        self.movieid_fc = nn.Linear(1, self.hidden_embedding_size)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.ll1=nn.Linear(self.hidden_embedding_size+768*2, 128)
        self.softmax1=nn.Softmax(dim=1)
        self.ll2=nn.Linear(128, 128)
        self.softmax2=nn.Softmax(dim=1)
        
        
        
        
    def embed_forward(self):
        
        vectorz_movieid = self.movieid_fc(self.movieid)
        
        self.title_token = self.tokenizer(self.title, return_tensors='pt')
        self.title_embedding = self.bert(**self.title_token)[1]
        self.genres_token = self.tokenizer(self.genres, return_tensors='pt')
        self.genres_embedding = self.bert(**self.genres_token)[1]
        vectorz=torch.cat((vectorz_movieid, self.title_embedding, self.genres_embedding), dim=1)
        vectorz=self.softmax1(self.ll1(vectorz))
        vectorz=self.softmax2(self.ll2(vectorz))
        return vectorz