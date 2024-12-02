import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class userEmbeddingLayer(nn.Module):
    def __init__(self):
        super(userEmbeddingLayer, self).__init__()
        self.hidden_embedding_size = 16
        
        # 线性层定义
        self.userid_fc = nn.Linear(1, self.hidden_embedding_size)
        self.gender_fc = nn.Linear(1, int(self.hidden_embedding_size / 2))
        self.age_fc = nn.Linear(1, self.hidden_embedding_size)
        self.occupation_fc = nn.Linear(1, self.hidden_embedding_size)
        self.ll1 = nn.Linear(56, 128)
        self.softmax1 = nn.ReLU()
        self.ll2 = nn.Linear(128, 32)
        self.softmax2 = nn.Softmax(dim=0)
        
    def forward(self, record: str):
        # 对数据进行处理
        info_list = record.split("::")
        self.userid = torch.tensor(int(info_list[0])).unsqueeze(0).float()  # 转换为 1D 张量并转为 float
        self.gender = torch.tensor(1 if info_list[1] == 'M' else 0).unsqueeze(0).float()
        self.age = torch.tensor(int(info_list[2])).unsqueeze(0).float()
        self.occupation = torch.tensor(int(info_list[3])).unsqueeze(0).float()
        
        # 输入到全连接层
        vectorz_userid = self.userid_fc(self.userid)
        vectorz_gender = self.gender_fc(self.gender)
        vectorz_age = self.age_fc(self.age)
        vectorz_occupation = self.occupation_fc(self.occupation)
        
        # 拼接向量
        vectorz = torch.cat((vectorz_userid, vectorz_gender, vectorz_age, vectorz_occupation), dim=0)
        
        # 通过全连接层和 Softmax 层
        vectorz = self.softmax1(self.ll1(vectorz))
        vectorz = self.softmax2(self.ll2(vectorz))
        
        return vectorz


class moviesEmbeddingLayer(nn.Module):
    def __init__(self):
        super(moviesEmbeddingLayer, self).__init__()
        self.hidden_embedding_size = 64
        
        # 线性层和 BERT 模型初始化
        self.movieid_fc = nn.Linear(1, self.hidden_embedding_size)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.ll1 = nn.Linear(self.hidden_embedding_size + 768 * 2, 128)  # Bert 输出的维度为 768
        self.softmax1 = nn.ReLU()
        self.ll2 = nn.Linear(128, 32)
        self.softmax2 = nn.Softmax(dim=0)
        
    def forward(self, record: str):
        # 处理电影数据
        info_list = record.split("::")
        self.movieid = torch.tensor(int(info_list[0])).unsqueeze(0).float()  # 转换为 1D 张量并转为 float
        self.title = info_list[1]
        self.genres = info_list[2].replace("|", " ")
        
        # 输入到电影 ID 的全连接层
        vectorz_movieid = self.movieid_fc(self.movieid)
        
        # 使用 BERT 进行文本编码
        self.title_token = self.tokenizer(self.title, return_tensors='pt', padding=True, truncation=True)
        self.title_embedding = self.bert(**self.title_token).pooler_output  # 使用 pooler_output
        self.genres_token = self.tokenizer(self.genres, return_tensors='pt', padding=True, truncation=True)
        self.genres_embedding = self.bert(**self.genres_token).pooler_output  # 使用 pooler_output
        
        # 拼接向量
        vectorz = torch.cat((vectorz_movieid, self.title_embedding.squeeze(), self.genres_embedding.squeeze()), dim=0)
        
        # 通过全连接层和 Softmax 层
        vectorz = self.softmax1(self.ll1(vectorz))
        vectorz = self.softmax2(self.ll2(vectorz))
        
        return vectorz
