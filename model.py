import numpy as np
import torch
import torch.nn as nn
from layer import userEmbeddingLayer,moviesEmbeddingLayer


class recommendMaster(nn.Module):
    def __init__(self):
        super(recommendMaster, self).__init__()
        self.user_layer=userEmbeddingLayer()
        self.movie_layer=moviesEmbeddingLayer()
    
    def forward(self,user_info:str,movie_info:str):
        user_emb=self.user_layer(user_info)
        movie_emb=self.movie_layer(movie_info)
        likelihood=torch.dot(user_emb,movie_emb)
        return likelihood
