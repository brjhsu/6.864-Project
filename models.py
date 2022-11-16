# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:32:41 2022

@author: brian
"""

from utils import mean_pooling
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class SBERT:
    def __init__(self):
        """
        Initializes an SBERT model based on 
        https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens

        Returns
        -------
        None.

        """

        #Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.lm = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").cuda()
    
    def embed_queries(self, queries, return_tokens_and_mask = False):
        """
        Given a list of strings (queries) we tokenize them, run them through the LM, and apply mean pooling to get a
        single embedding for each query

        Parameters
        ----------
        queries : TYPE
            DESCRIPTION.

        Returns
        -------
        query_embeddings : TYPE
            DESCRIPTION.

        """
        # Embed the training queries using an LM and create centroids based on the train
        encoded_query = self.tokenizer(queries,return_token_type_ids=False, padding=True, max_length=128, return_tensors='pt')
        input_ids = encoded_query['input_ids'].cuda()
        attn_mask = encoded_query['attention_mask'].cuda()
        
        with torch.no_grad():
            output_query = self.lm(input_ids = input_ids, attention_mask = attn_mask)
        query_embeddings = mean_pooling(output_query, attn_mask)
        if return_tokens_and_mask:
            return query_embeddings, input_ids, attn_mask
        else:
            return query_embeddings


class SiameseNet(nn.Module):
    def __init__(self, lm):
      super(SiameseNet, self).__init__()
      torch.manual_seed(6864)
      self.lm = lm
      self.embedding_dim = 768
      self.output_size = 128

      self.lin = nn.Linear(self.embedding_dim, 512)
      self.lin1 = nn.Linear(512, 256)
      self.out = nn.Linear(256, self.output_size)
      self.dropout = nn.Dropout(0.02)

    def freeze_lm(self): # freeze the SBERT model such that we only train the classifier 
      for param in self.lm.parameters():
        param.requires_grad = False
    def unfreeze_lm(self):
      for param in self.lm.parameters():
        param.requires_grad = True

    def FF(self, embeddings): # Feed forward embedding 
      x = F.relu(self.lin(embeddings))
      x = self.dropout(x)
      x = F.relu(self.lin1(x))
      x = self.out(x)
      return x

    def forward_one(self, encoded_query): # define the same forward function for the embeddings to produce embedding representation
      output = self.lm(**encoded_query)
      embeddings = mean_pooling(output, encoded_query['attention_mask']) # pool -> (batch size, hidden_size)
      x = self.FF(embeddings)
      return x

    def forward(self, anc_encoded, pos_encoded, neg_encoded): # takes in anchor, positive, and negative
      anc_emb = self.forward_one(anc_encoded) # pool -> (batch size, 32)
      pos_emb = self.forward_one(pos_encoded)
      neg_emb = self.forward_one(neg_encoded)
      return anc_emb, pos_emb, neg_emb
   

class PseudoSiameseNet(nn.Module):
    def __init__(self, lm):
      super(PseudoSiameseNet, self).__init__()
      torch.manual_seed(6864)
      self.lm = lm
      self.embedding_dim = 768
      self.output_size = 128

      self.FF_lins = nn.ModuleList([nn.Linear(self.embedding_dim, 512),
                                    nn.Linear(512, 256)])
      self.FF_out = nn.Linear(256, self.output_size)

      self.FF_cent_lins = nn.ModuleList([nn.Linear(self.embedding_dim, 512),
                                    nn.Linear(512, 256)])
      self.FF_cent_out = nn.Linear(256, self.output_size)

      self.dropout = nn.Dropout(0.02)

    def freeze_lm(self): # freeze the SBERT model such that we only train the classifier 
      for param in self.lm.parameters():
        param.requires_grad = False
    def unfreeze_lm(self):
      for param in self.lm.parameters():
        param.requires_grad = True

    def FF(self, x): # Feed forward embedding 
      for lin in self.FF_lins:
        x = F.relu(lin(x))
        x = self.dropout(x)
      x = self.FF_out(x)
      return x

    def FF_cent(self, x): # Feed forward embedding 
      for lin in self.FF_cent_lins:
        x = F.relu(lin(x))
        x = self.dropout(x)
      x = self.FF_cent_out(x)
      return x

    def forward_one(self, encoded_query): # define the same forward function for the embeddings to produce embedding representation
      output = self.lm(**encoded_query)
      embeddings = mean_pooling(output, encoded_query['attention_mask']) # pool -> (batch size, hidden_size)
      x = self.FF(embeddings)
      return x

    def forward(self, anc_encoded, pos_encoded, neg_encoded): # takes in anchor, positive, and negative
      anc_emb = self.forward_one(anc_encoded) # pool -> (batch size, 32)
      pos_emb = self.forward_one(pos_encoded)
      neg_emb = self.forward_one(neg_encoded)
      return anc_emb, pos_emb, neg_emb