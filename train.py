# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:56:12 2022

@author: brian
"""
import torch
import numpy as np
from data_processing import BankingDataLoader
from models import SBERT, SiameseNet
from utils import TripletLoss, generate_batch

# Load in training data
banking_data_loader = BankingDataLoader(r'C:\Users\brian\OneDrive\Documents\GitHub\6.864-Project')
(train_queries, train_labels), (val_queries, val_labels), (holdout_queries, holdout_labels), (test_queries, test_labels) = banking_data_loader.load_banking_data()
  

# Define training parameters 
n_classes = 30
num_epochs = 40
num_batches = 10
learning_rate = 5e-4
weight_decay = 1e-7
optimizer_eps = 1e-6
triplet_eps = 6
n_pos = 5
warmup_rate = 0.05
ques_max_length = 64
max_grad_norm = 5
ctx_max_length = 448
batch_size = n_pos*n_classes*2 # (x2 because we have both hard p/n and centroid p/n)

# Calculating the number of warmup steps
num_training_cases = num_batches*n_pos*n_classes
t_total = (num_training_cases // batch_size + 1) * num_epochs
ext_warmup_steps = int(warmup_rate * t_total)

# Create model 
SBERT_model = SBERT()
Siamese = SiameseNet(SBERT_model.lm).cuda()

# Freeze language model to only train classification layer
Siamese.freeze_lm()
# Siamese.unfreeze_lm()

# Embed the training queries of the (query, label) tuple and create centroids based on the train
query_embeddings, train_input_ids, train_attn_mask = SBERT_model.embed_queries(train_queries, return_tokens_and_mask=True)

# Embed each intent in the set of possible labels
label_map = banking_data_loader.get_label_map()
label_map_inv = {v:k for k,v in label_map.items()}
intent_set = list(label_map.values())
intent_embeddings_raw = SBERT_model.embed_queries(intent_set)
intent_centroids = {}
intent_centroids_lm = torch.zeros(intent_embeddings_raw.shape).cuda()

i=0
for label in set(train_labels):
  ix = np.where(np.array(train_labels) == label)[0]
  intent_centroids[label] = torch.mean(query_embeddings[ix,:], dim = 0)
  intent_centroids_lm[i,:] = torch.mean(query_embeddings[ix,:], dim = 0)
  i+=1


# Initialize triplet loss
loss_fn = TripletLoss(eps = triplet_eps)

# Initializing an AdamW optimizer
learning_rate = 3e-4
ext_optim = torch.optim.Adam(Siamese.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Important to make sure all inputs (except train_labels) is on the same device 
Siamese.train()

batch_loop = 2 # number of times we pass through each batch 
anneal_epsilon = True

torch.manual_seed(6864)
for epoch in range(num_epochs):
  for batch in range(num_batches):

    # Zero out gradient
    Siamese.zero_grad() 

    # Generate batch
    batch_anc_enc, batch_pos_enc, batch_neg_enc, pos_centroid, neg_centroid = generate_batch(Siamese, train_input_ids, train_attn_mask, train_labels, 
                                                                                             intent_centroids, n_pos, batch*epoch)

    for _ in range(batch_loop):
      # Encode anc/pos/neg
      anc_emb, pos_emb, neg_emb = Siamese(batch_anc_enc, batch_pos_enc, batch_neg_enc)

      # Embed centroids
      pos_centroid_emb = Siamese.FF(pos_centroid.cuda())
      neg_centroid_emb = Siamese.FF(neg_centroid.cuda())

      # compute the loss 
      main_loss = loss_fn(anc_emb, pos_emb, neg_emb)
      # print("Hard Pos/Neg Loss: {}".format(main_loss))

      centroid_loss = loss_fn(anc_emb, pos_centroid_emb, neg_centroid_emb)
      # print("Centroid Pos/Neg Loss: {}".format(centroid_loss))

      total_loss = main_loss + centroid_loss # concatenate the loss 

      total_loss.backward()
      # torch.nn.utils.clip_grad_norm_(Siamese.parameters(), max_grad_norm)
      ext_optim.step()
      # ext_sche.step() # Update learning rate for better convergence
    

    if (epoch % 2 == 0) & (batch == num_batches-1):
      print("Epoch {} batch {} total loss: {}".format(epoch, batch, total_loss))


# Evaluate the model
Siamese.eval()

with torch.no_grad():
    # Embed all labels with the trained network 
    intents_embeddings = torch.zeros((n_classes, Siamese.output_size)).cuda()
    for i in range(len(intent_set)):
      lab = label_map_inv[intent_set[i]]
      intents_embeddings[i,:] = Siamese.FF(intent_centroids[lab].cuda())
    
    # Test in a loop 
    top_k = 1 # the number of items predict
    n_correct_SNN = 0
    
    for ix in range(len(val_queries)):
      query = val_queries[ix]
      label = label_map[val_labels[ix]]
    
      # LM + SNN's FF layer 
      encoded_query = SBERT_model.tokenizer(query, padding=True, truncation=True,return_token_type_ids=False, max_length=128, return_tensors='pt')
      encoded_query['input_ids']=encoded_query['input_ids'].cuda()
      encoded_query['attention_mask']=encoded_query['attention_mask'].cuda()
      query_embeddings_SNN = Siamese.forward_one(encoded_query)
      top_intents = torch.topk(torch.norm(query_embeddings_SNN-intents_embeddings, 2, dim = 1), k = top_k, largest = False).indices.squeeze()
      if top_k == 1:
        top_intents = [top_intents]
      top_intents_set = []
      for i in range(len(top_intents)):
        label_ix = intent_set[top_intents[i].item()]
        top_intents_set.append(label_ix)
      if label in top_intents_set:
        n_correct_SNN+=1
    
    print("SNN: {}".format(n_correct_SNN/len(val_queries)))