# -*- coding: utf-8 -*-
"""
Defines auxillary functions that are used in the main training loop including
Mean pooling: for pooling embeddings
TripletLoss: the main loss that is used for the Siamese neural network
generate_batch: the batch generation function for the Siamese neural network
"""

import torch 
import torch.nn as nn
from torch.nn.modules.distance import PairwiseDistance
import numpy as np

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class TripletLoss(nn.Module):
    def __init__(self, eps): # eps is some margin 
        super(TripletLoss, self).__init__()
        self.eps = eps
        self.dist_fcn = PairwiseDistance(p=2)

    def forward(self, anc_emb, pos_emb, neg_emb):
        pos_dist = self.dist_fcn(anc_emb, pos_emb)
        neg_dist = self.dist_fcn(anc_emb, neg_emb)

        hinge_dist = torch.clamp(self.eps + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist) # take the average distance in the batch to be the loss  
        return loss
    
# Wrap the batch generation into a function 
def generate_batch(model, train_input_ids, train_attn_mask, train_labels, centroids, n_pos, random_state):
    """
    Model: the Siamese net model that will be used to generate embeddings
    train_input_ids: ALL tokenized input ids across the training set
    train_attention mask: ALL tokenized attention masks across the training set
    train_labels: ALL labels across the training set
    centroids: A dictionary that maps {label: centroid embedding}  
    n_pos: The number of examples to draw from each class in train_labels
    random_state: The seed to be set 
    """
    
    batch_ix = np.array([])
    np.random.seed(random_state)
    label_set = list(set(train_labels))
    for label in label_set:
      label_ix = np.where(np.array(label)==np.array(train_labels))[0]
      label_ix = np.random.choice(label_ix,size=n_pos, replace = False) 
      batch_ix = np.concatenate([batch_ix, label_ix]).astype(np.int32)
    batch_size = len(batch_ix)
    batch_anc_enc = {'input_ids': train_input_ids[batch_ix], 'attention_mask': train_attn_mask[batch_ix]}
    # Embed everything in the sample and obtain the distance matrix
    embeddings = model.forward_one(batch_anc_enc)
    emb_dists = torch.cdist(embeddings,embeddings, p=2)
    # find the indices of emb_dists with the respective max (if same class) or min (if different class) for the distance
    pos_samples = torch.zeros((batch_size)).long()
    neg_samples = torch.zeros((batch_size)).long()
    # Get the embedding of the same label centroid as well as some embedding of a (random) different label centroid
    # Unlike pos/neg samples we'll just create this directly (not through input_ids and attention_mask)
    emb_shape = centroids[label_set[0]].shape[0] # 768 is the shape of each centroid 
    pos_centroid_samples = torch.zeros((batch_size,emb_shape))
    neg_centroid_samples = torch.zeros((batch_size,emb_shape))
    
    batch_labels = np.array(train_labels)[batch_ix]
    np.random.seed(random_state)
    for i in range(batch_size): # just do brute-force over all selected samples for each label - it's a small matrix, otherwise it's too difficult to deal with indices
      lab = batch_labels[i]
      pos_ix = np.where(lab == batch_labels)[0]
      pos_ix = np.setdiff1d(pos_ix, i) 
      neg_ix = np.where(lab != batch_labels)[0]
      pos_dists = emb_dists[i,pos_ix]
      neg_dists = emb_dists[i,neg_ix]
      pos_samples[i] = pos_ix[torch.argmax(pos_dists)] # Hard-positive
      neg_samples[i] = neg_ix[torch.argmin(neg_dists)] # Hard-negative
      pos_centroid_samples[i] = centroids[lab] # Centroid-positive
      diff_labels = [x for x in label_set if x != lab] # Randomly pick a different centroid to designate as a negative 
      neg_centroid_samples[i] = centroids[ np.random.choice(diff_labels, size=1, replace = False)[0] ] # Centroid-negative
    
    # now generate a batch-based set of (anchor, hard-positive, hard-negative) triplets
    pos_input_ids, pos_attention_mask = batch_anc_enc['input_ids'][pos_samples], batch_anc_enc['attention_mask'][pos_samples]
    neg_input_ids, neg_attention_mask = batch_anc_enc['input_ids'][neg_samples], batch_anc_enc['attention_mask'][neg_samples]
    
    batch_pos_enc = {'input_ids': pos_input_ids, 'attention_mask': pos_attention_mask}
    batch_neg_enc = {'input_ids': neg_input_ids, 'attention_mask': neg_attention_mask}
    
    return batch_anc_enc, batch_pos_enc, batch_neg_enc, pos_centroid_samples, neg_centroid_samples