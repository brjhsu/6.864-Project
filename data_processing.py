# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import re
import random
import os
from datasets import load_dataset

class BankingDataLoader:
    def __init__(self, file_dir):
        """
        Defines standard constants to use for standardized data loading

        Returns
        -------
        None

        """
        self._seed = 6864
        self._np_seed = 42
        self._n_classes = 30
        self._n_samples = 20
        self._val_size = 75
        self._n_testing = 1000
        os.chdir(file_dir)
        
    def load_dataset(self, n_classes):
        """
        Parameters
        ----------
        n_classes : The number of label classes to keep

        Returns
        -------
        Filtered banking dataset as a dictionary with keys (text, labels)

        """
        banking = load_dataset('banking77')
        # Need to remap all labels
        label_tbl = pd.read_csv('banking77.csv')
        label_map = {}
        
        # for simplicity only keep some 30 labels in training/testing data
        random.seed(self._seed)
        labels_to_keep = random.sample(list(range(len(label_tbl))), k= n_classes)
        
        for i in range(len(label_tbl)):
            label = label_tbl['Label'].iloc[i]
            label = re.sub(r'\_',' ',label)
            ix = label_tbl['Label_ix'].iloc[i]
            if ix in labels_to_keep:
              label_map.update({ix: label})
        
        self.label_map = label_map
        
        # filter training set to only intents we want 
        return banking.filter(lambda row: row['label'] in labels_to_keep)      
              
    def create_splits(self, banking, n_samples=20, val_size=75):
        """
        Creates splits of data based 

        Parameters
        ----------
        banking : dataset of text and labels
        n_samples : number of samples to retain for each label class
        val_size : size of the validation data

        Returns
        -------
        train, val, test, holdout datsets

        """
        # First create a relatively small training set in light of few-shot learning. We collect k samples from each label
        label_counter = {}
        train_queries, train_labels = [],[]
        test_queries, test_labels = [], []
        for i in range(len(banking['train'])):
          text = banking['train'][i]['text']
          label = banking['train'][i]['label']
          if label not in label_counter:
            label_counter.update({label: 1})
            train_queries.append(text)
            train_labels.append(label)
          else:
            if label_counter[label] < n_samples:
              train_queries.append(text)
              train_labels.append(label)
              label_counter[label] += 1
            else:
              test_queries.append(text)
              test_labels.append(label)

        np.random.seed(self._np_seed)
        test_ix = np.random.choice(len(test_queries), self._n_testing) 
        mask = np.ones(len(test_queries), dtype=bool)
        mask[test_ix] = False
        holdout_ix = np.where(mask)[0]
        # Everything else from test_ix goes into the holdout
        holdout_queries = banking['test']['text']
        holdout_labels = banking['test']['label']
        holdout_queries += [test_queries[i] for i in holdout_ix.tolist()]
        holdout_labels += [test_labels[i] for i in holdout_ix.tolist()]
        # Create testing set of 1000
        test_queries=[test_queries[i] for i in test_ix.tolist()]
        test_labels=[test_labels[i] for i in test_ix.tolist()]
        
        train = pd.DataFrame({'text': train_queries, 'label': train_labels})
        test = pd.DataFrame({'text': test_queries, 'label': test_labels})
        holdout = pd.DataFrame({'text': holdout_queries, 'label': holdout_labels})
        
        train_queries, train_labels = list(train['text']), list(train['label'])
        val_queries, val_labels = list(holdout['text'][:val_size]), list(holdout['label'][:val_size])
        holdout_queries, holdout_labels = list(holdout['text'][val_size:]), list(holdout['label'][val_size:])
        test_queries, test_labels = list(test['text']), list(test['label'])
        return (train_queries, train_labels), (val_queries, val_labels), (holdout_queries, holdout_labels), (test_queries, test_labels)
    
    def get_label_map(self):
        return self.label_map
    
    def load_banking_data(self):
        banking = self.load_dataset(self._n_classes)
        return self.create_splits(banking, self._n_samples, self._n_testing)
    
"""
Usage below
"""
def main():
    banking_data_loader = BankingDataLoader(r'C:\Users\brian\OneDrive\Documents\GitHub\6.864-Project')
    (train_queries, train_labels), (val_queries, val_labels), (holdout_queries, holdout_labels), (test_queries, test_labels) = banking_data_loader.load_banking_data()
  
if __name__=="__main__":
    main()