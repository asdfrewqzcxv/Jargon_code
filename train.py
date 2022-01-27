import pickle
import networkx as nx
import numpy as np
from jargon import *
from learn_jargon import *

import sys, os
#import json


my_datasets = ['BA-shapes', 'Tree-Cycle', 'cora', 'citeseer', 'pubmed']

my_argv = sys.argv
dataset_name = my_argv[1]


if not (dataset_name) in my_datasets:
  print("Use the dataset in {}".format(my_datasets))
  raise ("Unvalid dataset name")

datasets = dataset_name


#datasets = 'BA-shapes'
#datasets = 'Tree-Cycle'
#datasets = 'cora'
#datasets = 'citeseer'
#datasets = 'pubmed'

#with open('datasets/{}/A.pickle'.format(datasets),'rb') as f:
#  A = pickle.load(f)
with open('datasets/{}/X.pickle'.format(datasets),'rb') as f:
  X = pickle.load(f)
with open('datasets/{}/Y.pickle'.format(datasets),'rb') as f:
  Y = pickle.load(f)
with open('datasets/{}/tr.pickle'.format(datasets),'rb') as f:
  train_nodes = pickle.load(f)
with open('datasets/{}/va.pickle'.format(datasets),'rb') as f:
  val_nodes = pickle.load(f)
with open('datasets/{}/te.pickle'.format(datasets),'rb') as f:
  test_nodes = pickle.load(f)

print("Load Data Done")
 
label_len = len(Y[0])
feature_len = len(X[0])
node_len = len(X)



if os.path.isfile('datasets/{}/succ_node_to_nodes.pickle'.format(datasets)):
  with open('datasets/{}/succ_node_to_nodes.pickle'.format(datasets),'rb') as f:
    succ_node_to_nodes = pickle.load(f) 
  with open('datasets/{}/pred_node_to_nodes.pickle'.format(datasets),'rb') as f:
    pred_node_to_nodes = pickle.load(f) 
  with open('datasets/{}/node_to_nodes.pickle'.format(datasets),'rb') as f:
    node_to_nodes = pickle.load(f) 

else:
  print("There is no node to nodes map we generate it")
  succ_node_to_nodes = {}
  pred_node_to_nodes = {}
  node_to_nodes = {}

  for i in range(node_len):
    succ_node_to_nodes[i] = set()
    pred_node_to_nodes[i] = set()
    node_to_nodes[i] = set()
  
  for i in range(node_len):
    for j in range(node_len):
      if A[i][j] > 0 and i != j:
        succ_node_to_nodes[i].add(j)
        pred_node_to_nodes[j].add(i)
        node_to_nodes[j].add(i)
        node_to_nodes[i].add(j)
  
  with open('datasets/{}/succ_node_to_nodes.pickle'.format(datasets), 'wb') as f:
    pickle.dump(succ_node_to_nodes, f, pickle.HIGHEST_PROTOCOL)
  with open('datasets/{}/pred_node_to_nodes.pickle'.format(datasets), 'wb') as f:
    pickle.dump(pred_node_to_nodes, f, pickle.HIGHEST_PROTOCOL)
  with open('datasets/{}/node_to_nodes.pickle'.format(datasets), 'wb') as f:
    pickle.dump(node_to_nodes, f, pickle.HIGHEST_PROTOCOL)


total_train_node_len = len(train_nodes)


data_set_complexity = feature_len * node_len
#print(data_set_complexity)
node_label = {}
label_nodes = {}


for node in range(node_len):
  for label in range(label_len):
    if not label in label_nodes:
      label_nodes[label] = set() 
    if Y[node][label] > 0:
      label_nodes[label].add(node)
      node_label[node] = label



feature_len = len(X[0])


nodes_len = len(train_nodes)
feature_list = []

#print(X)

for i, node in enumerate(X):
  feature_list.append(X[i][0])

feature_list = sorted(feature_list)
feature_list2 = copy.deepcopy(feature_list)
feature_list2.reverse()
min_max_feat = [(feature_list[0], feature_list[len(feature_list) - 1])]

if len(set(feature_list2)) == 2:
  is_one_hot = True
else:
  is_one_hot = False

if datasets == 'BA-Community':
  feature_list_ = []

  for i, node in enumerate(X):
    feature_list_.append(X[i][1])

  feature_list_ = sorted(feature_list_)
  feature_list2_ = copy.deepcopy(feature_list_)
  feature_list2_.reverse()
  min_max_feat_ = [(feature_list_[0], feature_list_[len(feature_list_) - 1])]




print("Is one hot : {}".format(is_one_hot))


#train
for my_label in range(label_len):
  print()
  print("============================")
  print("Learning sentences for label : {}".format(my_label))
  print("============================")
  print()
  parameter = Parameter()
  parameter.dict = dict()
  parameter.succ_node_to_nodes = succ_node_to_nodes
  parameter.pred_node_to_nodes = pred_node_to_nodes
  parameter.X_arr = X
  parameter.labeled_nodes =label_nodes[my_label] & train_nodes
  parameter.original_labeled_nodes = label_nodes[my_label] & train_nodes
  parameter.train_nodes = train_nodes
  parameter.filtered_nodes = train_nodes
  parameter.data_set_complexity = data_set_complexity
  #parameter.features = features
  #''' 
  if datasets == 'BA-Community':
    parameter.feature_list_ = feature_list_
    parameter.feature_list2_ = feature_list2_
    parameter.min_max_feature_ = min_max_feat_

  parameter.feature_list = feature_list
  parameter.feature_list2 = feature_list2
  parameter.min_max_feature = min_max_feat
  
  parameter.is_one_hot = is_one_hot
  #'''
  if datasets == 'BA-shapes' or datasets == 'Tree-Cycle':
    parameter.epsilon = 10
  else:
    parameter.epsilon = 0.1

  #simple data
  if data_set_complexity < 10000:
    parameter.chosen_depth = 3 
  elif data_set_complexity < 100000:
    parameter.chosen_depth = 2 
  else:
    parameter.chosen_depth = 1 
  print("Chosen Depth : {}".format(parameter.chosen_depth)) 
  #complex graph
  if node_len > 10000:
    parameter.is_complex_graph = True
  else:
    parameter.is_complex_graph = False 
  print("chosen epsilon : {}".format(parameter.epsilon))
  sentences = learn_sentences(parameter)
  with open('datasets/{}/learned_sentences/learned_sentences_for_{}.pickle'.format(datasets, my_label), 'wb') as f:
    pickle.dump(sentences, f, pickle.HIGHEST_PROTOCOL)
  
#sys.exit()
  

