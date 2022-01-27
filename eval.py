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




print("Load Adj Map Done")

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

default_sentence = Sentence()
default_sentence.absList = [{}]
default_sentence.root = 0

val_test_nodes = val_nodes | test_nodes

def find_max(my_list):
  max_idx = -1
  max_val = 0

  for i in range(len(my_list)):
    if my_list[i] > max_val:
      max_idx = i
      max_val = my_list[i]

  return max_idx

def find_max2(my_list):
  max_idx = -1
  max_val = 0

  for i in range(len(my_list)):
    if my_list[i][0] > max_val:
      max_idx = i
      max_val = my_list[i][0]

  return max_idx



my_label_len = []
for my_label in range(label_len):
  my_label_len.append(len(label_nodes[my_label]))

max_idx = find_max(my_label_len)

sentence_to_chosen_train_nodes_len = {}


for my_label in range(label_len):
  print("Processing label {}".format(my_label))
  #with open('datasets/{}_new/learned_sentences/learned_sentences_for_{}.pickle'.format(datasets,my_label),'rb') as f:
  with open('datasets/{}/learned_sentences/learned_sentences_for_{}.pickle'.format(datasets,my_label),'rb') as f:
    sentences = pickle.load(f) 

  if my_label == max_idx:
    new_sentence = Sentence ()
    new_sentence.absList = [{}]
    new_sentence.root = 0
    sentences.add(new_sentence)
 
  check_redundant_sentence = set()
  
  print()
  print("Label : {}".format(my_label))
  print()

  for _, sentence in enumerate(sentences):

    key = (sentence.root, json.dumps(sentence.absList))
    if key in check_redundant_sentence:
      continue
    else:
      check_redundant_sentence.add(key)
    nodes = eval_sentence(sentence, succ_node_to_nodes, pred_node_to_nodes, X)
    chosen_val_test_nodes = nodes & val_test_nodes
    if len(nodes & val_test_nodes) > 0 :
      sentence_to_chosen_train_nodes_len[sentence] = [my_label, chosen_val_test_nodes, len(nodes & train_nodes & label_nodes[my_label]), len(nodes & train_nodes)]
    score = float(len(nodes & train_nodes & label_nodes[my_label]) / (len(nodes & train_nodes) + 0.1))
    #print()
    #print("Sentence :{}".format(sentence))
    #print("Score: {}".format(score))


#sys.exit()



#epsilons = [0.1, 5, 10, 50, 100]
#epsilons = [0.1, 10, 100]
epsilons = [10, 100, 1000]

best_epsilon = -10
best_accuracy = 0

for _, epsilon in enumerate(epsilons):
  
  node_scores = {}
  for _, node in enumerate(val_test_nodes):  
    node_scores[node] = []
    for i in range(label_len):
      node_scores[node].append([0])
 
  #for _, [my_label, chosen_val_test_nodes, correctly_chosen_train_nodes, chosen_train_nodes] in enumerate(sentence_to_chosen_train_nodes_len):
  for _, sentence in enumerate(sentence_to_chosen_train_nodes_len):
    my_label = sentence_to_chosen_train_nodes_len[sentence][0]
    chosen_val_test_nodes = sentence_to_chosen_train_nodes_len[sentence][1]
    correctly_chosen_train_nodes = sentence_to_chosen_train_nodes_len[sentence][2]
    chosen_train_nodes = sentence_to_chosen_train_nodes_len[sentence][3]    
    
    score = float(correctly_chosen_train_nodes / (chosen_train_nodes + epsilon))
    #print("Score: {}".format(score))
    #print("Score : {}".format(score))
    for _, node in enumerate(chosen_val_test_nodes):
      node_scores[node][my_label].append(score) 


  for _, node in enumerate(val_test_nodes):  
    for my_label in range(label_len):
      node_scores[node][my_label].sort(reverse = True)


  known_nodes = train_nodes | val_nodes
  known_node_label = {}
  accurately_classified_nodes = 0 
  for _, node in enumerate(known_nodes):
    if node in node_label:
      known_node_label[node] = node_label[node]
  for _, node in enumerate(val_nodes):
    my_sentences_scores = copy.deepcopy(node_scores[node])
    first = find_max2(my_sentences_scores)
    if first == node_label[node]:
      accurately_classified_nodes = accurately_classified_nodes + 1 

  if accurately_classified_nodes >= best_accuracy:
    best_epsilon = epsilon 
    best_accuracy = accurately_classified_nodes
  print()
  print("current epsilon : {}".format(epsilon))
  print("current score : {}".format(accurately_classified_nodes))
  print()
  print()


#best_epsilon = 10
print("Best epsilon : {}".format(best_epsilon))



print()
print()
print()
print()

 
node_scores = {}
for _, node in enumerate(val_test_nodes):  
  node_scores[node] = []
  for i in range(label_len):
    node_scores[node].append([0])
 
for _, sentence in enumerate(sentence_to_chosen_train_nodes_len):
  my_label = sentence_to_chosen_train_nodes_len[sentence][0]
  chosen_val_test_nodes = sentence_to_chosen_train_nodes_len[sentence][1]
  correctly_chosen_train_nodes = sentence_to_chosen_train_nodes_len[sentence][2]
  chosen_train_nodes = sentence_to_chosen_train_nodes_len[sentence][3]    
    
  score = float(correctly_chosen_train_nodes / (chosen_train_nodes + best_epsilon))
  #print("Score : {}".format(score))
  for _, node in enumerate(chosen_val_test_nodes):
    node_scores[node][my_label].append(score) 


for _, node in enumerate(val_test_nodes):  
  for my_label in range(label_len):
    node_scores[node][my_label].sort(reverse = True)



#sys.exit()







print("Determine hyper parameter h")
best_h = -0.6
best_accuracy = 0 

for _, my_h in enumerate([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]):
  known_nodes = train_nodes | val_nodes
  known_node_label = {}
  accurately_classified_nodes = 0 
  for _, node in enumerate(known_nodes):
    if node in node_label:
      known_node_label[node] = node_label[node]
  for _, node in enumerate(val_nodes):
    my_sentences_scores = copy.deepcopy(node_scores[node])
    belief = []
    for i in range(label_len):
      belief.append(0)
    adj_nodes = node_to_nodes[node]
    for _, adj_node in enumerate(adj_nodes & known_nodes):
      if adj_node in node_label:
        belief[known_node_label[adj_node]] = belief[known_node_label[adj_node]] + my_h
    for k in range(label_len):
      my_sentences_scores[k][0] = my_sentences_scores[k][0] + belief[k]

    first = find_max2(my_sentences_scores)
    if first == node_label[node]:
      accurately_classified_nodes = accurately_classified_nodes + 1 

  if accurately_classified_nodes >= best_accuracy:
    best_h = my_h
    best_accuracy = accurately_classified_nodes
  print()
  print("current h : {}".format(my_h))
  print("current score : {}".format(accurately_classified_nodes))

  known_nodes.add(node)
  known_node_label[node] = first


print("Best h : {}".format(best_h))

h = best_h


known_nodes = val_nodes | train_nodes
known_node_label = {}
for _, node in enumerate(known_nodes):
  if node in node_label:
    known_node_label[node] = node_label[node]




wrong_nodes = set()

accurately_classified_nodes = 0
for _, node in enumerate(test_nodes):
  my_sentences_scores = copy.deepcopy(node_scores[node])
  belief = []
  for i in range(label_len):
    belief.append(1)
  adj_nodes = node_to_nodes[node]
 
 
  for _, adj_node in enumerate(adj_nodes & known_nodes):
    if adj_node in node_label:
      belief[known_node_label[adj_node]] = belief[known_node_label[adj_node]] + h 
  for k in range(label_len):
    my_sentences_scores[k][0] = my_sentences_scores[k][0] + belief[k]

  first = find_max2(my_sentences_scores)

  if first == node_label[node]:
    accurately_classified_nodes = accurately_classified_nodes + 1

  else:
    wrong_nodes.add(node)

  known_nodes.add(node)
  known_node_label[node] = first



accuracy = float(accurately_classified_nodes/len(test_nodes))
print()
print("==============================================================")
print("Test Nodes : {}".format(len(test_nodes)))
print("Accurately Classified Nodes : {}".format(accurately_classified_nodes))
print("Accuracy : {}".format(accuracy))
print("==============================================================")

print(wrong_nodes)
