import pickle
#import networkx as nx
import numpy as np
from jargon import *
from learn_jargon import *

import sys, os
#import json


datasets = 'BA-shapes'

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

with open('datasets/{}/succ_node_to_nodes.pickle'.format(datasets),'rb') as f:
  succ_node_to_nodes = pickle.load(f) 
with open('datasets/{}/pred_node_to_nodes.pickle'.format(datasets),'rb') as f:
  pred_node_to_nodes = pickle.load(f) 
with open('datasets/{}/node_to_nodes.pickle'.format(datasets),'rb') as f:
  node_to_nodes = pickle.load(f) 

node_label = {}
label_nodes = {}
node_len = len(X)
label_len = len(Y[0])



for node in range(node_len):
  for label in range(label_len):
    if not label in label_nodes:
      label_nodes[label] = set()
    if Y[node][label] > 0:
      label_nodes[label].add(node)
      node_label[node] = label


print("=====================================")
print("BA-shapes dataset")
print()
print()


print("Label 0")
print()
sentence1 = Sentence()
sentence1.absList = [ {0: (12.0, 999.0)}, {}, {0: (12.0, 999.0)}]
sentence1.root = 1
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 0 : {}".format(len(label_nodes[0])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[0])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[0])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[0])/len(label_nodes[0])))


print("Label 1")
print()
sentence1 = Sentence()
sentence1.absList = [ {0: (4.0, 999.0)}, {0: (3.0, 4.0)}, {0: (2.0, 2.0)}]
sentence1.root = 1
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 1 : {}".format(len(label_nodes[1])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[1])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[1])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[1])/len(label_nodes[1])))



print("Label 2")
print()
sentence1 = Sentence()
sentence1.absList = [ {0: (2.0, 2.0)}, {0: (2.0, 2.0)}]
sentence1.root = 0 
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 2 : {}".format(len(label_nodes[2])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[2])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[2])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[2])/len(label_nodes[2])))




print("Label 3")
print()
sentence1 = Sentence()
sentence1.absList = [ {0: (4.0, 999.0)}, {0: (2.0, 2.0)}, {0: (3.0, 4.0)}]
sentence1.root = 1
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 3 : {}".format(len(label_nodes[3])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[3])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[3])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[3])/len(label_nodes[3])))




datasets = 'Tree-Cycle'

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

with open('datasets/{}/succ_node_to_nodes.pickle'.format(datasets),'rb') as f:
  succ_node_to_nodes = pickle.load(f) 
with open('datasets/{}/pred_node_to_nodes.pickle'.format(datasets),'rb') as f:
  pred_node_to_nodes = pickle.load(f) 
with open('datasets/{}/node_to_nodes.pickle'.format(datasets),'rb') as f:
  node_to_nodes = pickle.load(f) 

node_label = {}
label_nodes = {}
node_len = len(X)
label_len = len(Y[0])



for node in range(node_len):
  for label in range(label_len):
    if not label in label_nodes:
      label_nodes[label] = set()
    if Y[node][label] > 0:
      label_nodes[label].add(node)
      node_label[node] = label

print()
print()
print("=====================================")
print("Tree-Cycle dataset")
print()
print()


print("Label 0")
print()
sentence1 = Sentence()
sentence1.absList = [ {0: (3.0, 3.0)}, {0: (3.0, 5.0)}]
sentence1.root = 1
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 0 : {}".format(len(label_nodes[0])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[0])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[0])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[0])/len(label_nodes[0])))

print()
sentence1 = Sentence()
sentence1.absList = [ {0: (-999.0, 1.0)}]
sentence1.root = 0
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 0 : {}".format(len(label_nodes[0])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[0])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[0])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[0])/len(label_nodes[0])))


print()
print()
print("Label 1")
print()
sentence1 = Sentence()
sentence1.absList = [{}, {0: (-999.0, 2.0)}, {0: (-999.0, 2.0)}]
sentence1.root = 0 
print("Sentence : {}".format(sentence1.absList))
print("Subject word idx : {}".format(sentence1.root))

chosen_nodes1 = eval_sentence(sentence1, node_to_nodes, node_to_nodes, X)

print("Nodes in label 1 : {}".format(len(label_nodes[1])))
print("Chosen nodes : {}".format(len(chosen_nodes1)))
print("Correctly chosen nodes: {}".format(len(chosen_nodes1 & label_nodes[1])))
print("Accuracy : {}".format(len(chosen_nodes1 & label_nodes[1])/len(chosen_nodes1)))
print("Recall : {}".format(len(chosen_nodes1 & label_nodes[1])/len(label_nodes[1])))










