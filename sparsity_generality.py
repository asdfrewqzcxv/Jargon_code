import pickle
import networkx as nx
import numpy as np
from jargon import *
from learn_jargon import *

import sys, os
import json



#datasets = 'BA-shapes'
datasets = 'Tree-Cycle'
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


def find_max3(my_list):
  max_idx = -1
  max_val = 0

  for i in range(len(my_list)):
    if my_list[i][0][0] > max_val:
      max_idx = i
      max_val = my_list[i][0][0]

  return max_idx




my_label_len = []
for my_label in range(label_len):
  my_label_len.append(len(label_nodes[my_label]))

max_idx = find_max(my_label_len)

sentence_to_chosen_train_nodes_len = {}



for my_label in range(label_len):
  print("Processing label {}".format(my_label))
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
      #print()
      #print("Sentence : {}".format(sentence.absList))
      #print("Sentence Root : {}".format(sentence.root))
      check_redundant_sentence.add(key)
    nodes = eval_sentence(sentence, succ_node_to_nodes, pred_node_to_nodes, X)
    chosen_val_test_nodes = nodes & val_test_nodes
    if len(nodes & val_test_nodes) > 0 :
      sentence_to_chosen_train_nodes_len[sentence] = [my_label, chosen_val_test_nodes, len(nodes & train_nodes & label_nodes[my_label]), len(nodes & train_nodes)]


 
node_scores = {}
for _, node in enumerate(val_test_nodes):  
  node_scores[node] = []
  for i in range(label_len):
    node_scores[node].append([(0,0)])
 
for _, sentence in enumerate(sentence_to_chosen_train_nodes_len):
  my_label = sentence_to_chosen_train_nodes_len[sentence][0]
  chosen_val_test_nodes = sentence_to_chosen_train_nodes_len[sentence][1]
  correctly_chosen_train_nodes = sentence_to_chosen_train_nodes_len[sentence][2]
  chosen_train_nodes = sentence_to_chosen_train_nodes_len[sentence][3]    
    
  score = float(correctly_chosen_train_nodes / (chosen_train_nodes + 10))
  #print("Score : {}".format(score))
  my_tuple = (score, sentence)
  for _, node in enumerate(chosen_val_test_nodes):
    node_scores[node][my_label].append(my_tuple) 
    #node_scores[node][my_label].append(score) 


for _, node in enumerate(val_test_nodes):  
  for my_label in range(label_len):
    node_scores[node][my_label].sort(reverse = True, key = lambda tup: tup[0])


if datasets == 'cora' or datasets == 'citeseer' or datasets == 'Tree-Cycle': 
  h = 0.5

if datasets == 'BA-shapes':
  h = 0.2

if datasets == 'pubmed':
  h = 0.4



known_nodes = val_nodes | train_nodes
known_node_label = {}
for _, node in enumerate(known_nodes):
  if node in node_label:
    known_node_label[node] = node_label[node]



sen_label_dict = {}


wrong_nodes = set()

test_nodes1 = list(test_nodes)
test_nodes1.sort()

print(len(test_nodes1))
cnt = 0

#distinct_sentences = set()


sparsity_sum = 0

my_nodes = 0

None_explanation = 0


accurately_classified_nodes = 0
for _, node in enumerate(test_nodes):


  nodes_in_one_hop = set([node]) | node_to_nodes[node]
  nodes_in_two_hop = set()
  
  for _, val in enumerate(nodes_in_one_hop):
    nodes_in_two_hop = nodes_in_two_hop | node_to_nodes[val]

  nodes_in_three_hop = set()

  for _, val in enumerate(nodes_in_two_hop):
    nodes_in_three_hop = nodes_in_three_hop | node_to_nodes[val]


  edges = set()
  for _, val in enumerate(nodes_in_three_hop): 
    successors = node_to_nodes[val] & nodes_in_three_hop
    for _, val_ in enumerate(successors):
      my_edge = (val, val_)
      edges.add(my_edge)

  num_edges = len(edges)/2



  #for _, node in enumerate(test_nodes1):
  my_sentences_scores = copy.deepcopy(node_scores[node])
  belief = []
  for i in range(label_len):
    belief.append(1)
  adj_nodes = node_to_nodes[node]
 
 
  for _, adj_node in enumerate(adj_nodes & known_nodes):
    if adj_node in node_label:
      belief[known_node_label[adj_node]] = belief[known_node_label[adj_node]] + h 
  for k in range(label_len):
    new_score = my_sentences_scores[k][0][0] + belief[k]
    sentence = my_sentences_scores[k][0][1]
    new_tuple = (new_score, sentence)
    my_sentences_scores[k][0] = new_tuple





  first = find_max3(my_sentences_scores)
  if my_sentences_scores[first][0][1] != 0:

    my_nodes = my_nodes + 1
    sentence_len = len(my_sentences_scores[first][0][1].absList) - 1
    if num_edges > 0:
      sparsity_sum = sparsity_sum + (1 - (sentence_len / num_edges))
    else:
      sparsity_sum = sparsity_sum + 1
          

    key = (first, json.dumps(my_sentences_scores[first][0][1].absList))
    if key in sen_label_dict:
      sen_label_dict[key] = sen_label_dict[key] + 1
    
    else:
      sen_label_dict[key] = 1
    '''
    print()
    print("Label : {}".format(node_label[node]))
    print("First : {}".format(node_label[node]))
    print("Sentence : {}".format(my_sentences_scores[first][0][1].absList))
    print("Sentence root : {}".format(my_sentences_scores[first][0][1].root))
    #print("Sentence : {}".format(len(my_sentences_scores[first][0][1].absList) - 1))
    print(node)
    '''
    cnt = cnt + (len(my_sentences_scores[first][0][1].absList) - 1)

  else:
    None_explanation = None_explanation + 1
    sparsity_sum = sparsity_sum + 1


  if first == node_label[node]:
    accurately_classified_nodes = accurately_classified_nodes + 1

  else:
    wrong_nodes.add(node)

  known_nodes.add(node)
  known_node_label[node] = first


#print(wrong_nodes)
my_list = []
for i, val in enumerate(sen_label_dict):
  my_list.append((val,sen_label_dict[val]))

my_list.sort(reverse = True, key = lambda tup : tup[1])





accuracy = float(accurately_classified_nodes/len(test_nodes))
print()
print("==============================================================")
print("Test Nodes : {}".format(len(test_nodes)))
print("Accurately Classified Nodes : {}".format(accurately_classified_nodes))
print("Accuracy : {}".format(accuracy))
print("==============================================================")
print()

print("Sparsity : {}".format(sparsity_sum/len(test_nodes)))
print("Generality : {}".format(len(test_nodes)/len(sen_label_dict)))

f = open("length.py", 'w')
f.write("total_len = 0\n")

for i, val in enumerate (my_list):
  (my_val , gen) = val
  (label, sentence) = my_val
  #print(gen)
  cmd = 'total_len = total_len + (len({}))\n'.format(sentence)
  f.write(cmd) 

f.write("print(total_len/{})".format(len(my_list)))
 
f.close()




