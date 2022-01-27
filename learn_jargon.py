from jargon import *
import copy
import json



def learn_sentences(parameter):
  i = 1 
  sentences = set()
  parameter.original_labeled_nodes = parameter.labeled_nodes
  parameter.covered_nodes = set()
  print("labeled nodes : "+str(len(parameter.labeled_nodes)))
  print("COVERED nodes : "+str(len(parameter.covered_nodes)))
  print("ORIGINAL_LABELED_NODES nodes : "+str(len(parameter.original_labeled_nodes)))
  while(len(parameter.labeled_nodes)>0):
    print()
    print()
    print("Outer iteration : "+str(i))
    print ("")
    print ("--------------------------------------------")
    '''new_sentence = learn_a_sentence(node_to_nodes, X_arr, labeled_nodes, parameter.feature_results)'''
    (new_sentences, new_sentence) = learn_a_sentence(parameter)
    print ("--------------------------------------------")
    print ("")
    print ("============================================")
    print ("Learned Best Sentence : "+str(new_sentence.absList))
    print ("Learned Best Sentence root : "+str(new_sentence.root))
    #chosen_nodes = filter_eval_sentence(new_sentence, parameter.node_to_nodes, parameter.X_arr, parameter.original_labeled_nodes, parameter.original_labeled_nodes) & parameter.train_nodes
    chosen_nodes = eval_sentence(new_sentence, parameter.succ_node_to_nodes, parameter.pred_node_to_nodes, parameter.X_arr) & parameter.train_nodes 
    print ("# of chosen Nodes : "+str(len((chosen_nodes & parameter.train_nodes))))
    print ("Chosen Labeled Nodes : "+str(chosen_nodes & parameter.original_labeled_nodes))
    print ("# of chosen Non Labeled Nodes : "+str(len((chosen_nodes - parameter.original_labeled_nodes))))
    parameter.covered_nodes = parameter.covered_nodes | (parameter.original_labeled_nodes & chosen_nodes)
    print ("COVERED nodes : "+str(len(parameter.covered_nodes)))
    print ("============================================")
    
    if len(chosen_nodes & parameter.original_labeled_nodes) == len(parameter.original_labeled_nodes):
      #Found a very good sentence
      print("Find a very good sentence")
      return set([new_sentence])

    #sentences.add(new_sentence)
    sentences = sentences | new_sentences 
    parameter.labeled_nodes = parameter.labeled_nodes - chosen_nodes 
    print("left nodes : "+str(len(parameter.labeled_nodes)))
    i = i+1
    if parameter.is_complex_graph and (len(parameter.labeled_nodes) < len(parameter.original_labeled_nodes) * 0.1):
      print("It is too complex graph, we stop learning here")
      return sentences
  return sentences



def learn_a_sentence(parameter):
  sentence = Sentence()
  sentence.root = 0 
  sentence.absList = [{}]
  
  current_sentence_score = -1.0
  new_sentence_score = 0.0
  current_sentence = sentence
  new_sentence = sentence
  new_sentences = set()
  i = 1
  parameter.filtered_nodes = parameter.train_nodes
  chosen_nodes = parameter.filtered_nodes
  while(new_sentence_score > current_sentence_score):
    current_sentence = new_sentence
    current_sentence_score = new_sentence_score
    print("")
    print("")
    print("---------------------------------------------------------------------")
    print("Inner iteration : "+str(i))
    if parameter.is_one_hot == True:
      (new_sentence, new_sentence_score) = specify_binary(current_sentence, parameter)
    else:
      (new_sentence, new_sentence_score) = specify(current_sentence, parameter)
    
    new_sentences.add(new_sentence) 
    print("new_sentence : "+str(len(new_sentence.absList)))
    print("new_sentence root : "+str(new_sentence.root))
    print("new_sentence score : : "+str(new_sentence_score))
    chosen_nodes = eval_sentence(new_sentence, parameter.succ_node_to_nodes, parameter.pred_node_to_nodes , parameter.X_arr) & parameter.train_nodes 
    parameter.filtered_nodes = chosen_nodes
    print("# of chosen Nodes : "+str(len(chosen_nodes)))
    print("Chosen Labeled Nodes : "+str(chosen_nodes & parameter.original_labeled_nodes))
    if len(chosen_nodes) == len(chosen_nodes & parameter.original_labeled_nodes):
      print()
      print("All nodes are labeled ones")
      break 
    print("---------------------------------------------------------------------")
    i = i+1
    if parameter.is_complex_graph and (new_sentence_score > 0.9):
      print ("============================================")
      print ("Learned Best Sentence : "+str(new_sentence.absList))
      print ("Learned Best Sentence root : "+str(new_sentence.root))
      print ("Learned Best score : "+str(new_sentence_score))
      print ("============================================")
      return (new_sentences, new_sentence)

  print ("============================================")
  print ("Learned Best Sentence : "+str(new_sentence.absList))
  print ("Learned Best Sentence root : "+str(new_sentence.root))
  print ("Learned Best score : "+str(new_sentence_score))
  print ("============================================")
  return (new_sentences, new_sentence)

def specify(sentence, parameter):
  best_score = score(sentence, parameter)
  best_sentence = sentence
  print ("Curret Best sentence : " + str(best_sentence.absList))
  print ("Curret Best score : " + str(best_score))

  print ("sentence len : " + str(len(sentence.absList)))
  print ("features len : " + str(len(parameter.X_arr[0])))
  candidate_sentences = set([sentence]) 
  for depth in range(parameter.chosen_depth):
    print()
    print("=====================================")
    print("Depth : {}".format(depth))
    print("=====================================")
    print()
    new_candidate_sentences = set()
    for _, sentence in enumerate(candidate_sentences): 
    
      for i in range(len(sentence.absList)):
        for j in range(len(parameter.X_arr[0])):
          abs_node = sentence.absList[i]
          if j in abs_node:
            bot = abs_node[j][0]
            top = abs_node[j][1]
            top_idx = parameter.feature_list.index(top)
            #bot_idx = parameter.feature_list.index(bot)

            bot_idx = parameter.feature_list2.index(bot)
            bot_idx = len(parameter.feature_list) - bot_idx
            #if top_idx == bot_idx + 1:
            #  continue
            new_sentence = copy.deepcopy(sentence)
            #if top_idx == bot_idx + 1 :
            if parameter.feature_list[top_idx -1] == bot :
              #print("adj")
              new_sentence.absList[i][j] = (bot, bot)
            else:
              new_sentence.absList[i][j] = (bot, parameter.feature_list[int((top_idx+bot_idx)/2)])
              
            new_score = score(new_sentence, parameter)
            if depth < parameter.chosen_depth-1:
              new_candidate_sentences.add(new_sentence)
            if (new_score > best_score):
              best_score = new_score
              best_sentence = new_sentence
              print ("Current Best sentence : " + str(best_sentence.absList))
              print ("Current Best score : " + str(best_score))
              #sys.exit()
              

            new_sentence = copy.deepcopy(sentence)
            
            #if top_idx == bot_idx + 1 :
            if parameter.feature_list[top_idx -1] == bot :
              #print("adj")
              new_sentence.absList[i][j] = (top, top)
            else:
              new_sentence.absList[i][j] = (parameter.feature_list[int((top_idx+bot_idx)/2)], top)
            new_score = score(new_sentence, parameter)
            if depth < parameter.chosen_depth-1:
              new_candidate_sentences.add(new_sentence)
            if (new_score > best_score):
              best_score = new_score
              best_sentence = new_sentence
              print ("Current Best sentence : " + str(best_sentence.absList))
              print ("Current Best score : " + str(best_score))
              #sys.exit()
            #do This
          else:    
            top = parameter.min_max_feature[0][1] 
            bot = parameter.feature_list[int(len(parameter.feature_list)/2)] 
            new_sentence = copy.deepcopy(sentence)
            new_sentence.absList[i][j] = (bot, top)
            new_score = score(new_sentence, parameter)
            if depth < parameter.chosen_depth-1:
              new_candidate_sentences.add(new_sentence)
            '''if (new_score >= best_score):'''
            if (new_score > best_score):
              best_score = new_score
              best_sentence = new_sentence
              print ("Current Best sentence : " + str(best_sentence.absList))
              print ("Current Best score : " + str(best_score))
            #bot = 0.0
            bot = parameter.min_max_feature[0][0] 
            top = parameter.feature_list[int(len(parameter.feature_list)/2)] 
            new_sentence = copy.deepcopy(sentence)
            new_sentence.absList[i][j] = (bot, top)
            new_score = score(new_sentence, parameter)
            if depth < parameter.chosen_depth-1:
              new_candidate_sentences.add(new_sentence)
            '''if (new_score >= best_score):'''
            if (new_score > best_score):
              best_score = new_score
              best_sentence = new_sentence
              print ("Current Best sentence : " + str(best_sentence.absList))
              print ("Current Best score : " + str(best_score))
      if len(sentence.absList) < 3:
        for j in range(len(parameter.X_arr[0])):
          new_node = {}
          bot = parameter.feature_list[int(len(parameter.feature_list)/2)] 
          top = parameter.min_max_feature[0][1] 
          #top = 1
          new_node[j] = (bot,top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.append(new_node)
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          '''if (new_score >= best_score):'''
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))

          new_node = {}
          #bot = 0.0 
          bot = parameter.min_max_feature[0][0] 
          top = parameter.feature_list[int(len(parameter.feature_list)/2)] 
          new_node[j] = (bot, top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.append(new_node)
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          '''if (new_score >= best_score):'''
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))



        for j in range(len(parameter.X_arr[0])):
          new_node = {}
          bot = parameter.feature_list[int(len(parameter.feature_list)/2)] 
          top = parameter.min_max_feature[0][1] 
          #top = 1.0
          new_node[j] = (bot,top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.insert(0,new_node)
          new_sentence.root = new_sentence.root+1 
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          '''if (new_score >= best_score):'''
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))


          new_node = {}
          #bot = 0.0 
          bot = parameter.min_max_feature[0][0] 
          top = parameter.feature_list[int(len(parameter.feature_list)/2)] 
          new_node[j] = (bot,top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.insert(0,new_node)
          new_sentence.root = new_sentence.root+1 
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          '''if (new_score >= best_score):'''
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))
    candidate_sentences = new_candidate_sentences
  print ("The Best sentence's list : " + str(best_sentence.absList))
  print ("The Best sentence's root : " + str(best_sentence.root))
  print ("The Best sentence's score : " + str(best_score))
  return (best_sentence, best_score) 



def specify_binary(sentence, parameter):
  best_score = score(sentence, parameter)
  best_sentence = sentence
  print ("Curret Best sentence : " + str(best_sentence.absList))
  print ("Curret Best score : " + str(best_score))

  print ("sentence len : " + str(len(sentence.absList)))
  print ("features len : " + str(len(parameter.X_arr[0])))
  candidate_sentences = set([sentence]) 
  for depth in range(parameter.chosen_depth):
    print()
    print("=====================================")
    print("Depth : {}".format(depth))
    print("=====================================")
    print()
    new_candidate_sentences = set()
    for _, sentence in enumerate(candidate_sentences): 
      for i in range(len(sentence.absList)):
        '''
        if i == sentence.root:
          feat_set = parameter.features[0]
        elif i == (len(sentence.absList) - sentence.root + 1) or i == (len(sentence.absList) - sentence.root - 1):
          feat_set = parameter.features[1]
        else : 
          feat_set = parameter.features[2]
        for _, j in enumerate(feat_set):
        '''
        for j in range(len(parameter.X_arr[0])):
          abs_node = sentence.absList[i]
          if j in abs_node:
            continue
          else:
            bot = 0.5 
            top = 1
            new_sentence = copy.deepcopy(sentence)
            new_sentence.absList[i][j] = (bot, top)
            new_score = score(new_sentence, parameter)
            if depth < parameter.chosen_depth-1:
              new_candidate_sentences.add(new_sentence)
            if (new_score > best_score):
              best_score = new_score
              best_sentence = new_sentence
              print ("Current Best sentence : " + str(best_sentence.absList))
              print ("Current Best score : " + str(best_score))
            
            bot = 0.0
            top = 0.5
            new_sentence = copy.deepcopy(sentence)
            new_sentence.absList[i][j] = (bot, top)
            new_score = score(new_sentence, parameter)
            if depth < parameter.chosen_depth-1:
              new_candidate_sentences.add(new_sentence)
            if (new_score > best_score):
              best_score = new_score
              best_sentence = new_sentence
              print ("Current Best sentence : " + str(best_sentence.absList))
              print ("Current Best score : " + str(best_score))
      #'''
      if len(sentence.absList) < 3:
        '''
        if len(sentence.absList) == 1:
          feat_set = parameter.features[1]
        if len(sentence.absList) == 2 and sentence.root == 1:
          feat_set = parameter.features[1]
        if len(sentence.absList) == 2 and sentence.root == 0:
          feat_set = parameter.features[2]
        for _, j in enumerate(feat_set):
        '''  
        for j in range(len(parameter.X_arr[0])):
          new_node = {}
          bot = 0.5 
          top = 1
          new_node[j] = (bot,top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.append(new_node)
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))
          new_node = {}
          bot = 0.0 
          top = 0.5 
          new_node[j] = (bot, top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.append(new_node)
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))

        '''
        if len(sentence.absList) == 1:
          feat_set = parameter.features[1]
        if len(sentence.absList) == 2 and sentence.root == 0:
          feat_set = parameter.features[1]
        if len(sentence.absList) == 2 and sentence.root == 1:
          feat_set = parameter.features[2]
        for _, j in enumerate(feat_set):
        '''


        for j in range(len(parameter.X_arr[0])):
          new_node = {}
          bot = 0.5 
          top = 1.0
          new_node[j] = (bot,top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.insert(0,new_node)
          new_sentence.root = new_sentence.root+1 
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))

          new_node = {}
          bot = 0.0 
          top = 0.5
          new_node[j] = (bot,top)
          new_sentence = copy.deepcopy(sentence)
          new_sentence.absList.insert(0,new_node)
          new_sentence.root = new_sentence.root+1 
          new_score = score(new_sentence, parameter)
          if depth < parameter.chosen_depth-1:
            new_candidate_sentences.add(new_sentence)
          if (new_score > best_score):
            best_score = new_score
            best_sentence = new_sentence
            print ("Current Best sentence : " + str(best_sentence.absList))
            print ("Current Best score : " + str(best_score))
      #'''
    candidate_sentences = new_candidate_sentences
  print ("The Best sentence's list : " + str(best_sentence.absList))
  print ("The Best sentence's root : " + str(best_sentence.root))
  print ("The Best sentence's score : " + str(best_score))
  return (best_sentence, best_score) 




def score(sentence, parameter):
  key = (sentence.root, json.dumps(sentence.absList))
  if key in parameter.dict:
    nodes = parameter.dict[key]
  else:  
    nodes = filter_eval_sentence(sentence, parameter.succ_node_to_nodes, parameter.pred_node_to_nodes, parameter.X_arr, parameter.filtered_nodes)
    parameter.dict[key] = nodes
 
  #print()
  #print("Sentence : {}".format(sentence.absList))
  #print("Node Len : {}".format(len(nodes)))
  nodes = nodes & parameter.train_nodes
  #print("Score : {}".format(score))
  if len(parameter.labeled_nodes & nodes) == 0:  
    return 0.001
  #score = float(len(parameter.original_labeled_nodes & nodes)) / float(len(nodes)+0.1)
  #score = float(len(parameter.original_labeled_nodes & nodes)) / float(len(nodes)+0.1)
  #score = float(len(parameter.original_labeled_nodes & nodes)) / float(len(nodes)+0.1)
  #score = float(len(parameter.original_labeled_nodes & nodes)) / float(len(nodes)+10)
  score = float(len(parameter.original_labeled_nodes & nodes)) / float(len(nodes) + parameter.epsilon)
  return score
  #return float(len(parameter.original_labeled_nodes & nodes)) / float(len(nodes)+0.1)






