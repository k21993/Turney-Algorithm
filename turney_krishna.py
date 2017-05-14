# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:06:50 2017

@author: Admin
"""
from nltk import pos_tag
import re
import ast
import math
import os
import numpy as np

import string
from sklearn.model_selection import KFold
"""REPLACE WITH POS/NEG FILES PATH"""

path_pos=r"C:\Users\Admin\Desktop\TAMU 2 SEM\NLP\Assignment2\sentimentAnalyzer\data\imdb1\pos"
path_neg=r"C:\Users\Admin\Desktop\TAMU 2 SEM\NLP\Assignment2\sentimentAnalyzer\data\imdb1\neg"

file_names=os.listdir(path_pos)+os.listdir(path_neg)

"""Extracting phrases based on POS tag rules for all files"""
contents=[]
for c, f in enumerate(file_names):
    if c<1000:
        with open (path_pos+ "\\" +f) as txt:
    
              for line in txt:
                  contents.append(line)
        
            
    if c>=1000:
        with open (path_neg+ "\\" +f) as txt:
            for line in txt:
                  contents.append(line)
                  
res='\n'.join(contents).split()
pos_tags=pos_tag(res)
"""Store all phrases that satisfy the conditions in tag_pattern"""
tag_pattern=[]     
def find_pattern(postag):
   
       
    for k in range(len(postag)-2):
        
        if( postag[k][1]=="JJ" and postag[k+1][1]=="JJ" and postag[k+2][1]!="NN" and postag[k+2][1]!="NNS"):
            tag_pattern.append("".join(postag[k][0])+" "+"".join(postag[k+1][0]))
        
        if( (postag[k][1]=="NN" or postag[k][1]=="NNS") and postag[k+1][1]=="JJ" and postag[k+2][1]!="NN" and postag[k+2][1]!="NNS"):
            tag_pattern.append("".join(postag[k][0])+" "+"".join(postag[k+1][0]))
        
        if( (postag[k][1]=="RB" or postag[k][1]=="RBR" or postag[k][1]=="RBS") and postag[k+1][1]=="JJ" and postag[k+2][1]!="NN" and postag[k+2][1]!="NNS"):
            tag_pattern.append("".join(postag[k][0])+" "+"".join(postag[k+1][0]))

        
        if( (postag[k][1]=="RB" or postag[k][1]=="RBR" or postag[k][1]=="RBS") and (postag[k+1][1]=="VB" or postag[k+1][1]=="VBN" or postag[k+1][1]=="VBD" or postag[k+1][1]=="VBG")):
            tag_pattern.append("".join(postag[k][0])+" "+"".join(postag[k+1][0]))

            
            
        if( postag[k][1]=="JJ" and postag[k+1][1]=="NN" ) or ( postag[k][1]=="JJ" and postag[k+1][1]=="NNS" ):
            tag_pattern.append("".join(postag[k][0])+" "+"".join(postag[k+1][0]))
#
find_pattern(pos_tags)
#
"""mat_phrase_great: numpy matrix of hits between each phrase and thw word great, mat_phrase_poor:hits between phrase and poor in each file"""
"""mat_phrase_count: matrix storing 1 if a phrase is present in a file. used for adding corresponding SOs later."""
"""hits_great stores total hits of great in training set for each fold, correspondingly hits poor stores poor hits"""
tag_pattern=list(set(tag_pattern))
mat_phrase_great= np.zeros((len(tag_pattern), len(file_names)), dtype="int8")
mat_phrase_poor= np.zeros((len(tag_pattern), len(file_names)), dtype="int8")
mat_phrase_count=np.zeros((len(tag_pattern),len(file_names)), dtype="int8")
hits_great=[]
hits_poor=[]
for cnt, fi in enumerate(file_names):
    if cnt<1000:
        with open (path_pos+ "\\" +fi) as cf:
            txt=cf.read()
            txt = "".join(l for l in txt if l not in string.punctuation)
            file_list=txt.split()
            hits_great.append(file_list.count("great"))
            hits_poor.append(file_list.count("poor"))
            
            for j in range(len(tag_pattern)):
                all_hit_phrase_index=[]

                hits_phrase_great=0
                hits_phrase_poor=0
                if (tag_pattern[j] in txt):
                    mat_phrase_count[j][cnt]=1
                    
                    try:
                          
                          for w in (file_list):
                              if (w==tag_pattern[j].split()[0]):
                                  ind=file_list.index(w)
                                  if(file_list[ind+1]==tag_pattern[j].split()[1]):
                                      #print(ind)
                                      all_hit_phrase_index.append(ind)
            
                          
                          for ids in (all_hit_phrase_index):
                              #print(all_hit_index)
                              for words in file_list[ids-10 :ids+11]:
                                if words=="great":
                                    hits_phrase_great+=1

                                if words=="poor":
                                    hits_phrase_poor+=1
                                    
                          mat_phrase_great[j][cnt]=hits_phrase_great
                          mat_phrase_poor[j][cnt]=hits_phrase_poor
                    except:
                            pass
                        
                
#    
    if cnt>=1000:
        with open (path_neg+ "\\" +fi) as cf:
            txt=cf.read()
            file_list=txt.split()
            hits_great.append(file_list.count("great"))
            hits_poor.append(file_list.count("poor"))
            for j in range(len(tag_pattern)):
                all_hit_phrase_index=[]
                hits_phrase_great=0
                hits_phrase_poor=0
                if (tag_pattern[j] in txt):
                    mat_phrase_count[j][cnt]=1
                        
                    try:
                          
                          for w in (file_list):
                              if (w==tag_pattern[j].split()[0]):
                                  ind=file_list.index(w)
                                  if(file_list[ind+1]==tag_pattern[j].split()[1]):
                                      #print(ind)
                                      all_hit_phrase_index.append(ind)
                          for ids in (all_hit_phrase_index):
                            #print(all_hit_phrase_index)
                            for words in file_list[ids-10 :ids+11]:
                                if words=="great":
                                    hits_phrase_great+=1

                                if words=="poor":
                                    hits_phrase_poor+=1
                                    
                                    
                          mat_phrase_great[j][cnt]=hits_phrase_great
                          mat_phrase_poor[j][cnt]=hits_phrase_poor
            
                    except:
                            pass


acc_all=[]
def semantic_orientation(p_hit_great, p_hit_poor, hits_gr, hits_po):
    num=(p_hit_great*float(hits_po))+0.01
    den=(p_hit_poor*float(hits_gr))+0.01
    so=np.log2(np.divide(num, den))
    acc=0.0
    fold_no=0
    for f in test:
        polarity=0.0
        
        
        if f<1000:
            correct_label="positive"
            
        if f>=1000:
            correct_label="negative"
        for p in range(len(so)):
            
            if mat_phrase_count[p][f]==1:
                polarity+=so[p]
                
        if (polarity>=0.0):
            pred="positive"
            
        else:
            pred="negative"
     
        if(pred==correct_label):
             acc+=1
             
    acc=acc/float(200)
    acc_all.append(acc)
    
    print("[INFO] Fold accuracy: %r" %(acc))               
    fold_no+=1
 
### 3.dat is all tags, 2.tag is unique tags   
#mat_phrase_great.dump("mat_g3.dat")
#mat_phrase_great = np.load("mat_g2.dat")
#np.array(tag_pattern).dump("selected_phrases")                     
#mat_phrase_poor.dump("mat_po2.dat")
#mat_g = np.load("mat_po3.dat")
#mat_phrase_count.dump("mat_ph_counts1.dat")

"""10 Fold Cross-Validation"""

kf=KFold(n_splits=10, shuffle=True)
for train, test in kf.split(file_names):  ##for each fold in the 10 fold CV
   tr_great= mat_phrase_great[: , train[0]:train[-1]]
   phrase_hit_great=np.sum(tr_great, axis=1)
   tr_poor= mat_phrase_poor[: , train[0]:train[-1]]
   phrase_hit_poor=np.sum(tr_poor, axis=1)
   
   hits_gr=sum(hits_great[train[0]:train[-1]])
   hits_po=sum(hits_poor[train[0]:train[-1]])
   
   semantic_orientation(phrase_hit_great, phrase_hit_poor, hits_gr, hits_po)   
acc_avg=sum(acc_all)/float(10)
print("[INFO] Accuracy: %r " %(acc_avg))
   
   
#    


    
    
    
    
    
    
    
