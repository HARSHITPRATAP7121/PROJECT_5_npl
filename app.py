# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 23:46:09 2022

@author: prata
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)

model1 = pickle.load(open('Naive-base5.pkl','rb'))
model2 = pickle.load(open('decision-tree5.pkl','rb')) 
model3 = pickle.load(open('Random-forest5.pkl','rb'))
model4 = pickle.load(open('k-nearest5.pkl','rb')) 
model5 = pickle.load(open('svm5.pkl','rb'))  

def review(text):
  df = pd.read_csv('NLP dataset 1.csv')
  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, 479):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  import re
  review = re.sub('[^a-zA-Z]', ' ', text)
  review=review.lower()
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  review = review.split()
  review1 = [word for word in review if not word in set(stopwords.words('english'))]
  from nltk.stem.porter import PorterStemmer
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
  review2 = ' '.join(review)
  X = cv.transform(review).toarray()
  input_pred1 = model1.predict(X)
  input_pred1 = input_pred1.astype(int)
  print(input_pred1)
  if input_pred1[0]==1:
    result1= "Review is Positive"
  else:
    result1="Review is negative" 
  return result1

  input_pred2 = model2.predict(X)
  input_pred2 = input_pred2.astype(int)
  print(input_pred2)
  if input_pred2[0]==1:
    result2= "Review is Positive"
  else:
    result2="Review is negative" 
  return result2

  input_pred3 = model3.predict(X)
  input_pred3 = input_pred3.astype(int)
  print(input_pred3)
  if input_pred3[0]==1:
    result3= "Review is Positive"
  else:
    result3="Review is negative" 
  return result3

  input_pred4 = model4.predict(X)
  input_pred4 = input_pred4.astype(int)
  print(input_pred4)
  if input_pred4[0]==1:
    result4= "Review is Positive"
  else:
    result4="Review is negative" 
  return result4

  input_pred5 = model5.predict(X)
  input_pred5 = input_pred5.astype(int)
  print(input_pred5)
  if input_pred5[0]==1:
    result5= "Review is Positive"
  else:
    result5="Review is negative" 
  return result5

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Summer Internship 2022</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
st.header("Text review system")
  
text = st.text_area("Write Text")

if st.button("Naive Bayes"):
  result1=review(text)
  st.success('Model has predicted {}'.format(result1))
if st.button("K-Nearest"):
  result4=review(text)
  st.success('Model has predicted {}'.format(result4))
if st.button("Random Forest"):
  result3=review(text)
  st.success('Model has predicted {}'.format(result3))
if st.button("Decision Tree"):
  result2=review(text)
  st.success('Model has predicted {}'.format(result2))
if st.button("SVM"):
  result5=review(text)
  st.success('Model has predicted {}'.format(result5))
      
if st.button("About"):
  st.subheader("Developed by Harshit Pratap")
  st.subheader(" Student, Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Summer Internship 2022 Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)