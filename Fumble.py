#loading necessary libraries
import streamlit as st
import pandas as pd
import pickle
from io import StringIO
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import re
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import html.parser
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import xlrd




def main():
   
    html_temp = """
    <div style="background-color:#184B44;padding:5px">
    <h1 style="text-align:center;"> Fumble </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    


st.markdown('##')

okc = pd.read_excel("D:/5th Sem/Project/Copy of User Details_Faf.xlsx")
st.title("Home")

if __name__=='__main__':
    main() 
    
st.markdown('##')  
    
if st.checkbox("Show data"):
    st.write(okc)
  
st.markdown('##')  
 
   
'''

Fill These List Of Questions To Get Your Match

'''   
st.markdown('##')

     
     

okc1 = pd.DataFrame() 

#def user_input_features():
name = st.text_input("Your Name Please","Type..") 
age = st.number_input("Age",18,79) 
status = st.selectbox(" Status", ["Single","in a relationship","Unknown"]) 
gender = st.selectbox("Gender", ["Female","Male"]) 
orientation = st.selectbox("Orientation", ["Straight","Bisexual","Gay"]) 
body_type = st.selectbox("body_Type", ["Fit","Average","Curvy","Thin","Overweight","Rather not say"]) 
education = st.selectbox("Education", ["college/university","Masters and above","other","Two-year college","High school","Med / Law school"]) 
ethnicity= st.selectbox("Ethnicity", ["White","Asian","Hispanic","African American","Mixed","Unknown","others"]) 
religion = st.selectbox("Religion", ["Agnosticism","Atheism","Christianity","Catholicism","Judaism","Buddhism","Islam","Hinduism","Unknown","others"]) 
smokes = st.selectbox("Smokes", ["Yes","No"]) 
drink = st.selectbox("Drink", ["Yes","No"]) 
diet = st.selectbox("Diet", ["Anything","Vegan","Vegetarian","Halal","Kosher","other"]) 
speaks = st.text_input("Speaks ","Type..") 
sign = st.selectbox("Sign", ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpion","Sagittarius","Capricorns","Aquarius","Pisces"]) 
offspring = st.selectbox("Offspring", ["Wants Kids","Does not want kids","Has kid","Unknown"]) 
drugs = st.selectbox("Drugs", ["Yes","No"]) 
height = st.number_input("Height \n (In inches)",30,100) 
income = st.number_input("Income",0,100000) 
pets = st.selectbox("Pets Preference", ["Likes Cats and Dogs","Dislikes Cats and Dogs","Likes only cats","Likes only Dogs","Unknown"]) 
job = st.selectbox("Job", ["Office/Professional","Science/Tech","Business Management","Creative"]) 
essay0 = st.text_input("My Self Summary","Type..") 
essay1 = st.text_input("What I Am Doing With My Life","Type..") 
essay2 = st.text_input("I am really good at ","Type..") 
essay3 = st.text_input(" The First Thing People Usually Notice About Me","Type..") 
essay4 = st.text_input(" Favourite Books, Movies, Show, Music, Food","Type..") 
essay5 = st.text_input(" The 6 Things That I Could Never Do Without","Type..") 
essay6 = st.text_input(" I Spend A Lot Of Time Thinking About","Type") 
essay7 = st.text_input(" On A Typical Friday Night I Am","Type..") 
essay8 = st.text_input(" The Most Private Thing I Am Willing To Admit","Type..") 
essay9 = st.text_input(" You Should Message Me If","Type..") 

   

data = {"Name": name, "age": age, 
     "gender": gender, "orientation":orientation,
     "status":status,"education":education, "ethnicity":ethnicity, 
                 "religion" : religion,"smokes":smokes, "drink":drink, "body_type":body_type,"diet":diet,"job":job,
                 "speaks":speaks,"sign":sign,"offspring":offspring,"drugs":drugs,"height":height,
                 "income":income,"pets":pets,"essay0":essay0,"essay1":essay1,"essay2":essay2,
                 "essay3":essay3,"essay4":essay4,"essay5":essay5,"essay6":essay6,"essay7":essay7,
                 "essay8":essay8,"essay9":essay9}

data = pd.DataFrame(data,index=[0])
data.columns = data.columns.str.lower()
okc.columns = okc.columns.str.lower()
features = pd.DataFrame(data) 
         

okc1 = pd.concat([features,okc],axis=0,join="inner",ignore_index=True) 


ok= okc1.copy(deep=True)


#Data cleaning and pre-processing
ok['essay']=ok[['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9']].apply(lambda x: ' '.join(x), axis=1)
ok.drop(['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9'],axis=1,inplace=True)

corpus_df = ok.copy(deep=True)



corpus_df['corpus'] = ok[['age', 'status', 'gender', 'orientation', 'body_type', 'diet', 'drink',
       'drugs', 'education', 'ethnicity', 'height', 'income', 'job',
       'offspring', 'pets', 'religion', 'sign', 'smokes', 'speaks', 'essay']].astype(str).agg(' '.join, axis=1)
corpus_df = corpus_df.astype(str)



corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('\n', ' '))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('nan', ' '))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("\'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("-'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("--'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("='", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("/", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(".", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(":", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(",", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("(", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(")", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("?", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("!", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(";", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('"', " "))
corpus_df['corpus'] = corpus_df['corpus'].str.replace('\d+', '')
corpus_list = corpus_df['corpus']


# vectorization
stemmer = SnowballStemmer("english")

tfidf = TfidfVectorizer(stop_words = "english", ngram_range = (1,3), max_df=0.8, min_df=0.2) 
corpus_tfidf = tfidf.fit(corpus_list)
corpus_2d = pd.DataFrame(tfidf.transform(corpus_list).todense(),
                   columns = tfidf.get_feature_names(),)
tfidf_vec = tfidf.fit_transform(corpus_list)

corpus_2d.head()
corpus_mat_sparse = csr_matrix(corpus_2d.values)

pd.set_option('display.max_columns',25)
pd.set_option('expand_frame_repr', False)

#Model specification
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute') # cosine takes 1-cosine( as cosine distance)
model_knn.fit(corpus_mat_sparse)

#recommendation algorithm (cosine)
def rec(query_index):
  distances, indices = model_knn.kneighbors(corpus_2d.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 21)

  for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}: \n'.format(corpus_2d.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2} \n '.format(i, corpus_2d.index[indices.flatten()[i]],distances.flatten()[i]))
  for i in indices:
    print(okc1.loc[i,:])
 
    
 
def rec(query_index):
  distances, indices = model_knn.kneighbors(corpus_2d.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
  result= pd.DataFrame()
  for i in indices:
    result= result.append(okc1.iloc[i,:])
  result['similarity distance']= distances.flatten()
  return result[['name',"similarity distance","age","status","orientation","body_type","ethnicity","religion","smokes","drink","diet","essay0"]]    

st.markdown('##')
st.markdown('##')   

if st.button("Join"):
    st.write("Your Matches are:")
    st.write(rec(0))
    




    

         

        


