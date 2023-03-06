#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install wordcloud


# In[5]:


#pip install -U spacy


# In[6]:


#!pip install sklearn


# In[7]:


#!pip install -U scikit-learn


# In[8]:


#pip install --upgrade scikit-learn


# In[9]:


#pip install pandas


# In[10]:


#pip install seaborn


# In[11]:


import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


df=pd.read_csv("C:/Users/Lenovo/Downloads/fake_job_postings.csv")


# In[13]:


df.info()


# In[14]:


df.shape


# In[15]:


df.isna().sum()


# In[16]:


df.head(50)


# In[17]:


c1=df.drop(['job_id','telecommuting','has_company_logo','employment_type','salary_range'],axis=1)


# In[18]:


c1.info()


# In[19]:


c1.isna().sum()


# In[20]:


c1


# In[21]:


c1.fillna('',inplace=True)


# In[22]:


c1


# In[23]:


plt.figure(figsize=(15,5))
sns.countplot(y='fraudulent',data=c1)
plt.show()


# In[24]:


#from above fig we can say that majority of jobs posted are non-fradulent while some of the jobs are fradulent


# In[25]:


c1.groupby('fraudulent')['fraudulent'].count()  ##to know exact count of fradulent & non-fradulent jobs in our dataset


# In[26]:


exp=dict(c1.required_experience.value_counts())


# In[27]:


exp


# In[28]:


del exp['']


# In[29]:


exp


# In[30]:


plt.figure(figsize=(10,5))
sns.set_theme(style="whitegrid")
plt.bar(exp.keys(),exp.values())
plt.title('No of jobs with experience',size=20)
plt.xlabel('Experience',size=10)
plt.ylabel('No of jobs',size=12)
plt.xticks(rotation=30)
plt.show()


# In[31]:


#jobs posted based on countries....



# In[32]:


def split(location):
    l=location.split(',')
    return l[0]
c1['country']=c1.location.apply(split)


# In[33]:


c1.head()


# In[34]:


countr=dict(c1.country.value_counts()[:14])


# In[35]:


countr


# In[36]:


plt.figure(figsize=(8,6))
plt.title('country-wise job posting',size=20)
plt.bar(countr.keys(),countr.values())
plt.ylabel('No of jobs',size=10)
plt.xlabel('countries')


# In[37]:


edu=dict(c1.required_education.value_counts()[:7])
del edu['']
edu


# In[38]:


plt.figure(figsize=(15,6))
plt.title('job posting based on education',size=20)
plt.bar(edu.keys(),edu.values())
plt.ylabel('No of jobs',size=10)
plt.xlabel('Education',size=10)


# In[39]:


print(c1[c1.fraudulent==0].title.value_counts()[:10]) #these are the titles when job posted was non-fradulent ---top10


# In[40]:


print(c1[c1.fraudulent==1].title.value_counts()[:10])


# In[41]:


c1['text']=c1['title']+ '' +c1['company_profile']+ '' +c1['description']+ '' +c1['requirements']+ '' +c1['benefits']


# In[42]:


c1['text']


# In[43]:


del c1['title']
del c1['location']
del c1['department']
del c1['company_profile']
del c1['description']
del c1['requirements']
del c1['benefits']
del c1['required_experience']
del c1['required_education']
del c1['industry']
del c1['function']
del c1['country']


# In[44]:


c1.head()


# In[45]:


del c1['has_questions']


# In[46]:


c1.head()


# In[47]:


fraudjobs_text=c1[c1.fraudulent==1].text
realjobs_text=c1[c1.fraudulent==0].text


# In[48]:


STOPWORDS=spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc=WordCloud(min_font_size=3, max_words=3000 ,width=1600, height=800 , stopwords=STOPWORDS).generate(str(" ".join(fraudjobs_text)))
plt.imshow(wc,interpolation= 'bilinear')


# In[49]:


STOPWORDS=spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc=WordCloud(min_font_size=3, max_words=3000 ,width=1600, height=800 , stopwords=STOPWORDS).generate(str(" ".join(realjobs_text)))
plt.imshow(wc,interpolation= 'bilinear')


# In[50]:


#pip install -U pip setuptools wheel


# In[51]:


#! python -m spacy download en_core_web_sm


# In[52]:


import spacy


# In[53]:


import en_core_web_sm
nlp = en_core_web_sm.load()


# In[54]:


punctuation =string.punctuation
nlp = spacy.load("en_core_web_sm")
stop_words=spacy.lang.en.stop_words.STOP_WORDS
parser=English()

def spacy_tokenizer(sentence):
    mytokens=parser(sentence)
    mytokens=[word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens=[word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self ,X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}
def clean_text(text):
    return text.strip().lower()


# In[55]:


c1['text']=c1['text'].apply(clean_text)


# In[56]:


cv=TfidfVectorizer(max_features=100) 
x=cv.fit_transform(c1['text'])
c2=pd.DataFrame(x.toarray(),columns=cv.get_feature_names_out())
c1.drop(['text'],axis=1,inplace=True)
main_c1=pd.concat([c2,c1],axis=1)


# In[57]:


main_c1.head()


# In[58]:


Y=main_c1.iloc[:,-1]
X=main_c1.iloc[:,:-1]


# In[59]:


X


# In[60]:


Y


# In[61]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# In[62]:


X_train.shape ,X_test.shape,y_train.shape ,y_test.shape


# In[63]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion ="entropy")
model=rfc.fit(X_train,y_train)


# In[64]:


print(X_test)


# In[65]:


pred=rfc.predict(X_test)
score=accuracy_score(y_test,pred)


# In[66]:


score*100


# In[67]:


print("Classification Report\n")
print(classification_report(y_test,pred))


# In[68]:


from sklearn.svm import SVC


# In[100]:


svc=SVC(kernel='poly', random_state=0)


# In[101]:


svc.fit(X_train, y_train) 


# In[102]:


y_pred1=svc.predict(X_test)


# In[103]:


print(classification_report(y_test,y_pred1))


# In[104]:


score=accuracy_score(y_test,y_pred1)
score*100


# In[ ]:




