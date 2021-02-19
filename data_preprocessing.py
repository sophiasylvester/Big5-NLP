#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing:
# 
# ### Features (as in paper):
# 1. 11.140 ngram features: tf and tf-idf weighted word and character ngrams stemmed with Porter's stemmer
# 2. type-token ratio
# 3. ratio of comments in English
# 4. ratio of British english vs. American English words
# 5. 93 features from LIWC 
# 6. 26 PSYCH features (Preotiuc: Paraphrase Database and NRC Psycholinguistics Database)
# 
# ### Columns (from the description of the dataset):
# 1. 'global':[7,10], #subreddits_commented, subreddits_commented_mbti, num_comments
# 2. 'liwc':[10,103], #liwc
# 3. 'word':[103,3938], #top1000 word ngram (1,2,3) per dimension based on chi2
# 4. 'char':[3938,7243], #top1000 char ngrams (2,3) per dimension based on chi2
# 5. 'sub':[7243,12228], #number of comments in each subreddit
# 6. 'ent':[12228,12229], #entropy
# 7. 'subtf':[12229,17214], #tf-idf on subreddits
# 8. 'subcat':[17214,17249], #manually crafted subreddit categories
# 9. 'lda50':[17249,17299], #50 LDA topics
# 10. 'posts':[17299,17319], #posts statistics
# 11. 'lda100':[17319,17419], #100 LDA topics
# 12. 'psy':[17419,17443], #psycholinguistic features
# 13. 'en':[17443,17444], #ratio of english comments
# 14. 'ttr':[17444,17445], #type token ratio
# 15. 'meaning':[17445,17447], #additional pyscholinguistic features
# 16. 'time_diffs':[17447,17453], #commenting time diffs
# 17. 'month':[17453,17465], #monthly distribution
# 18. 'hour':[17465,17489], #hourly distribution
# 19. 'day_of_week':[17489,17496], #daily distribution
# 20. 'word_an':[17496,21496], #word ngrams selected by F-score
# 21. 'word_an_tf':[21496,25496], #tf-idf ngrams selected by F-score
# 22. 'char_an':[25496,29496], #char ngrams selected by F-score
# 23. 'char_an_tf':[29496,33496], #tf-idf char ngrams selected by F-score
# 24. 'brit_amer':[33496,33499], #british vs american english ratio
# 

# ## Import packages

# In[2]:


import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import bigrams, ngrams
import string
from string import punctuation
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer 
from num2words import num2words 
import pandas as pd
from empath import Empath
import random
random.seed(32)

# close nltk download window to continue


# ## Import data

# In[3]:


pandora = pd.read_csv('/home/sophia/ma_py/pandora_bigfive1000.csv')

authors = pd.read_csv('/home/sophia/ma_py/author_profiles.csv')

bigfive = authors[['author', 'agreeableness','openness','conscientiousness','extraversion','neuroticism']]
bigfive = bigfive.dropna()

pandoradf = pd.merge(pandora, bigfive, on='author', how='outer')
pandoradf = pandoradf.dropna()
pandoradf = pandoradf.reset_index()
pandoradf.tail()


# ## Feature extraction

# In[4]:


def choose_stopwordlist(df, mode):
    if mode == 'NLTK':
        stopwordList = stopwords.words('english')
    if mode == 'NLTK-neg':
        stopwordList = stopwords.words('english')
        stopwordList.remove('no')
        stopwordList.remove('nor')
        stopwordList.remove('not')
    return stopwordList

stopwordList = choose_stopwordlist(pandoradf, mode='NLTK-neg')

print(stopwordList)


# In[5]:


def create_features(workdata):

    # Total number of characters (including space)
    workdata['char_count'] = workdata['body'].str.len()

    # Total number of stopwords
    workdata['stopwords'] = workdata['body'].apply(lambda x: len([x for x in x.split() if x in stopwordList]))

    # Total number of punctuation or special characters
    workdata['total_punc'] = workdata['body'].apply(lambda x: len([x for x in x.split() for j in x if j in string.punctuation]))

    # Total number of numerics
    workdata['total_num'] = workdata['body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    # Total number of uppercase words
    workdata['total_uppercase'] = workdata['body'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    
    return workdata

featuredf = create_features(pandoradf)
featuredf.head()


# ### Create ngrams
# 
# 1. lower 
# 2. tokenize
# 3. numbers to words
# 4. delete special tokens

# In[6]:


def preprocessing(workdf):
    # lower, remove special characters, remove stopwords
    workdf['probody'] = workdf['body'].apply(lambda x: ' '.join([x.lower() for x in x.split() if x.isalnum()]))
    workdf['probody'] = workdf['probody'].apply(lambda x: ' '.join([x for x in x.split() if (x not in stopwordList)]))
    newbody = []
    # num2words
    for sentence in workdf['probody']:
        # string to list
        inputtext = sentence.split()
        numlist = []
        for i in range(len(inputtext)):
            if inputtext[i].isnumeric():
                numlist.append(i)
        for number in numlist:
            inputtext[number] = num2words(inputtext[number])
        
        # list to string
        celltext = ' '.join(inputtext)
        # tokenize
        celltext = word_tokenize(celltext)
        newbody.append(celltext)   
    workdf['tokens'] = newbody
    return workdf

preprocesseddf = preprocessing(featuredf)
print(preprocesseddf.iloc[2]['tokens'])
preprocesseddf.head()
preprocesseddf.info()


# In[7]:


# Porter Stemmer
def stemming(df):
    ps = PorterStemmer()
    df['tokens'] = df['tokens'].apply(lambda x:([ps.stem(word) for word in x]))
    return df

stemmeddf = stemming(preprocesseddf)
print(stemmeddf.iloc[1]['tokens'])
stemmeddf.head()


# In[8]:


# def apply_empath(df):

#     empath = Empath()
# #     df['empath'] = df['tokens'].apply(lambda x:([empath.analyze(sentence, normalize=True) for sentence in x]))
#     empathlist = df['tokens'].apply(lambda x:([empath.analyze(sentence, normalize=True) for sentence in x]))
#     empathdf = pd.DataFrame(empathlist)  
#     print(empathlist)
# #     for row in df:
# #         empathdict = empath.analyze(row, normalize=True)
# #     empathdf = pd.DataFrame.from_dict(data)
#     return df

# empathdf = apply_empath(stemmeddf)
# empathdf.head()



def apply_empath(df):
    empath = Empath()
    empathvalues = []
    for sentence in df['body']:
        empathvalues.append(empath.analyze(sentence, normalize=True))
    empathdf = pd.DataFrame(empathvalues)
    empathdf['author'] = df['author']

    newdf = pd.merge(df, empathdf, on='author', how='outer')
    return newdf

empdf = apply_empath(stemmeddf)
print(empdf.isnull().any().any())
empdf.head()



# empath = Empath()
# empathvalues = []
# for sentence in stemmeddf['tokens']:
#     empathvalues.append(empath.analyze(sentence, normalize=True))
# print(type(empathvalues))
# print(type(empathvalues[10]))
# empathdf = pd.DataFrame(empathvalues)
# empathdf.head()
# # newdf = stemmeddf.append(empathdf)
# # newdf.head()


# In[ ]:


def ngrams(df, n_min, n_max, ngramtype):
    # convert input from list to string
    ngrams = []
    inputtext = []
    for sentence in df['tokens']:
        text = ' '.join(sentence)
        inputtext.append(text)
    vectorizer = TfidfVectorizer(ngram_range=(n_min,n_max), analyzer=ngramtype) 
    vectors = vectorizer.fit_transform(inputtext)
    dense = vectors.todense()
    denselist = dense.tolist()
    names = vectorizer.get_feature_names()
    ngramdf = pd.DataFrame(denselist, columns=names)
    ngramdf['author'] = df['author']
    newdf = pd.merge(df, ngramdf, on='author', how='outer')
#     ngramdict = ngramdf.to_dict('index')
#     dict_items = list(ngramdict.items())    
    return newdf

# stemmeddf['wordngrams'] = ngrams(stemmeddf, 1, 3, 'word')
# stemmeddf['charngrams'] = ngrams(stemmeddf, 2, 3, 'char')
# stemmeddf.head()

wordngramsdf = ngrams(empdf, 1, 3, 'word')
print(wordngramsdf.isnull().any().any())
cwngramsdf = ngrams(wordngramsdf, 2, 3, 'char')
print(cwngramsdf.isnull().any().any())
# cwngramsdf.head()


# ## Empath
# 
# as a replacement for LIWC

# ## Word Lists

# Still needed are lists that comprise pronouns and stuff like that (see LIWC vs Empath). These lists can be created via empath

# ## Export dataframe

# In[ ]:


cwngramsdf.to_csv('features.csv', index=False)


# In[ ]:



