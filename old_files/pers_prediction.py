#!/usr/bin/env python
# coding: utf-8

# # All functions for personality prediction

# ## Prep

# In[1]:


# Import packages

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import bigrams, ngrams

import re
import string
from string import punctuation

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

import gensim
from gensim import corpora, models

from empath import Empath

from collections import Counter
from num2words import num2words
from lexicalrichness import LexicalRichness
import textblob


import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
tqdm.pandas()
import datetime
import random
random.seed(32)


# ## Prepare dataset

# In[2]:


# Import dataset with comments
df = pd.read_csv('/home/sophia/ma_py/pandora_bigfive.csv')

# Import dataset authors and delete not needed columns
authors = pd.read_csv('/home/sophia/ma_py/author_profiles.csv')
bigfive = authors[['author','agreeableness','openness','conscientiousness','extraversion','neuroticism']]
bigfive = bigfive.dropna()


# In[3]:


# Functions

# change language to numeric representation
def numeric_lang(df):
    # change lang to numerical representation
    language = df['lang'].values.tolist()
    language = set(language)
    language
    df['language']= np.select([df.lang == 'en', df.lang == 'es', df.lang == 'nl'], 
                            [0, 1, 2], 
                            default=3)
    # print(gramsdf['language'])
    df = df.drop(columns=['lang'])

    return df

# create time columns from UTC
def create_timecolumns(df):
    readable = []
    weekday = []
    month = []
    year = []
    for row in df['created_utc']:
        item = datetime.datetime.fromtimestamp(row)
        weekday_item = item.strftime('%A')
        readable_item = datetime.datetime.fromtimestamp(row).isoformat()
        month.append(str(readable_item[5:7]))
        year.append(str(readable_item[0:4]))
        readable.append(readable_item)
        weekday.append(weekday_item.lower())
    df['time'] = readable
    df['weekday'] = weekday
    df['month'] = month
    df['year'] = year
    return df

# coun occurences in time columns to get time distribution
def timecounter(lst, vocablst):
    if vocablst == 'weekday':
        vocab = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    elif vocablst == 'month':
        vocab = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    elif vocablst == 'year':
        vocab = ['2015', '2016', '2017', '2018', '2019']
    else:
        print("No valid input: vocab list")
    vectorizer = CountVectorizer(analyzer="word", vocabulary=vocab)
    vectors = vectorizer.fit_transform(lst)
    v = vectors.toarray()
    return v

# create a list of all subreddits in the dataset
lst = df['subreddit'].tolist()
lst = [item.lower() for item in lst]
subredditset = set(lst)
subredditlist = list(subredditset)

# count occurences of subreddits 
def subredditcounter(lst, subredditlst):
    vectorizer = CountVectorizer(analyzer="word", vocabulary=subredditlist)
    vectors = vectorizer.fit_transform(lst)
    v = vectors.toarray()
    return v

# aggregate dataset to get one row per author and create new columns for time and subreddit
def create_authordf(df): 
    df = numeric_lang(df)
    # body
    df['complete_body'] = df.groupby(['author'])['body'].transform(lambda x : ' '. join(str(x)))
    df['doc_body'] = df.groupby(['author'])['body'].transform(lambda x : '§'. join(str(x)))
    df['doc_body'] =  df['doc_body'].apply(lambda x: x.split("§"))
    
    # language
    df['lang'] = df['language'].apply(lambda x: str(x))
    df['all_lang'] = df.groupby(['author'])['lang'].transform(lambda x : len(set(x)))
    # created_utc
    df['utc_lst'] = df['created_utc'].apply(lambda x: str(x))
    df['all_utc'] = df.groupby(['author'])['utc_lst'].transform(lambda x : ' '. join(x))
    df['all_utc'] = df['all_utc'].apply(lambda x: x.split())
    # controversiality
    df['mean_controversiality'] = df.groupby(['author']).agg({'controversiality': ['mean']})
    df['mean_controversiality'] = df['mean_controversiality'].fillna(0)
    # gilded
    df['mean_gilded'] = df.groupby(['author']).agg({'gilded': ['mean']})
    df['mean_gilded'] = df['mean_gilded'].fillna(0)
    # number of subreddits
    df['num_subreddits'] = df.groupby(['author'])['subreddit'].transform(lambda x : ' '. join(x))
    df['num_subreddits'] = df['num_subreddits'].apply(lambda x: len(set(x.split())))
    # number of comments per subreddit
    df['subreddit'] = df['subreddit'].apply(lambda x: [x.lower()])
    df['subreddit'] = df['subreddit'].apply(lambda x: ''.join(x))
    df['subreddit_dist'] = df.groupby(['author'])['subreddit'].transform(lambda x : ' '. join(x))
    subreddit_predist = subredditcounter(df['subreddit'], subredditlist)
    subreddit_predist = subreddit_predist.tolist()
    df['subreddit_dist'] = subreddit_predist
    # time
    df = create_timecolumns(df)
    df['weekday_dist'] = df.groupby(['author'])['weekday'].transform(lambda x : ' '. join(x))
    weekday = timecounter(df['weekday_dist'], 'weekday')
    weekday = weekday.tolist()
    df['weekday_dist'] = weekday
    df['month_dist'] = df.groupby(['author'])['month'].transform(lambda x : ' '. join(x))
    month = timecounter(df['month_dist'], 'month')
    month = month.tolist()
    df['month_dist'] = month
    df['year_dist'] = df.groupby(['author'])['year'].transform(lambda x : ' '. join(x))
    year = timecounter(df['year_dist'], 'year')
    year = year.tolist()
    df['year_dist'] = year
    
    newdf = df[['author', 'complete_body', 'doc_body', 'all_utc', 'mean_controversiality', 
                'mean_gilded', 'num_subreddits', 'subreddit_dist', 'weekday_dist', 
                'month_dist', 'year_dist', 'all_lang']]
    newdf = newdf.sort_values(by='author')
    newdf = newdf.drop_duplicates(subset=['author'])
    return newdf

# get one column for each feature in the distributions of time and subreddit
weekday = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
month = ['january', 'february', 'march', 'april', 'may', 'june', 'juli', 'august', 'september', 'october', 'november', 'december']
year = ['2015', '2016', '2017', '2018', '2019']

def onecolumnperdatapoint(df, column, namelist):
    for i in range(len(namelist)):
#         df[namelist[i]] = df[column].apply(lambda x:[row[i] for row in x])
        df[namelist[i]] = df[column].apply(lambda x:[x[i]])
        df[namelist[i]] = [item[0] for item in df[namelist[i]]]
    return df


# In[4]:


# Wrapper for commentdf
def create_commentdf(df):
    pandora = create_authordf(df)
    pandora = onecolumnperdatapoint(pandora, 'weekday_dist', weekday)
    pandora = onecolumnperdatapoint(pandora, 'month_dist', month)
    pandora = onecolumnperdatapoint(pandora, 'year_dist', year)
    pandora = onecolumnperdatapoint(pandora, 'subreddit_dist', subredditlist)
    pandora.drop(['weekday_dist', 'month_dist', 'year_dist', 'subreddit_dist'], axis=1, inplace=True)
    return pandora


# In[ ]:


# create commentdf
print("Create comment df (name: pandora)...")
pandora = create_commentdf(df)

# merge commentdf and authordf
print("Merge commentdf and authordf....")
pandoradf = pandora.merge(bigfive, how='left', on=['author'])
pandoradf = pandoradf.sort_values(by='author')
pandoradf = pandoradf[pandoradf['agreeableness'].notna()]
pandoradf = pandoradf.reset_index()

# create binary representation of personality traits
def bigfive_cat(df):
    # change big five to binary representation
    df['agree'] = df['agreeableness'].apply(lambda x: 0 if x<50 else 1)
    df['openn'] = df['openness'].apply(lambda x: 0 if x<50 else 1)
    df['consc'] = df['conscientiousness'].apply(lambda x: 0 if x<50 else 1)
    df['extra'] = df['extraversion'].apply(lambda x: 0 if x<50 else 1)
    df['neuro'] = df['neuroticism'].apply(lambda x: 0 if x<50 else 1)
    return df

print("Create binary representations for each personality trait")
pandoradf = bigfive_cat(pandoradf)


# ## Preprocessing

# In[ ]:


# Functions

# define stopwordlist to use
def choose_stopwordlist(df, mode):
    if mode == 'NLTK':
        stopwordList = stopwords.words('english')
    if mode == 'NLTK-neg':
        stopwordList = stopwords.words('english')
        stopwordList.remove('no')
        stopwordList.remove('nor')
        stopwordList.remove('not')
    return stopwordList

# remove decontractions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# create sentence tokens
def senttokenize(df):
    sentbody = []
    for row in df['doc_body']:
        sentitem = []
        for item in row:
            sentences = sent_tokenize(item)
            sentitem.append(sentences)
        sentbody.append(sentitem)
    df['senttokens'] = sentbody
    return df

# lower words and remove special characters
def lower_special(df):
    newrow = []
    for row in tqdm(df['probody']):
        newcomment = []
        for comment in row:
            text_pre = ""
            for character in comment:
                if character.isalnum() or character.isspace():
                    character = character.lower()
                    text_pre += character
                else:
                    text_pre += " "
            newcomment.append(text_pre)
        newrow.append(newcomment)   
    df['probody'] = newrow
    return df

# remove stopwords
def remove_stopwords(df, stopwordList):
    newprobody = []
    for row in tqdm(df['probody']):
        newrowprobody = []
        for comment in row:
            words = [word for word in comment.split() if (word not in stopwordList)]
            newcomment = ' '.join(words)
            newrowprobody.append(newcomment)
        newprobody.append(newrowprobody)
    df['probody'] = newprobody
    return df

# change numbers to words and tokenize words
def num_tokenize(df):    
    newbody_complete = []
    newprobody_complete = []
    # num2words
    for row in tqdm(df['probody']):
        newbody = []
        newprobody = []
        for sentence in row:
            # string to list
            inputtext = sentence.split()
            numlist = []
            for i in range(len(inputtext)):
                if inputtext[i].isnumeric():
                    numlist.append(i)
            for number in numlist:
                inputtext[number] = num2words(inputtext[number])

            # list to string
            inputtext = [word for word in inputtext if word.isalpha()]
            celltext = ' '.join(inputtext)
            newprobody.append(celltext)
            # tokenize
            words = word_tokenize(celltext)
            newbody.append(words)
        newbody_complete.append(newbody)
        newprobody_complete.append(newprobody)
    df['probody'] = newprobody_complete
    df['tokens'] = newbody_complete
    return df

# Porter Stemmer
def stemming(df):
    ps = PorterStemmer()
    for row in tqdm(df['tokens']):
        for comment in row:
            words = [ps.stem(word) for word in comment]
            comment = ' '.join(words)
    return df

# bring columns of dataframe in correct order
def ordering(df):
    cols_tomove = ['index', 'author', 'complete_body', 'doc_body', 'probody', 'tokens', 'senttokens', 'agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism', 'agree', 'openn', 'consc', 'extra', 'neuro']
    orderdf  = df[cols_tomove + [col for col in df.columns if col not in cols_tomove]]
    return orderdf


# In[ ]:


# Wrapper

def preprocess(df):
    # adjust some column representations
    df = bigfive_cat(df)
    # choose stopwordlist with or without negation
    stopwordList = choose_stopwordlist(df, mode='NLTK-neg')
    # decontract abbreviations (e.g., n't to not)
    print("Decontract...")
    df['probody'] = df['doc_body'].apply(lambda x:([decontracted(x) for x in x]))
    # create sentence tokens
    print("Tokenize Sentences...")
    df = senttokenize(df)
    # lower, remove stopwords, num2words, tokenize
    print("Lower words and remove special characters...")
    df = lower_special(df)
    print("Remove stopwords...")
    df = remove_stopwords(df, stopwordList)
    print("Change numbers to words and tokenize words...")
    df = num_tokenize(df)
    # porters stemmer
    print("Porters Stemmer...")
    df = stemming(df)
    print("Order df...")
    df = ordering(df)
    print("Done!")
    return df

# apply preprocessing
predf = preprocess(pandoradf)

predf


# ## Extract features

# In[ ]:


# User features

# Preprocessing for LDA
def preprocess_lda(df):
    neglst = ["no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never", "nay"]
    inputlst = []
    for row in tqdm(df['tokens']):
        rowlst = []
        for comment in row:
            rowlst.append([word for word in comment if (word not in neglst)])
        inputlst.append(rowlst)
    return inputlst
# LDA for topics
def apply_lda(df, inputlst, number, name):
    print("Start LDA...")
    lst = []
    for row in tqdm(inputlst):
        if len(row) < 2:
            lst.append(-1)
        else:
            dictionary = corpora.Dictionary(row)
            corpus = [dictionary.doc2bow(text) for text in row]
            ldamodel = gensim.models.LdaMulticore(corpus, num_topics=number, id2word = dictionary, passes=20, workers=15)
            result = ldamodel.print_topics(num_topics=1, num_words=1)
            res = list(result)
            topic = [item[0] for item in res]
            lst.append(topic[0])
    df[name] = lst
    return df

# Wrapper
def extract_userfeatures(df):
    print("Preprocessing for LDA...")
    inputlst = preprocess_lda(df)
    print("LDA with fifty topics: ")
    df = apply_lda(df, inputlst, 50, "ldafifty")
    print("LDA with onehundred topics: ")
    df = apply_lda(df, inputlst, 100, "ldahundred")
    return df

# create df with user features
user_feat_df = extract_userfeatures(predf)


# In[ ]:


# Linguistic features (functions)

# other features that are not mentioned in the paper
def create_features(df):
    # Total number of characters (including space)
    df['char_count'] = df['complete_body'].str.len()
    # Total number of stopwords
    stopwordList = stopwords.words('english')
    df['stopwords'] = df['complete_body'].apply(lambda x: len([x for x in x.split() if x in stopwordList]))
    # Total number of punctuation or special characters
    df['total_punc'] = df['complete_body'].apply(lambda x: len([x for x in x.split() for j in x if j in string.punctuation]))
    # Total number of numerics
    df['total_num'] = df['complete_body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    # Total number of uppercase words
    df['total_uppercase'] = df['complete_body'].apply(lambda x: len([x for x in x.split() if x.isupper()]))    
    return df

# type token ratio
def typetokenratio(df):
    ratiolst = []
    for comment in df['complete_body']:
            lex = LexicalRichness(comment)
            if lex.words == 0:
                ratiolst.append(0)
            else:
                ratio = lex.ttr
                ratiolst.append(ratio)
    df['ttr'] = ratiolst
    return df

# words per sentence
def wordcounter(df):
    lengthscore = []
    for row in df['senttokens']:
        rowscore = []
        for comment in row:
            sentencescore = 0
            for senttoken in comment:
                length = len(senttoken.split())
                sentencescore += length
            sentencescore = sentencescore/len(comment)
        lengthscore.append(sentencescore)
        arr = np.array(lengthscore)
    df['words_per_sent'] = lengthscore
    return df

# words longer than six characters
def charcounter(df):
    charscore = []
    for row in df['tokens']:
        for comment in row:
            rowcharscore = 0
            lencomment = len(comment)
            if lencomment == 0:
                score = 0
            else:
                number = 0
                for token in comment:
                    length = len(token)
                    if length > 5:
                        number+=1
                score = number/lencomment
            rowcharscore += score
        rowcharscore = rowcharscore/len(row)
        charscore.append(rowcharscore)
    df['wordslongersix'] = charscore
    return df

# POS tagger
def tagging(df):
    past = [] #VPA
    presence = [] #VPR
    adverbs = [] #RB
    prepositions = [] #PREP
    pronouns = [] #PR
    for comment in df['complete_body']:
            text = comment.split()
            tags = nltk.pos_tag(text)
            counts = Counter(tag for word,tag in tags)
            total = sum(counts.values())
            pron = counts['PRP'] + counts['PRP$']
            verbspr = counts['VB'] + counts['VBG'] + counts['VBP'] + counts['VBZ'] + counts['MD']
            verbspa = counts['VBD'] + counts['VBN']
            preps = counts['IN'] + counts['TO']
            counts['PR'] = pron
            counts['PREP'] = preps
            counts['VPR'] = verbspr #present tense
            counts['VPA'] = verbspa #past tense
            if total == 0:
                allcounts = dict((word, float(count)/1) for word,count in counts.items())
            else:
                allcounts = dict((word, float(count)/total) for word,count in counts.items())
            try:
                past.append(allcounts['VPA'])
            except KeyError:
                past.append(0)
            try:
                presence.append(allcounts['VPR'])
            except KeyError:
                presence.append(0)
            try:
                adverbs.append(allcounts['RB'])
            except KeyError:
                adverbs.append(0)
            try:
                prepositions.append(allcounts['PREP'])
            except KeyError:
                prepositions.append(0)
            try:
                pronouns.append(allcounts['PR'])
            except KeyError:
                pronouns.append(0)
    df['pasttense'] = past
    df['presencetense'] = presence
    df['adverbs'] = adverbs
    df['prepositions'] = prepositions
    df['pronouns'] = pronouns
    return df

def ngrams(df, n_min, n_max, ngramtype):
    # convert input from list to string
    ngrams = []
    inputtext = []
    for row in df['tokens']:
        for comment in row:
            text = ' '.join(comment)
        inputtext.append(text)
    print("Length of inputtext: ", len(inputtext))
    vectorizer = TfidfVectorizer(ngram_range=(n_min,n_max), analyzer=ngramtype)
    print("Vectorize...")
    vectors = vectorizer.fit_transform(tqdm(inputtext))
    dense = vectors.todense()
    denselist = dense.tolist()
    print("Get feature names...")
    names = vectorizer.get_feature_names()
    print("Length of feature names: ", len(names))
    print("Create df...")
    ngramdf = pd.DataFrame(denselist, columns=names)
    ngramdf['author'] = df['author']
    return ngramdf

def merge_dfs(df1, df2, df3):
    cwngramsdf = pd.merge(df1, df2, on='author', how='inner', suffixes= (None, "_charngram"))
    gramsdf = pd.merge(df3, cwngramsdf, on='author', how='inner', suffixes= (None, "_ngram"))
    return gramsdf


# In[ ]:


# Wrapper for linguistic features

def extract_lin_features(df, create_ngrams):
    print("Create additional features...")
    df = create_features(df)
    print("Create ttr...")
    df = typetokenratio(df)
    print("Count words per sentence...")
    df = wordcounter(df)
    print("Count words with more than six letters...")
    df = charcounter(df)
    print("POS-Tagger...")
    df = tagging(df)
    print("number of rows df", len(df))
    
    if create_ngrams == "none":
        return df
    
    elif create_ngrams == "all":
        print("Ngrams...")
        print("Create word ngrams...")
        wordngramsdf = ngrams(df, 1, 3, "word")
        print("Create char ngrams...")
        charngramsdf = ngrams(df, 2, 3, "char")
        print("Merge df...")
        gramsdf = merge_dfs(wordngramsdf, charngramsdf, df)
        return gramsdf
    
    elif create_ngrams == "word":
        wordngrams = ngrams(df, 1, 3, 'word')
        wordngramsdf = pd.DataFrame(wordngrams)
        gramsdf = pd.merge(df, wordngramsdf, on='author', how='inner', suffixes=(None, "_ngram"))
        return gramsdf
    
# create dataframe with linguistic features

# without ngrams
# lin_feat_df = extract_lin_features(user_feat_df, "none")

# with all ngrams
lin_ngrams_df = extract_lin_features(user_feat_df, "all")

# wordngrams only
# lin_wordngrams_df = extract_lin_features(user_feat_df, "word")


# In[ ]:


# Wordlists (functions)

# Empath
# create new categories with empath
def new_cat():
    empath = Empath()
    social = empath.create_category("social",["mate","talk","they"])
    humans = empath.create_category("humans",["adult","baby","boy"])
    cognitive = empath.create_category("cognitive",["cause","know","ought"])
    insight = empath.create_category("insight",["think","know","consider"])
    causation = empath.create_category("causation",["because","effect","hence"])
    discrepancy = empath.create_category("discrepancy",["should","would","could"])
    tentative = empath.create_category("tentative",["maybe","perhaps","guess"])
    certainty = empath.create_category("certainty",["always","never", "proof"])
    inhibition = empath.create_category("inhibition",["block","constrain","stop"])
    inclusive = empath.create_category("inclusive",["and","with","include"])
    exclusive = empath.create_category("exclusive",["but","without","exclude"])
    perceptual = empath.create_category("perceptual",["observing","hear","feeling"])
    see = empath.create_category("see",["view","saw","seen"])
    feel = empath.create_category("feel",["feels","touch","feeling"])
    biological = empath.create_category("biological",["eat","blood","pain"])
    relativity = empath.create_category("relativity",["area","bend","go"])
    space = empath.create_category("space",["down","in","thin"])
    time = empath.create_category("time",["end","until","season"])
    agreement = empath.create_category("agreement", ["agree", "ok", "yes"])
    fillers = empath.create_category("fillers", ["like", "Imean", "yaknow"])
    nonfluencies = empath.create_category("nonfluencies", ["umm", "hm", "er"])
    conjunctions = empath.create_category("conjunctions", ["and", "but", "whereas"])
    quantifiers = empath.create_category("quantifiers", ["few", "many", "much"])
    numbers = empath.create_category("numbers", ["two", "fourteen", "thousand"])

def apply_empath(df):
    empath = Empath()
    print("Create new empath categories...")
    new_cat()
    print("Apply empath...")
    empathvalues = []
    empathcategories = ["swearing_terms", "social", "family", "friends", "humans", "emotional", "positive_emotion", "negative_emotion", "fear", "anger", "sadness", "cognitive", "insight", "causation", "discrepancy", "tentative", "certainty", "inhibition", "inclusive", "exclusive", "perceptual", "see", "hear", "feel", "biological", "body", "health", "sexual", "eat", "relativity", "space", "time", "work", "achievement", "leisure", "home", "money", "religion", "death" ,"agreement", "fillers", "nonfluencies"]
    for sentence in tqdm(df['complete_body']):
        empathvalues.append(empath.analyze(sentence, categories=empathcategories, normalize=True))
    empathdf = pd.DataFrame(empathvalues)
    empathdf['author'] = df['author']

    newdf = pd.merge(df, empathdf, on='author', how='inner', suffixes=(None, "_wordlist"))
    return newdf


# In[ ]:


# Import data for other wordlists
concretenessdf = pd.read_csv('/home/sophia/ma_py/psych_lists/concreteness.csv')
cdf = concretenessdf[['Conc.M']]
cmatrix = cdf.to_numpy()
concrete = concretenessdf['Word'].values.tolist()

happinessdf = pd.read_csv('/home/sophia/ma_py/psych_lists/happiness_ratings.csv')
hdf = happinessdf[['happiness_average']]
hmatrix = hdf.to_numpy()
happiness = happinessdf['word'].values.tolist()

cursedf = pd.read_csv('/home/sophia/ma_py/psych_lists/mean_good_curse.csv')
cudf = cursedf[['mean_good_curse']]
cumatrix = cudf.to_numpy()
curse = cursedf['word'].values.tolist()

sensorydf = pd.read_csv('/home/sophia/ma_py/psych_lists/sensory_experience_ratings.csv')
serdf = sensorydf[['Average SER']]
sermatrix = serdf.to_numpy()
ser = sensorydf['Word'].values.tolist()

alldf = pd.read_csv('/home/sophia/ma_py/psych_lists/sensory_ratings_all.csv')
newalldf = alldf[['Emotion', 'Polarity', 'Social', 'Moral', 'MotionSelf', 'Thought', 'Color', 'TasteSmell', 'Tactile', 'VisualForm', 'Auditory', 'Space', 'Quantity', 'Time', 'CNC', 'IMG', 'FAM']]
allmatrix = newalldf.to_numpy()
allsens = alldf['Word'].values.tolist()

valarodomdf = pd.read_csv('/home/sophia/ma_py/psych_lists/valence_arousal_dominence.csv')
vaddf = valarodomdf[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
vadmatrix = vaddf.to_numpy()
vad = valarodomdf['Word'].values.tolist()

mrcdf = pd.read_csv('/home/sophia/ma_py/psych_lists/mrclists_c_p.csv', sep='\t', names=['word', 'cmean', 'pmean'])
cpdf = mrcdf[['cmean', 'pmean']]
cpmatrix = cpdf.to_numpy()
mrc = mrcdf['word'].values.tolist()

# function for other wordlists

def counter(df, vocab):
    inputtext = []
    for row in df['complete_body']:
        text = ' '.join(row)
        inputtext.append(text)
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1,1), vocabulary = vocab)
    print("Vectorize...")
    vectors = vectorizer.fit_transform(tqdm(inputtext))
    v = vectors.toarray()
    return v

def multiply(matrix, ratings):
    # matrix multiplication 
    result = np.matmul(matrix, ratings)
    # divide each score with the number of words in the list to normalize
    result = result/(len(ratings))
    return result

def aggregator(df, vocab, ratings, name):
    count = counter(df, vocab)
    result = multiply(count, ratings)
    num_rows, num_cols = result.shape
    
    if num_cols ==1:
        df[name] = result
    else:
        resultdf = pd.DataFrame(result)
        for i in range(len(name)):
            # first i is zero
            column = name[i]
            df[column] = resultdf[i]
    return df


# In[ ]:


# wordlists created manually

negations = ["no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never", "nay"]
articles = ["a", "an", "the"]
future = ["will", "gonna"]

def list_counter(df, vocab, name):
    inputtext = []
    total = []
    for row in df['complete_body']:
        total.append(len(row))
        text = ' '.join(row)
        inputtext.append(text)
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1,1), vocabulary = vocab)
    print("Vectorize...")
    vectors = vectorizer.fit_transform(tqdm(inputtext))
    v = vectors.toarray()
    averagev = v.sum(axis=1)
    totalvector =  np.array(total)
    score = np.divide(averagev, totalvector)
    df[name] = score
    return df


# In[ ]:


# Wrapper for wordlists

def extract_wordlist_features(df):
    print("Empath...")
    empdf = apply_empath(df)
    # create scores for each word list and add them to df
    print("Count Wordlist Concreteness: \n")
    psychdf = aggregator(empdf, concrete, cmatrix, "concreteness")
    print("Count Wordlist Happiness: \n")
    psychdf = aggregator(empdf, happiness, hmatrix, "happiness")
    print("Count Wordlist Good_Curse: \n")
    psychdf = aggregator(empdf, curse, cumatrix, "good_curse")
    print("Count 17 further wordlists: \n")
    psychdf = aggregator(empdf, allsens, allmatrix, ['emotion', 'polarity', 'social', 'moral', 'motionself', 'thought', 'color', 'tastesmell', 'tactile', 'visualform', 'auditory', 'space', 'quantity', 'time', 'CNC', 'IMG', 'FAM'])
    print("Count Wordlist SER: \n")
    psychdf = aggregator(empdf, ser, sermatrix, "SER")
    print("Count Wordlists Valence, Arousal, Dominance: \n")
    psychdf = aggregator(empdf, vad, vadmatrix, ['valence', 'arousal', 'dominance'])
    print("Count Wordlist Negation: \n")
    psychdf = list_counter(empdf, negations, "negations")
    print("Count Wordlist Articles: \n")
    psychdf = list_counter(empdf, articles, "articles")
    print("Count Wordlist Future: \n")
    psychdf = list_counter(empdf, future, "future")
    print("Count Wordlists from MRC (2): \n")
    psychdf = aggregator(empdf, mrc, cpmatrix, ["mrc_cmean", "mrc_pmean"])
    
    return psychdf

psychdf = extract_wordlist_features(lin_ngrams_df)


# ## Classifier

# In[ ]:


# functions

# histogram of distribution of traits in dataset
def all_hist_true(df):
    plt.figure(figsize = (16, 8))
#     plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplot(2, 3, 1)
    plt.hist(df['agreeableness'], bins = 20)
    plt.title('Agreeableness')
    
    plt.subplot(2, 3, 2)
    plt.hist(df['openness'], bins = 20)
    plt.title('Openness')
    
    plt.subplot(2, 3, 3)
    plt.hist(df['conscientiousness'], bins = 20)
    plt.title('Conscientiousness')
    
    plt.subplot(2, 3, 4)
    plt.hist(df['extraversion'], bins = 20)
    plt.title('Extraversion')
    
    plt.subplot(2, 3, 5)
    plt.hist(df['neuroticism'], bins = 20)
    plt.title('Neuroticism')
    
    plt.suptitle("Histograms of the true trait values")
    plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4) 
    plt.show()

#split dataset in features and target variable depending on which trait to focus on
def trait(df, trait_name, startnumber):
    featurelist = df.columns.tolist()
    feature_cols = featurelist[startnumber:]
    x = df[feature_cols] 
    
    if trait_name == 'agree':
        y = df.agree
    elif trait_name == 'openn':
        y = df.openn
    elif trait_name == 'consc':
        y = df.consc
    elif trait_name == 'extra':
        y = df.extra
    elif trait_name == 'neuro':
        y = df.neuro       
    return x,y 

# create pipeline
def create_pipeline(x_train, y_train ,classifier):
    if classifier == "log":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('feature_selection',  SelectKBest(f_classif, k=30)),
          ('scaler', StandardScaler()),
          ('classification',LogisticRegression(n_jobs=-1))
        ])
        
    pipeline.fit(x_train, y_train)
    return pipeline

def get_names(x, pipeline):
    features = pipeline.named_steps['feature_selection']
    names = x.columns[features.get_support(indices=True)]
    return names

def get_pvalues(pipeline, x):
    features = pipeline.named_steps['feature_selection']
    pvalues = features.pvalues_
    dfpvalues = pd.DataFrame(features.pvalues_)
    dfscores = pd.DataFrame(features.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    # concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores, dfpvalues],axis=1)
    featureScores.columns = ['Specs','Score', 'P-Value']
    # plot
    fig, ax = plt.subplots()
    plt.hist(pvalues)
    plt.show()
    return featureScores

def scores(y_test, y_pred, presentationtype):
    if presentationtype == "scores":
        accuracy=metrics.accuracy_score(y_test, y_pred)
        precision=metrics.precision_score(y_test, y_pred)
        recall=metrics.recall_score(y_test, y_pred)
        f_one=metrics.f1_score(y_test, y_pred)
        return accuracy, precision, recall, f_one
    if presentationtype == "report":
        report = classification_report(y_test, y_pred)
        return report
    
def score_plot(logreg, y_test, x_test):
    lr_probs = logreg.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig, ax = plt.subplots()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.show()
    return lr_precision, lr_recall

def create_cnfmatrix(y_test, y_pred, plotting=True):
    cnfpipe_matrix = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    sumpositive = tp + fn
    sumnegative = fp + tn
    sumcorrect = tp + tn
    sumwrong = fp + fn
    sumall = tn+fp+fn+tp
    print("TN, FP, FN, TP: ", tn, fp, fn, tp, "\nSum: ", sumall, "\nSum correct predictions: ", 
          sumcorrect, "Percent: ", sumcorrect/sumall, "\nSum wrong predictions: ", sumwrong, "\tPercent: ",
          sumwrong/sumall, "\nSum actual positives: ", sumpositive, "\tPercent: ", sumpositive/sumall,
          "\nSum actual negatives: ", sumnegative, "\tPercent: ", sumnegative/sumall)
    
    if plotting:
        get_ipython().run_line_magic('matplotlib', 'inline')
        class_names=[0,1] # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnfpipe_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
        ax.xaxis.set_label_position("bottom")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()


# In[ ]:


# wrapper for classifier

def classify(df, trait_name, startnumber, plotting=True):
    print("Trait to predict: ", trait_name)
    x,y = trait(df, trait_name, startnumber)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    print("Number of authors in y_train: ", len(y_train))
    print("Number of authors in y_test: ", len(y_test))
    logpipe = create_pipeline(x_train, y_train, 'log')
    y_pred=logpipe.predict(x_test)
    print("Number of authors in y_pred: ", len(y_pred))
    names = get_names(x, logpipe)
    print("Names of the top", len(names), "features: \n", names, "\n")
    pvalues = get_pvalues(logpipe, x)
    print("\nP-Values: ")
    print(pvalues.nsmallest(30,'P-Value'))
    print("\n")
    cnfmatrix = create_cnfmatrix(y_test, y_pred, plotting=True) 
#     accuracy, precision, recall, f_one = scores(y_test, y_pred, "scores")
#     print("Scores:\nAccuracy:",accuracy, "\nPrecision:",precision, "\nRecall:",recall, "\nF1 score:",f_one)
    report = scores(y_test, y_pred, "report")
    print("Classification report: \n", report)
    lr_precision, lr_recall = score_plot(logpipe, y_test, x_test)
    print("\n \n \n")


# In[ ]:


psychdf.info(verbose=True)


# In[ ]:


start = 18
print ("Number of authors: ", len(psychdf))

# personality prediction on test set
all_hist_true(psychdf)
classify(psychdf, "agree", start, plotting=True)
classify(psychdf, "openn", start, plotting=True)
classify(psychdf, "consc", start, plotting=True)
classify(psychdf, "extra", start, plotting=True)
classify(psychdf, "neuro", start, plotting=True)


# In[ ]:


# Results for the train set

def classify_trainset(df, trait_name, startnumber, plotting=True):
    print("Trait to predict: ", trait_name)
    x,y = trait(df, trait_name, startnumber)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    logpipe = create_pipeline(x_train, y_train, 'log')
    y_pred=logpipe.predict(x_train)
    print("Number of authors in y_pred: ", len(y_pred))
    names = get_names(x, logpipe)
    print("Names of the top", len(names), "features: \n", names, "\n")
    pvalues = get_pvalues(logpipe, x)
#     print("p-values of", len(pvalues), "features: \n", pvalues, "\n")
    print("\nP-Values: ")
    print(pvalues.nsmallest(30,'P-Value'))
    print("\n")
    cnfmatrix = create_cnfmatrix(y_train, y_pred, plotting=True) 
#     accuracy, precision, recall, f_one = scores(y_test, y_pred, "scores")
#     print("Scores:\nAccuracy:",accuracy, "\nPrecision:",precision, "\nRecall:",recall, "\nF1 score:",f_one)
    report = scores(y_train, y_pred, "report")
    print("Classification report: \n", report)
    lr_precision, lr_recall = score_plot(logpipe, y_train, x_train)
#     print("Scores:\nLR_Precision:",lr_precision, "\nLR_Recall:",lr_recall)
    plt.show()
    print("\n \n")


# In[ ]:


classify_trainset(psychdf, "agree", start, plotting=True)
classify_trainset(psychdf, "openn", start, plotting=True)
classify_trainset(psychdf, "consc", start, plotting=True)
classify_trainset(psychdf, "extra", start, plotting=True)
classify_trainset(psychdf, "neuro", start, plotting=True)


# In[ ]:




