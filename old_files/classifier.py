#!/usr/bin/env python
# coding: utf-8

# # Classifier
# ## Import packages
import pandas as pd
from tqdm.notebook import tqdm
import random
random.seed(32)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, roc_auc_score, plot_roc_curve, confusion_matrix, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Import data
df = pd.read_pickle("wordlists_lin_feat_df_withoutuserfeat_allcomments.pkl")
df.head()
df.info()

# multilevel columns
lst1 = (5)*["data"]
lst9 = (15-5)*["traits"]
lst10 = ["data"]
lst2 = (21-17)*["global"]
lst3 = (45-21)*["time"]
lst4 = (16103-45)*["subreddits"]
lst5 = (16116-16103)*["extra_features"]
lst6 = (96308-16116)*["word_ngrams"]
lst7 = (103829-96308)*["char_ngrams"]
lst8 = (103889-103829)*["wordlists"]
headers = lst1 + lst9  + lst10 + lst2 + lst3 + lst4 +lst5 + lst6 + lst7 + lst8 
columns = df.columns.values
print(len(headers))
print(len(columns))
arrays = [headers] + [columns]
df.columns=pd.MultiIndex.from_arrays(arrays)

df['traits', 'agree5'] = df['traits', 'agreeableness'].apply(lambda x: 0 if x<20 else(1 if x>19 and x<40 else(2 if x>39 and x<60 else(3 if x>59 and x<80 else 4))))
df['traits', 'openn5'] = df['traits', 'openness'].apply(lambda x: 0 if x<20 else(1 if x>19 and x<40 else(2 if x>39 and x<60 else(3 if x>59 and x<80 else 4))))
df['traits', 'consc5'] = df['traits', 'conscientiousness'].apply(lambda x: 0 if x<20 else(1 if x>19 and x<40 else(2 if x>39 and x<60 else(3 if x>59 and x<80 else 4))))
df['traits', 'extra5'] = df['traits', 'extraversion'].apply(lambda x: 0 if x<20 else(1 if x>19 and x<40 else(2 if x>39 and x<60 else(3 if x>59 and x<80 else 4))))
df['traits', 'neuro5'] = df['traits', 'neuroticism'].apply(lambda x: 0 if x<20 else(1 if x>19 and x<40 else(2 if x>39 and x<60 else(3 if x>59 and x<80 else 4))))
df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')
smalldf = df[['traits', 'global', 'time', 'extra_features', 'word_ngrams', 'char_ngrams', 'wordlists']]
smalldf.info()

# histogram
n_bins = 20

def hist_true(df, trait):
    fig, ax = plt.subplots()
    plt.hist(df[trait], bins = 20)
    plt.title(trait, y=1.1)
    plt.xlabel("score")

def all_hist_true(df):
    plt.figure(figsize = (16, 16))
#     plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplot(3, 2, 1)
    plt.hist(df['traits', 'openness'], bins = 20)
    plt.title('Agreeableness')
    
    plt.subplot(3, 2, 2)
    plt.hist(df['traits', 'conscientiousness'], bins = 20)
    plt.title('Openness')
    
    plt.subplot(3, 2, 3)
    plt.hist(df['traits', 'extraversion'], bins = 20)
    plt.title('Conscientiousness')
    
    plt.subplot(3, 2, 4)
    plt.hist(df['traits', 'agreeableness'], bins = 20)
    plt.title('Extraversion')
    
    plt.subplot(3, 2, 5)
    plt.hist(df['traits', 'neuroticism'], bins = 20)
    plt.title('Neuroticism')
    
    plt.suptitle("Histograms of the true trait values")
    plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4) 
    plt.show()


# ## Trait

#split dataset in features and target variable depending on which trait to focus on
def trait(df, classes, trait_name):
    featuredf = df.drop(['data', 'traits'], axis=1, level=0)
    feature_cols = featuredf.columns.tolist()
    
    x = df[feature_cols] 
    
    if classes=='binary':
    
        if trait_name == 'agreeableness':
            y = df['traits', 'agree']
        elif trait_name == 'openness':
            y = df['traits', 'openn']
        elif trait_name == 'conscientiousness':
            y = df['traits', 'consc']
        elif trait_name == 'extraversion':
            y = df['traits', 'extra']
        elif trait_name == 'neuroticism':
            y = df['traits', 'neuro']   
    elif classes=='multi':
        if trait_name == 'agreeableness':
            y = df['traits', 'agree5']
        elif trait_name == 'openness':
            y = df['traits', 'openn5']
        elif trait_name == 'conscientiousness':
            y = df['traits', 'consc5']
        elif trait_name == 'extraversion':
            y = df['traits', 'extra5']
        elif trait_name == 'neuroticism':
            y = df['traits', 'neuro5'] 
    elif classes=='linear':
        if trait_name == 'agreeableness':
            y = df['traits', 'agreeableness']
        elif trait_name == 'openness':
            y = df['traits', 'openness']
        elif trait_name == 'conscientiousness':
            y = df['traits', 'conscientiousness']
        elif trait_name == 'extraversion':
            y = df['traits', 'extraversion']
        elif trait_name == 'neuroticism':
            y = df['traits', 'neuroticism']  
    return x,y 


# ## Classifier
def create_pipeline_cv(X_train, y_train, classifier, num_feat):
    if classifier == "log":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', LogisticRegression(n_jobs=-1))
        ])
    elif classifier == "multilog":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', LogisticRegression(multi_class='multinomial', n_jobs=-1))
        ])
    elif classifier == "mlp":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', MLPClassifier())
        ])
    elif classifier == "svm":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', svm.SVC())
        ])
    elif classifier == "svmlinear":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', svm.LinearSVC())
        ])
    elif classifier == "knn":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', KNeighborsClassifier(n_neighbors=1, n_jobs=-1))
        ])
    elif classifier == "linear":
        pipeline = Pipeline([
          ('variance_threshold', VarianceThreshold()),
          ('scaler', StandardScaler()),
          ('feature_selection',  SelectKBest(f_classif, k=num_feat)),
          ('classification', LinearRegression(n_jobs=-1))
        ])
    return pipeline


def get_names(x, pipeline):
    features = pipeline.named_steps['feature_selection']
    names = x.columns[features.get_support(indices=True)]
    return names

def get_pvalues(pipeline, x):
    features = pipeline.named_steps['feature_selection']
    pvalues = features.pvalues_
#     pvalues /= pvalues.max()
    dfpvalues = pd.DataFrame(features.pvalues_)
    dfscores = pd.DataFrame(features.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores, dfpvalues],axis=1)
    featureScores.columns = ['specs','score', 'pvalue']
    featureScores.sort_values(by='pvalue')
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(pvalues, bins=20)
    plt.title('All p-values')
    plt.subplot(1, 2, 2)
    smallpvalues = pvalues[pvalues<0.1]
    plt.hist(smallpvalues, bins=10)
    plt.title('Small p-values')
    
    plt.suptitle("Histograms of the p-values")
    plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4) 
    plt.show()
    return featureScores

def scores(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    return report

def score_plot(logreg, y_test, x_test):
    lr_probs = logreg.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
#     yhat = logreg.predict(x_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
#     lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig, ax = plt.subplots()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Classifier')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.show()
    return lr_precision, lr_recall

def create_cnfmatrix(clf, x_test, y_test, y_pred, plotting, detailed):
    cnfpipe_matrix = confusion_matrix(y_test, y_pred)
#     print(cnfpipe_matrix)
#     disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnfpipe_matrixcmap=plt.cm.Blues, normalize=normalize)
#     disp.plot() 
    if detailed:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
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
        plot_confusion_matrix(clf, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

def roc_aucscore(clf, x_test, y_test, classes, plotting, detailed):
    if detailed:
        print(roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovo'))
    
    if plotting and classes == 'binary':
        plot_roc_curve(clf, x_test, y_test)
        plt.title('ROC Curve', y=1.1)
        plt.show()

traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

def switching(trait):
    switcher={
            'openness':'all',
            'conscientiousness':'all',
            'agreeableness':'all',
            'extraversion':'all',
            'neuroticism':'all'
         }
    return switcher.get(trait,"Invalid")


# ## Wrapper with nested stratified cross validation

def get_params(classifier):
    if classifier == 'log':
        params = {'class_weight': [None, 'balanced'], 
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
                  'max_iter': [100, 200, 500, 1000]}
    if classifier == 'multilog':
        params = {'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced'], 
                  'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
                  'max_iter': [100, 200, 500, 1000]}
    elif classifier == 'mlp':
        params = {'hidden_layer_sizes': [(3,), (5,)]}
    elif classifier == 'svm':
        params = {'gamma': ['scale', 'auto'], 'class_weight': [None, 'balanced'],
                  'max_iter': [100, 200, 500, 1000]}
    return params


def classify_cv(df, classifier, classes):   
    for trait_name in traits:
        num_feat = switching(trait_name)
        print("Trait to predict: ", trait_name)
        x,y = trait(df, classes, trait_name)
        print(x.info())
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
        cv_outer = StratifiedKFold(n_splits=5)

        for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):
            train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
            train_target, val_target = y_train[train_idx], y_train[val_idx]

    #         model = LogisticRegression(random_state=7)
            clf = create_pipeline_cv(X_train, y_train, classifier, num_feat)
            cv_inner = StratifiedKFold(n_splits=5)
            params = get_params(classifier)
            gd_search = GridSearchCV(clf, params, scoring='f1_macro', n_jobs=-1, cv=cv_inner)
            gd_search = gd_search.fit(train_data, train_target)
            best_model = gd_search.best_estimator_

            classifier = best_model.fit(train_data, train_target)
            y_pred_prob = classifier.predict_proba(val_data)[:,1]
            f1_macro = f1_score(val_target, y_pred_prob, average='macro')

            print("Val Acc:",f1_macro , "Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)


#       # Training final model

#     model = LogisticRegression(random_state=7, C=0.001, class_weight='balanced', penalty='l2').fit(X_train, y_train)
#     y_pred_prob = model.predict_proba(X_test)[:,1]
#     print("AUC", metrics.roc_auc_score(y_test, y_pred_prob))
#     print(metrics.confusion_matrix(y_test, y_pred))

classify_cv(df, 'mlp', 'binary')


# ## Histogram of true traits
def check_imbalance(df, traits):
    length = len(df)
    o = df['traits', 'openn']
    c = df['traits', 'consc']
    e = df['traits', 'extra']
    a = df['traits', 'agree']
    n = df['traits', 'neuro']
    binarylst = [o, c, e, a, n]
    o5 = df['traits', 'openn5']
    c5 = df['traits', 'consc5']
    e5 = df['traits', 'extra5']
    a5 = df['traits', 'agree5']
    n5 = df['traits', 'neuro5']
    multilst = [o5, c5, e5, a5, n5]
    
    result = []
    for trait in binarylst: 
        result.append(np.bincount(trait) / length)
    result5 = []
    for trait in multilst:
        result5.append(np.bincount(trait) / len(trait))
    
    print("Distribution of the true trait values in the classes (in %):\n")
    for i in range(len(traits)):
        print(traits[i], "\n\tBinary: ", result[i], "\n\t5 classes: ", result5[i], "\n")
    
#     result =np.bincount(o) / len(o)
#     result5 =np.bincount(o5) / len(o)
#     print("Openness\n\tBinary: ", result, "\n\t5 classes: ", result5)

    
check_imbalance(df, traits)
all_hist_true(df)