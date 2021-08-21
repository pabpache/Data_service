#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pablopacheco
znumber: z5222810
"""
import sys
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier

#Load data
try:
    path_train=sys.argv[1]
    path_test=sys.argv[2]
except:
    print('You must give path for training data and path for test data')
    sys.exit()

df=pd.read_csv(path_train)
df2=pd.read_csv(path_test)


#Drop outlier
#There is no negative revenues, but there are values close to zero and some small values. It will be assume
#that values smaller than 10,000 are wrong (could be another unit). So, those rows will be erased
#In addition, revenues above 2,000,000,000 will be considered outliers as Avatar (There have been just 5 films in
#the history that have surprass the 2B dollars in revenue)
df=df[(df['revenue']>10000) & (df['revenue']<2000000000)]

#rating should be 1,2 or 3
df=df[(df['rating']>0) & (df['rating']<4)]


"""
*******************************************************************************
Defining functions to extract information from json types and other utilities
*******************************************************************************
"""

#Receive the json 'cast' and make a list with the actors names or production company names or genres
def get_list_(x):
    list_=[]
    if x !='not_given': 
        x=json.loads(x)
        for i in range(len(x)):
            list_.append(x[i]['name'])
    return list_

#get the most n frequent actors
def get_list_top_n_actors(df,n):
    df['actors']=df['cast'].apply(get_list_)
    df_aux=df[['actors']].explode('actors')
    df_aux=df_aux.value_counts(ascending=False)
    return [df_aux[:n].index[i][0] for i in range(n)]

#get the most n frequent production companies
def get_list_top_n_prod(df,n):
    df['production']=df['production_companies'].apply(get_list_)
    df_aux=df[['production']].explode('production')
    df_aux=df_aux.value_counts(ascending=False)
    return [df_aux[:n].index[i][0] for i in range(n)]

#Receive a df and list of top (classes to encode) and the column name to encode
def personalised_encoding(x, top_list,col_name):
    for i in top_list:
        x[i]=x.apply(lambda x: 1 if i in x[col_name] else 0,axis=1)
        
#Receive a json 'crew'
def get_director_name(x):
    if x != 'not_given':
        x=json.loads(x)
        for i in range(len(x)):
            if x[i]['job']=='Director':
                return x[i]['name']
    else:
        return 'not_given'

def get_list_top_n_directors(df,n):
    df['director']=df['crew'].apply(get_director_name)
    return list(df['director'].value_counts(ascending=False)[:n].index)

#Function to separate components of release_date in three different columns
def expand_date(df):
    df[['year', 'month', 'day']] = df['release_date'].str.split('-', expand=True)
    for i in ['year', 'month', 'day']:
        df[i]=df[i].apply(lambda x: np.int64(x))
        
#Get the genres list that appear in the training data
def get_list_genres(df):
    df['genres_list']=df['genres'].apply(get_list_)
    df_aux=df[['genres_list']].explode('genres_list')
    return df_aux['genres_list'].unique()

#Dealing with potencial missing values in data
def missing_values(df):
    #dealing with missing values in release_date
    df['release_date'] = df['release_date'].fillna("0-0-0")
    #replace missing values in 'budget' by most_frequent value
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df['budget'] = imputer.fit_transform(df[['budget']])
    #Fill the NaN values for 'not_given' string
    for i in ['cast','crew','genres','production_companies']:
        df[i]=df[i].fillna('not_given')  
        

#Count how many actors/production companies (could be used for both) of a ranked group there are in a movie
def count_ranked(x,list_rank):
    counter=0
    for i in x:
        if i in list_rank:
            counter += 1
    return counter

#create columns that associated the quantity of ranked actors in every movie
def group_of_actors(df,list_act):
    df['tier1']=df.apply(lambda x: count_ranked(x['actors'],list_act),axis=1)


"""
*******************************************************************************
PRE-PROCESSING
*******************************************************************************
"""

#Pre-processing training data

#Deal with potential missing values
#If a row has more than 1 null values among these columns: ['cast','crew','budget','genres','production_companies','release_date'], the row is dropped
#otherwise the model is mislead
important_feautures=['cast','crew','budget','genres','production_companies','release_date']
to_drop=df[important_feautures].isnull().sum(axis=1).values>1
df.drop(df.index[to_drop],inplace=True)


missing_values(df)
#expand release_date
expand_date(df)

top_list_actors=get_list_top_n_actors(df,25)  #There are 33296 actors
top_list_dir=get_list_top_n_directors(df,60)     #There are 888
top_list_prod=get_list_top_n_prod(df,60)      #There are 2347 companies
list_genres=get_list_genres(df)

personalised_encoding(df,top_list_dir,'director')
personalised_encoding(df,top_list_prod,'production')
personalised_encoding(df,list_genres,'genres_list')
group_of_actors(df,top_list_actors)


list_columns=['budget','revenue','year','month','day','rating','runtime']
list_rank_actors=['tier1']
list_columns.extend(list_rank_actors)
list_columns.extend(top_list_dir)
list_columns.extend(top_list_prod)
list_columns.extend(list_genres)

df_train=df[list_columns]

#pre-processing test data

#Deal with potential missing values
missing_values(df2)

expand_date(df2)

df2['actors']=df2['cast'].apply(get_list_)
df2['director']=df2['crew'].apply(get_director_name)
df2['production']=df2['production_companies'].apply(get_list_)
df2['genres_list']=df2['genres'].apply(get_list_)

personalised_encoding(df2,top_list_dir,'director')
personalised_encoding(df2,top_list_prod,'production')
personalised_encoding(df2,list_genres,'genres_list')
group_of_actors(df2,top_list_actors)
list_columns_test=['movie_id']
list_columns_test.extend(list_columns)

df_test=df2[list_columns_test]


"""
*******************************************************************************
PART 1 - REGRESSION: Creating the model - Gradient Boosting Regressor
*******************************************************************************
"""

x_values=df_train.drop(['revenue','rating','runtime'],axis=1).values
y_values=df_train['revenue'].values

x_test_values=df_test.drop(['movie_id','revenue','rating','runtime'],axis=1).values
y_test_values=df_test['revenue'].values


#remove random_state
reg = GradientBoostingRegressor(random_state=0,n_estimators=100,learning_rate=0.1,min_samples_leaf=1,min_impurity_decrease=0.0)
model_part1=reg.fit(x_values, y_values)

y_pred = model_part1.predict(x_test_values)

df_part1_1=df_test[['movie_id']].copy()
df_part1_1['predicted_revenue']=y_pred.astype(np.int64).copy()
df_part1_1.to_csv('z5222810.PART1.output.csv',index=False)


# The mean squared error
MSE_part1='{:.2f}'.format(mean_squared_error(y_test_values, y_pred))

#Get the correlation between the prediction and the true value
y_pred_2=pd.Series(y_pred)
y_test_values_2=pd.Series(y_test_values)
correlation_part1='{:.2f}'.format(y_pred_2.corr(y_test_values_2))


df_part1_2=pd.DataFrame({'zid':['z5222810'],'MSE':[MSE_part1],'correlation':[correlation_part1]})
df_part1_2.to_csv('z5222810.PART1.summary.csv',index=False)


"""
*******************************************************************************
PART 2 - CLASSIFICATION: Creating the model - RandomForest
*******************************************************************************
"""

x_values_part2=df_train.drop(['revenue','rating'],axis=1).values
y_values_part2=df_train['rating'].values

x_test_values_part2=df_test.drop(['movie_id','revenue','rating'],axis=1).values
y_test_values_part2=df_test['rating'].values

rfc=RandomForestClassifier(n_estimators=100,class_weight='balanced_subsample',min_samples_leaf=0.01,max_features=0.9,random_state=0)
model_part2=rfc.fit(x_values_part2,y_values_part2)

y_pred_part2=model_part2.predict(x_test_values_part2)

avg_precision_part2='{:.2f}'.format(precision_score(y_test_values_part2, y_pred_part2,average='macro'))
avg_recall_part2= '{:.2f}'.format(recall_score(y_test_values_part2, y_pred_part2,average='macro'))
accuracy_part2= '{:.2f}'.format(accuracy_score(y_test_values_part2, y_pred_part2))

df_part2 = pd.DataFrame({'zid':['z5222810'],'average_precision':[avg_precision_part2],'average_recall':[avg_recall_part2], 'accuracy':[accuracy_part2]})
df_part2.to_csv('z5222810.PART2.summary.csv',index=False)

df_part2_2=df_test[['movie_id']].copy()
df_part2_2['predicted_rating']=y_pred_part2.astype(np.int64).copy()
df_part2_2.to_csv('z5222810.PART2.output.csv',index=False)


