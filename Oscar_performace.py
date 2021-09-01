#%%
import pandas as pd
import re
import logging
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns",200)
pd.set_option('display.max_colwidth', None)

#Reading Csv
oscar_df = pd.read_csv(r'C:\Users\mbvsuraj\Documents\Python_Scripts\Baby\oscars-demographics.csv',encoding='ISO 8859-1')
oscar_df.head()
#%%
oscar_df_sub = oscar_df[['birthplace', 'date_of_birth','race_ethnicity', 'year_of_award', 'award']]

oscar_df_sub.head(3)

oscar_df_sub['award'].unique()

for col in oscar_df_sub.columns:
    print(col)
#%%

#Date_of_birth cleaning

oscar_df_sub['ldob'] = oscar_df_sub['date_of_birth'].apply(len)

oscar_df_sub['ldob'].unique()

uncleaned_list = oscar_df_sub['ldob'].unique().tolist()

oscar_df_sub['date_of_birth'].unique()

oscar_df_sub['date_of_birth_mod'] = np.nan

for x in range(0,len(oscar_df_sub["date_of_birth"])):
    if oscar_df_sub['ldob'].iloc[x] == 9:
        oscar_df_sub['date_of_birth_mod'].iloc[x]= datetime.datetime.strptime(oscar_df_sub['date_of_birth'].iloc[x],'%d-%b-%y').strftime('%d-%b-%Y') 
    elif oscar_df_sub['ldob'].iloc[x] == 15:
        oscar_df_sub['date_of_birth_mod'].iloc[x]=oscar_df_sub['date_of_birth'].iloc[x][:11]
    elif oscar_df_sub['ldob'].iloc[x] == 4:
        oscar_df_sub['date_of_birth_mod'].iloc[x] = '01-Jan-'+oscar_df_sub['date_of_birth'].iloc[x]
    else:
        oscar_df_sub['date_of_birth_mod'].iloc[x]=oscar_df_sub['date_of_birth'].iloc[x]


#%%
oscar_df_sub[oscar_df_sub['birthplace'].isnull()==True]

##Adding country to existing birthplace column
for x in range(0,len(oscar_df_sub["birthplace"])):
    if len(oscar_df_sub['birthplace'].iloc[x])-oscar_df_sub['birthplace'].iloc[x].rfind(" ")  == 3:
        oscar_df_sub['birthplace'].iloc[x]=oscar_df_sub['birthplace'].iloc[x]+', USA'
        
#%%        

##Adding a country column
oscar_df_sub['country'] = np.nan

for x in range(0,len(oscar_df_sub["birthplace"])):
    oscar_df_sub['country'].iloc[x] = oscar_df_sub["birthplace"].iloc[x].split(",")[-1]
    
#%%
oscar_df_sub['year_of_award'].head()
#Adding Age
oscar_df_sub['Age_award'] = np.nan

def age_oscar(dob,award):
    date_format = "%d-%b-%Y"
    a = datetime.datetime.strptime(dob, "%d-%b-%Y")
    b = datetime.datetime.strptime(award, "%Y")
    delta = b - a
    return(int(round((delta.days)/365)))

for x in range(0,len(oscar_df_sub["Age_award"])):
    oscar_df_sub['Age_award'].iloc[x] = age_oscar(oscar_df_sub['date_of_birth_mod'].iloc[x],str(oscar_df_sub['year_of_award'].iloc[x]))

oscar_df_sub["Age_award"]=oscar_df_sub["Age_award"].astype(np.int64)
#%%
#QC of age
oscar_df_sub[oscar_df_sub["Age_award"]<0]

#%%
### Exploratory Data Analysis
##Checking null values
nv = oscar_df_sub.isnull().sum()

##Checking for duplicates rows based on all columns 
dups = oscar_df_sub.duplicated()
oscar_df_sub[dups].count()
#26 duplicate rows

#removing dulpicates rows based on all columns 
oscar_df_sub.drop_duplicates( keep = 'first', inplace = True)

#%%
#Dropping unneccesary column
oscar_df_sub.drop(columns=['ldob'],inplace=True)
#%%

# #Plots
# # Detecting whether data is skewed or not, this can be done by histograms for individual variable using sub plots
# fig, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7 , ax8, ax9)) = plt.subplots(3, 3,figsize=(12,12))
# axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
# cols = list(oscar_df_sub.columns)
# cols
# for n in range(0,len(axs)):
#     axs[n].hist(oscar_df_sub.iloc[:,n],bins=50)
#     axs[n].set_title('name={}'.format(cols[n])) 
#%%
#Data Visualization

#Most Oscar winners by country
oscar_df_sub['country'].value_counts().sort_values().plot(kind = 'barh',figsize=[20,10])

#Most Oscar winners by race,ethinicity
oscar_df_sub['race_ethnicity'].value_counts().sort_values().plot(kind = 'barh',figsize=[20,10])

oscar_df_sub['award'].unique()
oscar_df_sub[oscar_df_sub['award']=='Best Director']['Age_award'].mean()

#%%
###### Model Building 

oscar_df_sub['Age_bucket'] = np.nan
#Preparing data for age bucket
for x in range(0,len(oscar_df_sub["Age_award"])):
    if oscar_df_sub['Age_award'].iloc[x] <35:
        oscar_df_sub['Age_bucket'].iloc[x] = 'Bucket_1'
    if oscar_df_sub['Age_award'].iloc[x] <=35 and oscar_df_sub['Age_award'].iloc[x] <45  :
        oscar_df_sub['Age_bucket'].iloc[x] = 'Bucket_2'        
    if oscar_df_sub['Age_award'].iloc[x] <=45 and oscar_df_sub['Age_award'].iloc[x] <55:
        oscar_df_sub['Age_bucket'].iloc[x] = 'Bucket_3'    
    if oscar_df_sub['Age_award'].iloc[x] >=55:
        oscar_df_sub['Age_bucket'].iloc[x] = 'Bucket_4'

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
oscar_df_sub['award'].unique()

# Changing categorical variables of 'y'
award_map = {'Best Director': 1, 'Best Actress': 2, 'Best Actor': 3, 'Best Supporting Actor': 4, 'Best Supporting Actress': 5}
oscar_df_sub['award_num'] = oscar_df_sub['award'].map(award_map)
oscar_df_sub['award_num'].unique()
#%%

X = oscar_df_sub[['Age_bucket','race_ethnicity','country']].iloc[:]
y = oscar_df_sub['award_num'].iloc[:]

X = X.apply(lambda x: x.str.strip())

seed=50
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

X_train = X_train.fillna('na')
X_test = X_test.fillna('na')
#%%
X_train.isnull().sum()
y_train.isnull().sum()

#%%
X_train.dtypes
#create a list of categorical variables
features_to_encode = X_train.columns[X_train.dtypes==object].tolist()

#constructor to handle categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
col_trans = make_column_transformer(
                        (OneHotEncoder(handle_unknown='ignore'),features_to_encode),
                        remainder = 'passthrough'
                        )



#%%
#Train Data using Randon Forrest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      random_state=seed,
                      n_jobs=-1,
                      max_features='auto')


#%%

#combine our classifier and the constructor
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(col_trans, rf_classifier)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

#%%

#Evaluate the classifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

accuracy_score(y_test, y_pred)
print(f"The accuracy of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")

#%%

#Confusion Matrix

confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [ 1, 2, 3, 4, 5], columns = [ 1, 2, 3, 4, 5])

plt.figure(figsize=(8,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('RF Accuracy:{0:.3f} \n Roc_Auc:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')


