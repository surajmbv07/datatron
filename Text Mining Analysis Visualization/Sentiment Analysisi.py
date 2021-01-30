#%%

#%%
#youtube comments dataset
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
pd.set_option('display.max_columns', None)
#pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
#%%
#Further Text Cleaning
#removing punctuations 

def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase

#remove html tags

def removeHTMLTags(review):
    soup = BeautifulSoup(review, 'lxml')
    return soup.get_text()

#remove alphanumeric words
def removeAlphaNumericWords(review):
    return re.sub("\S*\d\S*", "", review).strip()

#remove special characters
def removeSpecialChars(review):
    return re.sub('[^a-zA-Z]', ' ', review)

#remove Stop words
def remove_stopwords(review):
    word_tokens = word_tokenize(review)
    filtered_sentence = [w for w in word_tokens if not w in set(stopwords.words('english'))]
    return " ".join(filtered_sentence)
#%%

def doTextCleaning(review):
    review = removeHTMLTags(review)
    review = removeApostrophe(review)
    review = removeAlphaNumericWords(review)
    review = removeSpecialChars(review) 
    review = remove_stopwords(review)
    
    review = review.lower()  # Lower casing
    review = review.split()  # Tokenization
    
    #Removing Stopwords and Lemmatization
    lmtzr = WordNetLemmatizer()
    review = [lmtzr.lemmatize(word, 'v') for word in review if not word in set(stopwords.words('english'))]
    
    review = " ".join(review)    
    return review
#%%
dataset = pd.read_csv(r'C:\Users\mbvsuraj\Documents\Python_Scripts\rrevif\anastasija_p\Yt_comment_Dataset_final.csv')
corpus = []   
dataset['comments'] = dataset['comments'].str.strip()
for index, row in dataset.iterrows():
    review = doTextCleaning(row['comments'])
    corpus.append(review)
dataset.shape
#%%
dataset['Cleaned_comments'] = corpus
dataset.head()
#%%
data = dataset.copy()
data.head()
#%%
from textblob import TextBlob
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix

#%%
#The dataset is unlabelled because of using API,
# you can only extract the comments but not the polarity. 
#Polarity is something that can identify the emotion of a particular 
#sentence by using the words present in that. This can be done using the TextBlob module of python, which provides a function to find polarity as follows,
data['polarity'] = data['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)
#Shuffle the data set
data = data.sample(frac=1).reset_index(drop=True)
data['pol_cat']  = ''
#Continuos to Categorical
data['pol_cat'][data.polarity > 0] = 1
data['pol_cat'][data.polarity <= 0] = -1

#%%

#### Bar Plot #####
data['pol_cat'].value_counts()
data['pol_cat'].value_counts().plot.bar()

#Create separate dataframes for Negative,Positive & Neutral comments
data_pos = data[data['pol_cat'] == 1]
data_pos = data_pos.reset_index(drop = True)

data_neg = data[data['pol_cat'] == -1]
data_neg = data_neg.reset_index(drop = True)

#%%
################ Machine Learning #########################
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(data['comments'],data['pol_cat'],test_size = 0.2,random_state = 102)

y_train=y_train.astype('int')
y_test=y_test.astype('int')
#%%
X_train.shape
X_test.shape
#%%
######## Apply Logistic Regression #########
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
vect = CountVectorizer()
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)

#%%
#Document Term Matrix
X = vect.fit_transform(corpus)
df= pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
df.to_excel(r'C:\Users\mbvsuraj\Documents\Python_Scripts\rrevif\anastasija_p\Document_Term_Matrix.xlsx')

#%%
tf_train.shape
#%%
Vocubulary_df = pd.DataFrame(vect.vocabulary_.items(), columns=['Word', 'Word_count'])
Vocubulary_df.to_excel(r'C:\Users\mbvsuraj\Documents\Python_Scripts\rrevif\anastasija_p\Word_count.xlsx')

#%%
#Sentiment Analysis Classification
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(tf_train,y_train)
#Accuracy of Train Data
lr.score(tf_train,y_train)
## 0.9894

#Accuracy of Test Data
lr.score(tf_test,y_test)
##0.738

expected = y_test
predicted = lr.predict(tf_test)

#%%
cf = metrics.confusion_matrix(expected,predicted,labels=[1,-1])
print(cf)
#[[111  64]
# [ 23 135]]
#%%
from sklearn import metrics
print(metrics.classification_report(expected,predicted))
s=metrics.classification_report(expected,predicted)
s.to_excel(r'C:\Users\mbvsuraj\Documents\Python_Scripts\rrevif\anastasija_p\Classification report.xlsx')





