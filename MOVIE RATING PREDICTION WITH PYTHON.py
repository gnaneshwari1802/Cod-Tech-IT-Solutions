
Sentiment Analysis of IMDB Movie Reviews

Problem Statement:

In this, we have to predict the number of positive and negative reviews based on sentiments by using different classification models.

Import necessary libraries

#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os
print(os.listdir())
import warnings
warnings.filterwarnings('ignore')
['.anaconda', '.bash_history', '.cache', '.conda', '.condarc', '.config', '.continuum', '.eclipse', '.gitconfig', '.idlerc', '.ipynb_checkpoints', '.ipython', '.jupyter', '.kaggle', '.keras', '.m2', '.matplotlib', '.mito', '.ms-ad', '.node_repl_history', '.p2', '.python_history', '.spyder-py3', '.streamlit', '.tooling', '.viminfo', '.vscode', '3D Objects', 'anaconda3', 'AppData', 'Application Data', 'BKMeans.ipynb', 'classifier.pkl', 'complete-guide-on-time-series-analysis-in-python.ipynb', 'Contacts', 'Cookies', 'CV1_ capture videos.ipynb', 'CV2_ Red color mask.ipynb', 'CV3_ Blue color mask.ipynb', 'CV4_ Green color mask.ipynb', 'CV5_ Every color except white mask.ipynb', 'Desktop', 'Documents', 'Downloads', 'eda-bank-loan-default-risk-analysis.ipynb', 'eda-bank-loan-default-risk-analysis.py', 'Fashion Mnist dataset.ipynb', 'fashion-mnist-with-tensorflow.ipynb', 'Favorites', 'IntelGraphicsProfiles', 'Jedi', 'jupyter-datatables.ipynb', 'jupyter-labs-eda-dataviz.ipynb', 'jupyter-labs-eda-sql-coursera_sqllite.ipynb', 'jupyter-labs-spacex-data-collection-api.ipynb', 'jupyter-labs-webscraping.ipynb', 'labs-jupyter-spacex-Data wrangling.ipynb', 'lab_jupyter_launch_site_location.ipynb', 'Links', 'Local Settings', 'LovelyPlots-Professional-Matplotlib.ipynb', 'Mercury App.ipynb', 'mitosheet.ipynb', 'model.png', 'ModelTraining.ipynb', 'Music', 'My Documents', 'my_data1.db', 'NetHood', 'Nocode-Pivot-Groupby.ipynb', 'NTUSER.DAT', 'ntuser.dat.LOG1', 'ntuser.dat.LOG2', 'NTUSER.DAT{53b39e88-18c4-11ea-a811-000d3aa4692b}.TM.blf', 'NTUSER.DAT{53b39e88-18c4-11ea-a811-000d3aa4692b}.TMContainer00000000000000000001.regtrans-ms', 'NTUSER.DAT{53b39e88-18c4-11ea-a811-000d3aa4692b}.TMContainer00000000000000000002.regtrans-ms', 'ntuser.ini', 'Number_plate_detection_code.ipynb', 'OneDrive', 'pdf to text extraction PyPDF2.ipynb', 'pdf to text extraction.ipynb', 'Pictures', 'pivottablejs.html', 'predict-monthly-milk-production.ipynb', 'PrintHood', 'pyspark basic introduction.ipynb', 'python tips and trics graphs.ipynb', 'Recent', 'requirements.txt', 'Saved Games', 'Searches', 'SendTo', 'sentiment-analysis-of-imdb-movie-reviews.ipynb', 'sentiment-analysis-of-imdb-movie-reviews.pkl', 'spacex_launch_geo.csv', 'SpaceX_Machine_Learning_Prediction_Part_5.jupyterlite.ipynb', 'speech recognition.ipynb', 'speech recognition.py', 'Start Menu', 'Templates', 'test_HSV_Color.ipynb', 'time-series-data-analysis-using-lstm-tutorial.ipynb', 'Untitled Folder', 'Untitled.ipynb', 'Untitled1.ipynb', 'Untitled2.ipynb', 'Untitled3.ipynb', 'Untitled4.ipynb', 'Untitled5.ipynb', 'urban-8k-deeplearning-pytorch.ipynb', 'Videos', '_netrc']
pip install wordcloud
Requirement already satisfied: wordcloud in c:\users\gnaneshwari\anaconda3\lib\site-packages (1.9.2)
Requirement already satisfied: numpy>=1.6.1 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from wordcloud) (1.26.1)
Requirement already satisfied: pillow in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from wordcloud) (10.1.0)
Requirement already satisfied: matplotlib in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from wordcloud) (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (1.1.1)
Requirement already satisfied: cycler>=0.10 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (4.43.1)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (1.4.5)
Requirement already satisfied: packaging>=20.0 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (23.2)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (3.1.1)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from matplotlib->wordcloud) (2.8.2)
Requirement already satisfied: six>=1.5 in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
Note: you may need to restart the kernel to use updated packages.
pip install textblob
Requirement already satisfied: textblob in c:\users\gnaneshwari\anaconda3\lib\site-packages (0.17.1)Note: you may need to restart the kernel to use updated packages.

Requirement already satisfied: nltk>=3.1 in c:\users\gnaneshwari\anaconda3\lib\site-packages (from textblob) (3.8.1)
Requirement already satisfied: click in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from nltk>=3.1->textblob) (8.1.7)
Requirement already satisfied: joblib in c:\users\gnaneshwari\anaconda3\lib\site-packages (from nltk>=3.1->textblob) (1.2.0)
Requirement already satisfied: regex>=2021.8.3 in c:\users\gnaneshwari\anaconda3\lib\site-packages (from nltk>=3.1->textblob) (2022.7.9)
Requirement already satisfied: tqdm in c:\users\gnaneshwari\anaconda3\lib\site-packages (from nltk>=3.1->textblob) (4.65.0)
Requirement already satisfied: colorama in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from click->nltk>=3.1->textblob) (0.4.6)
Import the training dataset

#importing the training data
imdb_data=pd.read_csv(r'E:\IMDB Dataset.csv/IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)
(50000, 2)
review	sentiment
0	One of the other reviewers has mentioned that ...	positive
1	A wonderful little production. <br /><br />The...	positive
2	I thought this was a wonderful way to spend ti...	positive
3	Basically there's a family where a little boy ...	negative
4	Petter Mattei's "Love in the Time of Money" is...	positive
5	Probably my all-time favorite movie, a story o...	positive
6	I sure would like to see a resurrection of a u...	positive
7	This show was an amazing, fresh & innovative i...	negative
8	Encouraged by the positive comments about this...	negative
9	If you like original gut wrenching laughter yo...	positive
Exploratery data analysis

#Summary of the dataset
imdb_data.describe()
review	sentiment
count	50000	50000
unique	49582	2
top	Loved today's show!!! It was a variety and not...	positive
freq	5	25000
Sentiment count

#sentiment count
imdb_data['sentiment'].value_counts()
sentiment
positive    25000
negative    25000
Name: count, dtype: int64
We can see that the dataset is balanced.

Spliting the training dataset

#split the dataset  
#train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)
(40000,) (40000,)
(10000,) (10000,)
Text normalization

pip install nltk
Requirement already satisfied: nltk in c:\users\gnaneshwari\anaconda3\lib\site-packages (3.8.1)Note: you may need to restart the kernel to use updated packages.

Requirement already satisfied: click in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from nltk) (8.1.7)
Requirement already satisfied: joblib in c:\users\gnaneshwari\anaconda3\lib\site-packages (from nltk) (1.2.0)
Requirement already satisfied: regex>=2021.8.3 in c:\users\gnaneshwari\anaconda3\lib\site-packages (from nltk) (2022.7.9)
Requirement already satisfied: tqdm in c:\users\gnaneshwari\anaconda3\lib\site-packages (from nltk) (4.65.0)
Requirement already satisfied: colorama in c:\users\gnaneshwari\appdata\roaming\python\python311\site-packages (from click->nltk) (0.4.6)
import nltk
nltk.download()
showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
True
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\Gnaneshwari\AppData\Roaming\nltk_data...
[nltk_data]   Unzipping corpora\stopwords.zip.
!python -m nltk.downloader stopwords
# import nltk
# nltk.download('stopwords')
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
Removing html strips and noise text

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)
Removing special characters

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_special_characters)
**Text stemming **

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(simple_stemmer)
Removing stopwords

#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_stopwords)
{'again', 'just', 'i', 'their', 'have', 'over', 'in', 'on', 'so', 'with', 'ma', "wasn't", 'the', "wouldn't", 'itself', "weren't", 'myself', 'me', 'because', 'up', 'didn', 'whom', 'ours', 'there', 'it', 'hasn', "shouldn't", 'won', 'no', 'wouldn', 'other', 'into', "shan't", 'but', 'isn', 'he', 'if', "hasn't", 'how', 'these', 'of', 'can', 'aren', 'once', 'an', 'not', 'out', 'had', 't', "mightn't", 'has', 'are', "you've", 'yourself', 'do', 'under', 'his', 'themselves', 'by', 'about', 'between', 'off', 'any', 'down', 'you', 's', "doesn't", 'and', 'as', 'what', 'through', 'we', 'further', 'they', "haven't", "don't", "aren't", 'does', 'doesn', 'at', 'having', 'each', 'which', 'my', "should've", 'mightn', "you'd", "didn't", 'after', "you're", 'needn', 'both', 're', 'couldn', 'ourselves', 'too', 'ain', "isn't", 'being', 'shan', 'where', "mustn't", 'hadn', 'below', 'shouldn', 'will', 'a', 'don', 'why', 'll', 'yourselves', 'her', 'most', 'haven', 'wasn', 'here', 'this', 'herself', 'yours', 'few', 'against', 'did', 'theirs', 'until', 'from', 'them', 'all', 'be', 'him', 'now', 've', "that'll", 'hers', 'is', 'then', 'd', 'y', 'm', 'himself', 'before', 'to', 'doing', 'or', 'above', 'during', 'more', 'weren', 'she', "you'll", "needn't", 'am', 'its', "hadn't", "won't", 'your', 'when', "couldn't", 'such', 'should', 'while', 'mustn', 'were', 'some', 'same', 'for', 'only', 'those', 'was', "it's", 'nor', 'o', 'own', 'than', 'that', "she's", 'who', 'been', 'our', 'very'}
Normalized train reviews

#normalized train reviews
norm_train_reviews=imdb_data.review[:40000]
norm_train_reviews[0]
#convert dataframe to string
#norm_train_string=norm_train_reviews.to_string()
#Spelling correction using Textblob
#norm_train_spelling=TextBlob(norm_train_string)
#norm_train_spelling.correct()
#Tokenization using Textblob
#norm_train_words=norm_train_spelling.words
#norm_train_words
'one review ha mention watch 1 oz episod youll hook right thi exactli happen meth first thing struck oz wa brutal unflinch scene violenc set right word go trust thi show faint heart timid thi show pull punch regard drug sex violenc hardcor classic use wordit call oz nicknam given oswald maximum secur state penitentari focus mainli emerald citi experiment section prison cell glass front face inward privaci high agenda em citi home manyaryan muslim gangsta latino christian italian irish moreso scuffl death stare dodgi deal shadi agreement never far awayi would say main appeal show due fact goe show wouldnt dare forget pretti pictur paint mainstream audienc forget charm forget romanceoz doesnt mess around first episod ever saw struck nasti wa surreal couldnt say wa readi watch develop tast oz got accustom high level graphic violenc violenc injustic crook guard wholl sold nickel inmat wholl kill order get away well manner middl class inmat turn prison bitch due lack street skill prison experi watch oz may becom comfort uncomfort viewingthat get touch darker side'
Normalized test reviews

#Normalized test reviews
norm_test_reviews=imdb_data.review[40000:]
norm_test_reviews[45005]
##convert dataframe to string
#norm_test_string=norm_test_reviews.to_string()
#spelling correction using Textblob
#norm_test_spelling=TextBlob(norm_test_string)
#print(norm_test_spelling.correct())
#Tokenization using Textblob
#norm_test_words=norm_test_spelling.words
#norm_test_words
'read review watch thi piec cinemat garbag took least 2 page find somebodi els didnt think thi appallingli unfunni montag wasnt acm humour 70 inde ani era thi isnt least funni set sketch comedi ive ever seen itll till come along half skit alreadi done infinit better act monti python woodi allen wa say nice piec anim last 90 second highlight thi film would still get close sum mindless drivelridden thi wast 75 minut semin comedi onli world semin realli doe mean semen scatolog humour onli world scat actual fece precursor joke onli mean thi handbook comedi tit bum odd beaver niceif pubesc boy least one hand free havent found playboy exist give break becaus wa earli 70 way sketch comedi go back least ten year prior onli way could even forgiv thi film even made wa gunpoint retro hardli sketch clown subtli pervert children may cut edg circl could actual funni come realli quit sad kept go throughout entir 75 minut sheer belief may save genuin funni skit end gave film 1 becaus wa lower scoreand onli recommend insomniac coma patientsor perhap peopl suffer lockjawtheir jaw would final drop open disbelief'
**Bags of words model **

It is used to convert text documents to numerical vectors or bag of words.

#Count vectorizer for bag of words
# cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cv = CountVectorizer(min_df=1, max_df=1.0, binary=False, ngram_range=(1, 3))

#transformed train reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
#transformed test reviews
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names
BOW_cv_train: (40000, 6983231)
BOW_cv_test: (10000, 6983231)
Term Frequency-Inverse Document Frequency model (TFIDF)

It is used to convert text documents to matrix of tfidf features.

#Tfidf vectorizer
tv = TfidfVectorizer(min_df=1, max_df=1.0, use_idf=True, ngram_range=(1, 3))

#transformed train reviews
tv_train_reviews=tv.fit_transform(norm_train_reviews)
#transformed test reviews
tv_test_reviews=tv.transform(norm_test_reviews)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)
Tfidf_train: (40000, 6983231)
Tfidf_test: (10000, 6983231)
Labeling the sentiment text

#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data=lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)
(50000, 1)
Split the sentiment tdata

#Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)
[[1]
 [1]
 [1]
 ...
 [1]
 [0]
 [0]]
[[0]
 [0]
 [0]
 ...
 [0]
 [0]
 [0]]
Modelling the dataset

Let us build logistic regression model for both bag of words and tfidf features

#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
print(lr_tfidf)
LogisticRegression(C=1, max_iter=500, random_state=42)
LogisticRegression(C=1, max_iter=500, random_state=42)
Logistic regression model performane on test dataset

#Predicting the model for bag of words
lr_bow_predict=lr.predict(cv_test_reviews)
print(lr_bow_predict)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
print(lr_tfidf_predict)
[0 0 0 ... 0 0 0]
[0 0 0 ... 1 0 0]
Accuracy of the model

#Accuracy score for bag of words
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
lr_bow_score : 0.847
lr_tfidf_score : 0.8868
Print the classification report

#Classification report for bag of words 
lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])
print(lr_bow_report)

#Classification report for tfidf features
lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)
              precision    recall  f1-score   support

    Positive       0.81      0.90      0.85      4993
    Negative       0.89      0.80      0.84      5007

    accuracy                           0.85     10000
   macro avg       0.85      0.85      0.85     10000
weighted avg       0.85      0.85      0.85     10000

              precision    recall  f1-score   support

    Positive       0.89      0.88      0.89      4993
    Negative       0.88      0.89      0.89      5007

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Confusion matrix

#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,lr_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,lr_tfidf_predict,labels=[1,0])
print(cm_tfidf)
[[3981 1026]
 [ 504 4489]]
[[4459  548]
 [ 584 4409]]
Stochastic gradient descent or Linear support vector machines for bag of words and tfidf features

#training the linear svm
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)
#fitting the svm for bag of words
svm_bow=svm.fit(cv_train_reviews,train_sentiments)
print(svm_bow)
#fitting the svm for tfidf features
svm_tfidf=svm.fit(tv_train_reviews,train_sentiments)
print(svm_tfidf)
SGDClassifier(max_iter=500, random_state=42)
SGDClassifier(max_iter=500, random_state=42)
Model performance on test data

#Predicting the model for bag of words
svm_bow_predict=svm.predict(cv_test_reviews)
print(svm_bow_predict)
#Predicting the model for tfidf features
svm_tfidf_predict=svm.predict(tv_test_reviews)
print(svm_tfidf_predict)
[0 0 0 ... 0 0 0]
[0 0 0 ... 1 0 0]
Accuracy of the model

#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)
print("svm_bow_score :",svm_bow_score)
#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(test_sentiments,svm_tfidf_predict)
print("svm_tfidf_score :",svm_tfidf_score)
svm_bow_score : 0.8433
svm_tfidf_score : 0.8869
Print the classification report

#Classification report for bag of words 
svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])
print(svm_bow_report)
#Classification report for tfidf features
svm_tfidf_report=classification_report(test_sentiments,svm_tfidf_predict,target_names=['Positive','Negative'])
print(svm_tfidf_report)
              precision    recall  f1-score   support

    Positive       0.80      0.91      0.85      4993
    Negative       0.90      0.77      0.83      5007

    accuracy                           0.84     10000
   macro avg       0.85      0.84      0.84     10000
weighted avg       0.85      0.84      0.84     10000

              precision    recall  f1-score   support

    Positive       0.90      0.88      0.89      4993
    Negative       0.88      0.90      0.89      5007

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Plot the confusion matrix

#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,svm_tfidf_predict,labels=[1,0])
print(cm_tfidf)
[[3878 1129]
 [ 438 4555]]
[[4497  510]
 [ 621 4372]]
Multinomial Naive Bayes for bag of words and tfidf features

#training the model
mnb=MultinomialNB()
#fitting the svm for bag of words
mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)
print(mnb_bow)
#fitting the svm for tfidf features
mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)
print(mnb_tfidf)
MultinomialNB()
MultinomialNB()
Model performance on test data

#Predicting the model for bag of words
mnb_bow_predict=mnb.predict(cv_test_reviews)
print(mnb_bow_predict)
#Predicting the model for tfidf features
mnb_tfidf_predict=mnb.predict(tv_test_reviews)
print(mnb_tfidf_predict)
[0 0 0 ... 0 0 0]
[0 0 0 ... 0 0 0]
Accuracy of the model

#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)
#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(test_sentiments,mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)
mnb_bow_score : 0.8783
mnb_tfidf_score : 0.8892
Print the classification report

#Classification report for bag of words 
mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])
print(mnb_bow_report)
#Classification report for tfidf features
mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])
print(mnb_tfidf_report)
              precision    recall  f1-score   support

    Positive       0.85      0.91      0.88      4993
    Negative       0.91      0.84      0.87      5007

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000

              precision    recall  f1-score   support

    Positive       0.88      0.90      0.89      4993
    Negative       0.90      0.88      0.89      5007

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Plot the confusion matrix

#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,mnb_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,mnb_tfidf_predict,labels=[1,0])
print(cm_tfidf)
[[4218  789]
 [ 428 4565]]
[[4391  616]
 [ 492 4501]]
Let us see positive and negative words by using WordCloud.

Word cloud for positive review words

#word cloud for positive review words
plt.figure(figsize=(10,10))
positive_text=norm_train_reviews[1]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show
<function matplotlib.pyplot.show(close=None, block=None)>

Word cloud for negative review words

#Word cloud for negative review words
plt.figure(figsize=(10,10))
negative_text=norm_train_reviews[8]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
negative_words=WC.generate(negative_text)
plt.imshow(negative_words,interpolation='bilinear')
plt.show
<function matplotlib.pyplot.show(close=None, block=None)>

Conclusion:

We can observed that both logistic regression and multinomial naive bayes model performing well compared to linear support vector machines.
Still we can improve the accuracy of the models by preprocessing data and by using lexicon models like Textblob.
import pickle

# Pickle the trained models
with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\lr_bow_model.pkl', 'wb') as file:
    pickle.dump(lr_bow, file)

with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\lr_tfidf_model.pkl', 'wb') as file:
    pickle.dump(lr_tfidf, file)

with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\svm_bow_model.pkl', 'wb') as file:
    pickle.dump(svm_bow, file)

with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\svm_tfidf_model.pkl', 'wb') as file:
    pickle.dump(svm_tfidf, file)

with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\mnb_bow_model.pkl', 'wb') as file:
    pickle.dump(mnb_bow, file)

with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\mnb_tfidf_model.pkl', 'wb') as file:
    pickle.dump(mnb_tfidf, file)

# Pickle the vectorizers
with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\count_vectorizer.pkl', 'wb') as file:
    pickle.dump(cv, file)

with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tv, file)

# Pickle the dataframes
with open(r'E:\sentiment-analysis-of-imdb-movie-reviews\imdb_data.pkl', 'wb') as file:
    pickle.dump(imdb_data, file)

# Pickle the other variables or objects as needed
 
 
 
 
