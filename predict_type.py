import pickle
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import warnings
warnings.filterwarnings('ignore')


##########################################################
lemmatiser = WordNetLemmatizer()

# Remove the stop words for speed
import nltk
nltk.download("stopwords")
useless_words = stopwords.words("english")

# Remove these from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# Splitting the MBTI personality into 4 letters and binarizing it

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]

#To show result output for personality prediction
def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

##########################################################
def pre_process_text(data_2, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_data = len(data_2)
    i = 0

    for row in data_2.iterrows():
        # check code working
        # i+=1
        # if (i % 500 == 0 or i == 1 or i == len_data):
        #     print("%s of %s rows" % (i, len_data))

        # Remove and clean comments
        posts = row[1].posts

        # Remove url links
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        # Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()

        # Remove multiple letter repeating words
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        # Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        # transform mbti to binary vector
        type_labelized = translate_personality(row[1].type)  # or use lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        # the cleaned data temp is passed here
        list_posts.append(temp)

    # returns the result
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


##########################################################

# Load the saved model

# Load the the model for E/I
with open('Saved_Models/model_1.pkl', 'rb') as f:
    model_1 = pickle.load(f)

# Load the the model for N/S
with open('Saved_Models/model_2.pkl', 'rb') as f:
    model_2 = pickle.load(f)

# Load the the model for F/T
with open('Saved_Models/model_3.pkl', 'rb') as f:
    model_3 = pickle.load(f)

# Load the the model for J/P
with open('Saved_Models/model_4.pkl', 'rb') as f:
    model_4 = pickle.load(f)

models = [model_1, model_2, model_3, model_4]

# Load the count vectorizer
with open('Saved_Models/CountVector_1.pkl', 'rb') as f:
    cntizer = pickle.load(f)

# Load the tfidf vectorizer
with open('Saved_Models/TFizer_1.pkl', 'rb') as f:
    tfizer = pickle.load(f)


def make_ready_for_prediction(my_posts):
    # The type is just a dummy so that the data prep function can be reused
    mydata = pd.DataFrame(data={'type': ['ENTJ'], 'posts': [my_posts]})

    my_posts, dummy = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf = tfizer.transform(my_X_cnt).toarray()
    return my_X_tfidf


def predict_type(my_posts):
    # Prepare the data
    my_X_tfidf = make_ready_for_prediction(my_posts)
    
    # Predict the personality type
    result = []
    for i in range(4):
        result.append(models[i].predict(my_X_tfidf)[0])

    # Translate the binary result to mbti personality
    my_result = translate_back(result)
    return my_result