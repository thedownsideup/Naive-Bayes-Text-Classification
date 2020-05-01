#!/usr/bin/env python
# coding: utf-8

# ### Artificial Intelligence Project #3 : Naive Bayes Text Classification
# Mahsa Eskandari Ghadi         
# Student No. 810196597

# In this project we use <b>Naive Bayes</b> to classify the news by their short descriptions. Our data has 3 categories: Travel, Business and Style&Beauty.<br>
# Our approach to this text classification is the <b>Bag of Words</b> model. In this type of modeling we don't care about the order of the words or the grammar of the sentence. We just work with a bunch of words and how many times they appear so it's a "bag of words" in the literal sense.

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import re
import time
get_ipython().system('pip install prettytable')
from prettytable import PrettyTable
get_ipython().system('pip install tabulate')
get_ipython().system('set TABULATE_INSTALL=lib-only')
from tabulate import tabulate
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer


# In[2]:


TRAIN_DATA = "data.csv"
TEST_DATA = "test.csv"


# <font color="9AC0CD"><b>How is a clean data achieved here? </b></font> <br>
# - Step1 : Remove all the non-alphabetic characters such as "!, &, #, ^, @ , ...".
# - Step2 : Change all of the uppercase letters to lowercase to have a more consistent data. <br>
# - Step3 : Extract the words
# - Step4 : Stem the words
# 
# <font color="10B3B4"><b>What is <b>Stemming</b>? </b></font> <br>
# With stemming, words are reduced to their word stems. A word stem need not be the same root as a dictionary-based morphological root, it just is an equal to or smaller form of the word. For example “cooking,” and “cooked” all to the same stem of “cook.” [1] <br>
# 
# <font color="10B3B4"><b>What is <b>lemmatization</b>? and what is it's difference with stemming? </b></font> <br>
# Stemming is definitely the simpler of the two approaches. Lemmatization is a more calculated process and it involves resolving words to their dictionary form for example resolving "is" and "are" to “be”. <br>
# 
# Stemmers are generally more popular in text classifications and at first I used the Snowball Stemmer, you can read more about snowball stemmer and other types of stemmers -> [1]
# 
# I decided to give lemmatization a chance as well, In my code it didn't make much difference in precision and accuracy and recall which are all calculated in the end. <br>

# In[3]:


def extract_words (text):
    stop_words = set(stopwords.words('english'))
#     stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer() 
    alphabet = string.ascii_lowercase
    words = re.sub('[^{}]'.format(alphabet), ' ', text.lower()).split()
    clean_words = [word for word in words if word not in stop_words]
    stemmed = [lemmatizer.lemmatize(word) for word in clean_words]
#     stemmed = [stemmer.stem(word) for word in clean_words]
    return stemmed


# <font color="D7BDE2"><b>What is <b>Oversampling</b> and why do we do it? </b></font> <br>
# Oversampling involve introducing a bias to select more samples from one class than from another, to compensate for an imbalance that is either already present in the data, or likely to develop if a purely random sample were taken. [2] <br>
# The values of recall and precision shouldn't have more than a 10% difference looking between. For example at first the precision for TRAVEL was 0.92 and for BUSINESS it was 0.77 and that's a 15% difference (bad).
# Steps I took for oversampling:
# - Findout all the sizes of data there is in the dataset for each class.
# - Find the maximum
# - Randomly sample (maximum - class data size) from the current class
# - Append the result of last step to the data
# 
# So now we have the amount of data for all of the categories. <br>
# <b>Oversampling made a big difference in the precision and recall values and made them much closer to eachother comparing in classes.</b> BUSINESS's precision increased to 0.89 making the diffrence 0.3.

# In[4]:


def over_sample (categories, dataset):
    data_sizes = []

    for category in categories:
        data = dataset[dataset['category'] == category]
        data_sizes.append(data.shape[0])

    for index, category in enumerate(categories):
        data = dataset[dataset['category'] == category]
        dataset = dataset.append(data.sample(n = max(data_sizes)-data_sizes[index]))
        
    return dataset


# <font color="10B3B4"><b>Why is it better to get 80% of each category and put them all together rather than just using 80% of the whole data?</b></font> <br> If we have more of one category to train our model with in testing the model has more knowledge about that category and the other categories might have the same characteristics but the model doesn't know that and puts that characteristics along side of that specific category, but when we have the same amount of data to learn for all categories all of them have the same chance. <br>
# In this project however, I tested it both ways and again the accuracy and precisions didn't change at all.
# 

# In[5]:


def load_data (filename): 

    #split data to train and test
    data = pd.read_csv('data.csv')
    data_no_nans = data.dropna(subset=['short_description'])
    clean_data = data_no_nans.reset_index(drop=True)

    data = pd.DataFrame(clean_data, columns = ['category', 'short_description'])

    categories = clean_data['category'].unique()
    
    data = over_sample(categories, data)
    
    test_df = pd.read_csv('test.csv')
    test_df = data.dropna(subset=['short_description'])
    test_df = test_df.reset_index(drop=True)
    test_df = pd.DataFrame(test_df, columns = ['short_description'])

    df_list = []
    for category in categories:
        df_list.append(data[clean_data['category'] == category])
        
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    
    
    for df in df_list:
        train_set = train_set.append(df[:int(df.shape[0]*0.8)], ignore_index=True)
        test_set = test_set.append(df[int(df.shape[0]*0.8):], ignore_index=True)
    
    return train_set, test_set, test_df


# In[22]:


def count_words (text):
    extracted = []

    for sentence in text:
        extracted += extract_words(sentence)

    words_df = pd.DataFrame(extracted)
    words_count = words_df[0].value_counts()    
    words_count_dict = words_count.to_dict()

    return words_count_dict, extracted


# In[28]:


def write_to_outputcsv(text):
    df = pd.DataFrame(text)  
    df.to_csv('output.csv') 


# #### The Naive Bayes Classifier: <br>
# Has 4 methods: classify(which works as a main function), learn, predict and calculate accuracy. All of which are self explainatory by their names. I'll explain the flow: <br>
# <br>
# ##### Part A: Learning: <br>
# - For each category, Extract the words and make the bag of words dictionary(words and their frequencies)
# - For each category calculate P(Category) <br>
# and we're done with learning:D <br>
# 
# ##### Part B: Predicting: Using Lidstone Smoothing [3] : <br>
# 
# <font color="73C6B6"><b>If the word "Tehran" appears in the the learning corpus only once, what will our system predict?</b></font> <br> This is where smoothing can help us. For example if  "Newyork" appears only once in and that one time it was in TRAVEL it doesn't mean that when we see "Newyork" in the test descriptions we should immediately classify the description as TRAVEL it could be about stock markets and wall street in that case it should be classified as BUSINESS. We need to allow this possibility and let the model know that this could be another category. If we didn't use smoothing the probability of "Newyork" would've been zero for BUSINESS in this example.<br>
# 
# - For every short description in the testing set, calculate this probability for every category there is: <br><br>
# $ \hat{P}(Word_i|Class_j) = \frac{N_{{w_i}{c_j}} + \alpha}{{N_{c_j}} + \alpha d} $
# 
# 
# Where:
# 
# - $N_{{w_i}{c_j}}$ : Number of times $word_i$ appears in $class_j$ <font color="9AC0CD">category_dicts[idx][word]</font>
# - $N_{c_j}$ : Total count of all words in $class_j$ <font color="9AC0CD">len(category_word_lists[idx]</font>
# - $\alpha$ : Parameter for additive smoothing
# - $d$ : Number of words in total (i = [1,2, ... , d]) <font color="9AC0CD">len(words)</font>
# 
# Then: <br>
# 
# $ P(Word|Class) = P(x_1|Class) \times P(x_2|Class) \times ... P(x_n|Class) \times P(Class)$ 
# 
# THIS WAS A BAG OF WORDS APPROACH, another approach can be TF-IDF: <br>
# <font color="D7BDE2"><b>-----------------------------------------------------------------------------------------------------------------------------------------------------------------------</b></font><br>
# <font color="73C6B6"><b>What is <b>td-idf</b>? and how could it be used in a naive bayes problem? </b></font> <br>
# 
# TF-IDF stands for <b>“Term Frequency — Inverse Document Frequency”</b>. This is a technique to quantify a word in documents, we generally compute a <b>weight</b> to each word which signifies <b>the importance of the word</b> in the document and corpus. This method is a widely used technique in Information Retrieval and Text Mining. [4] <br><br>
# $TF-IDF = Term \ \ Frequency \ \ (TF) * Inverse \ \ Document \ \ Frequency \ \ (IDF)$ <br><br>
# 
# - <font color="D7BDE2"><b>Term Frequency: </b></font> the number of occurances of a word in a document (frequency). Highly depends on the length of the document and the generality of word. (removing stopwords helps with this)
# - <font color="D7BDE2"><b>Document Frequency: </b></font> number occurrences of the term(word) in documents. Basically defines the importance of the document in the corpus. Meaning here the importance of the short description in all of the short descriptions. (how much can this specific short decription help us to classify short descriptions later)
# 
# - <font color="D7BDE2"><b>Inverse Document Frequency: </b></font>: N/df. Depicts how much a word can be informative. For stop words it'll be really small as expected for these type of words. For example "is" is a word that can be used in any sentence and it should'nt make a difference when classifying. It's not like if the frequency of "is" is more in the TRAVEL category means that when we have a lot of "is"'s it must be TRAVEL:)) so this takes care of that. 
# 
# ##### Part C: Calculate Accuracy: <br>
# - $Recall = \frac{CorrectDetectedCategory}{CategoryInTesting}$ <br>
# - $Precision = \frac{CorrectDetectedCategory}{Predicted Category}$ <br>
# - $Accuracy = \frac{AllCorrectDetected}{Total}$ <br>
# 
# calculate_accuracy also prints the confusion matrix, which is a table that depicts how many of the predicted category were actually correct and how many were mistaken for which other category.

# In[29]:


class Naive_Bayes_Classifier:
    
    def __init__(self, alpha, is_two, is_test):
        self.training_df, self.testing_df, self.test_df = load_data(TRAIN_DATA) #whole df
        self.alpha = alpha
        self.is_two = is_two
        self.is_test = is_test
        
    def classify(self):
        
        if self.is_two:
            self.training_df.drop(self.training_df[self.training_df['category'] == 'STYLE & BEAUTY'].index, inplace=True)
            self.training_df = self.training_df.reset_index(drop=True)
            self.testing_df.drop(self.testing_df[self.testing_df['category'] == 'STYLE & BEAUTY'].index , inplace=True)
            self.testing_df = self.testing_df.reset_index(drop=True)
            phase = 'phase1'
        else:
            phase = 'phase2'
        
        self.categories = self.training_df['category'].unique()
        
        all_words_dict, extracted = count_words(self.training_df['short_description'])
        self.unique_words = pd.DataFrame(extracted)[0].unique()
        
        category_dicts = []
        category_word_lists = []
        class_prob_list= []
        for category in self.categories:
            category_dict, category_words = self.learn(category)
            category_dicts.append(category_dict)
            category_word_lists.append(category_words)
            class_prob_list.append(self.training_df[self.training_df['category'] == category].size / self.training_df.size)
            
        
        predictions = self.predict(category_dicts, category_word_lists, class_prob_list)
    
        if self.is_test:
            write_to_outputcsv(predictions)
        else:
            self.calculate_accuracy(predictions, phase)
        
    def learn (self, category):

        train_class = self.training_df[self.training_df['category'] == category]
        count_dict, extracted = count_words(train_class['short_description'])

        return count_dict, extracted
            
            
    def predict (self, category_dicts, category_word_lists, class_prob_list):
        predictions = []
        descriptions = self.testing_df['short_description']
        if self.is_test:
            descriptions = self.test_df['short_description']
        
        scores_dict = {}
        for index, description in enumerate(descriptions):
            words = extract_words(description)
            scores = []
            for idx, category in enumerate(self.categories):
                score = 1
                for word in words:
                    if word in category_dicts[idx]: 
                        Nwc = category_dicts[idx][word]
                    else:
                        Nwc = 0
                    score *= (Nwc + self.alpha) / (len(category_word_lists[idx]) + self.alpha*len(words))

                scores.append(score*class_prob_list[idx])
                    
            scores_dict[index] = scores

        for des in scores_dict:
            predictions.append(self.categories[scores_dict[des].index(max(scores_dict[des]))])

        return predictions
    
    def calculate_accuracy (self, predictions, phase):
        
        table = []
        
        matrix = []

        all_correct = 0
        for category in self.categories:
            
            matrix_dict = {}
            for _category in self.categories:
                matrix_dict[_category] = 0
                
            correct_count = 0
            actuals = self.testing_df['category']
            
            for index, actual in enumerate(actuals):
                if actual == category:
                    if actual == predictions[index]:
                        correct_count+=1 #tp
    
                    matrix_dict[predictions[index]]+=1
            
            matrix.append(matrix_dict)
            all_correct+=correct_count
            
            recall = correct_count / self.testing_df[self.testing_df['category'] == category].shape[0]
            precision = correct_count / predictions.count(category)
            table.append([category, recall, precision])

        accuracy = all_correct / self.testing_df.shape[0]
        print(tabulate(table, headers=[phase, "Recall", "Precision"], tablefmt="pretty"))
        print("Accuracy = ", accuracy)
        
        if not self.is_two:

            confusion_matrix = PrettyTable()
            confusion_matrix.add_row([' ', ' ', 'Predicted', 'Predicted', 'Predicted'])
            confusion_matrix.add_row([' ', ' '] + list(self.categories))
            for index, dic in enumerate(matrix):
                dic = OrderedDict([(el, dic[el]) for el in self.categories])
                confusion_matrix.add_row(['Actual'] + [self.categories[index]] + list(dic.values()))
            print(confusion_matrix)


# <font color="73C6B6"><b>Is <b>precision</b> everything?</b></font> <br>
# No. In modeling a problem we need to make sure that the model has learned from the training data properly and that is why precision is good. Because of the uncertain nature of data, sometimes the model results in a better precision and accuracy but hasn't really learned from the data and that's why it performs poorly when the data is varied. [5] For example when we have 10000 of one class and 200 of another, when the model classifies almost all of them as the first one and that gives it the precision will be 98%. But if we give this model 200 of the first one and 10000 of the second one, the model again predicts almost all of it as the first one, this time precision is 0.02. So this is why we need to checkout recalls. Recall refers to the percentage of total relevant results correctly classified by your algorithm. Where percision is the percentage of your results which are relevant. [6]<br>

# In[30]:


t1 = time.time()
naive_classifier = Naive_Bayes_Classifier(0.5, is_two=True, is_test=False)
naive_classifier.classify()
t2 = time.time()
print(t2-t1, 's')


# In[31]:


t1 = time.time()
naive_classifier = Naive_Bayes_Classifier(0.1, is_two=False, is_test=False)
naive_classifier.classify()
t2 = time.time()
print(t2-t1, 's')


# In[32]:


naive_classifier = Naive_Bayes_Classifier(0.1, is_two=False, is_test=True)
naive_classifier.classify()


# ### References: <br>
# [1] https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8 <br>
# [2] https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis <br>
# [3] https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b <br>
# [4] https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089 <br>
# [5] https://towardsdatascience.com/is-accuracy-everything-96da9afd540d <br>
# [6] https://towardsdatascience.com/precision-vs-recall-386cf9f89488
