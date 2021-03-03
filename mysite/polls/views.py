from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .models import DataDjango, DataDjango2
from collections import Counter
import numpy as np
import pandas as pd
from json import dumps
import datetime
import re
import emoji
import string
import operator
import functools
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import texttable as tt



# Create your views here.
def index(request):

    context = {}
    month = []
    year = ['2020']
    
    arrayCreated =[]
    arrayCreatedYear =[]
    arrayCreated2 =[]
    arrayCreatedYear2 =[]

    dayArray = []
    dayArray2 = []

    # open data month
    with open('polls/month.csv', 'r') as file_month:
        for line_month in file_month:
            clear_line_month = line_month.replace("\n", '').strip()
            month.append(clear_line_month) 

    # Emosi Bahagia Maps
    locationsArrayFix = []
    list_data = DataDjango.objects.all()
    for field in list_data:

        # Data Location
        locationsArray= []
        locations = field.locations
        locationsArray.append(locations)

        # Data Created
        created_at = field.created_at
        punctuation_created_at = created_at.split()

        # Day
        dateParse = datetime.datetime.strptime(created_at,'%a %b %d %H:%M:%S +0000 %Y').strftime('%d/%m/%Y')
        dayArray.append(dateParse)

        # Open data provinsi id
        with open('polls/provinsi-id.csv', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').strip()
                province, code_province = clear_line.split(',')
                if province in locationsArray:
                    locationsArrayFix.append(code_province)

        # tokenized data created at and compare to month data and year data
        for tokenized_created_at in punctuation_created_at:
            if tokenized_created_at in month:
                arrayCreated.append(tokenized_created_at)

            if tokenized_created_at in year:
                arrayCreatedYear.append(tokenized_created_at)

    # maps
    counterArray = Counter(locationsArrayFix)
    valueArray = counterArray.values()
    text2 =[]
    for text in counterArray:
        text2.append(text)
    value2 =[]
    for value in valueArray:
        value2.append(value)
    fixVar = [[f,c] for f,c in zip(text2,value2)]
    context["happy_emotion_maps"] = fixVar

    # Bar
    countCreated = Counter(arrayCreated)
    valueArray = countCreated.values()
    valueCreated = []
    for value_created in valueArray:
        valueCreated.append(value_created)
    context["value_happy_bar"] = valueCreated
    monthCreated = []
    for month_created in countCreated:
        monthCreated.append(month_created)
    context["month_happy_bar"] = monthCreated

    # Pie
    countCreatedYear = Counter(arrayCreatedYear)
    valueArrayYear = countCreatedYear.values()
    for value_year_happy in valueArrayYear:
        print("Happy Emotion")

    # Line
    countDay = Counter(dayArray)
    valueArrayDay = countDay.values()
    arrayValueDay = []
    for valueDay in valueArrayDay:
        arrayValueDay.append(valueDay)
    print(arrayValueDay)
    arrayDifference = []
    for x, y in zip(arrayValueDay[0::], arrayValueDay[1::]):
        z = (y - x)
        a = int((z/x)*100)
        arrayDifference.append(a)
    arrayCreatedDay = []
    for dataDay in countDay:
        arrayCreatedDay.append(dataDay)
    context["value_happy_day_line"] = arrayDifference
    context["data_happy_day_line"] = arrayCreatedDay 
 

    ### Emosi Tidak Bahagia Maps ###
    locationsArrayFix2 = []
    list_data2 = DataDjango2.objects.all()
    for field2 in list_data2:

        # Data Location
        locationsArray2= []
        locations2 = field2.locations
        locationsArray2.append(locations2)

        # Data Created
        created_at2 = field2.created_at
        punctuation_created_at2 = created_at2.split()

        # Day
        dateParse2 = datetime.datetime.strptime(created_at2,'%a %b %d %H:%M:%S +0000 %Y').strftime('%d/%m/%Y')
        dayArray2.append(dateParse2)

        with open('polls/provinsi-id.csv', 'r') as file2:
            for line2 in file2:
                clear_line2 = line2.replace("\n", '').strip()
                province2, code_province2 = clear_line2.split(',')
                if province2 in locationsArray2:
                    locationsArrayFix2.append(code_province2)

        # tokenized data created at and compare to month data and year
        for tokenized_created_at2 in punctuation_created_at2:
            if tokenized_created_at2 in month:
                arrayCreated2.append(tokenized_created_at2)

            if tokenized_created_at2 in year:
                arrayCreatedYear2.append(tokenized_created_at)

    # Maps
    counterArray2 = Counter(locationsArrayFix2)
    valueArray2 = counterArray2.values()
    text3 =[]
    for text2 in counterArray2:
        text3.append(text2)
    value3 =[]
    for value2 in valueArray2:
        value3.append(value2)
    fixVar2 = [[f2,c2] for f2,c2 in zip(text3,value3)]
    context["unhappy_emotion_maps"] = fixVar2

    # Bar
    countCreated2 = Counter(arrayCreated2)
    valueArray2 = countCreated2.values()
    valueCreated2 = []
    for value_created2 in valueArray2:
        valueCreated2.append(value_created2)
    context["value_unhappy_bar"] = valueCreated2
    monthCreated2 = []
    for month_created2 in countCreated2:
        monthCreated2.append(month_created2)
    context["month_unhappy_bar"] = monthCreated2

    # Pie
    countCreatedYear2 = Counter(arrayCreatedYear2)
    valueArrayYear2 = countCreatedYear2.values()
    for value_year_unhappy in valueArrayYear2:
        print("Unhappy Emotion")

    total = value_year_happy + value_year_unhappy
    percentHappy = (value_year_happy / total)*100
    percentUnhappy = (value_year_unhappy / total)*100

    context["value_happy_year_pie"] = percentHappy
    context["value_unhappy_year_pie"] = percentUnhappy

    # Line
    countDay2 = Counter(dayArray2)
    valueArrayDay2 = countDay2.values()

    arrayValueDay2 = []
    for valueDay2 in valueArrayDay2:
        arrayValueDay2.append(valueDay2)
    print(arrayValueDay2)
    arrayDifference2 = []
    for x2, y2 in zip(arrayValueDay2[0::], arrayValueDay2[1::]):
        z2 = (y2 - x2)
        a2 = int((z2/x2)*100)
        arrayDifference2.append(a2)
    arrayCreatedDay2 = []
    for dataDay2 in countDay2:
        arrayCreatedDay2.append(dataDay2)
    context["value_unhappy_day_line"] = arrayDifference2
    context["data_unhappy_day_line"] = arrayCreatedDay2

    return render(request, 'polls/index.html',context)

# Function to Clean
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|(_[A-Za-z0-9]+)|(\w+:\/\/\S+)|(\d+)|"
                           "(\s([@#][\w_-]+)|(#\\S+))|((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", " ", tweet).replace(",","").replace(".","").replace("?","").replace("!","").replace("/","").replace("&","").replace(":","").replace("_","").replace("@","").replace("#","").split())


with open('polls/dataset-after-training2.csv', newline='') as csvfile:
    TRAINING_DATA = list(csv.reader(csvfile))


class NaiveBayes:

    def __init__(self, data, vocab):
        self._displayHelper = DisplayHelper(data, vocab)
        self._vocab = vocab

        # LabelArray
        labelArray = []
        for i in range(1, len(data)):
            labelArray.append(data[i][1])
        self._label = np.array(labelArray)

        # docArray
        docArray = []
        for i in range(1, len(data)):
            docArray.append(self.map_doc_to_vocab(data[i][0].split()))
        self._doc = np.array(docArray)
        self.calc_prior_prob().calc_cond_probs()

    def calc_prior_prob(self):
        sum = 0

        # Laplacian Smoothing
        for i in self._label:
            if ("-".__eq__(i)): sum += 1;
        self._priorProb = sum / len(self._label)
        self._displayHelper.set_priors(sum, len(self._label))
        return self

    def calc_cond_probs(self):
        pProbNum = np.ones(len(self._doc[0]));
        nProbNum = np.ones(len(self._doc[0]))
        pProbDenom = len(self._vocab);
        nProbDenom = len(self._vocab)
        for i in range(len(self._doc)):
            if "-".__eq__(self._label[i]):
                nProbNum += self._doc[i]
                nProbDenom += sum(self._doc[i])
            else:
                pProbNum += self._doc[i]
                pProbDenom += sum(self._doc[i])
        self._negProb = np.log(nProbNum / nProbDenom)
        self._posProb = np.log(pProbNum / pProbDenom)
        self._displayHelper.display_calc_cond_probs(nProbNum, pProbNum, nProbDenom, pProbDenom)
        return self

    # Function classify label
    def classify(self, doc):

        global sentiment
        sentiment = "-"
        nLogSums = doc @ self._negProb + np.log(self._priorProb)
        pLogSums = doc @ self._posProb + np.log(1.0 - self._priorProb)
        self._displayHelper.display_classify(doc, pLogSums, nLogSums)
        if pLogSums > nLogSums:
            sentiment = "Emosi Bahagia"

        if pLogSums < nLogSums:
            sentiment = "Emosi Tidak Bahagia"

        if pLogSums == nLogSums:
            sentiment = "Netral"
        return "text classified as (" + sentiment + ") label"


    def map_doc_to_vocab(self, doc):
        mappedDoc = [0] * len(self._vocab)
        for d in doc:
            counter = 0
            for v in self._vocab:
                if (d.__eq__(v)): mappedDoc[counter] += 1
                counter += 1
        return mappedDoc


# Class display
class DisplayHelper:
    def __init__(self, data, vocab):
        self._vocab = vocab
        self.print_training_data(data)

    # print training data table
    def print_training_data(self, data):
        table = tt.Texttable()
        table.header(data[0])
        for i in range(1, data.__len__()): table.add_row(data[i])

        # Print table data training
        # print(table.draw().__str__())

    def set_priors(self, priorNum, priorDenom):
        self._priorNum = priorNum
        self._priorDenom = priorDenom

    def display_classify(self, sentiment, posProb, negProb):

        # N-Gram Feature
        # Happy Label
        temp = "N-Gram Data Training Happy Emotion Label = (" + \
               (self._priorDenom - self._priorNum).__str__() + "/" + self._priorDenom.__str__() + ")"
        for i in range(0, len(sentiment)):
            if sentiment[i] == 1:
                temp = temp
        print(temp)

        # Unhappy Label
        temp = "N-Gram Data Training Unhappy Emotion Label = (" + self._priorNum.__str__() \
               + "/" + self._priorDenom.__str__() + ")"
        for i in range(0, len(sentiment)):
            if sentiment[i] == 1:
                temp = temp
        print(temp)

        # Probabilitas sentiment Naive Bayes Method
        print("Probabilitas of (Happy Emotion) = ", np.exp(posProb))
        print("Probabilitas of (Unhappy Emotion) = ", np.exp(negProb))

    # function to display calculation word probability
    def display_calc_cond_probs(self, nProbNum, pProbNum, nProbDenom, pProbDenom):

        # Array Calculation Unhappy Emotion
        nProb = []
        nProb.append("P(w|Unhappy Emotion)")
        for i in range(0, len(self._vocab)):
            nProb.append((int)(nProbNum[i]).__str__() + "/" + nProbDenom.__str__())

        # Array Calculation Happy Emotion
        pProb = []
        pProb.append("P(w|Happy Emotion)")
        for i in range(0, len(self._vocab)):
            pProb.append((int)(pProbNum[i]).__str__() + "/" + pProbDenom.__str__())

        tempVocab = []
        tempVocab.append("")
        for i in range(0, len(self._vocab)): tempVocab.append(self._vocab[i])

        # Limit row table
        table = tt.Texttable(1000000)
        table.header(tempVocab)
        table.add_row(pProb)
        table.add_row(nProb)

        # print table calculation Frequency data training
        # print(table.draw().__str__())

        self._nProbNum = nProbNum
        self._pProbNum = pProbNum
        self._nProbDenom = nProbDenom
        self._pProbDenom = pProbDenom


# Function input command line
def handle_command_line(nb):
    entryClean = text
    hasil = (nb.classify(np.array(nb.map_doc_to_vocab(entryClean.lower().split()))))
    print("Hasil Klasifikasi :",sentiment)
    
# Prepare data training to lower case
def prepare_data():
    data = []
    for i in range(0, len(TRAINING_DATA)):
        data.append([TRAINING_DATA[i][0].lower(), TRAINING_DATA[i][1]])
    return data


# Split data training beetwen text and label
def prepare_vocab(data):
    vocabSet = set([])
    for i in range(1, len(data)):
        for word in data[i][0].split(): vocabSet.add(word)
    return list(vocabSet)


def checkemotion(request):

    global text

    context2 = {}

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    with open('polls/stopwordsfix.csv', 'r') as file:
        stopwords = []
        for line in file:
            clear_line = line.replace("\n", '').strip()
            stopwords.append(clear_line)

    stopwords_list = []
    after_stopwords = []


    text = request.GET['teks_input']

    # cleaning process
    gas = text.strip()
    blob = clean_tweet(gas)
    print("Text Cleaning :", blob)

    # split text and emoticon
    em_split_emoji = emoji.get_emoji_regexp().split(blob)
    em_split_whitespace = [substr.split() for substr in em_split_emoji]
    em_split = functools.reduce(operator.concat, em_split_whitespace)
    strSplit = ' '.join(em_split)
    print("Text Split Emoticon and Text :", strSplit)

    # lowering case process
    lower_case = strSplit.lower()
    print("Text Lower Case :", lower_case)

    # convert emoticon process
    punctuationText = lower_case.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = punctuationText.split()
    for tokenized_words_emoticon in tokenized_words:
        arrayTokenizingEmoticon = []
        arrayTokenizingEmoticon.append(tokenized_words_emoticon)
        with open('polls/EmojiCategory-People.csv', 'r', encoding='utf-8') as fileEmoticon:
            for lineEmoticon in fileEmoticon:
                clear_line_emoticon = lineEmoticon.replace("\n", '').strip()
                emoticon, convert = clear_line_emoticon.split(',')
                if emoticon in arrayTokenizingEmoticon:
                    # emoticon_detection.append(emoticon)
                    tokenized_words.append(convert)
                    print("Emoticon Convert :", emoticon, "to", convert)
    strEmoticonConvert = ' '.join(tokenized_words)
    print("Text Emoticon Convert :", strEmoticonConvert)

    # stemming process
    hasilStemmer = stemmer.stem(strEmoticonConvert)
    print("Text Stemming :", hasilStemmer)

    # stop words process
    punctuationText2 = hasilStemmer.translate(str.maketrans('', '', string.punctuation))
    tokenized_words2 = punctuationText2.split()
    for tokenized_words3 in tokenized_words2:
        if tokenized_words3 not in stopwords:
            stopwords_list.append(stopwords)
            after_stopwords.append(tokenized_words3)

    strTextFix = ' '.join(after_stopwords)
    print("Text After Stop Words : ", strTextFix)

    entryClean = strTextFix
    
    data = prepare_data()
    handle_command_line(NaiveBayes(data, prepare_vocab(data)))

    print(sentiment)

    context2["output"] = sentiment


    context2["output1"] = text
    return render(request, 'polls/checkemotion.html',context2)



