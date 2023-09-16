from django.shortcuts import render, redirect
from .forms import DataForm
from .models import Data
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
import torch
import pickle
import os
from django.conf import settings
import tensorflow as tf
from keras.models import load_model
from transformers import TFBertModel, BertTokenizer
from keras.utils import pad_sequences
import numpy as np
import h5py
import pandas as pd
from .ml_model.test1 import *
from keras.utils import pad_sequences
from tqdm import tqdm

# bert_model_name = 'bert-base-uncased'

# tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = DataForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('dashboard-predictions')
    else:

        cols = ['INFJ',
                'ENTP',
                'INTP',
                'INTJ',
                'ENTJ',
                'ENFJ',
                'INFP',
                'ENFP',
                'ISFP',
                'ISTP',
                'ISFJ',
                'ISTJ',
                'ESTP',
                'ESFP',
                'ESTJ',
                'ESFJ']

        colnames = ['sentence']
        colnames = colnames+cols

        

        # with tf.keras.utils.custom_object_scope({"TFBertModel": TFBertModel}):
        #     model = load_model("dashboard/ml_model/bert_uncased_model_new.h5")

        model = create_model()
        
        bert_model_name = 'bert-base-uncased'

        tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
        
        df_predict = pd.DataFrame(columns = colnames)
        sentence = "Hello World"

        df_predict.loc[0, 'sentence'] = sentence

        # sentences = pd.Series(sentences)
        sentence_inputs = tokenize_sentences(df_predict['sentence'], tokenizer, 512)
        
        sentence_inputs = pad_sequences(sentence_inputs, maxlen=512, dtype="long", value=0, truncating="post", padding="post")
        prediction = model.predict(sentence_inputs)
        df_predict.loc[0, cols] = prediction
        data = df_predict.loc[0, cols].to_dict()
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        print(df_predict)
        print(df_predict.loc[0, cols])
        # Get the top 3 values
        top_3_values = sorted_data[:3]

        # Print the top 3 values
        # for key, value in top_3_values:
        #     print(f"{key}: {round(value,2) * 100}")


    return render(request, 'dashboard/index.html')


def predictions(request):
    predicted_sports = Data.objects.all()
    context = {
        'predicted_sports': predicted_sports
    }
    return render(request, 'dashboard/predictions.html', context)