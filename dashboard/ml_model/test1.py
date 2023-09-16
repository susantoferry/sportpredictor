#Initialize Bert tokenizer and masks
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import transformers
from nltk.stem import WordNetLemmatizer
import plotly.express as px
from tqdm import tqdm
import regex as re
from keras import backend as K
from transformers import BertTokenizer
from keras.utils import pad_sequences

bert_model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
MAX_LEN = 512

def create_model():
    input_word_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32,
                                           name="input_word_ids")
    bert_layer = transformers.TFBertModel.from_pretrained('bert-base-uncased')
    bert_outputs = bert_layer(input_word_ids)[0]
    pred = tf.keras.layers.Dense(16, activation='softmax')(bert_outputs[:,0,:])

    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00002), metrics=['accuracy'])
    return model

def tokenize_sentences(sentences, tokenizer, max_seq_len = MAX_LEN):
    tokenized_sentences = []
    truncation_strategy = 'only_first'

    for sentence in tqdm(sentences):
        tokenized_sentence = tokenizer.encode(
                            sentence,                  # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_len,  # Truncate all sentences.
                            truncation=truncation_strategy
                    )

        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences