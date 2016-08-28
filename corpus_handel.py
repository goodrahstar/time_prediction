# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:02:45 2016

@author: rahulkumar
"""

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'", "", string)
    return string.strip().lower()


def load_data_and_labels(data):
    """
    Load data from the provided dataframe
    Returns splitted words for each sentences.
    """    
    x_text =[]
 
    x_text = list(data['destinationbranch'] + ' ' +data['originbranch'])
                     
    data = [clean_str(str(sent)) for sent in x_text] 
    data = [s.split(" ") for s in data]     
    
    return [data]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences) + 1
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]  
    vocabulary_inv = list(sorted(vocabulary_inv))  
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences], dtype = 'float')
    return [x]


def load_data(data):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    
    sentences = load_data_and_labels(data) 
    sentences = sentences[0]
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentences_padded, vocabulary)
    return [x, vocabulary, vocabulary_inv] 




#x_test, vocabulary, vocabulary_inv = load_data()
