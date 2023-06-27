import pandas as pd
from collections import Counter
import numpy as np

# create dictionary
def Make_dict(data_train_content: pd.DataFrame):
    all_word: list = []
    for content in data_train_content:
        word = str(content).split()
        all_word += word
    dictionary = Counter(all_word)
    list_dict_remove  = dictionary.keys()
    for key in list(list_dict_remove):
        if key.isalpha() == False:
            del dictionary[key]
        elif len(key) == 1:
            del dictionary[key]
    dictionary = dictionary.most_common(3000)
    return dictionary

# create matrix, label
def extract_features(data_train: pd.DataFrame):
    feature_matrix = np.zeros((data_train.shape[0], 3000))
    train_label = np.array(list(data_train['Label']))
    docId = 0
    for content in data_train['Content']:
        content_used: str = ""
        list_line = list(str(content).split('\n'))
        for i in range(1, len(list_line)):
            content_used += list_line[i]
        words = content_used.split()
        for word in words:
            wordId = 0
            for i, dic in enumerate(dictionary):
                if dic[0] == word:
                    wordId = i
                    feature_matrix[docId, wordId] = words.count(word)
        docId += 1
    return feature_matrix, train_label