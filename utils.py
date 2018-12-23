from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from keras.utils import to_categorical

# Data loading
def loadtoDF(filename):
    with open(filename) as f: 
        matrix = [line for line in f]
    matrix = [_.split() for _ in matrix]
    df = pd.DataFrame(matrix, columns=['token', 'POS', "Entity-POS", 'Entity'])
    print("DF Shape: ", df.shape)
    return df

def getTokenLabels(df):
    tokens = df['token'].tolist()
    tokens = list(filter(None, tokens))
    labels = df['Entity'].tolist()
    labels = list(filter(None, labels))
    pos = df['POS'].tolist()
    pos = list(filter(None, pos))
    entitypos = df['Entity-POS'].tolist()
    entitypos = list(filter(None, entitypos))
    tagged_tokens = list(zip(tokens, pos, entitypos, labels))
    return np.array(tagged_tokens)

def data_split(tagged_tokens, num_tokens):
    trunc_num = tagged_tokens.shape[0] % num_tokens
    if trunc_num != 0:
        temp = tagged_tokens[:-trunc_num]
    else:
        temp = tagged_tokens
    temp = np.reshape(temp, [-1, num_tokens, 4])
    sentences = temp[:,:,0]
    poses = temp[:,:,1]
    entityposes = temp[:,:,2]
    sentence_tags = temp[:,:,3]
    token_tr, token_te, pos_tr, pos_te, entpos_tr, entpos_te, labels_tr, labels_te = train_test_split(
        sentences, poses, entityposes, sentence_tags, test_size=0.2)
    split_data = {'token_tr': token_tr, 'token_te': token_te, 
                  'pos_tr': pos_tr, 'pos_te': pos_te, 
                  'entpos_tr': entpos_tr, 'entpos_te': entpos_te, 
                  'labels_tr': labels_tr, 'labels_te': labels_te}
    return split_data

def getWordTagIndexDictionaries(train_sentences, train_poses, train_entposes, train_tags):
    words, poses, entposes, tags = set([]), set([]), set([]), set([])
 
    for s in train_sentences:
        for w in s:
            words.add(w.lower())

    for ps in train_poses:
        for p in ps:
            poses.add(p)
            
    for ps in train_entposes:
        for p in ps:
            entposes.add(p)
            
    for ts in train_tags:
        for t in ts:
            tags.add(t)
            
    word2index = {w: i + 1 for i, w in enumerate(list(words))}
    word2index['-OOV-'] = 0  # The special value used for OOVs

    pos2index = {t: i + 0 for i, t in enumerate(list(poses))}
    entpos2index = {t: i + 0 for i, t in enumerate(list(entposes))}
    
    tag2index = {t: i + 0 for i, t in enumerate(list(tags))}
    
    index2word = dict((v,k) for k,v in word2index.items())
    index2pos = dict((v,k) for k,v in pos2index.items())
    index2entpos = dict((v,k) for k,v in entpos2index.items())
    index2tag = dict((v,k) for k,v in tag2index.items())
    
    dicts = {'word2index': word2index, 
             'pos2index': pos2index, 
             'entpos2index': entpos2index, 
             'tag2index': tag2index, 
             'index2word': index2word, 
             'index2pos': index2pos, 
             'index2entpos': index2entpos, 
             'index2tag': index2tag}
    
    return dicts

def sentenceWord2Index(sentences, word2index):
    indexedSentences = []
    for s in sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        indexedSentences.append(s_int)
    return np.array(indexedSentences)

def sentencePos2Index(poses, pos2index):
    indexedPoses = []
    for s in poses:
        indexedPoses.append([pos2index[t] for t in s])
    return np.array(indexedPoses)

def sentenceEntPos2Index(entposes, entpos2index):
    indexedEntPoses = []
    for s in entposes:
        indexedEntPoses.append([entpos2index[t] for t in s])
    return np.array(indexedEntPoses)

def sentenceTag2Index(tags, tag2index):
    indexedTags = []
    for s in tags:
        indexedTags.append([tag2index[t] for t in s])
    return np.array(indexedTags)

def load_data(filename, num_tokens):
    df = loadtoDF(filename)
    tagged_tokens = getTokenLabels(df)
    split_data = data_split(tagged_tokens, num_tokens)
    indexDicts = getWordTagIndexDictionaries(
                    split_data['token_tr'], split_data['pos_tr'], split_data['entpos_tr'], split_data['labels_tr'])
    
    train_sentences_X = sentenceWord2Index(split_data['token_tr'], indexDicts['word2index'])
    test_sentences_X = sentenceWord2Index(split_data['token_te'], indexDicts['word2index'])

    train_poses = sentencePos2Index(split_data['pos_tr'], indexDicts['pos2index'])
    test_poses = sentencePos2Index(split_data['pos_te'], indexDicts['pos2index'])

    train_entposes = sentenceEntPos2Index(split_data['entpos_tr'], indexDicts['entpos2index'])
    test_entposes = sentenceEntPos2Index(split_data['entpos_te'], indexDicts['entpos2index'])

    train_tags_y = sentenceTag2Index(split_data['labels_tr'], indexDicts['tag2index'])
    test_tags_y = sentenceTag2Index(split_data['labels_te'], indexDicts['tag2index'])
    
    indexedSplitData = {'train_sentences_X': train_sentences_X, 'test_sentences_X': test_sentences_X,
                     'train_poses': train_poses, 'test_poses': test_poses,
                     'train_entposes': train_entposes, 'test_entposes': test_entposes,
                     'train_tags_y': train_tags_y, 'test_tags_y': test_tags_y}
    
    return indexedSplitData, indexDicts

# For model
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

