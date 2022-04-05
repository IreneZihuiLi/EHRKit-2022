import glob
import os
import logging
from time import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from collections import defaultdict
from gensim import corpora, models, similarities
from pprint import pprint
from gensim.models import Word2Vec
import multiprocessing
import re
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)
def train_word2vec():

    #---loading training data---#
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(__file__)
    #path = '/home/lily/ch956/EHRKit/EHRKit/output/*/*.txt'
    path = os.path.join(cwd,'output/discharge_summary/*.txt') ##temporary, need to organize directory Structure
    print(path)
    files = glob.glob(path)
    raw_docs = []
    for num,name in enumerate(files):
        #using the first 50 documents for training
        if num > 50:
            break
        try:
            with open(name) as f:
                raw_t = f.read()
                raw_docs.append(raw_t)
        except KeyboardInterrupt:
            exit()
        except FileNotFoundError:
            print("File not found!")

        except:
            print("Error occurred!")
    #---preprocessing---#
    ##remove common words and tokenize
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(doc)).lower() for doc in raw_docs)
    t = time()
    texts = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    texts = list(filter(None, texts))

    ## remove words that appear only once
    texts = [row.split() for row in texts]

    #---training---#
    cores = multiprocessing.cpu_count()
    #step 1: setting up params, model still uninitialized
    w2v_model = Word2Vec(min_count=20,window=2,size=300,sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20,workers=cores-1)

    #step 2: building vocab from training texts, initializing model
    t = time()
    w2v_model.build_vocab(texts, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    #step 3:training
    t = time()
    w2v_model.train(texts, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.init_sims(replace=True)

    try:
        os.makedirs(cwd+"/models/")
        print(cwd+'/models/')
        w2v_model.save(cwd+"/models/discharge_model")
    except FileExistsError:
        w2v_model.save(cwd+"/models/discharge_model")
if __name__ == '__main__':
    train_word2vec()
