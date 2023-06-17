from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import numpy as np
import nltk
import string
import re

factory = StemmerFactory()
stemmer = factory.create_stemmer()

class Process:
    # Mengambil daftar stopwords dalam bahasa Indonesia dari NLTK
    LIST_STOPWORDS = stopwords.words('indonesian')

    # Menambahkan stopwords tambahan yang umum dalam teks bahasa Indonesia
    LIST_STOPWORDS.extend(['yg', 'dg', 'rt', 'dgn', 'ny', 'gt', 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&', 'yah', 'no', 'je', 'om', 'pru', 'sch',
                        'injirrr', 'ah', 'oena', 'bu', 'eh', 'xac', 'anjir'])

    LIST_STOPWORDS = set(LIST_STOPWORDS)    
    
    def __init__(self,frame: pd.DataFrame):
        # Mendownload tokenizer dari nltk
        nltk.download('punkt')
        self.frame = frame
    
    def remove_pattern(self,text, pattern):
        # Menghapus pola tertentu dalam teks menggunakan regex
        r = re.findall(pattern, str(text))
        for i in r:
            text = re.sub(i, '', str(text))
        return text
    
    def cleaning(self,text):
        # Membersihkan teks dengan melakukan beberapa substitusi dan penghapusan karakter khusus
        text = re.sub(r'\$\w*', '', text)
        text = re.sub(r'^rt[\s]+', '', text)
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
        text = re.sub('&quot;'," ", text)
        text = re.sub(r"\d+", " ", str(text))
        text = re.sub(r"\b[a-zA-Z]\b", "", str(text))
        text = re.sub(r"[^\w\s]", " ", str(text))
        text = re.sub(r'(.)\1+', r'\1\1', text)
        text = re.sub(r"\s+", " ", str(text))
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-zA-z0-9]', ' ', str(text))
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'\s\s+', ' ', text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'^b[\s]+', '', text)
        text = re.sub(r'^link[\s]+', '', text)
        return text

    def word_tokenize_wrapper(self,text):
        # Membungkus fungsi word_tokenize() dari nltk untuk digunakan dalam dataframe
        return word_tokenize(text)
    

    def stopwords_removal(self,text):
        # Menghapus stopwords dari teks
        return [word for word in text if word not in self.LIST_STOPWORDS]
    
    def stemming(self,text):
        term_dict = {}

        for text in self.frame['stop_words']:
            for term in text:
                if term not in term_dict:        
                    term_dict[term] = ' '

        for term in term_dict:
            term_dict[term] = stemmer.stem(term)

        text = [term_dict[term] for term in text]
        return ' '.join(text)

    def fit_stemming(self,text):
        text = np.array(text)
        text = ' '.join(text)

        return text

    def process(self):

        self.frame['remove_user'] = np.vectorize(self.remove_pattern)(self.frame['text'], "@[\w]*")
        self.frame['text_cleaning'] = self.frame['remove_user'].apply(self.cleaning)
        self.frame['case_folding'] = self.frame['text_cleaning'].str.lower()
        self.frame['tokenizing'] = self.frame['case_folding'].apply(lambda x: self.word_tokenize_wrapper(x.lower()))
        self.frame['stop_words'] = self.frame['tokenizing'].apply(self.stopwords_removal)    

        def stemmed_wrapper(term):
            return stemmer.stem(term)

        term_dict = {}

        for Text in self.frame['stop_words']:
            for term in Text:
                if term not in term_dict:
                    term_dict[term] = ' '

        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)

        # Memulai stemming
        def apply_stemmed_term(Text):
            return [term_dict[term] for term in Text]

        self.frame['stemming'] = self.frame['stop_words'].apply(apply_stemmed_term)

        def fit_stemming(text):
            text = np.array(text)
            text = ' '.join(text)
    
            return text

        self.frame['stemming'] = self.frame['stemming'].apply(lambda x: fit_stemming(x))
        self.frame.drop_duplicates(subset = "stemming", keep = 'first', inplace = True)

        return self.frame
