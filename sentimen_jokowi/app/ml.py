from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from nltk import word_tokenize
from numpy import set_printoptions
from sys import maxsize
import pandas as pd
import numpy as np
set_printoptions(threshold=maxsize)

label_encoder = LabelEncoder()

class Classification:
    sentimen = None
    tfidf_transformer = None
    df_idf :pd.DataFrame = []
    tf_idf :pd.DataFrame = []
    tf_df_idf = None
    word_count = None
    predicted_svm = None
    report = None
    score_svm = None
    confussion_matrix = None
    model_svm = None
    pred_svm = None
    test_svm = None
    testing_svm = None
    fix_idf = None
    calc_df = None

    def __init__(self,frame: pd.DataFrame):
        self.frame = frame
        self.frame["polarity"] = label_encoder.fit_transform(self.frame["sentimen"])
        self.sentimen = self.frame["sentimen"].value_counts()

        x = self.frame["stemming"]
        y = self.frame["polarity"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1396) 

        text = self.frame['stemming']

        #instantiate CountVectorizer() 
        cv = CountVectorizer() 

        # this steps generates word counts for the words in your docs 
        word_count_vector = cv.fit_transform(self.frame["stemming"])

        #IDF transformer
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
        self.tfidf_transformer.fit(word_count_vector)


        #Assigning CountVectorizer to variable
        tfidf = TfidfVectorizer().fit(self.frame.apply(lambda x: np.str_(x)))

        tf_idf_vector = self.tfidf_transformer.fit_transform(word_count_vector)
        first_document_vector = tf_idf_vector[1]
    
        tfidf_train = tfidf.fit_transform(x_train).toarray()
        tfidf_test = tfidf.transform(x_test).toarray()

        self.model_svm = SVC(kernel='linear').fit(tfidf_train, y_train) 
        self.predicted_svm = self.model_svm.predict(tfidf_test)
        
        self.pred_svm = np.array(self.predicted_svm)
        self.test_svm = np.array(y_test)
        self.testing_svm = np.array(x_train)
        
        self.report  = classification_report(y_test, self.predicted_svm,output_dict=True)
        self.score_svm = accuracy_score(self.predicted_svm, y_test)
        self.confussion_matrix =  confusion_matrix(y_test, self.predicted_svm)

        self.frame["tokenized_stemming"] = self.frame["stemming"].apply(self.tokenize_with_quotes)
        self.frame["TF_dict"] = self.frame["tokenized_stemming"].apply(self.calc_TF)

        self.calc_df = self.calc_DF(self.frame["TF_dict"])
        n_document = len(self.frame)
        IDF = self.calc_IDF(n_document, self.calc_df)
        
        sorted_DF = sorted(self.calc_df.items(), key=lambda kv: kv[1], reverse=True)
        unique_term = [item[0] for item in sorted_DF]

        self.frame["TF-IDF_dict"] = self.frame["TF_dict"].apply(self.calc_TF_IDF,args=(IDF,))
        self.frame["TF_IDF_Vec"] = self.frame["TF-IDF_dict"].apply(self.calc_TF_IDF_Vec,args=(unique_term,))

        
        TF_IDF_Vec_List = np.array(self.frame["TF_IDF_Vec"].to_list())

        # Sum element vector in axis=0
        sums = TF_IDF_Vec_List.sum(axis=0)

        self.tf_df_idf = pd.DataFrame(self.tfidf_transformer.idf_, columns=["idf"])
        self.tf_df_idf["words"] = cv.get_feature_names_out()
        self.tf_df_idf["term-frequency"] = word_count_vector.toarray().sum(axis=0)
        self.tf_df_idf["tfidf"] = sums
        self.tf_df_idf = self.tf_df_idf.reindex(columns=["words","term-frequency","idf","tfidf"])

    def tokenize_with_quotes(self,text):
        tokens = word_tokenize(text)
        tokens_with_quotes = ['"' + token + '"' for token in tokens]
        return tokens_with_quotes

    def calc_TF(self,document):
        TF_dict = {}
        for term in document:
            if term in TF_dict:
                TF_dict[term] += 1
            else:
                TF_dict[term] = 1
        # Computes tf for each word
        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(document)
        return TF_dict

    def calc_DF(self,tfDict):
        count_DF = {}
        # Run through each document's tf dictionary and increment countDict's (term, doc) pair
        for document in tfDict:
            for term in document:
                if term in count_DF:
                    count_DF[term] += 1
                else:
                    count_DF[term] = 1
        return count_DF
    
    def calc_IDF(self,__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict
    
    def calc_TF_IDF(self,TF,IDF):
        TF_IDF_Dict = {}
        #For each word in the review, we multiply its tf and its idf.
        for key in TF:
            TF_IDF_Dict[key] = TF[key] * IDF[key]
        return TF_IDF_Dict
    
    def calc_TF_IDF_Vec(self,__TF_IDF_Dict,unique_term):
        TF_IDF_vector = [0.0] * len(unique_term)

        # For each unique word, if it is in the review, store its TF-IDF value.
        for i, term in enumerate(unique_term):
            if term in __TF_IDF_Dict:
                TF_IDF_vector[i] = __TF_IDF_Dict[term]
        return TF_IDF_vector