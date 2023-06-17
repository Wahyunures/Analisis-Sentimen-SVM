from googletrans import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import json

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
translator = Translator()


class Labeling:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
    
    def labeling(self):
        # Membaca file JSON yang berisi kamus kata dengan sentimen dari file '_json_sentiwords_id.txt'
        with open('app/_json_sentiwords_id.txt') as f:
            data2 = f.read()
        sentiment = json.loads(data2)
        sid.lexicon.update(sentiment)

        # Menghitung skor sentimen menggunakan SentimentIntensityAnalyzer dari library nltk
        self.frame['score'] = self.frame['stemming'].apply(lambda x: sid.polarity_scores(str(x)))
    
        def condition(c):
            # Menentukan label sentimen berdasarkan skor sentimen
            if c >= 0.0000:
                return "positif"
            else:
                return 'negatif'

        # Menambahkan kolom 'compound' yang berisi nilai komposit dari skor sentimen
        self.frame['compound'] = self.frame['score'].apply(lambda score_dict: score_dict['compound'])
        
        # Menambahkan kolom 'sentimen' yang berisi label sentimen berdasarkan skor sentimen
        self.frame['sentimen'] = self.frame['compound'].apply(condition)
        
        # Mengubah tipe data kolom 'score' menjadi string
        self.frame["score"] = self.frame["score"].astype(str)

        return self.frame


    # ''' konsep dari labelling dengan based lexicon to indonesia '''

    # Fungsi __init__(self, frame: pd.DataFrame) merupakan konstruktor kelas Labeling yang menerima argumen berupa DataFrame yang akan dilabeli.
    # Fungsi labeling(self) digunakan untuk melakukan proses pelabelan sentimen pada DataFrame yang diberikan.
    # Pada bagian ini, file JSON berisi kamus kata dengan sentimen dibaca dan dimuat ke dalam variabel sentiment menggunakan modul json. Kamus kata ini akan digunakan untuk memperbarui kamus sentimen dalam objek sid.
    # lexicon yang merupakan bagian dari SentimentIntensityAnalyzer dari library nltk.
    # DataFrame yang diberikan memiliki kolom stemming, dan fungsi apply() digunakan untuk mengaplikasikan fungsi lambda pada setiap nilai di kolom tersebut. Fungsi lambda tersebut akan menggunakan sid.polarity_scores() untuk menghitung skor sentimen dari teks yang ada di kolom stemming.
    # Fungsi condition(c) adalah fungsi lambda yang digunakan untuk menentukan label sentimen berdasarkan skor sentimen. Jika skor sentimen lebih besar atau sama dengan 0.0000, maka teks dianggap positif, jika tidak, maka dianggap negatif.
    # Selanjutnya, kolom baru compound ditambahkan ke DataFrame yang berisi nilai komposit dari skor sentimen yang dihasilkan oleh sid.polarity_scores().
    # Kemudian, kolom sentimen ditambahkan ke DataFrame dengan menggunakan fungsi apply() dan fungsi lambda untuk menerapkan fungsi condition(c) ke setiap nilai di kolom compound. Kolom ini akan berisi label sentimen (positif atau negatif) berdasarkan skor sentimen.
    # finally, tipe data kolom score diubah menjadi string menggunakan astype(str).
    # DataFrame yang sudah dilabeli sentimennya kemudian dikembalikan.