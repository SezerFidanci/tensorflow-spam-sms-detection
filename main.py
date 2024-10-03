import re

import numpy as np
import pandas as pd
import nltk
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.python.keras.layers import Embedding, LSTM, Dense

from tensorflow.python.keras.models import Sequential, load_model

nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split

WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')

dataset = pd.read_csv('./TurkishSMSCollection.csv',delimiter=';')
dataset['Message'] = dataset['Message'].apply(lambda x: re.sub('[,\.!?:()"]', '', x))
dataset['Message'] = dataset['Message'].apply(lambda x: x.lower())
dataset['Message'] = dataset['Message'].apply(lambda x: x.strip())

def token(values):
    words = nltk.tokenize.word_tokenize(values)
    filtered_words = [word for word in words if word not in stop_word_list]
    not_stopword_doc = " ".join(filtered_words)
    return not_stopword_doc

dataset['Message'] = dataset['Message'].apply(lambda x: token(x))

messages = dataset['Message'].values.tolist()
group = dataset['Group'].values.tolist()

x_train, x_test, y_train, y_test = train_test_split(messages, group, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(messages)


x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)


have_a_model = True
MODEL_PATH = 'spam_model.h5'
if have_a_model:
    model = load_model(MODEL_PATH)
    print("Found a model!")
else:
    model = Sequential()
    embedding_size = 50
    model.add(Embedding(input_dim=10000,
                        output_dim=embedding_size,
                        input_length=max_tokens,
                        name='embedding_layer'))

    model.add(LSTM(units=16, return_sequences=True))
    model.add(LSTM(units=8, return_sequences=True))
    model.add(LSTM(units=4))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x=x_train_pad, y=np.array(y_train), validation_split=0.25, epochs=8, batch_size=256)
    model.save(MODEL_PATH)
    model.summary()

result = model.evaluate(x_test_pad, np.array(y_test))


texts = [
    "DIGITURKTEN FIRSAT! SiZE OZEL YIL SONUNA KADAR 200 UZERi ULUSAL HD YAYIN BELGESEL+SPOR+DiZi+YETiSKiN iLK AY LiG TV AYDA SADECE 10 TL.HEMEN ARAYIN 02122129070",
    "TAKIMININ YILDIZLARI GOL ATSIN DOGOBET ILE KAZANCIN ARTSIN! DERBIYE OZEL ORAN VE BONUSLAR SENI BEKLIYOR! https://t2m.io/ZWh75Za",
    "selam nerdesin",
    "yarın cafeye gelecek misin",
    "Acilisa Ozel Uye olan Herkese 1000TL Hemen gel Gun Sonuna Kadar Gecerli firsati kacirma! Limitsiz cekim Bayspin de Detaylar : https://cutt.ly/4wOAvL5p",
    "Denizli tabanli teknoloji girisimi Uzum Teknoloji kitle fonlama kampanyasi ile Fongogo'da yatirim ariyor! Basariyla tamamlanan 2 kampanyanin ardindan bir girisim daha Fongogo'da yayinda! Bagimsiz degerleme: 86 Milyon TL Fongogo kampanya degerlemesi: 70 Milyon TL Hedef Fon: 4.200.000 TL 25.000 TL uzeri yatirimlara %10 bonus pay. Hemen yatirima baslayin! https://yatirim.fongogo.com/Project/ProjectList B356",
    "Degerli musterimiz, bakiyenize 175 TL yuklediniz. Iyi gunlerde kullanin. 14.12.2023 09:47 tarihine kadar 20;40;60 TL alt limitinde ikinci yukleme yapabilmek icin http://vftr.co/liratopup i ziyaret edebilirsiniz. B003",
    "Ihtiyaclariniz icin avantajli faiz oraniyla ve dosya masrafi odemeden 200.000 TL'ye kadar kredi kullanmak isterseniz cep subemizin Krediler adimindan hemen bir basvuru yapabilirsiniz! Tanitim SMS'leri almak istemiyorsaniz IPTAL yazip 6770'e gonderebilirsiniz. Mersis: 0388002333400576 B016",

]
tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
predictions = model.predict(tokens_pad)

for i in range(0,len(predictions)):
    print("(Spam tahmini {}) {}".format(predictions[i][0],texts[i]))
