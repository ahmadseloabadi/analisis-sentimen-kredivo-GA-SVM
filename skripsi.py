#import library
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import re
import numpy as np
import random as rd
import seaborn as sns

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

import Genetic_Algorithm as svm_hp_opt
import base64

from google_play_scraper import Sort, reviews_all, app

# Set page layout and title
st.set_page_config(page_title="Kredivoo", page_icon="img/style/logo.png")

#custom style
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }

    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('img/style/backgroundd.png')

def sidebar(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size: contain;
          background
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
sidebar('img/style/sidebar.png')
st.markdown(f"""
      <style>
      .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3, .css-1629p8f h4, .css-1629p8f h5, .css-1629p8f h6, .css-1629p8f span {{
          background-color: white;
          color: black;
          border-radius: 20px;
          text-align: center;
          
      }}
      .css-184tjsw p , .css-1offfwp p{{
          background-color: white;
          color: black;
          border-radius: 10px;
          padding:8px;
      }}
      .css-18ni7ap{{
          background: rgb(0,0,0,0.1);
           }}
      </style>
      """,
      unsafe_allow_html=True,)

#import dataset
ulasan = pd.read_csv('data/dataset/dataset_bersih.csv')
ulasan=ulasan.dropna()

vectorizer = TfidfVectorizer()
Encoder = LabelEncoder()

#scrapping google play store
def scrapping_play_store():
    result = reviews_all(
    'com.finaccel.android',
    sleep_milliseconds=0, # defaults to 0
    lang='id', # defaults to 'en'
    country='id', # defaults to 'us'
    sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT , you can use Sort.NEWEST to get newst reviews
    )
    df_busu = pd.DataFrame(np.array(result),columns=['review'])
    df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist()))
    new_df = df_busu[['userName', 'score','at', 'content']]
    sorted_df = new_df.sort_values(by='at', ascending=False)
    sorted_df.to_csv("scraping_data_kredivo.csv", index = False) 
    return sorted_df

def create_sentiment_column(dataset_path):
    count=False
    # Load dataset
    data = pd.read_csv(dataset_path)
    st.write(data)
    count_edit=st.number_input('masukan banyak data yang akan dilabeli',step=1,value=len(data)-1,min_value=1,max_value=len(data)-1)
    if st.button("menambahkan kolom sentimen"):
        data['sentimen'] = ''
        
    
    with st.form("my_form"):
        st.write('pastikan penyimpan dengan cara mendownload hasil labeling')        
        # Display the selected rows for editing
        edited_df = data.loc[:count_edit].copy()
        st.write('silahkan labeli data :)')
        edited = st.data_editor(edited_df, num_rows="dynamic",column_config={
        "sentimen": st.column_config.SelectboxColumn(
            "sentimen",
            help="kelas sentimen",
            width="medium",
            options=[
                "positif",
                "netral",
                "negatif",
            ],        
        )},)
        # Every form must have a submit button.
        submitted = st.form_submit_button("selesai")    
        if submitted:
            st.write('terima kasih :)')


# text preprosessing
def cleansing(kalimat_baru): 
    kalimat_baru = re.sub(r'@[A-Za-a0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r'#[A-Za-z0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"http\S+",' ',kalimat_baru)
    kalimat_baru = re.sub(r'[0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", kalimat_baru)
    kalimat_baru = re.sub(r"\b[a-zA-Z]\b", " ", kalimat_baru)
    kalimat_baru = kalimat_baru.strip(' ')
    # menghilangkan emoji
    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    kalimat_baru =clearEmoji(kalimat_baru)
    def replaceTOM(ulasan):
        pola = re.compile(r'(.)\1{2,}', re.DOTALL)
        return pola.sub(r'\1', ulasan)
    kalimat_baru=replaceTOM(kalimat_baru)
    return kalimat_baru
def casefolding(kalimat_baru):
    kalimat_baru = kalimat_baru.lower()
    return kalimat_baru
def tokenizing(kalimat_baru):
    kalimat_baru = word_tokenize(kalimat_baru)
    return kalimat_baru
def slangword (kalimat_baru):
    kamusSlang = eval(open("data/kamus/slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in kalimat_baru:
        filter_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata.lower())
        if filter_slang.startswith('tidak_'):
          kata_depan = 'tidak_'
          kata_belakang = kata[6:]
          kata_belakang_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata_belakang.lower())
          kata_hasil = kata_depan + kata_belakang_slang
          content.append(kata_hasil)
        else:
          content.append(filter_slang)
    kalimat_baru = content
    return kalimat_baru
def handle_negation(kalimat_baru):
    negation_words = ["tidak", "bukan", "tak", "tiada", "jangan", "gak",'ga']
    new_words = []
    prev_word_is_negation = False
    for word in kalimat_baru:
        if word in negation_words:
            new_words.append("tidak_")
            prev_word_is_negation = True
        elif prev_word_is_negation:
            new_words[-1] += word
            prev_word_is_negation = False
        else:
            new_words.append(word)
    return new_words
def stopword (kalimat_baru):
    daftar_stopword = stopwords.words('indonesian')
    daftar_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d",'gb','ahk','g','anjing','ga','gua','nder']) 
    # Membaca file teks stopword menggunakan pandas
    txt_stopword = pd.read_csv("data/kamus/stopwords.txt", names=["stopwords"], header=None)

    # Menggabungkan daftar stopword dari NLTK dengan daftar stopword dari file teks
    daftar_stopword.extend(txt_stopword['stopwords'].tolist())

    # Mengubah daftar stopword menjadi set untuk pencarian yang lebih efisien
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        cleaned_words = []
        for word in words:
            # Memisahkan kata dengan tambahan "tidak_"
            if word.startswith("tidak_"):
                cleaned_words.append(word[:5])
                cleaned_words.append(word[6:])
            elif word not in daftar_stopword:
                cleaned_words.append(word)
        return cleaned_words
    kalimat_baru = stopwordText(kalimat_baru)
    return kalimat_baru 
def stemming(kalimat_baru):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Lakukan stemming pada setiap kata
    stemmed_words = [stemmer.stem(word) for word in kalimat_baru]
    return stemmed_words
vectorizer = TfidfVectorizer()

def output_tfidf(dataset):
    # Create CountVectorizer instance
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(dataset)

    # Create TfidfTransformer instance
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_count)

    # Create TfidfVectorizer instance
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf_vectorized = tfidf_vectorizer.fit_transform(dataset)

    # Get the feature names from CountVectorizer or TfidfVectorizer
    feature_names = count_vectorizer.get_feature_names_out()  # or tfidf_vectorizer.get_feature_names()

    # Create a dictionary to store the results
    results = {"Ulasan": [], "Term": [], "TF": [], "IDF": [], "TF-IDF": []}

    # Loop over the documents
    for i in range(len(dataset)):
        # Add the document to the results dictionary
        results["Ulasan"].extend([f" ulasan{i+1}"] * len(feature_names))
        # Add the feature names to the results dictionary
        results["Term"].extend(feature_names)
        # Calculate the TF, IDF, and TF-IDF for each feature in the document
        for j, feature in enumerate(feature_names):
            tf = X_count[i, j]
            idf = tfidf_transformer.idf_[j]  # or X_tfidf_vectorized.idf_[j]
            tf_idf_score = X_tfidf[i, j]  # or X_tfidf_vectorized[i, j]
            # Add the results to the dictionary
            results["TF"].append(tf)
            results["IDF"].append(idf)
            results["TF-IDF"].append(tf_idf_score)
    # Convert the results dictionary to a Pandas dataframe
    df = pd.DataFrame(results)

    #filter nilai term
    newdf = df[(df.TF != 0 )]
    # Save the results to a CSV file
    newdf.to_csv("hasil_TF_IDF.csv", index=False)
    return newdf

def pembagian_data(kolom_ulasan,kolom_label):
    x=kolom_ulasan
    y=kolom_label
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)
    return X_train, X_test, Y_train, Y_test

def pembobotan_tfidf(X_train, X_test, Y_train, Y_test):
    
    x_train = vectorizer.fit_transform(X_train)
    x_test = vectorizer.transform(X_test)
    
    y_train = Encoder.fit_transform(Y_train)
    y_test = Encoder.fit_transform(Y_test)
    return y_train, y_test,x_train, x_test

def model_svm(C,gamma,x_train,y_train):
    if C is not None:
        modelsvm = svm.SVC(C=C,gamma=gamma)
    else:
        modelsvm = svm.SVC()
    modelsvm.fit(x_train, y_train)
    return modelsvm

def output_dataset(dataset,kolom_ulasan,kolom_label):
    X_train, X_test, Y_train, Y_test=pembagian_data(kolom_ulasan,kolom_label)
    # Mengambil tanggal paling kecil dan paling besar
    tanggal_terkecil = datetime.strptime(dataset['at'].min(), '%Y-%m-%d %H:%M:%S')
    tanggal_terbesar = datetime.strptime(dataset['at'].max(), '%Y-%m-%d %H:%M:%S')

    st.title('Dataset ulasan aplikasi KREDIVO')
    st.write(f'Dataset ulasan aplikasi Kredivoo didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tangga {tanggal_terkecil.date()} hingga {tanggal_terbesar.date()} dengan jumlah ulasan sebanyak {len(dataset)}')

    st.write('berikut merupakan link pada google play store untuk  aplikasi [kredivoo](https://play.google.com/store/apps/details?id=com.finaccel.android&hl=id&showAllReviews=true) ')

    st.subheader('Tabel dataset ulasan palikasi kredivo')
    def filter_sentiment(dataset, selected_sentiment):
        return dataset[dataset['sentimen'].isin(selected_sentiment)]

    sentiment_map = {'positif': 'positif', 'negatif': 'negatif', 'netral': 'netral'}
    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
    filtered_data = filter_sentiment(dataset, selected_sentiment)
    st.dataframe(filtered_data)

    # Hitung jumlah kelas dataset
    st.write("Jumlah kelas sentimen:  ")
    kelas_sentimen = dataset['sentimen'].value_counts()
    # st.write(kelas_sentimen)
    datneg,datnet, datpos  = st.columns(3)
    with datpos:
        st.markdown("Positif")
        st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
    with datnet:
        st.markdown("Netral")
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kelas_sentimen[2]}</h1>", unsafe_allow_html=True)
    with datneg:
        st.markdown("Negatif")
        st.markdown(f"<h1 style='text-align: center; color: aqua;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
    #membuat diagram
    data = {'sentimen': ['negatif', 'netral', 'positif'],
    'jumlah': [kelas_sentimen[1], kelas_sentimen[2], kelas_sentimen[0]]}
    datasett = pd.DataFrame(data)
    # Membuat diagram pie interaktif
    fig = px.pie(datasett, values='jumlah', names='sentimen', title='Diagram kelas sentimen')
    st.plotly_chart(fig)
    with st.expander('pembagian dataset') :
        st.write(f"pembagian dataset dilakukan dengan skala 80:20, dimana 80%  menjadi data training sedangkan 20% menjadi data testing dari total dataset yaitu {len(dataset)}")
        st.write(f'Jumlah data training sebanyak {len(X_train)} data ,data training dapat dilihat pada tabel berikut')
        datatrain=pd.concat([X_train, Y_train], axis=1)
        st.dataframe(datatrain)
        st.write(f'Jumlah data testing sebanyak {len(X_test)} data,data testing dapat dilihat pada tabel berikut')
        datatest=pd.concat([X_test, Y_test], axis=1)
        st.dataframe(datatest)

kfold=5
kolom_ulasan =ulasan['Stopword Removal']
kolom_label =ulasan['sentimen']
X_train, X_test, Y_train, Y_test=pembagian_data(kolom_ulasan,kolom_label)
y_train, y_test,x_train, x_test=pembobotan_tfidf(X_train, X_test, Y_train, Y_test)

def class_report(model,x_test,y_test):
    # Using the model to predict the labels of the test data
    y_pred = model.predict(x_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write('plot confusion matrix')
    # Membuat plot matriks kebingungan
    f, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(f)
#side bar
with st.sidebar :
    selected = option_menu('sentimen analisis',['Home','Text preprocessing','Algoritma Genetika','Pengujian','Report'])

if(selected == 'Home') :
    tab1,tab2=st.tabs(['Main','Scrapping'])
    with tab1 :
        st.title('OPTIMASI PARAMETER METODE SUPPORT VECTOR MACHINE DENGAN ALGORITMA GENETIKA PADA ULASAN APLIKASI KREDIVO ')
        st.write('Diantara banyaknya aplikasi kredit dan pinjaman online ataupun pinjaman uang tunai yang ada di Indonesia, salah satu yang paling banyak diunduh dan diminati oleh masyarakat Indonesia adalah aplikasi Kredivo. Kredivo termasuk pelopor aplikasi kredit dan pinjaman online di Indonesia dimana aplikasi tersebut mendeklarasikan diri sebagai aplikasi dengan solusi kredit yang memberikan kemudahan untuk melakukan pembayaran')
        image = Image.open('img/home/Kredivo.jpeg')
        st.image(image)
        st.subheader('support vector machine')
        st.write('Support Vector Machine (SVM) merupakan algoritma terbaik diantara beberapa algoritma lainnya seperti Naïve Bayes, Decision Trees, dan Random Forest karena mampu mengkomputasi data dengan dimensi tinggi sehingga tingkat akurasi yang dihasilkan lebih baik (Andrean, 2024). Metode SVM juga memiliki kemampuan untuk mengatasi overfitting dan tidak membutuhkan data yang terlalu besar dengan hasil akurasi tinggi serta tingkat kesalahan yang relatif kecil (Harafani & Maulana, 2019) Selain banyaknya kelebihan yang dimiliki, metode SVM masih memiliki kelemahan yaitu sulitnya menentukan parameter yang optimal (Istiadi & Rahman, 2020). Jika menggunakan parameter default hasil akurasi dan klasifikasi SVM tidak akan semaksimal apabila menggunakan pemilihan parameter yang tepat.')
        st.image('img/home/svm.png')
        st.subheader('Algoritma Genetika')
        st.write('Algoritma Genetika adalah metode heuristik yang dikembangkan berdasarkan prinsip-prinsip genetika dan proses seleksi alam teori evolusi Darwin. Algoritma Genetika memiliki metode yang dapat menangani masalah optimasi nonlinier yang berdimensi tinggi dan sangat berguna untuk diimplementasikan pada saat range terbaik dari parameter SVM sama sekali tidak diketahui. Algoritma Genetika dapat menghasilkan parameter SVM yang optimal secara bersamaan. Tujuan utama algoritma ini yaitu mendapatkan populasi baru yang lebih baik dibandingkan populasi sebelumnya (Kusumaningrum, 2017).')
        st.image('img/home/GA.jpg')
    with tab2:
        st.write('scrapping data ulasan aplikasi kredivo pada google play store')
        if st.button('start scrapping') :
            hasilscraping=scrapping_play_store()
            # Mengambil tanggal paling kecil dan paling besar
            scraping = pd.read_csv('scraping_data_kredivo.csv')
            tanggal_terkecil = datetime.strptime(scraping['at'].min(), '%Y-%m-%d %H:%M:%S')
            tanggal_terbesar = datetime.strptime(scraping['at'].max(), '%Y-%m-%d %H:%M:%S')
            
            st.write(f'hasil scraping ulasan aplikasi Kredivoo didapatkan dari scrapping pada google play store dengan jarak data yang diambil pada tangga {tanggal_terkecil.date()} hingga {tanggal_terbesar.date()} dengan jumlah ulasan sebanyak {len(scraping)}')

            st.dataframe(scraping)
elif(selected == 'Text preprocessing') :
    tab1,tab2,tab3=st.tabs(['Dataset','Labeling','Text preprosesing'])
    with tab1 :
        opsi_dataset = st.selectbox('option',('statis', 'dinamis'),key="datasetoption")
        
        if(opsi_dataset == 'statis') :
            dataset = pd.read_csv('data/dataset/dataset_ulasan_kredivoo.csv')
            kolom_ulasan =ulasan['Stopword Removal']
            kolom_label =ulasan['sentimen']
            output_dataset(dataset,kolom_ulasan,kolom_label)
            
        if(opsi_dataset == 'dinamis') :
            file_labeling = st.file_uploader("masukan data yg sudah dilabeli", key="datasettt", type='csv')
            if file_labeling is not None:
                file = pd.read_csv(file_labeling)
                st.dataframe(file)
                ulasan = st.text_input('masukan nama kolom hasil stopword removal',value="Stopword Removal")
                label = st.text_input('masukan nama kolom labeling data',value="sentimen")
                if st.button("tampilkan dataset"):
                    kolom_ulasan =file[ulasan]
                    kolom_label =file[label]
                    output_dataset(file,kolom_ulasan,kolom_label)
            
    with tab2 :
        file_labeling = st.file_uploader("masukan data yang akan dilabeli", key="labeling", type='csv')
        if file_labeling is not None:
            create_sentiment_column(file_labeling)
            
            
    with tab3 :
        opsi_prepro = st.selectbox('option',('statis', 'dinamis'),key="preprocessing")
        prepro=False
        if(opsi_prepro == 'statis') :
            st.title('Text preprosesing')
            st.header('cleansing')#----------------
            st.text('membersihkan data dari angka ,tanda baca,dll.')
            cleansing = pd.read_csv('data/preprocessing/cleansing.csv')
            st.write(cleansing)
            st.header('casefolding')#----------------
            st.text('mengubahan seluruh huruf menjadi kecil (lowercase) yang ada pada dokumen.')
            casefolding = pd.read_csv('data/preprocessing/casefolding.csv')
            st.write(casefolding)
            st.header('tokenizing')#----------------
            st.text('menguraikan kalimat menjadi token-token atau kata-kata.')
            tokenizing = pd.read_csv('data/preprocessing/tokenizing.csv')
            st.write(tokenizing)
            st.header('stemming')#----------------
            st.text(' merubahan kata yang berimbuhan menjadi kata dasar. ')
            stemming = pd.read_csv('data/preprocessing/stemming.csv')
            st.write(stemming)
            st.header('negasi')#----------------
            st.text(' menangani kata yang mengandung kata negasi  ')
            negasi = pd.read_csv('data/preprocessing/negasi.csv')
            st.write(negasi)
            st.header('word normalization')#----------------
            st.text('mengubah penggunaan kata tidak baku menjadi baku')
            word_normalization = pd.read_csv('data/preprocessing/word normalization.csv')
            st.write(word_normalization)
            st.header('stopword')#----------------
            st.text('menyeleksi kata yang tidak penting dan menghapus kata tersebut.')
            stopword = pd.read_csv('data/preprocessing/stopword.csv')
            st.write(stopword)
            st.title('Pembobotan TF-IDF')
            st.text('pembobotan pada penelitan ini menggunakan tf-idf')
            tfidf = pd.read_csv('data/preprocessing/hasil TF IDF.csv')
            st.dataframe(tfidf,use_container_width=True)

        if(opsi_prepro == 'dinamis') :
            uploaded_file = st.file_uploader("masukan data yang akan dilakukan preprocessing", key="preprocewssing", type='csv')

            if uploaded_file is not None:

                dataset = pd.read_csv(uploaded_file)
                st.write('proses untuk menampilkan hasil text preprocessing dan tf-idf mungkin membutuhkan waktu yang cukup lama')
                kolom = st.text_input('masukan nama kolom ulasan/review pada data yang di input',value="content")
                st.write('tampilan dataset yang di input')
                st.write(dataset)

                if st.button('start text preprocessing') :
                    dataset['Cleansing']= dataset[kolom].apply(cleansing)
                    st.write('tampilan hasil cleansing')
                    cleansing = dataset[[kolom,'Cleansing']]
                    st.dataframe(cleansing)

                    dataset['CaseFolding']= dataset['Cleansing'].apply(casefolding)
                    st.write('tampilan hasil casefolding')
                    casefolding= dataset[['Cleansing','CaseFolding']]
                    st.dataframe(casefolding)
                    
                    dataset['Tokenizing']= dataset['CaseFolding'].apply(tokenizing)
                    st.write('tampilan hasil tokenizing')
                    tokenizing= dataset[['CaseFolding','Tokenizing']]
                    st.dataframe(tokenizing)
                    
                    dataset['stemming']= dataset['Tokenizing'].apply(stemming)
                    st.write('tampilan hasil stemming')
                    stemming= dataset[['Tokenizing','stemming']]
                    st.dataframe(stemming)

                    dataset['negasi']= dataset['stemming'].apply(handle_negation)
                    st.write('tampilan hasil negasi')
                    negasi= dataset[['stemming','negasi']]
                    st.dataframe(negasi)

                    dataset['wordnormalization']= dataset['negasi'].apply(slangword)
                    st.write('tampilan hasil word normalization')
                    wordnormalization= dataset[['negasi','wordnormalization']]
                    st.dataframe(wordnormalization)

                    dataset['stopword']= dataset['wordnormalization'].apply(stopword)
                    st.write('tampilan hasil stopword')
                    stopword= dataset[['wordnormalization','stopword']]
                    #merubah list ke str 
                    dataset['Stopword Removal'] = dataset['stopword'].apply(' '.join)
                    st.dataframe(stopword)
                    dataset.to_csv('hasil_preprocessing.csv',index=False)
                    prepro=True
                    
                if st.button('start pembobotan TF-IDF') :
                    dataset=pd.read_csv('hasil_preprocessing.csv')
                    datasett=dataset['Stopword Removal'] 
                    start_tfidf=output_tfidf(datasett)
                    st.write('tampilan hasil pembobotan TF-IDF')
                    st.dataframe(start_tfidf,use_container_width=True)
                    # Save the results to a CSV file
                    
elif(selected == 'Algoritma Genetika') :

    st.title('Algoritma Genetika')
    st.text('jumlah populasi : 40')
    st.text('jumlah generasi : 20')
    st.text('jumlah crosover rate : 0.6')
    st.text('jumlah mutation rate :0.1')
    st.text('range parameter c :1-50')
    st.text('range parameter gamma :0.1-1')
    st.write('pencarian parameter metode svm')
    ga = pd.read_csv('data/pengujian/GA_result.csv')
    st.dataframe(ga,use_container_width=True)

    best1 = ga['C_best'].iloc[-1]
    best2 = ga['Gamma_best'].iloc[-1]
    best3 = ga['Fitness_best'].iloc[-1]
    
    best_convergenn = ga.loc[(ga['C_con'] == best1) & (ga['Gamma_con'] == best2) & (ga['Fitness_con'] == best3)]
    akurasinn=1-best_convergenn['Fitness_best'].values[0]
    
    st.write(f"Kesimpulan dari tabel di atas yaitu : Algoritma Genetika terbaik Pada Generasi ke- { best_convergenn['generasi'].values[0]} dengan nilai C :{round(best_convergenn['C_con'].values[0],4)} ,nilai Gamma {round(best_convergenn['Gamma_con'].values[0],4)},dan nilai Fitness {best_convergenn['Fitness_con'].values[0]}sehingga mendapatkan akurasi sebesar {akurasinn*100:.2f}%")
    


    probcros = st.number_input('probabilitas crossover',format="%0.1f",value=0.6,step=0.1)
    probmutasi = st.number_input('probabilitas mutasi',format="%0.1f",value=0.2,step=0.1)
    populasi = st.number_input('banyak populasi',step=1,value=6)
    generasi = st.number_input('banyak generasi',step=1,value=3)

    if populasi % 2 != 0:  # Cek apakah x tidak habis dibagi 2
        populasi += 1 

    if st.button('algoritma genetika') :
        with st.expander('data generasi') :
            x=x_train
            y=y_train
            prob_crsvr = probcros 
            prob_mutation = probmutasi 
            population = populasi
            generations = generasi 

            kfold = 5

            x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
                                0,1,1,1,0,0,1,0,1,1,1,0]) 


            pool_of_solutions = np.empty((0,len(x_y_string)))

            best_of_a_generation = np.empty((0,len(x_y_string)+1))

            for i in range(population):
                rd.shuffle(x_y_string)
                pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))

            gen = 1 
            gene=[]
            c_values = []
            gamma_values = []
            fitness_values = []
            cb_values = []
            gammab_values = []
            fitnessb_values = []
            for i in range(generations): 
                
                new_population = np.empty((0,len(x_y_string)))
                
                new_population_with_fitness_val = np.empty((0,len(x_y_string)+1))
                
                sorted_best = np.empty((0,len(x_y_string)+1))
                
                st.write()
                st.write()
                st.write("Generasi ke -", gen) 
                
                family = 1 
                
                for j in range(int(population/2)): 
                    
                    st.write()
                    st.write("populasi ke -", family) 
                    
                    parent_1 = svm_hp_opt.find_parents_ts(pool_of_solutions,x=x,y=y)[0]
                    parent_2 = svm_hp_opt.find_parents_ts(pool_of_solutions,x=x,y=y)[1]
                    
                    child_1 = svm_hp_opt.crossover(parent_1,parent_2,prob_crsvr=prob_crsvr)[0]
                    child_2 = svm_hp_opt.crossover(parent_1,parent_2,prob_crsvr=prob_crsvr)[1]
                    
                    mutated_child_1 = svm_hp_opt.mutation(child_1,child_2, prob_mutation=prob_mutation)[0]
                    mutated_child_2 = svm_hp_opt.mutation(child_1,child_2,prob_mutation=prob_mutation)[1]
                    
                    fitness_val_mutated_child_1 = svm_hp_opt.fitness(x=x,y=y,chromosome=mutated_child_1,kfold=kfold)[2]
                    fitness_val_mutated_child_2 = svm_hp_opt.fitness(x=x,y=y,chromosome=mutated_child_2,kfold=kfold)[2]
                    
                    

                    mutant_1_with_fitness_val = np.hstack((fitness_val_mutated_child_1,mutated_child_1)) 
                    
                    mutant_2_with_fitness_val = np.hstack((fitness_val_mutated_child_2,mutated_child_2)) 
                    
                    
                    new_population = np.vstack((new_population,
                                                mutated_child_1,
                                                mutated_child_2))
                    
                    
                    new_population_with_fitness_val = np.vstack((new_population_with_fitness_val,
                                                            mutant_1_with_fitness_val,
                                                            mutant_2_with_fitness_val))
                    
                    st.write(f"Parent 1:",str(parent_1))
                    st.write(f"Parent 2:",str(parent_2))
                    st.write(f"Child 1:",str(child_1))
                    st.write(f"Child 2:",str(child_2))
                    st.write(f"Mutated Child 1:",str(mutated_child_1))
                    st.write(f"Mutated Child 2:",str(mutated_child_2))
                    st.write(f"nilai fitness 1:",fitness_val_mutated_child_1)
                    st.write(f"nilai fitness 2:",fitness_val_mutated_child_2)

                    family = family+1
                pool_of_solutions = new_population
                
                sorted_best = np.array(sorted(new_population_with_fitness_val,
                                                        key=lambda x:x[0]))
                
                best_of_a_generation = np.vstack((best_of_a_generation,
                                                sorted_best[0]))
                
                
                sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                                    key=lambda x:x[0]))

                gen = gen+1 
                

                best_string_convergence = sorted_best[0]
                best_string_bestvalue = sorted_best_of_a_generation[0]
                final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_convergence[0:],kfold=kfold)
                final_solution_best = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_bestvalue[0:],kfold=kfold)
                
                gene.append(gen-1)

                c_values.append(final_solution_convergence[0])
                gamma_values.append(final_solution_convergence[1])
                fitness_values.append(final_solution_convergence[2])

                cb_values.append(final_solution_best[0])
                gammab_values.append(final_solution_best[1])
                fitnessb_values.append(final_solution_best[2])
                # create a dictionary to store the data
                results = {'generasi':gene,
                        'C_con': c_values,
                        'Gamma_con': gamma_values,
                        'Fitness_con': fitness_values,
                        'C_best': cb_values,
                        'Gamma_best': gammab_values,
                        'Fitness_best': fitnessb_values}
            sorted_last_population = np.array(sorted(new_population_with_fitness_val,key=lambda x:x[0]))

            sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,key=lambda x:x[0]))

            sorted_last_population[:,0] = (sorted_last_population[:,0]) # get accuracy instead of error
            sorted_best_of_a_generation[:,0] = (sorted_best_of_a_generation[:,0])
            best_string_convergence = sorted_last_population[0]

            best_string_overall = sorted_best_of_a_generation[0]

            # to decode the x and y chromosomes to their real values
            final_solution_convergence = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_convergence[1:],
                                                                    kfold=kfold)

            final_solution_overall = svm_hp_opt.fitness(x=x,y=y,chromosome=best_string_overall[1:],
                                                                kfold=kfold)
        # st.write(" C (Best):",str(round(final_solution_overall[0],4)) )# real value of x
        # st.write(" Gamma (Best):",str(round(final_solution_overall[1],4))) # real value of y
        # st.write(" fitness (Best):",str(round(final_solution_overall[2],4)) )# obj val of final chromosome
        # st.write()
        # st.write("------------------------------")
        result=pd.DataFrame(results)
        st.write(result)
        result.to_csv("hasil_optimasi_GA.csv",index=False)
        gabest1 = result['C_best'].iloc[-1]
        gabest2 = result['Gamma_best'].iloc[-1]
        gabest3 = result['Fitness_best'].iloc[-1]

        best_convergen = result.loc[(result['C_con'] == gabest1) & (result['Gamma_con'] == gabest2) & (result['Fitness_con'] == gabest3)]
        akurasi=1-best_convergen['Fitness_con'].values[0]
        st.write(f"Kesimpulan dari tabel di atas yaitu : Algoritma Genetika terbaik Pada Generasi { best_convergen['generasi'].values[0]} denngan nilai C :{round(best_convergen['C_con'].values[0],4)} ,nilai Gamma {round(best_convergen['Gamma_con'].values[0],4)},dan nilai Fitness {best_convergen['Fitness_con'].values[0]} sehingga mendapatkan akurasi sebesar {akurasi*100:.2f}%")

elif(selected == 'Pengujian') :
    tab1,tab2 =st.tabs(['pengujian klasifikasi','Pengujian metode SVM'])

    with tab2 :
        st.title('metode svm')
        st.write('pengujian nilai kernel')
        pengujian_kernel = pd.read_csv('data/pengujian/hasil_svm_kernel.csv')
        st.dataframe(pengujian_kernel,use_container_width=True)
        kernelbest = pengujian_kernel.loc[(pengujian_kernel['accuracy']==pengujian_kernel['accuracy'].max())]
        
        
        st.write(f"dari percobaan diatas dapat disimpulkan bahwa kernel yg efektif adalah pada kernel {kernelbest['kernel'].values[0]} dan dengan nilai akurasi {round(kernelbest['accuracy'].values[0],2)}")
        st.write('pengujian nilai C')
        pengujian_C = pd.read_csv('data/pengujian/hasil_svm_C.csv')
        st.dataframe(pengujian_C,use_container_width=True)
        Cbest = pengujian_C.loc[(pengujian_C['accuracy']==pengujian_C['accuracy'].max())]
        
        
        st.write(f"dari percobaan diatas dapat disimpulkan bahwa nilai c yg efektif adalah pada nilai C {Cbest['c'].values[0]} dan dengan nilai akurasi {round(Cbest['accuracy'].values[0],2)}")
        with st.expander('penjelasan nilai C'):
            st.write('parameter C (juga dikenal sebagai parameter penalti) adalah faktor penting yang mempengaruhi kinerja dan perilaku model SVM. Parameter C mengendalikan trade-off antara penalti kesalahan klasifikasi dan lebar margin.Nilai C yang lebih kecil akan menghasilkan margin yang lebih lebar, dan menjadikan model tidak peka terhadap data dan kelasahan klasifikasi(UNDERFITTING). Sebaliknya, nilai C yang lebih besar akan menghasilkan margin yang lebih sempit, menjadikan model lebih peka terhadap data dan tingkat kesalahan klasifikasi.(OVERFITTING)')
        st.write('pengujian nilai Gamma')
        pengujian_Gamma = pd.read_csv('data/pengujian/hasil_svm_gamma.csv')
        st.dataframe(pengujian_Gamma,use_container_width=True)
        Gammabest = pengujian_Gamma.loc[(pengujian_Gamma['accuracy']==pengujian_Gamma['accuracy'].max())]
        
        
        st.write(f"dari percobaan diatas dapat disimpulkan bahwa nilai Gamma yg efektif adalah pada nilai Gamma {Gammabest['gamma'].values[0]} dan dengan nilai akurasi {round(Gammabest['accuracy'].values[0],2)}")
        
        with st.expander('penjelasan nilai gamma') :
            st.write('Parameter gamma dalam metode Support Vector Machine (SVM) mengontrol pengaruh dari satu sampel data terhadap pembentukan garis pemisah atau decision boundary.jika nilai gamma terlalu kecil makamodel akan terlalu sederhana dan menjadikan model menjadi tidak peka terhadap data dan ada kemungkinan overgeneralisasi, di mana model SVM dapat mengabaikan pola yang signifikan dalam data atau gagal memisahkan kelas yang berbeda secara efektif (UNDERFITTING). ,jika gamma terlalu besar maka model akan terlalu sensitif/detail terhadap data sehingga ada risiko overfitting, di mana model SVM dapat terlalu memperhatikan atau "menghafal" data pelatihan, dan performa pada data pengujian menjadi buruk.')
    with tab1 :
        option = st.selectbox('METODE',('SVM', 'GA-SVM'))
        document = st.text_input('masukan kalimat',value="Saran aja buat kredivo kl bisa limit nya jgn di batasi per 30hari,3bulan,6bulan,12bulan.. Jd kl mau blnj agak susah..kadang linit 30hari g mencukupi.apalagi yg 12 bulan udah pake sekali g bisa pake lagi,kudu lunas dl.🤣🤣🤭🤭 Saran aja,kyk di sebelah limit global sekian juta sy malah suka pake yg dr Aku*******sering sy pake.mau blnj tgl pilih tenor nya. Dari 1bulan,3bulan,6bulan,12bulan.")
        kcleansing = cleansing(document)
        kcasefolding = casefolding(kcleansing)
        ktokenizing = tokenizing(kcasefolding)
        kstemming = stemming(ktokenizing)
        knegasi= handle_negation(kstemming)
        kslangword = slangword(knegasi)
        kstopword = stopword(kslangword)
        kdatastr = str(kstopword)
        ktfidf =vectorizer.transform([kdatastr])
        if (option == 'SVM') :
            if st.button('klasifikasi') :
                st.write('Hasil pengujian dengan metode',option)
                # Making the SVM Classifer
                svmbiasa = model_svm(None,None,x_train,y_train)
                
                predictions = svmbiasa.predict(ktfidf)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stemming :',str(kstemming))
                st.write('hasil negasi :',str(knegasi))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stopword :',str(kstopword))
                if not kstemming:
                    st.write("Maaf mohon inputkan kalimat lagi :)")
                elif predictions == 2:
                    st.write(f"karena nilai prediksi adalah 1 maka termasuk kelas Sentimen Positif")
                elif predictions == 0:
                    st.write(f"karena nilai prediksi adalah -1 maka termasuk kelas Sentimen Negatif")
                elif predictions == 1:
                    st.write(f"karena nilai prediksi adalah 0 maka termasuk kelas Sentimen netral")
            else:
                st.write('hasil akan tampil disini :)') 
        elif (option == 'GA-SVM') :
            GAbaru=pd.read_csv('hasil_optimasi_GA.csv')
            bestCbaru = GAbaru['C_best'].iloc[-1]
            bestGammabaru = GAbaru['Gamma_best'].iloc[-1]
            GAlama=pd.read_csv('data/pengujian/GA_result.csv')
            bestClama = GAlama['C_best'].iloc[-1]
            bestGammalama = GAlama['Gamma_best'].iloc[-1]


            optionparam = st.selectbox('parameter yang digunakan',('lama', 'baru'))
            if (optionparam == 'baru') :
                if st.button('klasifikasi') :
                    st.write(f'Hasil pengujian dengan metode',option,f'parameter yang digunakan C={bestCbaru:.2f} dan gamma={bestGammabaru:.2f}')
                    svmGAbaru = model_svm(bestCbaru,bestGammabaru,x_train,y_train)
                    # Making the SVM Classifer
                    predictions = svmGAbaru.predict(ktfidf)
                    st.write('hasil cleansing :',str(kcleansing))
                    st.write('hasil casefolding :',str(kcasefolding))
                    st.write('hasil tokenizing :',str(ktokenizing))
                    st.write('hasil stopword :',str(kstopword))
                    st.write('hasil word normalization :',str(kslangword))
                    st.write('hasil stemming :',str(kstemming))
                    if not kstemming:
                        st.write("Maaf mohon inputkan kalimat lagi :)")
                    elif predictions == 2:
                        st.write(f"karena nilai prediksi adalah 1 maka termasuk kelas Sentimen Positif")
                    elif predictions == 0:
                        st.write(f"karena nilai prediksi adalah -1 maka termasuk kelas Sentimen Negatif")
                    elif predictions == 1:
                        st.write(f"karena nilai prediksi adalah 0 maka termasuk kelas Sentimen netral")
                else:
                    st.write('hasil akan tampil disini :)')  
            elif (optionparam == 'lama') :
                if st.button('klasifikasi') :
                    st.write(f'Hasil pengujian dengan metode',option,f'parameter yang digunakan C={bestClama:.2f} dan gamma={bestGammalama:.2f}')
                    svmGAlama = model_svm(bestClama,bestGammalama,x_train,y_train)
                    # Making the SVM Classifer
                    predictions = svmGAlama.predict(ktfidf)
                    st.write('hasil cleansing :',str(kcleansing))
                    st.write('hasil casefolding :',str(kcasefolding))
                    st.write('hasil tokenizing :',str(ktokenizing))
                    st.write('hasil stopword :',str(kstopword))
                    st.write('hasil word normalization :',str(kslangword))
                    st.write('hasil stemming :',str(kstemming))
                    if not kstemming:
                        st.write("Maaf mohon inputkan kalimat lagi :)")
                    elif predictions == 2:
                        st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen Positif")
                    elif predictions == 0:
                        st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen Negatif")    
                    elif predictions == 1:
                        st.write(f"karena nilai prediksi adalah {predictions} maka termasuk kelas Sentimen netral")
                else:
                    st.write('hasil akan tampil disini :)') 

elif(selected == 'Report') :
    st.header('Halaman Report')
    st.write('evaluasi model menggunakan confusion matrix')
    optionreport = st.selectbox('option',('statis', 'dinamis'),key='report')
    if (optionreport == 'statis') :
        st.subheader('tampilan confusion matrix metode SVM ')
        st.image('data/evaluasi/class_report_svm.png')
        st.subheader('tampilan confusion matrix metode GA-SVM ')
        st.image('data/evaluasi/class_report_gasvm.png')
    elif (optionreport == 'dinamis') :
        GAbaru=pd.read_csv('hasil_optimasi_GA.csv')
        bestCbaru = GAbaru['C_best'].iloc[-1]
        bestGammabaru = GAbaru['Gamma_best'].iloc[-1]
        st.subheader('hasil evaluasi metode svm')
        svmbiasa = model_svm(None,None,x_train,y_train)
        class_report(svmbiasa,x_test,y_test)
        st.subheader('hasil evaluasi metode gasvm')
        svmGAbaru = model_svm(bestCbaru,bestGammabaru,x_train,y_train)
        class_report(svmGAbaru,x_test,y_test)
