# pages/3_Demo_Klasifikasi.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Demo Klasifikasi", page_icon="🚀")

# hide default navbar
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Load Assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("bilstm_model.keras")
    with open("tokenizer.pickle", "rb") as handle: tokenizer = pickle.load(handle)
    with open("stopwords.pickle", "rb") as handle: stopwords = pickle.load(handle)
    with open("max_len.pickle", "rb") as handle: max_len = pickle.load(handle)
    return model, tokenizer, stopwords, max_len

# Fungsi preprocessing dan interpretasi
def text_lower(text): return text.lower()
def normalize_and_clean_text(text):
    text = re.sub(r'<.*?>', ' ', text); text = re.sub(r'[^a-zA-Z\s]', ' ', text); text = re.sub(r'\s+', ' ', text)
    return text.strip()
def remove_stopwords(text, stopwords_set):
    words = text.split()
    return " ".join([word for word in words if word not in stopwords_set])

def interpret_binary_to_string_label(binary_predictions):
    label_map = {
        (1, 0, 0): "hukum", (0, 1, 0): "politik", (0, 0, 1): "nasional",
        (1, 1, 0): "hukum-politik", (1, 0, 1): "hukum-nasional",
        (0, 1, 1): "politik-nasional", (1, 1, 1): "hukum-politik-nasional",
        (0, 0, 0): "lainnya"
    }
    final_labels = [label_map.get(tuple(pred), "lainnya") for pred in binary_predictions]
    return final_labels

# UI Aplikasi
st.title("🚀 Uji Coba Klasifikasi Berita")
st.markdown("Masukkan sebuah kalimat berita untuk diprediksi kategorinya.")

model, tokenizer, all_stopwords, MAX_LEN = load_assets()
labels = ['hukum', 'politik', 'nasional']

user_input = st.text_area("", height=200)

if st.button("Prediksi Kategori"):
    if user_input:
        # 1. Lakukan preprocessing
        clean_text = remove_stopwords(normalize_and_clean_text(text_lower(user_input)), all_stopwords)

        # 2. Tampilkan hasil preprocessing di dalam expander
        with st.expander("Lihat Teks Setelah Preprocessing"):
            st.info(f"Teks berikut adalah versi bersih dari input Anda yang akan dianalisis oleh model:")
            st.text(clean_text)
        
        # 3. Lanjutkan proses prediksi
        sequence = tokenizer.texts_to_sequences([clean_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        with st.spinner('Model sedang menganalisis...'):
            probabilities = model.predict(padded_sequence)[0]
            predicted_binary = (probabilities > 0.5).astype(int)
            predicted_label = interpret_binary_to_string_label([predicted_binary])[0]
        
        st.success(f"**Prediksi Kategori: {predicted_label.upper()}**")
        
        st.write("---")
        st.subheader("Rincian Probabilitas per Label:")
        
        cols = st.columns(len(labels))
        for i, label in enumerate(labels):
            with cols[i]:
                st.metric(label=label.capitalize(), value=f"{probabilities[i]:.2%}")

    else:
        st.warning("Mohon masukkan teks berita terlebih dahulu.")

# Footer
footer_html = """
<style>
.footer {
    position: fixed;
    right: 0;
    bottom: 0;
    width: 100%;
    background-color: #A0522D; /* Warna cokelat */
    color: white;
    text-align: center;
    padding: 0px;  /* Hilangkan padding bawah */
    margin: 0;
    font-size: 16px;
    line-height: 32px;  /* Bisa diatur agar tinggi teks tetap proporsional */
}
</style>
<div class="footer">
    <p style="margin: 0;">© 2025 Chairal Octavyanz Tanjung - 140810210030</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
