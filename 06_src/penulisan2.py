
'''
# Final Project: Pelatihan model BiLSTM dengan teknik cross-validation
- Nama  : Chairal Octavyanz Tanjung
- NPM   : 140810210030
- Email : chairal21001@mail.unpad.ac.id
'''

# Impor library
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import random
import pickle
from wordcloud import WordCloud

# Sastrawi & Stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# TensorFlow dan Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Scikit-learn, Scikit-multilearn 
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    multilabel_confusion_matrix
)

# Mengatur environment
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")

# Pemuatan Dataset
# Untuk dataset 1 (Hukum, Politik, Nasional)
df = pd.read_csv('../datasets/dataset_berita_multilabel1.csv')

# Untuk dataset 1 (Hukum, Politik, Ekonomi)
df = pd.read_csv('../datasets/dataset_berita_multilabel2.csv')

# File Stopword
df_sw = pd.read_csv('../datasets/stopwordbahasa.csv', header=None)

# Fungsi-fungsi preprocessing
def text_lower(text): return text.lower()
def normalize_and_clean_text(text):
    text = re.sub(r'<.*?>', ' ', text); text = re.sub(r'[^a-zA-Z\s]', ' ', text); text = re.sub(r'\s+', ' ', text)
    return text.strip()

stopword_factory = StopWordRemoverFactory()
sastrawi_stopwords = set(stopword_factory.get_stop_words())
custom_stopwords = set(df_sw[0])
all_stopwords = sastrawi_stopwords.union(custom_stopwords)

def remove_stopwords(text, stopwords_set):
    words = text.split()
    return " ".join([word for word in words if word not in stopwords_set])

print("Memulai proses preprocessing teks...")
df['cleaned_konten'] = df['konten'].astype(str).progress_apply(text_lower)
df['cleaned_konten'] = df['cleaned_konten'].progress_apply(normalize_and_clean_text)
df['cleaned_konten'] = df['cleaned_konten'].progress_apply(lambda x: remove_stopwords(x, all_stopwords))
print("Preprocessing teks selesai.")

# Inisialisasi Fitur
labels = ['hukum', 'politik', 'ekonomi']
X = df['cleaned_konten'].values
y = df[labels].values

print(f"\nBentuk data fitur (X): {X.shape}")
print(f"Bentuk data label (y): {y.shape}\n\n")

df.head()

# Data Splitting 80:20
print("Membagi data menjadi data latih dan uji (80:20) menggunakan iterative split...")
X_train, y_train, X_test, y_test = iterative_train_test_split(X.reshape(-1, 1), y, test_size=0.2)

# Mengembalikan X_train dan X_test ke bentuk 1D array
X_train = X_train.ravel()
X_test = X_test.ravel()

print(f"\nJumlah data latih awal: {len(X_train)}")
print(f"Jumlah data uji: {len(X_test)}")

# Mengubah vektor label menjadi string
y_train_str = np.array(['-'.join(map(str, row)) for row in y_train])

# Hitung frekuensi masing-masing kombinasi
y_train_str_counts = pd.Series(y_train_str).value_counts()

# Menampilkan Distribusi
print("\nDistribusi label pada data latih:")
print(pd.DataFrame(y_train, columns=labels).sum())

print("\nDistribusi Kombinasi label:")
print(y_train_str_counts)


# Tokenization and Padding
VOCAB_SIZE = 5000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')


# Menggabungkan data training dan testing awal untuk di-split ulang oleh K-Fold
X_combined = np.concatenate((X_train_pad, X_test_pad), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)


# Inisialisasi K-Fold
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# List untuk menyimpan skor dari setiap fold
fold_histories = []
fold_no = 1

# Membuat Loop K-Fold
for train_index, val_index in mskf.split(X_combined, y_combined):
    print(f"--- Melatih Fold ke-{fold_no} ---")

    # Membagi data untuk fold saat ini
    X_train_fold, X_val = X_combined[train_index], X_combined[val_index]
    y_train_fold, y_val = y_combined[train_index], y_combined[val_index]

    # Membangun ulang model untuk setiap fold
    def create_model():
        model = Sequential([
            Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dropout(0.4),
            Dense(len(labels), activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    model_kfold = create_model()

    # Inisialiasasi callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Melatih Model
    print(f"\nMelatih model BiLSTM Multilabel untuk fold {fold_no}...")
    history = model_kfold.fit(
        X_train_fold,
        y_train_fold,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Menyimpan Hasil
    scores = model_kfold.evaluate(X_val, y_val, verbose=0)
    print(f"Skor untuk fold {fold_no}: {model_kfold.metrics_names[0]} = {scores[0]}; {model_kfold.metrics_names[1]} = {scores[1]}")
    fold_histories.append(scores)

    fold_no += 1

# Hasil Cross-Validation
print("\n--- Hasil Rata-Rata Cross-Validation (10-Fold) ---")

# Ubah list of scores menjadi array numpy untuk perhitungan mudah
scores_array = np.array(fold_histories)

# Hitung rata-rata dan standar deviasi
mean_loss = np.mean(scores_array[:, 0])
std_loss = np.std(scores_array[:, 0])
mean_accuracy = np.mean(scores_array[:, 1])
std_accuracy = np.std(scores_array[:, 1])

print(f"Rata-rata Loss: {mean_loss:.4f} (+/- {std_loss:.4f})")
print(f"Rata-rata Akurasi: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")


# Pengujian dan Evaluasi Model
# Inisialiasi fungsi predict
y_pred_proba = modelk_fold.predict(X_test_pad)

# Mengonversi ke biner
y_pred_binary = (y_pred_proba > 0.5).astype(int)

# Evaluasi Metrik Multilabel Standar
print("--- Laporan Evaluasi Model Multilabel ---")
subset_accuracy = accuracy_score(y_test, y_pred_binary)
print(f"🎯 Subset Accuracy (Kecocokan Penuh): {subset_accuracy:.4f}")

hamming = hamming_loss(y_test, y_pred_binary)
print(f"🔻 Hamming Loss (Rata-rata Kesalahan per Label): {hamming:.4f}")

# Skor ROC-AUC
print("\n--- Skor ROC-AUC (Per Label & Rata-rata) ---")
try:
    roc_auc_macro = roc_auc_score(y_test, y_pred_proba, average='macro')
    roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, average='weighted')
    print(f"📈 ROC-AUC (Macro Average): {roc_auc_macro:.4f}")
    print(f"📈 ROC-AUC (Weighted Average): {roc_auc_weighted:.4f}\n")
    print("Skor ROC-AUC per Label:")
    for i, label in enumerate(labels):
        roc_label = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        print(f"   - {label.capitalize():<10}: {roc_label:.4f}")
except ValueError:
    print("Tidak dapat menghitung ROC-AUC.")

# Laporan Klasifikasi per Label
print("\n--- Laporan Klasifikasi (Per Label Individu) ---")
print(classification_report(y_test, y_pred_binary, target_names=labels, zero_division=0))


# Confusion Matrix per Label
print("\n--- Confusion Matrix (Per Label Individu) ---")
cm_per_label = multilabel_confusion_matrix(y_test, y_pred_binary)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Confusion Matrix untuk Setiap Label Individu', fontsize=16)

for i, (matrix, label) in enumerate(zip(cm_per_label, labels)):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=[f'Bukan {label}', label],
                yticklabels=[f'Bukan {label}', label])
    axes[i].set_title(f'Kategori: {label.capitalize()}', fontsize=14)
    axes[i].set_ylabel('Label Aktual')
    axes[i].set_xlabel('Label Prediksi')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Interpretasi menjadi 7 kombinasi label
def interpret_binary_to_string_label(binary_predictions):
    # Untuk dataset 1 (Hukum, Politik, Nasioal)
    label_map = {
        (1, 0, 0): "hukum", (0, 0, 1): "nasional", (0, 1, 0): "politik",
        (1, 0, 1): "hukum-nasional", (1, 1, 0): "hukum-politik",
        (0, 1, 1): "politik-nasional", (1, 1, 1): "hukum-politik-nasional",
        (0, 0, 0): "lainnya"
    }

    # Untuk dataset 2 (Hukum, Politik, Ekonomi)
    label_map = {
        (1, 0, 0): "hukum", (0, 0, 1): "ekonomi", (0, 1, 0): "politik",
        (1, 0, 1): "hukum-ekonomi", (1, 1, 0): "hukum-politik",
        (0, 1, 1): "politik-ekonomi", (1, 1, 1): "hukum-politik-ekonomi",
        (0, 0, 0): "lainnya"
    }
    final_labels = [label_map.get(tuple(pred), "lainnya") for pred in binary_predictions]
    return np.array(final_labels)

# Inisialisasi fungsi label kombinasi
y_pred_combination = interpret_binary_to_string_label(y_pred_binary)
y_true_combination = interpret_binary_to_string_label(y_test)
all_possible_classes = sorted(list(set(y_true_combination) | set(y_pred_combination)))

# Laporan Evaluasi 7 Label Kombinasi
print("\n\n--- Laporan Evaluasi Berdasarkan 7 Kombinasi Label (Format Tabel) ---")
report_dict = classification_report(y_true_combination, y_pred_combination, labels=all_possible_classes, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
display(report_df.round(2))

# Confusion Matrix 7 Label Kombinasi
print("\n📈 Confusion Matrix (Berdasarkan 7 Kombinasi Label):")
cm_combination = confusion_matrix(y_true_combination, y_pred_combination, labels=all_possible_classes)
plt.figure(figsize=(12, 9))
sns.heatmap(cm_combination, annot=True, fmt='d', cmap='Greens',
            xticklabels=all_possible_classes, yticklabels=all_possible_classes)
plt.title('Confusion Matrix (Hasil Interpretasi)', fontsize=16)
plt.ylabel('Label Aktual', fontsize=12)
plt.xlabel('Label Prediksi', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()