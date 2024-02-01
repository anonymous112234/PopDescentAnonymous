import os
import urllib.request
import zipfile
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split

# Download and extract the IWSLT14 dataset
def download_and_extract_iwslt14():
    url = "https://wit3.fbk.eu/archive/2014-01//texts/de/en/de-en.tgz"
    file_name = "iwslt14_de_en.tgz"
    download_path = "./data/"
    
    os.makedirs(download_path, exist_ok=True)
    
    # Download the file
    urllib.request.urlretrieve(url, os.path.join(download_path, file_name))
    
    # Extract the contents
    with tarfile.open(os.path.join(download_path, file_name), "r:gz") as tar:
        tar.extractall(download_path)

# Load and preprocess the data
def load_and_preprocess_iwslt14(data_path="./data"):
    # Download and extract the dataset
    download_and_extract_iwslt14()
    
    # Read the data from the extracted files
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    # Get the list of files for training
    train_files = [
        os.path.join(data_path, 'IWSLT14.TED.dev2010.de-en.de'),
        os.path.join(data_path, 'IWSLT14.TED.tst2010.de-en.de'),
        os.path.join(data_path, 'IWSLT14.TED.tst2011.de-en.de'),
        os.path.join(data_path, 'IWSLT14.TED.tst2012.de-en.de'),
        os.path.join(data_path, 'IWSLT14.TED.tst2013.de-en.de'),
        os.path.join(data_path, 'IWSLT14.TED.tst2014.de-en.de'),
    ]
    
    # Combine all training data
    raw_german_text = [read_file(file) for file in train_files]
    
    # Get the list of files for testing
    test_files = [
        os.path.join(data_path, 'IWSLT14.TED.dev2010.de-en.en'),
        os.path.join(data_path, 'IWSLT14.TED.tst2010.de-en.en'),
        os.path.join(data_path, 'IWSLT14.TED.tst2011.de-en.en'),
        os.path.join(data_path, 'IWSLT14.TED.tst2012.de-en.en'),
        os.path.join(data_path, 'IWSLT14.TED.tst2013.de-en.en'),
        os.path.join(data_path, 'IWSLT14.TED.tst2014.de-en.en'),
    ]
    
    # Combine all testing data
    raw_english_text = [read_file(file) for file in test_files]
    
    return raw_german_text, raw_english_text

# Example usage
raw_german_text, raw_english_text = load_and_preprocess_iwslt14()

# Preprocess the text data
def preprocess_text(text):
    # Add start and end tokens to each sentence
    text = ["<start> " + sentence + " <end>" for sentence in text]
    return text

preprocessed_german_text = preprocess_text(raw_german_text)
preprocessed_english_text = preprocess_text(raw_english_text)

# Split the data into training and testing sets
german_train, german_test, english_train, english_test = train_test_split(
    preprocessed_german_text, preprocessed_english_text, test_size=0.2, random_state=42
)

# Tokenize the text
def tokenize_text(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(text)
    
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    
    return padded_sequences, tokenizer

# Tokenize the German text
german_sequences, german_tokenizer = tokenize_text(german_train)
german_test_sequences = german_tokenizer.texts_to_sequences(german_test)
german_test_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(
    german_test_sequences, padding='post', maxlen=german_sequences.shape[1]
)

# Tokenize the English text
english_sequences, english_tokenizer = tokenize_text(english_train)
english_test_sequences = english_tokenizer.texts_to_sequences(english_test)
english_test_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(
    english_test_sequences, padding='post', maxlen=english_sequences.shape[1]
)
