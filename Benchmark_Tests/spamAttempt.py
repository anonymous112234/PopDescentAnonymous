import tensorflow as tf
import tensorflow_datasets as tfds

# Load the SPAM SMS dataset from TFDS
# dataset, info = tfds.load('sms_spam', split='train', with_info=True)
ds, info = tfds.load('huggingface:sms_spam/plain_text')

# Create a function to preprocess the text data
def preprocess(example):
    text = example['text']
    label = tf.cast(example['label'], tf.float32)
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r'<[^>]+>', ' ')  # Remove HTML tags
    text = tf.strings.regex_replace(text, r'[^a-zA-Z ]', '')  # Remove non-alphabetic characters
    text = tf.strings.strip(text)
    return text, label

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess)

# Split the dataset into training and testing sets
train_size = int(0.8 * info.splits['train'].num_examples)
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Tokenize and vectorize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_dataset.map(lambda text, label: text))
vocab_size = len(tokenizer.word_index) + 1

train_dataset = train_dataset.map(lambda text, label: (tokenizer.texts_to_sequences(text), label))
test_dataset = test_dataset.map(lambda text, label: (tokenizer.texts_to_sequences(text), label))

# Pad sequences for consistent input size
train_dataset = train_dataset.map(lambda text, label: (tf.keras.preprocessing.sequence.pad_sequences(text), label))
test_dataset = test_dataset.map(lambda text, label: (tf.keras.preprocessing.sequence.pad_sequences(text), label))

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=None),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset.batch(32), epochs=5)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset.batch(32))
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
