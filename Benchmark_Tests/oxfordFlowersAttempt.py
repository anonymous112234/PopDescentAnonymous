import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the dataset from the local directory
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'oxford_flowers102',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)





# Function to preprocess and resize images
def preprocess_image(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing to the datasets
train_ds = train_ds.map(preprocess_image)
val_ds = val_ds.map(preprocess_image)
test_ds = test_ds.map(preprocess_image)

# Create batches and shuffle the datasets
batch_size = 32
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)


# Define a function to extract images and labels
def extract_images_labels(images, labels):
    return images, labels

# Apply the function to the dataset
processed_dataset = train_ds.map(extract_images_labels)

# Iterate through the processed dataset
i = 0
for images, labels in processed_dataset.take(100):  # Adjust the number of batches you want to take
    print(i)
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    i+=1




# Define a simple CNN model
model = Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(102, activation='softmax')  # Adjust this based on the number of classes in your dataset
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Define EarlyStopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

# Train the model
model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[early_stopping])

# Evaluate the model on the train set
train_loss, train_accuracy = model.evaluate(train_ds)
print(f'Train loss: {train_loss}, Train accuracy: {train_accuracy}')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# # Save the trained model
# model.save('/path/to/your/model/oxford_flowers_model.h5')


