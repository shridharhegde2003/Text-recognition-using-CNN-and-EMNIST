# train_model.py

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def main():
    """Main function to train and evaluate the model."""

    print("--- Starting Model Training ---")

    # --- 1. Load and Preprocess the EMNIST Dataset ---
    print("\n[1/6] Loading and preprocessing EMNIST dataset...")
    
    # Using a 20% subset for training and 50% for testing to speed up the process.
    # To use the full dataset, change split to: ['train', 'test']
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=['train[:20%]', 'test[:50%]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32` and corrects orientation."""
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.transpose(image, perm=[1, 0, 2]) # Correct EMNIST orientation
        return image, label

    BATCH_SIZE = 128
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    num_classes = ds_info.features['label'].num_classes
    print(f"Dataset loaded. Number of classes: {num_classes}")

    # --- 2. Build the CNN Model ---
    print("\n[2/6] Building the CNN model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # --- 3. Train the Model ---
    print("\n[3/6] Starting model training for 10 epochs...")
    history = model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
    )
    print("Training complete.")

    # --- 4. Plot and Save Training History ---
    print("\n[4/6] Plotting and saving training history graphs...")
    history_data = history.history
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'], label='Training Accuracy')
    plt.plot(history_data['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'], label='Training Loss')
    plt.plot(history_data['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_filename = 'training_history.png'
    plt.savefig(plot_filename)
    print(f"Saved training history plot to '{plot_filename}'")
    plt.close()

    # --- 5. Evaluate Model and Generate Reports ---
    print("\n[5/6] Evaluating model and generating reports...")
    loss, accuracy = model.evaluate(ds_test)
    print(f'Final Test Accuracy: {accuracy*100:.2f}%')

    y_pred_probs = model.predict(ds_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([y for x, y in ds_test], axis=0)

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print('\nClassification Report:')
    print(report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("Saved classification report to 'classification_report.txt'")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(25, 25))
    sns.heatmap(cm, annot=False, xticklabels=class_names, yticklabels=class_names, cmap='viridis')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_filename = 'confusion_matrix.png'
    plt.savefig(cm_filename)
    print(f"Saved confusion matrix to '{cm_filename}'")
    plt.close()

    # --- 6. Save the Final Model ---
    print("\n[6/6] Saving the final model...")
    model.save('htr_model.h5')
    print("--- Model training complete. Model saved as htr_model.h5 ---")

if __name__ == '__main__':
    main()
