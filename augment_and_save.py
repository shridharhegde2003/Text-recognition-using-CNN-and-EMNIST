import pandas as pd
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
INPUT_CSV = 'dataset/English.csv'
IMAGE_DIR = 'dataset/img'
OUTPUT_DIR = 'dataset/augmented_img'
OUTPUT_CSV = 'dataset/augmented_English.csv'
NUM_AUGMENTATIONS_PER_IMAGE = 5 # Create 5 new images for each original

# --- Main Script ---
def augment_and_save():
    """Reads the dataset, creates augmented images, and saves them to a new directory and CSV."""
    # 1. Load the original dataset
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at '{INPUT_CSV}'. Please make sure it exists.")
        return

    # 2. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 3. Set up the data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

    new_rows = []

    print(f"Starting augmentation for {len(df)} images...")

    # 4. Loop through each image in the original dataset
    for index, row in df.iterrows():
        image_path = os.path.join(IMAGE_DIR, row['image'])
        label = row['label']

        try:
            # Load the original image
            img = Image.open(image_path)
            img_array = np.array(img)
            # Reshape for the generator
            img_array = img_array.reshape((1,) + img_array.shape + (1,))

            # Generate and save new augmented images
            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                augmented_img_array = batch[0].reshape(batch[0].shape[:-1])
                augmented_img = Image.fromarray(augmented_img_array.astype('uint8'))

                # Create a new filename
                base_filename = os.path.splitext(row['image'])[0]
                new_filename = f"{base_filename}_aug_{i}.png"
                save_path = os.path.join(OUTPUT_DIR, new_filename)
                augmented_img.save(save_path)

                # Add the new image info to our list
                new_rows.append({'image': f'augmented_img/{new_filename}', 'label': label})

                i += 1
                if i >= NUM_AUGMENTATIONS_PER_IMAGE:
                    break # Stop after creating the desired number of augmentations

        except FileNotFoundError:
            print(f"Warning: Could not find image {image_path}. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred with {image_path}: {e}")
            continue

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1} / {len(df)} images...")

    # 5. Create the new DataFrame and save to CSV
    augmented_df = pd.DataFrame(new_rows)
    # Combine original and augmented data
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)

    print("\nAugmentation complete!")
    print(f"New CSV saved to: {OUTPUT_CSV}")
    print(f"Total images in new dataset: {len(combined_df)}")

if __name__ == '__main__':
    augment_and_save()
