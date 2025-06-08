import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import zipfile
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image 

#Download the dataset
dataset_url = "https://github.com/NoctPear/521-project/raw/main/Water%20Spinach%20Dataset.zip"
cache_path = r"C:\grz\VSCode\521" 
archive = tf.keras.utils.get_file("water_spinach_dataset.zip", 
                                  origin=dataset_url,
                                  extract=False,
                                  cache_dir=cache_path)

with zipfile.ZipFile(archive, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(archive))
data_dir = pathlib.Path(os.path.dirname(archive)) / "Water Spinach Dataset"

# Define preprocessing parameters
img_height = 224
img_width = 224
batch_size = 32

# Define system files to filter out
system_files = ['desktop.ini', 'thumbs.db', '.ds_store']

# Process and standardize all images
print("Processing images to standard format...")
processed_count = 0

for subdir in data_dir.iterdir():
    if subdir.is_dir():
        print(f"Processing directory: {subdir.name}")
        for img_path in subdir.iterdir():
            # Skip system files
            if img_path.name.lower() in system_files:
                print(f"Skipped system file: {img_path.name}")
                continue
                
            # Process only image files
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                try:
                    # Open the image
                    img = Image.open(img_path)
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to standard dimensions
                    img = img.resize((img_width, img_height))
                    
                    # Save as JPG (overwrite if already JPG)
                    new_path = img_path.with_suffix('.jpg')
                    img.save(new_path, quality=95)
                    
                    # Delete original if it was not JPG
                    if img_path.suffix.lower() != '.jpg' and img_path != new_path:
                        os.remove(img_path)
                        print(f"Converted: {img_path.name} -> {new_path.name}")
                    
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")

print(f"Processed {processed_count} images to standard format (224×224 JPG)")

# Print dataset statistics
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total images after preprocessing: {image_count}")

# Count images per class
class_counts = {}
for subdir in data_dir.iterdir():
    if subdir.is_dir():
        count = len(list(subdir.glob('*.jpg')))
        class_counts[subdir.name] = count
        print(f"Class '{subdir.name}': {count} images")

# Display a few sample images after preprocessing
plt.figure(figsize=(10, 10))
sample_count = min(9, image_count)
cols = 3
rows = (sample_count + cols - 1) // cols

sample_paths = list(data_dir.glob('*/*.jpg'))[:sample_count]
for i, sample_path in enumerate(sample_paths):
    img = PIL.Image.open(sample_path)
    ax = plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"{sample_path.parent.name}\n{img.width}×{img.height}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Get class names directly from data directory
class_names = [subdir.name for subdir in data_dir.iterdir() if subdir.is_dir()]
class_names = ['Dried_Aging' if name == 'Dried or Aging' else 'Rotten_Spoiled' 
               if name == 'Rotten or Spoiled' else name for name in class_names]
print(f"Class names: {class_names}")

# Data Augmentation
def generate_standard_augmentations(image):
    """Generate standard augmentations: original, horizontal flip, rotation, vertical flip, zoom"""
    augmentations = []
    height, width = image.shape[0], image.shape[1]
    
    # (a) Original image
    augmentations.append(image)
    
    # (b) Horizontal flip
    h_flip = tf.image.flip_left_right(image)
    augmentations.append(h_flip)
    
    # (c) Rotation (90 degrees)
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, 0)
    rotated = tf.image.rot90(image_tensor)
    rotation = tf.squeeze(rotated, 0)
    augmentations.append(rotation)
    
    # (d) Vertical flip
    v_flip = tf.image.flip_up_down(image)
    augmentations.append(v_flip)
    
    # (e) Zoom (central crop and resize back)
    center_h, center_w = height // 2, width // 2
    crop_size = int(min(height, width) * 0.75)
    half_crop = crop_size // 2
    zoom = image[center_h-half_crop:center_h+half_crop, center_w-half_crop:center_w+half_crop]
    zoom = tf.image.resize(zoom, [height, width])
    augmentations.append(zoom)
    
    return augmentations

# Visualize standard augmentations for a sample image
plt.figure(figsize=(15, 10))
# Get a sample image directly from the data directory
sample_path = next(data_dir.glob('*/*.jpg'))
sample_img = np.array(PIL.Image.open(sample_path).resize((img_width, img_height))) / 255.0
augmentations = generate_standard_augmentations(sample_img)

for i, aug in enumerate(augmentations):
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(aug)
    if i == 0:
        plt.title("(a) Original")
    elif i == 1:
        plt.title("(b) Horizontal Flip")
    elif i == 2:
        plt.title("(c) Rotation (90°)")
    elif i == 3:
        plt.title("(d) Vertical Flip")
    else:
        plt.title("(e) Zoom")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("Standard Data Augmentation Techniques", y=0.95)
plt.show()

# Save standard augmentations for ALL images
print("Saving standard augmentations for ALL images...")
save_dir = pathlib.Path(r"C:\grz\VSCode\521\all_augmented_data")
save_dir.mkdir(exist_ok=True)

# Create subdirectories for each class
for class_name in class_names:
    (save_dir / class_name).mkdir(exist_ok=True)

# Save standard augmentations for ALL images
image_count = 0
augmentation_count = 0
image_groups = {}  # Track original images and their augmentations

# Process each class directory
for subdir in data_dir.iterdir():
    if subdir.is_dir():
        # Get standardized class name
        if subdir.name == 'Dried or Aging':
            class_name = 'Dried_Aging'
        elif subdir.name == 'Rotten or Spoiled':
            class_name = 'Rotten_Spoiled'
        else:
            class_name = subdir.name
            
        # Process each image in this class
        for img_path in subdir.glob('*.jpg'):
            # Load and preprocess the image
            img = PIL.Image.open(img_path).resize((img_width, img_height))
            img_array = np.array(img) / 255.0  # Normalize to 0-1
            
            # Generate standard augmentations
            augmentations = generate_standard_augmentations(img_array)
            
            # Create a group ID for tracking related augmentations
            group_id = f"img_{image_count:04d}"
            image_groups[group_id] = class_name
            
            # Save all augmentations
            for j, aug in enumerate(augmentations):
                # Convert TensorFlow tensor to NumPy array if needed
                if isinstance(aug, tf.Tensor):
                    aug_array = aug.numpy()
                else:
                    aug_array = aug
                
                # Scale to uint8 range
                aug_array = (aug_array * 255).astype(np.uint8)
                aug_img = PIL.Image.fromarray(aug_array)
                aug_img.save(save_dir / class_name / f"{group_id}_aug_{j}.jpg")
                augmentation_count += 1
            
            image_count += 1
            if image_count % 20 == 0:
                print(f"Processed {image_count} original images...")

print(f"Saved {augmentation_count} augmented images from {image_count} original images to {save_dir}")
