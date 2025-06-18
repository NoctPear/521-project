# 1. Standardized all images to 224×224 JPG format.
# 2. Implemented 5 standard augmentation techniques (original, horizontal flip, 90° rotation, vertical flip, zoom) for all images.
# 3. Created a fixed train/val/test split (60:10:30) while ensuring all augmented versions of the same image stay in the same split.
# 4. Saved the organized dataset to three separate directories for training, validation, and testing.
# The code first augments all images and then splits the dataset, but uses "image_groups" to ensure that all augmented versions of the same original image are assigned to the same dataset.

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
import shutil
import datetime

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

# Load the augmented images and split into train/val/test datasets
print("Loading augmented data and splitting into train/val/test sets...")

# Get list of all image groups from the saved mapping
group_ids = list(image_groups.keys())
print(f"Found {len(group_ids)} original image groups")

# Use fixed seed for reproducibility
np.random.seed(42)
# Shuffle the group IDs
np.random.shuffle(group_ids)

# Split with ratio 6:1:3
train_ratio, val_ratio, test_ratio = 0.6, 0.1, 0.3
train_size = int(len(group_ids) * train_ratio)
val_size = int(len(group_ids) * val_ratio)

train_groups = group_ids[:train_size]
val_groups = group_ids[train_size:train_size + val_size]
test_groups = group_ids[train_size + val_size:]

print(f"Training groups: {len(train_groups)}")
print(f"Validation groups: {len(val_groups)}")
print(f"Testing groups: {len(test_groups)}")

# Function to load images from a list of group IDs
def load_images_by_groups(group_ids, class_names):
    images = []
    labels = []
    class_to_index = {name: i for i, name in enumerate(class_names)}
    
    for group_id in group_ids:
        class_name = image_groups[group_id]
        class_index = class_to_index[class_name]
        
        # Load all 5 augmentations for this group
        for aug_idx in range(5):
            img_path = save_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
            if img_path.exists():
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                
                images.append(img_array)
                labels.append(class_index)
    
    return np.array(images), np.array(labels)

# Load train, validation and test images
X_train, y_train = load_images_by_groups(train_groups, class_names)
X_val, y_val = load_images_by_groups(val_groups, class_names)
X_test, y_test = load_images_by_groups(test_groups, class_names)

# Normalize pixel values to [0,1]
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Count samples per class in each split
def count_per_class(labels, class_count):
    counts = np.zeros(class_count, dtype=int)
    for label in labels:
        counts[label] += 1
    return counts

train_counts = count_per_class(y_train, len(class_names))
val_counts = count_per_class(y_val, len(class_names))
test_counts = count_per_class(y_test, len(class_names))

print("\nClass distribution:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {train_counts[i]} train, {val_counts[i]} val, {test_counts[i]} test")

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch and optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.batch(batch_size).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

# Print dataset sizes
print(f"\nNumber of training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Number of validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")

# Visualize one batch of training data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.tight_layout()
plt.suptitle("Sample Training Images", y=0.95)
plt.show()

# Save train, validation, and test sets to separate directories
print("Saving train, validation, and test sets to separate directories...")

# Define the directories for the split datasets
split_dir = pathlib.Path(r"C:\grz\VSCode\521\split_datasets")
train_dir = split_dir / "train"
val_dir = split_dir / "val"
test_dir = split_dir / "test"

# Create the directories
split_dir.mkdir(exist_ok=True)
train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# Create class subdirectories in each split directory
for class_name in class_names:
    (train_dir / class_name).mkdir(exist_ok=True)
    (val_dir / class_name).mkdir(exist_ok=True)
    (test_dir / class_name).mkdir(exist_ok=True)

# Save training images
print("Saving training images...")
for group_id in train_groups:
    class_name = image_groups[group_id]
    for aug_idx in range(5):
        src_path = save_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
        if src_path.exists():
            dst_path = train_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
            shutil.copy(src_path, dst_path)

# Save validation images
print("Saving validation images...")
for group_id in val_groups:
    class_name = image_groups[group_id]
    for aug_idx in range(5):
        src_path = save_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
        if src_path.exists():
            dst_path = val_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
            shutil.copy(src_path, dst_path)

# Save test images
print("Saving test images...")
for group_id in test_groups:
    class_name = image_groups[group_id]
    for aug_idx in range(5):
        src_path = save_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
        if src_path.exists():
            dst_path = test_dir / class_name / f"{group_id}_aug_{aug_idx}.jpg"
            shutil.copy(src_path, dst_path)

# Count how many images were saved for each set
train_count = sum(len(list((train_dir / class_name).glob("*.jpg"))) for class_name in class_names)
val_count = sum(len(list((val_dir / class_name).glob("*.jpg"))) for class_name in class_names)
test_count = sum(len(list((test_dir / class_name).glob("*.jpg"))) for class_name in class_names)

print(f"Saved {train_count} training images, {val_count} validation images, and {test_count} test images")
print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}")
print(f"Test directory: {test_dir}")

# Save a metadata file with information about the split
with open(split_dir / "split_info.txt", "w") as f:
    f.write(f"Dataset split info (ratio 6:1:3)\n")
    f.write(f"Date created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Train groups: {len(train_groups)} ({train_count} images)\n")
    f.write(f"Validation groups: {len(val_groups)} ({val_count} images)\n")
    f.write(f"Test groups: {len(test_groups)} ({test_count} images)\n\n")
    f.write("Class distribution:\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {train_counts[i]} train, {val_counts[i]} val, {test_counts[i]} test\n")

# Define path for binary classification dataset
binary_split_dir = pathlib.Path(r"C:\grz\VSCode\521\binary_split_dataset")

# Define binary class mapping
binary_mapping = {
    'Fresh': 'good',
    'Wilted': 'good',
    'Dried_Aging': 'defect',
    'Rotten_Spoiled': 'defect'
}

print("Reorganizing dataset into binary classification (defect/non-defect)...")

# Create directory structure
for split in ['train', 'val', 'test']:
    for binary_class in ['good', 'defect']:
        os.makedirs(binary_split_dir / split / binary_class, exist_ok=True)

# Process each split
for split in ['train', 'val', 'test']:
    source_split_dir = split_dir / split
    
    # Process each original class
    for orig_class in class_names:
        orig_class_dir = source_split_dir / orig_class
        
        # Skip if directory doesn't exist
        if not orig_class_dir.exists():
            continue
        
        # Get the binary class for this original class
        binary_class = binary_mapping[orig_class]
        target_class_dir = binary_split_dir / split / binary_class
        
        # Copy all images with prefix to avoid name conflicts
        img_count = 0
        for img_path in orig_class_dir.glob('*.jpg'):
            target_path = target_class_dir / f"{orig_class}_{img_path.name}"
            shutil.copy(img_path, target_path)
            img_count += 1
            
        print(f"Copied {img_count} images from {split}/{orig_class} to {split}/{binary_class}")

# Count images in the reorganized binary dataset
good_train = len(list((binary_split_dir / 'train' / 'good').glob('*.jpg')))
defect_train = len(list((binary_split_dir / 'train' / 'defect').glob('*.jpg')))
good_val = len(list((binary_split_dir / 'val' / 'good').glob('*.jpg')))
defect_val = len(list((binary_split_dir / 'val' / 'defect').glob('*.jpg')))
good_test = len(list((binary_split_dir / 'test' / 'good').glob('*.jpg')))
defect_test = len(list((binary_split_dir / 'test' / 'defect').glob('*.jpg')))

print("\nBinary Classification Dataset Summary:")
print(f"Training set: {good_train + defect_train} images ({good_train} good, {defect_train} defect)")
print(f"Validation set: {good_val + defect_val} images ({good_val} good, {defect_val} defect)")
print(f"Test set: {good_test + defect_test} images ({good_test} good, {defect_test} defect)")
print(f"Total: {good_train + defect_train + good_val + defect_val + good_test + defect_test} images")
print(f"Binary dataset directory: {binary_split_dir}")

# Save metadata about the binary split
with open(binary_split_dir / "binary_split_info.txt", "w") as f:
    f.write(f"Binary Dataset split info\n")
    f.write(f"Date created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Class mapping:\n")
    f.write(f"  - good (non-defect): Fresh, Wilted\n")
    f.write(f"  - defect: Dried_Aging, Rotten_Spoiled\n\n")
    f.write(f"Training set: {good_train + defect_train} images ({good_train} good, {defect_train} defect)\n")
    f.write(f"Validation set: {good_val + defect_val} images ({good_val} good, {defect_val} defect)\n")
    f.write(f"Test set: {good_test + defect_test} images ({good_test} good, {defect_test} defect)\n")
