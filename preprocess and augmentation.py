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

# Create TensorFlow datasets from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names from dataset
class_names = train_ds.class_names
class_names = ['Dried_Aging' if name == 'Dried or Aging' else 'Rotten_Spoiled' 
               if name == 'Rotten or Spoiled' else name for name in class_names]
print(f"Class names: {class_names}")

# Create pixel value normalization layer (scale to 0-1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Check normalization result
image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
print(f"Normalized pixel range: min={np.min(first_image):.4f}, max={np.max(first_image):.4f}")

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create test dataset from validation set
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

# Print dataset sizes
print(f"Number of training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Number of validation batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")

# # Data augmentation 
# print("Setting up data augmentation...")

# # Reset training dataset to before performance optimizations
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

# # Create a data augmentation layer with transformations
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal_and_vertical"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.05),
# ])

# # Create a proper preprocessing model
# preprocessing_model = tf.keras.Sequential([
#     data_augmentation,
#     normalization_layer
# ])

# # Apply combined preprocessing to training data
# train_ds = train_ds.map(
#     lambda x, y: (preprocessing_model(x), y),
#     num_parallel_calls=AUTOTUNE
# )

# # Apply only normalization to validation and test data
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# # Split validation and test again
# val_batches = tf.data.experimental.cardinality(val_ds)
# test_ds = val_ds.take(val_batches // 2)
# val_ds = val_ds.skip(val_batches // 2)

# # Optimize dataset performance
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# # Visualize augmented and normalized images
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         if i < len(images):
#             ax = plt.subplot(3, 3, i + 1)
#             # Images are already preprocessed in the dataset
#             plt.imshow(images[i])
#             plt.title(class_names[labels[i]])
#             plt.axis("off")
# plt.tight_layout()
# plt.suptitle("Augmented and Normalized Images", y=0.95)
# plt.show()

# print("Data augmentation applied to training set.")

# # Save augmented images to disk
# print("Saving augmented image examples...")
# save_dir = pathlib.Path(r"C:\grz\VSCode\521\augmented_examples")
# save_dir.mkdir(exist_ok=True)

# # Take a batch of images and save augmented versions
# for images, labels in train_ds.take(1):
#     for i in range(min(9, len(images))):
#         img_array = images[i].numpy()
#         # Convert from float [0,1] back to uint8 [0,255] for saving
#         img_array = (img_array * 255).astype(np.uint8)
#         img = PIL.Image.fromarray(img_array)
#         class_name = class_names[labels[i].numpy()]
#         img.save(save_dir / f"augmented_{class_name}_{i}.jpg")

# print(f"Saved augmented image examples to {save_dir}")

# # Save all augmented images to disk
# print("Saving all augmented images...")
# save_dir = pathlib.Path(r"C:\grz\VSCode\521\augmented_images")
# save_dir.mkdir(exist_ok=True)

# # Create subdirectories for each class
# for class_name in class_names:
#     (save_dir / class_name).mkdir(exist_ok=True)

# # Process and save all augmented images from training set
# image_count = 0
# for images, labels in train_ds:
#     for i in range(len(images)):
#         img_array = images[i].numpy()
#         # Convert from float [0,1] back to uint8 [0,255] for saving
#         img_array = (img_array * 255).astype(np.uint8)
#         img = PIL.Image.fromarray(img_array)
#         class_name = class_names[labels[i].numpy()]
#         img.save(save_dir / class_name / f"augmented_{image_count:04d}.jpg")
#         image_count += 1

# print(f"Saved {image_count} augmented images to {save_dir}")

# Create targeted data augmentation with crops and zooms

# ——————————————————————————————————————————————————————————

# def generate_crops_and_zooms(image, num_crops=4):
#     """Generate multiple crops/zoomed views from a single image"""
#     crops = []
#     height, width = image.shape[0], image.shape[1]
    
#     # Original image
#     crops.append(image)
    
#     # Center crop (zoom in)
#     center_h, center_w = height // 2, width // 2
#     crop_size = min(height, width) // 2
#     center_crop = image[center_h-crop_size:center_h+crop_size, 
#                         center_w-crop_size:center_w+crop_size]
#     center_crop = tf.image.resize(center_crop, [height, width])
#     crops.append(center_crop)
    
#     # Top-left crop
#     top_left = image[:height//2, :width//2]
#     top_left = tf.image.resize(top_left, [height, width])
#     crops.append(top_left)
    
#     # Bottom-right crop
#     bottom_right = image[height//2:, width//2:]
#     bottom_right = tf.image.resize(bottom_right, [height, width])
#     crops.append(bottom_right)
    
#     return crops

# # Visualize crops and zooms for a sample image
# plt.figure(figsize=(12, 12))
# for images, labels in train_ds.take(1):
#     sample_img = images[0].numpy()
#     crops = generate_crops_and_zooms(sample_img)
    
#     for i, crop in enumerate(crops):
#         ax = plt.subplot(2, 2, i + 1)
#         plt.imshow(crop)
#         if i == 0:
#             plt.title("Original")
#         elif i == 1:
#             plt.title("Center Zoom")
#         elif i == 2:
#             plt.title("Top-Left Crop")
#         else:
#             plt.title("Bottom-Right Crop")
#         plt.axis("off")
# plt.tight_layout()
# plt.suptitle("Targeted Crops and Zooms Example", y=0.95)
# plt.show()

# # Save targeted augmentations
# print("Saving targeted crop augmentations...")
# save_dir = pathlib.Path(r"C:\grz\VSCode\521\targeted_augmentations")
# save_dir.mkdir(exist_ok=True)

# # Create subdirectories for each class
# for class_name in class_names:
#     (save_dir / class_name).mkdir(exist_ok=True)

# # Save targeted augmentations for training images
# image_count = 0
# for images, labels in train_ds.take(20):  # Limit to first few batches
#     for i in range(len(images)):
#         img = images[i].numpy()
#         class_name = class_names[labels[i].numpy()]
        
#         # Generate crops
#         crops = generate_crops_and_zooms(img)
        
#         # Save all crops
#         for j, crop in enumerate(crops):
#             crop_array = (crop * 255).astype(np.uint8)
#             crop_img = PIL.Image.fromarray(crop_array)
#             crop_img.save(save_dir / class_name / f"img_{image_count:04d}_crop_{j}.jpg")
        
#         image_count += 1

# print(f"Saved targeted augmentations for {image_count} images to {save_dir}")

# ————————————————————————————————————————————————————————————

# Create standardized data augmentation based on the requirements
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
for images, labels in train_ds.take(1):
    sample_img = images[0].numpy()
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

# Save standard augmentations
print("Saving standard augmentations...")
save_dir = pathlib.Path(r"C:\grz\VSCode\521\standard_augmentations")
save_dir.mkdir(exist_ok=True)

# Create subdirectories for each class
for class_name in class_names:
    (save_dir / class_name).mkdir(exist_ok=True)

# Save standard augmentations for training images
image_count = 0
augmentation_count = 0

for images, labels in train_ds:
    for i in range(len(images)):
        img = images[i].numpy()
        class_name = class_names[labels[i].numpy()]
        
        # Generate standard augmentations
        augmentations = generate_standard_augmentations(img)
        
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
            aug_img.save(save_dir / class_name / f"img_{image_count:04d}_aug_{j}.jpg")
            augmentation_count += 1
        
        image_count += 1

print(f"Saved {augmentation_count} augmented images from {image_count} original images to {save_dir}")
