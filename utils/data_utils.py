"""
Data loading and preprocessing utilities for chest X-ray images.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image


def create_data_generators(train_dir, val_dir, test_dir, 
                          img_size=(224, 224), batch_size=32,
                          augment=True):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        img_size: Target image size (height, width)
        batch_size: Number of samples per batch
        augment: Whether to apply data augmentation to training set
    
    Returns:
        train_generator, val_generator, test_generator
    """
    
    # Training data augmentation (helps model generalize)
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,                    # Normalize pixel values to [0, 1]
            rotation_range=20,                 # Randomly rotate up to 20 degrees
            width_shift_range=0.1,             # Randomly shift horizontally
            height_shift_range=0.1,            # Randomly shift vertically
            horizontal_flip=True,              # Random horizontal flip
            zoom_range=0.1,                    # Random zoom
            brightness_range=[0.8, 1.2],       # Random brightness adjustment
            fill_mode='nearest'                # Fill empty pixels with nearest value
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Validation and test: only rescaling (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',      # Binary: NORMAL vs PNEUMONIA
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False             # Don't shuffle validation/test
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    print("\n" + "="*70)
    print("DATA GENERATORS CREATED")
    print("="*70)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Classes: {train_generator.class_indices}")
    print("="*70 + "\n")
    
    return train_generator, val_generator, test_generator


def calculate_class_weights(train_generator):
    """
    Calculate class weights for imbalanced dataset.
    
    Gives higher weight to minority class so model pays more attention to it.
    
    Args:
        train_generator: Training data generator
    
    Returns:
        Dictionary of class weights {0: weight_0, 1: weight_1}
    """
    # Get all labels from training set
    labels = train_generator.classes
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n" + "="*70)
    print("CLASS WEIGHTS (for imbalanced data)")
    print("="*70)
    for class_idx, weight in class_weight_dict.items():
        class_name = list(train_generator.class_indices.keys())[class_idx]
        count = np.sum(labels == class_idx)
        print(f"Class {class_idx} ({class_name}): Weight = {weight:.4f}, Samples = {count}")
    print("="*70 + "\n")
    
    return class_weight_dict


def analyze_dataset_distribution(train_dir, val_dir, test_dir):
    """
    Analyze and visualize class distribution across splits.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
    
    Returns:
        DataFrame with distribution statistics
    """
    import matplotlib.pyplot as plt
    
    def count_images(directory):
        """Count images in each class folder"""
        counts = {}
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                counts[class_name] = len(os.listdir(class_path))
        return counts
    
    train_counts = count_images(train_dir)
    val_counts = count_images(val_dir)
    test_counts = count_images(test_dir)
    
    # Create DataFrame
    classes = list(train_counts.keys())
    data = {
        'Class': classes,
        'Train': [train_counts[c] for c in classes],
        'Validation': [val_counts[c] for c in classes],
        'Test': [test_counts[c] for c in classes]
    }
    df = pd.DataFrame(data)
    df['Total'] = df['Train'] + df['Validation'] + df['Test']
    
    print("\n" + "="*70)
    print("DATASET DISTRIBUTION")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    
    # Visualize distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart by split
    df.set_index('Class')[['Train', 'Validation', 'Test']].plot(
        kind='bar', ax=axes[0], color=['steelblue', 'orange', 'green']
    )
    axes[0].set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_xlabel('Class')
    axes[0].legend(title='Split')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart of total distribution
    axes[1].pie(df['Total'], labels=df['Class'], autopct='%1.1f%%', 
                colors=['lightcoral', 'lightskyblue'], startangle=90)
    axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df


def create_sample_visualization(generator, num_samples=16, save_path='../results/sample_images.png'):
    """
    Visualize sample images from dataset.
    
    Args:
        generator: Data generator
        num_samples: Number of images to display
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of images
    images, labels = next(generator)
    
    # Class names (reverse mapping from indices)
    class_names = {v: k for k, v in generator.class_indices.items()}
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        label_name = class_names[int(labels[i])]
        axes[i].set_title(f'{label_name}', fontsize=11, fontweight='bold')
    
    plt.suptitle('Sample Chest X-Ray Images', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Sample visualization saved to {save_path}")


def preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array ready for model input
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
