"""
CNN model architectures for pneumonia detection.
Includes custom CNN and VGG16 transfer learning implementations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16


def create_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a custom CNN architecture for pneumonia detection.
    
    Architecture:
    - 4 convolutional blocks with increasing filters (32 -> 256)
    - Batch normalization after each conv layer
    - MaxPooling for spatial dimension reduction
    - Dropout for regularization
    - Dense layers for classification
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of output classes (2 for binary classification)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='Custom_Pneumonia_CNN')
    
    # Block 1: Initial feature extraction
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                           input_shape=input_shape, name='conv1_1'))
    model.add(layers.BatchNormalization(name='bn1_1'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
    model.add(layers.BatchNormalization(name='bn1_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool1'))
    model.add(layers.Dropout(0.25, name='dropout1'))
    
    # Block 2: Deeper features
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
    model.add(layers.BatchNormalization(name='bn2_1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
    model.add(layers.BatchNormalization(name='bn2_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool2'))
    model.add(layers.Dropout(0.25, name='dropout2'))
    
    # Block 3: More complex patterns
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
    model.add(layers.BatchNormalization(name='bn3_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
    model.add(layers.BatchNormalization(name='bn3_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool3'))
    model.add(layers.Dropout(0.25, name='dropout3'))
    
    # Block 4: High-level features
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(layers.BatchNormalization(name='bn4_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(layers.BatchNormalization(name='bn4_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool4'))
    model.add(layers.Dropout(0.25, name='dropout4'))
    
    # Classification head
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(512, activation='relu', name='fc1'))
    model.add(layers.BatchNormalization(name='bn_fc1'))
    model.add(layers.Dropout(0.5, name='dropout_fc1'))
    model.add(layers.Dense(256, activation='relu', name='fc2'))
    model.add(layers.BatchNormalization(name='bn_fc2'))
    model.add(layers.Dropout(0.5, name='dropout_fc2'))
    
    # Output layer
    if num_classes == 2:
        # Binary classification: single neuron with sigmoid
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
    else:
        # Multi-class: neurons equal to classes with softmax
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


def create_vgg16_transfer_learning(input_shape=(224, 224, 3), num_classes=2, 
                                   freeze_base=True, fine_tune_from=15):
    """
    Create VGG16-based transfer learning model.
    
    Uses pretrained VGG16 weights from ImageNet and adds custom classification head.
    Option to freeze base layers or fine-tune top layers.
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes
        freeze_base: Whether to freeze VGG16 base layers
        fine_tune_from: Layer index from which to start fine-tuning (if not frozen)
    
    Returns:
        Compiled Keras model
    """
    # Load pretrained VGG16 without top layers (ImageNet weights)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,  # Remove classification head
        input_shape=input_shape
    )
    
    # Freeze base model layers if specified
    if freeze_base:
        base_model.trainable = False
        print(f"Base model frozen: {len(base_model.layers)} layers")
    else:
        # Fine-tune only top layers
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        print(f"Fine-tuning from layer {fine_tune_from} onwards")
    
    # Create new model
    model = models.Sequential(name='VGG16_Transfer_Learning')
    model.add(base_model)
    
    # Add custom classification head
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(512, activation='relu', name='fc1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(256, activation='relu', name='fc2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.Dropout(0.5, name='dropout2'))
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


def compile_model(model, learning_rate=0.001, class_weights=None):
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        class_weights: Optional class weights for imbalanced datasets
    
    Returns:
        Compiled model
    """
    # Adam optimizer with learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Determine loss function based on output layer
    if model.output_shape[-1] == 1:
        # Binary classification
        loss = 'binary_crossentropy'
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    else:
        # Multi-class classification
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"\nModel compiled successfully!")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Loss: {loss}")
    print(f"Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
    
    return model


def get_model_summary(model, save_path=None):
    """
    Print and optionally save model architecture summary.
    
    Args:
        model: Keras model
        save_path: Optional path to save summary as text file
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    model.summary()
    
    if save_path:
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"\nModel summary saved to {save_path}")
    
    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"\nParameter Count:")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {non_trainable_params:,}")
    print(f"  Total: {total_params:,}")
    print("="*70)
