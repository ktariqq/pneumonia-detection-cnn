"""
Visualization utilities for model evaluation and interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from src import model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

PURPLE = "#7C3AED"
LIGHT_PURPLE = "#A78BFA"
SOFT_PURPLE = "#C4B5FD"
DARK_BG = "#0B0B14"
TEXT = "#E9D5FF"
GRID = "#2E1065"

sns.set_theme(style="darkgrid")

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.edgecolor": LIGHT_PURPLE,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.alpha": 0.3,
    "legend.facecolor": DARK_BG,
    "legend.edgecolor": LIGHT_PURPLE
})

def plot_training_history(history, save_path='../results/training_history.png'):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Keras History object from model.fit()
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'],
                label='Train Loss',
                linewidth=2,
                color=PURPLE)
    axes[0, 0].plot(history.history['val_loss'],
                label='Val Loss',
                linewidth=2,
                color=LIGHT_PURPLE)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'],
                label='Train Accuracy',
                linewidth=2,
                color=PURPLE)
    axes[0, 1].plot(history.history['val_accuracy'],
                label='Val Accuracy',
                linewidth=2,
                color=LIGHT_PURPLE)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'],
                label='Train Precision',
                linewidth=2,
                color=PURPLE)
        axes[1, 0].plot(history.history['val_precision'],
                label='Val Precision',
                linewidth=2,
                color=LIGHT_PURPLE)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'],
                label='Train Recall',
                linewidth=2,
                color=PURPLE)
        axes[1, 1].plot(history.history['val_recall'],
                label='Val Recall',
                linewidth=2,
                color=LIGHT_PURPLE)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {save_path}")


def evaluate_model_comprehensive(model, test_generator, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class names
    
    Returns:
        Dictionary containing predictions and metrics
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Get predictions
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Test Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision:      {precision*100:.2f}%")
    print(f"Recall:         {recall*100:.2f}%")
    print(f"F1-Score:       {f1*100:.2f}%")
    
    # Classification report
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs.flatten(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true, y_pred, class_names=['NORMAL', 'PNEUMONIA'],
                         save_path='../results/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / cm[i].sum() * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
             ha='center', va='center', fontsize=10, color=TEXT)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_pred_probs, save_path='../results/roc_curve.png'):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        save_path: Path to save plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=PURPLE, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=LIGHT_PURPLE, lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved to {save_path}")
    print(f"AUC Score: {roc_auc:.4f}")
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_pred_probs, 
                                save_path='../results/precision_recall_curve.png'):
    """
    Plot Precision-Recall curve.
    
    Important for imbalanced datasets (like medical imaging).
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        save_path: Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color=PURPLE, lw=2,
         label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Precision-Recall curve saved to {save_path}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    return avg_precision


def visualize_predictions(model, test_generator, num_samples=16,
                         save_path='../results/sample_predictions.png'):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        num_samples: Number of samples to display
        save_path: Path to save plot
    """
    test_generator.reset()
    images, labels = next(test_generator)
    predictions = model.predict(images[:num_samples])
    
    class_names = {v: k for k, v in test_generator.class_indices.items()}
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        true_class = class_names[int(labels[i])]
        pred_class = class_names[int(predictions[i] > 0.5)]
        confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
        
        # Green if correct, red if wrong
        color = 'green' if true_class == pred_class else 'red'
        axes[i].set_title(
            f'True: {true_class}\nPred: {pred_class}\nConf: {confidence*100:.1f}%',
            color=color, fontsize=10, fontweight='bold'
        )
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Predictions visualization saved to {save_path}")


def generate_gradcam_heatmaps(model, test_generator, num_samples=8,
                              save_path='../results/gradcam.png'):

    print("\nGenerating Grad-CAM visualizations...")

    test_generator.reset()
    images, labels = next(test_generator)

    images = images[:num_samples]
    labels = labels[:num_samples]

    class_names = {v: k for k, v in test_generator.class_indices.items()}

    # Choose correct layer name
    if "VGG16" in model.name:
        last_conv_layer = "block5_conv3"
    else:
        last_conv_layer = "conv4_2"

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))

    predictions = model.predict(images)

    for i in range(num_samples):

        heatmap = make_gradcam_heatmap(images[i:i+1], model, last_conv_layer)

        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="viridis")
        axes[i, 1].set_title("Heatmap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(images[i])
        axes[i, 2].imshow(heatmap, cmap="viridis", alpha=0.5)

        true_class = class_names[int(labels[i])]
        pred_class = class_names[int(predictions[i] > 0.5)]

        color = "green" if true_class == pred_class else "red"

        axes[i, 2].set_title(
            f"True: {true_class} | Pred: {pred_class}",
            color=color
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved to {save_path}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap + 1e-8)

    return heatmap.numpy()