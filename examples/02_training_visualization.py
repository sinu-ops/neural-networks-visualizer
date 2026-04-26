"""
Example 2: Training Visualization
...
"""

# ========== CRITICAL: Set backend FIRST ==========
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.ion()
# ================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ... rest of imports ...

import matplotlib.pyplot as plt
from src.model_builder import build_model
from src.data_utils import generate_dataset
from src.training_tracker import TrainingVisualizer


def main():
    """Run training visualization example."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: TRAINING VISUALIZATION")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/4] 📊 Generating dataset...")
    X_train, X_test, y_train, y_test, X_all, y_all = generate_dataset(
        dataset_type='moons',
        n_samples=300
    )
    
    # Step 2: Build neural network model
    print("\n[Step 2/4] 🏗️  Building neural network...")
    print("   Architecture: [2 input → 64 → 32 → 16 → 2 output]")
    model = build_model(
        layer_sizes=[2, 64, 32, 16, 2],
        learning_rate=0.001
    )
    print("✅ Model built successfully!")
    
    # Step 3: Train the model
    print("\n[Step 3/4] 🚂 Training model (100 epochs)...")
    print("   This may take 30-60 seconds...")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0  # Don't show progress bar
    )
    print("✅ Training complete!")
    
    # Step 4: Visualize training progress
    print("\n[Step 4/4] 📈 Plotting training curves...")
    
    # Create tracker and update with training history
    tracker = TrainingVisualizer()
    
    for epoch, (loss, acc, val_loss, val_acc) in enumerate(
        zip(
            history.history['loss'],
            history.history['accuracy'],
            history.history['val_loss'],
            history.history['val_accuracy']
        )
    ):
        tracker.update(epoch, loss, acc, val_loss, val_acc)
    
    # Plot the curves
    fig = tracker.plot_training_curves()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n📈 Final Training Metrics:")
    print(f"   Loss:     {final_train_loss:.4f}")
    print(f"   Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    
    print(f"\n📈 Final Validation Metrics:")
    print(f"   Loss:     {final_val_loss:.4f}")
    print(f"   Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    
    # Check for overfitting
    print(f"\n🔍 Overfitting Analysis:")
    accuracy_diff = final_train_acc - final_val_acc
    if accuracy_diff < 0.05:
        print(f"   ✅ Good! Minimal overfitting (diff: {accuracy_diff:.4f})")
    elif accuracy_diff < 0.15:
        print(f"   ⚠️  Moderate overfitting (diff: {accuracy_diff:.4f})")
    else:
        print(f"   ❌ High overfitting (diff: {accuracy_diff:.4f})")
    
    print("\n" + "="*70)
    print("✅ EXAMPLE 2 COMPLETE!")
    print("="*70)
    print("\nYou just visualized the training process!")
    print("- Blue line:   Training loss/accuracy")
    print("- Red line:    Validation loss/accuracy")
    print("- Ideal:       Both lines go down/up together\n")


if __name__ == "__main__":
    main()