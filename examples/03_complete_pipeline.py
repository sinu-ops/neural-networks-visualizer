"""
Example 3: Complete End-to-End Pipeline
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

import matplotlib.pyplot as plt
from src.network_visualizer import NeuralNetworkVisualizer
from src.training_tracker import TrainingVisualizer
from src.layer_analyzer import LayerAnalyzer
from src.model_builder import build_model
from src.data_utils import generate_dataset


def main():
    """Run complete end-to-end pipeline."""
    
    print("\n" + "="*70)
    print("COMPLETE END-TO-END NEURAL NETWORK PIPELINE")
    print("="*70)
    
    # ============================================================
    # STEP 1: Generate Data
    # ============================================================
    print("\n[Step 1/7] 📊 Generating dataset...")
    X_train, X_test, y_train, y_test, X_all, y_all = generate_dataset(
        dataset_type='moons',
        n_samples=300
    )
    print("✅ Dataset generated!")
    
    # ============================================================
    # STEP 2: Build Model
    # ============================================================
    print("\n[Step 2/7] 🏗️  Building neural network...")
    layer_sizes = [2, 64, 32, 16, 2]
    model = build_model(layer_sizes=layer_sizes, learning_rate=0.001)
    print(f"✅ Model built with architecture: {layer_sizes}")
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
    # ============================================================
    # STEP 3: Visualize Architecture
    # ============================================================
    print("\n[Step 3/7] 🖼️  Visualizing network architecture...")
    visualizer = NeuralNetworkVisualizer(layer_sizes=layer_sizes)
    fig = visualizer.plot_network_architecture(
        title="Network Architecture (Before Training)"
    )
    plt.show()
    print("✅ Architecture visualization complete!")
    
    # ============================================================
    # STEP 4: Train Model
    # ============================================================
    print("\n[Step 4/7] 🚂 Training model...")
    print("   Training for 100 epochs...")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )
    print("✅ Training complete!")
    
    # ============================================================
    # STEP 5: Plot Training Curves
    # ============================================================
    print("\n[Step 5/7] 📈 Plotting training curves...")
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
    
    fig = tracker.plot_training_curves()
    plt.show()
    print("✅ Training curves plotted!")
    
    # ============================================================
    # STEP 6: Analyze Layer Weights
    # ============================================================
    print("\n[Step 6/7] 🔍 Analyzing layer weights...")
    analyzer = LayerAnalyzer()
    weights_dict = {
        f'Layer {i+1}': layer.get_weights()[0]
        for i, layer in enumerate(model.layers) if layer.get_weights()
    }
    fig = analyzer.plot_weight_distribution(weights_dict)
    plt.show()
    print("✅ Weight analysis complete!")
    
    # ============================================================
    # STEP 7: Plot Decision Boundary
    # ============================================================
    print("\n[Step 7/7] 🌍 Plotting decision boundary...")
    fig = visualizer.plot_decision_boundary(
        model=model,
        X=X_all,
        y=y_all,
        title="Decision Boundary (After Training)"
    )
    plt.show()
    print("✅ Decision boundary plotted!")
    
    # ============================================================
    # EVALUATION
    # ============================================================
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n🎯 Training Metrics:")
    print(f"   Loss:     {train_loss:.4f}")
    print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    print(f"\n🎯 Test Metrics:")
    print(f"   Loss:     {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\n📊 Performance Summary:")
    print(f"   Total Parameters: {model.count_params():,}")
    print(f"   Total Epochs: 100")
    print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    print("\nYou successfully:")
    print("  ✅ Generated synthetic data")
    print("  ✅ Built a neural network")
    print("  ✅ Visualized the architecture")
    print("  ✅ Trained the model")
    print("  ✅ Plotted training curves")
    print("  ✅ Analyzed layer weights")
    print("  ✅ Visualized decision boundaries")
    print("\n")


if __name__ == "__main__":
    main()