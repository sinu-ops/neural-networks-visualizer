"""
Example 4: Advanced Analysis and Comparison
...
"""

# ========== CRITICAL: Set backend FIRST ==========
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.ion()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


import matplotlib.pyplot as plt
from src.model_builder import build_model
from src.data_utils import generate_dataset
from src.advanced_features import AdvancedVisualizations


def main():
    """Run advanced analysis example."""
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS AND MODEL COMPARISON")
    print("="*70)
    
    # ============================================================
    # STEP 1: Generate Data
    # ============================================================
    print("\n[Step 1/5] 📊 Generating dataset...")
    X_train, X_test, y_train, y_test, X_all, y_all = generate_dataset(
        dataset_type='moons',
        n_samples=300
    )
    print("✅ Dataset ready!")
    
    # ============================================================
    # STEP 2: Build Multiple Models
    # ============================================================
    print("\n[Step 2/5] 🏗️  Building multiple models with different architectures...\n")
    
    architectures = {
        'Small\n(16 units)': [2, 16, 2],
        'Medium\n(64 units)': [2, 64, 32, 16, 2],
        'Large\n(256 units)': [2, 256, 128, 64, 32, 2],
    }
    
    models = {}
    for name, layer_sizes in architectures.items():
        print(f"   Building: {name.replace(chr(10), ' ')}")
        print(f"   Architecture: {layer_sizes}")
        
        model = build_model(layer_sizes=layer_sizes)
        
        print(f"   Training...")
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        models[name] = model
        print(f"   ✅ Complete!\n")
    
    # ============================================================
    # STEP 3: Compare Models
    # ============================================================
    print("\n[Step 3/5] 🏆 Comparing model performance...")
    advanced = AdvancedVisualizations()
    fig = advanced.compare_models(models, X_test, y_test)
    plt.show()
    print("✅ Model comparison complete!")
    
    # ============================================================
    # STEP 4: Confusion Matrix
    # ============================================================
    print("\n[Step 4/5] 🔍 Plotting confusion matrix for best model...")
    best_model = models['Medium\n(64 units)']
    y_pred = best_model.predict(X_test, verbose=0)
    
    fig = advanced.plot_confusion_matrix(
        y_test, y_pred,
        title="Confusion Matrix - Medium Model (Test Set)"
    )
    plt.show()
    print("✅ Confusion matrix plotted!")
    
    # ============================================================
    # STEP 5: Feature Importance
    # ============================================================
    print("\n[Step 5/5] 📊 Analyzing feature importance...")
    fig = advanced.plot_feature_importance(
        best_model,
        feature_names=['Feature 1\n(X-axis)', 'Feature 2\n(Y-axis)']
    )
    plt.show()
    print("✅ Feature importance analysis complete!")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nModel Comparison Results:")
    for name, model in models.items():
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"   {name}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n" + "="*70)
    print("✅ ADVANCED ANALYSIS COMPLETE!")
    print("="*70)
    print("\nYou explored:")
    print("  ✅ Building multiple architectures")
    print("  ✅ Comparing model performance")
    print("  ✅ Confusion matrices")
    print("  ✅ Feature importance analysis")
    print("\n")


if __name__ == "__main__":
    main()