"""
Example 1: Basic Network Architecture Visualization

Demonstrates how to visualize a neural network's structure.
Shows different network architectures side by side.

Run with: python -m examples.01_basic_visualization
"""

# ========== CRITICAL: Set backend FIRST, before any plotting ==========
import matplotlib
matplotlib.use('TkAgg')  # ← MUST BE HERE FIRST!

import matplotlib.pyplot as plt
plt.ion()  # Interactive mode ON
# ========================================================================

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network_visualizer import NeuralNetworkVisualizer


def main():
    """Run basic visualization example."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC NETWORK ARCHITECTURE VISUALIZATION")
    print("="*70)
    
    # Define different network architectures to visualize
    architectures = {
        'Small Network': [2, 16, 2],
        'Medium Network': [2, 64, 32, 16, 2],
        'Deep Network': [2, 128, 64, 32, 16, 8, 2]
    }
    
    print("\n📊 Creating visualizations for different network architectures...\n")
    
    # Create and display each architecture
    for name, layer_sizes in architectures.items():
        print(f"🏗️  Visualizing: {name}")
        print(f"   Architecture: {layer_sizes}")
        
        # Create visualizer
        visualizer = NeuralNetworkVisualizer(layer_sizes=layer_sizes)
        
        # Plot network architecture
        fig = visualizer.plot_network_architecture(
            title=f"{name}\nLayers: {layer_sizes}"
        )
        
        plt.tight_layout()
        plt.show(block=False)  # Don't block execution
        plt.pause(1)  # Wait 1 second before next plot
        
        print(f"✅ {name} visualization complete!\n")
    
    print("="*70)
    print("✅ EXAMPLE 1 COMPLETE!")
    print("="*70)
    print("\n🎮 TOOLBAR BUTTONS ARE NOW ACTIVE!")
    print("   Click on the plots and use these buttons:")
    print("   🏠 Home   - Reset view")
    print("   ⬅️  Back   - Previous view")
    print("   ➡️  Forward - Next view")
    print("   ✚ Zoom   - Click and drag to zoom in")
    print("   ⊟ Pan    - Click and drag to move plot")
    print("   💾 Save   - Save plot as image (.png)")
    print("\n📌 Keep plots open to use buttons!")
    print("   Close all plot windows when done.\n")


if __name__ == "__main__":
    main()
    
    # Keep all plots open and responsive
    plt.show(block=True)