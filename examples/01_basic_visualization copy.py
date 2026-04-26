"""
Example 1: Basic Network Architecture Visualization

Demonstrates how to visualize a neural network's structure.
Shows different network architectures side by side.

Run with: python examples/01_basic_visualization.py
"""

import sys
import matplotlib.pyplot as plt


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
        plt.show()
        
        print(f"✅ {name} visualization complete!\n")
    
    print("="*70)
    print("✅ EXAMPLE 1 COMPLETE!")
    print("="*70)
    print("\nYou just visualized 3 different neural network architectures!")
    print("- Small:  2 input → 16 hidden → 2 output")
    print("- Medium: 2 input → 64 → 32 → 16 → 2 output")
    print("- Deep:   2 input → 128 → 64 → 32 → 16 → 8 → 2 output")
    print("\n")


if __name__ == "__main__":
    main()
   