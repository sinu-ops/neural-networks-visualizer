"""
Neural Network Visualizer - Streamlit Web App

Interactive web interface for neural network visualization.
Run with: streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.network_visualizer import NeuralNetworkVisualizer
from src.training_tracker import TrainingVisualizer
from src.model_builder import build_model
from src.data_utils import generate_dataset
from src.layer_analyzer import LayerAnalyzer
from src.advanced_features import AdvancedVisualizations

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Neural Network Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Title and Description
# ============================================================
st.markdown("""
# 🧠 Neural Network Visualizer

An interactive tool for understanding neural networks through beautiful visualizations!

**Explore how neural networks work:**
- 🏗️ Visualize network architectures with neurons and connections
- 📈 Watch training progress in real-time with loss and accuracy curves
- 🌍 See decision boundaries that separate different classes
- 📊 Analyze layer weights and compare multiple models
""")

st.markdown("---")

# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Choose an option:", [
    "🏗️ Network Architecture",
    "📈 Training Visualization",
    "🌍 Decision Boundary",
    "📊 Advanced Analysis",
    "ℹ️ About"
])

# ============================================================
# PAGE 1: Network Architecture
# ============================================================
if page == "🏗️ Network Architecture":
    st.header("🏗️ Network Architecture Visualization")
    
    st.write("""
    Design and visualize different neural network architectures!
    
    Adjust the number of hidden layers and neurons per layer to see how the network structure changes.
    """)
    
    # Two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configure Architecture")
        
        # Number of hidden layers
        num_hidden_layers = st.slider(
            "Number of hidden layers:",
            min_value=1,
            max_value=5,
            value=2,
            help="More layers = deeper network"
        )
        
        # Layer sizes
        layer_sizes = [2]  # Input layer (fixed)
        
        for i in range(num_hidden_layers):
            neurons = st.slider(
                f"Hidden layer {i+1}:",
                min_value=8,
                max_value=256,
                value=64,
                step=8,
                help=f"Number of neurons in hidden layer {i+1}"
            )
            layer_sizes.append(neurons)
        
        layer_sizes.append(2)  # Output layer (fixed)
        
        # Display summary
        st.markdown("**📊 Architecture Summary:**")
        st.info(f"""
        - **Input Neurons:** {layer_sizes[0]}
        - **Hidden Layers:** {num_hidden_layers}
        - **Output Neurons:** {layer_sizes[-1]}
        - **Total Layers:** {len(layer_sizes)}
        - **Layer Structure:** {' → '.join(map(str, layer_sizes))}
        """)
        
        # Button to generate
        generate_btn = st.button(
            "🎨 Generate Visualization",
            key="arch_gen",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        if generate_btn:
            st.write("Creating network visualization...")
            
            try:
                # Create visualizer and plot
                visualizer = NeuralNetworkVisualizer(layer_sizes=layer_sizes)
                fig = visualizer.plot_network_architecture(
                    title=f"Network Architecture\n{' → '.join(map(str, layer_sizes))}"
                )
                
                # Display plot
                st.pyplot(fig, use_container_width=True)
                
                st.success("✅ Visualization created successfully!")
                
                # Display info
                st.markdown("**📌 About This Visualization:**")
                st.write("""
                - **Circles** represent neurons
                - **Blue lines** show positive weight connections
                - **Red lines** show negative weight connections
                - **Line thickness** indicates weight magnitude
                - **Darker colors** indicate stronger connections
                """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================
# PAGE 2: Training Visualization
# ============================================================
# ============================================================
# PAGE 2: Training Visualization
# ============================================================
elif page == "📈 Training Visualization":
    st.header("📈 Training Progress Visualization")
    
    st.write("Train a neural network and visualize how it learns!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Training Settings")
        
        # Dataset selection
        dataset_type = st.selectbox(
            "Select Dataset:",
            ["moons", "circles"],
            key="training_dataset"
        )
        
        # Model architecture
        arch_dict = {
            "Small (16 units)": [2, 16, 2],
            "Medium (64 units)": [2, 64, 32, 16, 2],
            "Large (256 units)": [2, 256, 128, 64, 32, 2],
        }
        
        arch_choice = st.selectbox(
            "Network Architecture:",
            list(arch_dict.keys()),
            key="training_arch"
        )
        
        arch_value = arch_dict[arch_choice]
        
        # Training parameters
        epochs = st.slider(
            "Epochs:",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="training_epochs"
        )
        
        learning_rate = st.select_slider(
            "Learning Rate:",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            key="training_lr"
        )
        
        batch_size = st.select_slider(
            "Batch Size:",
            options=[8, 16, 32, 64],
            value=32,
            key="training_batch"
        )
        
        st.markdown("**📊 Summary:**")
        st.info(f"""
        - Dataset: {dataset_type}
        - Architecture: {arch_choice}
        - Epochs: {epochs}
        - Learning Rate: {learning_rate}
        - Batch Size: {batch_size}
        """)
        
        # Train button
        train_btn = st.button(
            "🚂 Train Model",
            key="train_model_btn",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        if train_btn:
            try:
                # Create placeholders for progress
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Step 1: Generate data
                status_placeholder.info("📊 Step 1/4: Generating dataset...")
                progress_placeholder.progress(20)
                
                X_train, X_test, y_train, y_test, _, _ = generate_dataset(
                    dataset_type=dataset_type,
                    n_samples=300
                )
                
                # Step 2: Build model
                status_placeholder.info("🏗️ Step 2/4: Building model...")
                progress_placeholder.progress(40)
                
                model = build_model(
                    layer_sizes=arch_value,
                    learning_rate=learning_rate
                )
                
                # Step 3: Train model
                status_placeholder.info(f"🚂 Step 3/4: Training for {epochs} epochs...")
                progress_placeholder.progress(60)
                
                # Train with verbose=0 to avoid clutter
                history = model.fit(
                    X_train, 
                    y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    
                )
                
                # Step 4: Visualize
                status_placeholder.info("📈 Step 4/4: Creating visualization...")
                progress_placeholder.progress(80)
                
                # Create manual plot instead of using TrainingVisualizer
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot 1: Loss
                epochs_list = list(range(epochs))
                axes[0].plot(epochs_list, history.history['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
                axes[0].plot(epochs_list, history.history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
                axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
                axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
                axes[0].set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
                axes[0].legend(fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Accuracy
                axes[1].plot(epochs_list, history.history['accuracy'], 'g-o', label='Train Accuracy', linewidth=2, markersize=4)
                axes[1].plot(epochs_list, history.history['val_accuracy'], 'orange', marker='s', label='Val Accuracy', linewidth=2, markersize=4)
                axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
                axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                axes[1].set_title('Training Accuracy Over Time', fontsize=13, fontweight='bold')
                axes[1].legend(fontsize=11)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                progress_placeholder.progress(100)
                status_placeholder.empty()
                progress_placeholder.empty()
                
                # Display the plot
                st.pyplot(fig, use_container_width=True)
                
                st.success("✅ Training complete!")
                
                # Display metrics in columns
                st.subheader("📊 Training Metrics")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    final_train_loss = history.history['loss'][-1]
                    st.metric(
                        "Train Loss",
                        f"{final_train_loss:.4f}",
                        delta=f"{final_train_loss - history.history['loss'][0]:.4f}"
                    )
                
                with col_m2:
                    final_train_acc = history.history['accuracy'][-1]
                    st.metric(
                        "Train Accuracy",
                        f"{final_train_acc:.2%}",
                        delta=f"{final_train_acc - history.history['accuracy'][0]:+.2%}"
                    )
                
                with col_m3:
                    final_val_loss = history.history['val_loss'][-1]
                    st.metric(
                        "Val Loss",
                        f"{final_val_loss:.4f}",
                        delta=f"{final_val_loss - history.history['val_loss'][0]:.4f}"
                    )
                
                with col_m4:
                    final_val_acc = history.history['val_accuracy'][-1]
                    st.metric(
                        "Val Accuracy",
                        f"{final_val_acc:.2%}",
                        delta=f"{final_val_acc - history.history['val_accuracy'][0]:+.2%}"
                    )
                
                # Show interpretation
                st.markdown("**📌 How to Read:**")
                st.write("""
                - **Blue line:** Training performance (what model saw during training)
                - **Red line:** Validation performance (what model didn't see)
                - **Ideal:** Both lines go down (loss) or up (accuracy) together
                - **Problem:** Large gap between lines = overfitting (memorizing)
                """)
                
                # Evaluate on test set
                st.subheader("🧪 Test Set Performance")
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                
                col_test1, col_test2 = st.columns(2)
                with col_test1:
                    st.metric("Test Loss", f"{test_loss:.4f}")
                with col_test2:
                    st.metric("Test Accuracy", f"{test_acc:.2%}")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.write("**Debug Info:**")
                import traceback
                st.write(traceback.format_exc())

# ============================================================
# PAGE 3: Decision Boundary
# ============================================================
elif page == "🌍 Decision Boundary":
    st.header("🌍 Decision Boundary Visualization")
    
    st.write("""
    Visualize how the network learns to separate different classes!
    
    The colored regions show how the network classifies different parts of the input space.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Settings")
        
        dataset_type = st.selectbox(
            "Dataset:",
            ["moons", "circles"],
            help="Choose dataset type"
        )
        
        st.info(f"""
        **Dataset:** {dataset_type}
        
        The network will be trained to separate the {dataset_type} dataset into two classes.
        """)
        
        boundary_btn = st.button(
            "🌍 Generate Decision Boundary",
            key="decision_boundary",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        if boundary_btn:
            with st.spinner("Training network..."):
                try:
                    # Generate data
                    X_train, X_test, y_train, y_test, X_all, y_all = generate_dataset(
                        dataset_type=dataset_type,
                        n_samples=300
                    )
                    
                    # Build and train model
                    model = build_model([2, 64, 32, 16, 2])
                    model.fit(
                        X_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        verbose=0
                    )
                    
                    # Visualize decision boundary
                    visualizer = NeuralNetworkVisualizer([2, 64, 32, 16, 2])
                    fig = visualizer.plot_decision_boundary(
                        model=model,
                        X=X_all,
                        y=y_all,
                        title=f"Decision Boundary - {dataset_type.capitalize()} Dataset"
                    )
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    st.success("✅ Decision boundary visualized!")
                    
                    st.markdown("**📌 What You're Seeing:**")
                    st.write("""
                    - **Colored regions:** Network's predicted class for each area
                    - **Data points:** Actual training examples (colors show true class)
                    - **Boundary:** Where the network changes its prediction
                    - **Goal:** Boundary should match data distribution
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================
# PAGE 4: Advanced Analysis
# ============================================================
elif page == "📊 Advanced Analysis":
    st.header("📊 Advanced Analysis and Model Comparison")
    
    st.write("""
    Compare multiple model architectures and see which performs best!
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Settings")
        
        compare_btn = st.button(
            "🏆 Compare Models",
            key="compare_models",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        if compare_btn:
            with st.spinner("Training models... This may take a minute."):
                try:
                    # Generate data
                    X_train, X_test, y_train, y_test, _, _ = generate_dataset(
                        "moons",
                        n_samples=300
                    )
                    
                    # Train multiple models
                    models = {}
                    architectures = {
                        "Small\n(16 units)": [2, 16, 2],
                        "Medium\n(64 units)": [2, 64, 32, 16, 2],
                        "Large\n(256 units)": [2, 256, 128, 64, 32, 2],
                    }
                    
                    progress_placeholder = st.empty()
                    
                    for idx, (name, arch) in enumerate(architectures.items()):
                        progress_placeholder.text(f"Training {name.replace(chr(10), ' ')}...")
                        
                        model = build_model(arch)
                        model.fit(
                            X_train, y_train,
                            validation_split=0.2,
                            epochs=50,
                            verbose=0
                        )
                        models[name] = model
                    
                    progress_placeholder.empty()
                    
                    # Compare models
                    advanced = AdvancedVisualizations()
                    fig = advanced.compare_models(models, X_test, y_test)
                    st.pyplot(fig, use_container_width=True)
                    
                    st.success("✅ Model comparison complete!")
                    
                    # Display individual metrics
                    st.subheader("📈 Individual Model Metrics")
                    
                    metrics_data = []
                    for name, model in models.items():
                        loss, acc = model.evaluate(X_test, y_test, verbose=0)
                        metrics_data.append({
                            "Model": name.replace("\n", " "),
                            "Test Loss": f"{loss:.4f}",
                            "Test Accuracy": f"{acc:.2%}"
                        })
                    
                    st.table(metrics_data)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================
# PAGE 5: About
# ============================================================
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🧠 Neural Network Visualizer")
        
        st.write("""
        An interactive tool for understanding neural networks through beautiful visualizations.
        
        **Key Features:**
        - 🏗️ Visualize network architectures
        - 📈 Track training progress
        - 🌍 Explore decision boundaries
        - 📊 Compare multiple models
        - 🎨 Interactive plots with full control
        
        **Technologies Used:**
        - **TensorFlow/Keras** - Deep learning
        - **Matplotlib** - Plotting
        - **Streamlit** - Web interface
        - **Scikit-learn** - ML utilities
        - **NumPy/Pandas** - Data processing
        """)
    
    with col2:
        st.subheader("📚 Learn More")
        
        st.write("""
        **Resources:**
        - [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
        - [TensorFlow Docs](https://www.tensorflow.org)
        - [Streamlit Docs](https://docs.streamlit.io)
        
        **GitHub:**
        - [View Source Code](https://github.com/sinu-ops/neural-networks-visualizer)
        - [Report Issues](https://github.com/sinu-ops/neural-networks-visualizer/issues)
        
        **Author:**
        - GitHub: [@sinu-ops](https://github.com/sinu-ops)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>🧠 Neural Network Visualizer</strong></p>
        <p>Made with ❤️ by Sinu-Ops</p>
        <p>⭐ <a href='https://github.com/sinu-ops/neural-networks-visualizer'>Star on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Neural Network Visualizer | © 2024 | <a href='https://github.com/sinu-ops'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)