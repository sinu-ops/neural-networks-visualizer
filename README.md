# 🧠 Neural Network Visualizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow 2.7+](https://img.shields.io/badge/tensorflow-2.7%2B-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/sinu-ops/neural-networks-visualizer?style=social)](https://github.com/sinu-ops/neural-networks-visualizer)

> 🎓 An interactive web-based tool for understanding neural networks through beautiful visualizations. Perfect for students, educators, and ML enthusiasts!

**[🌐 Live Demo](https://neural-networks-visualizer-sinu.streamlit.app)** | **[📖 Documentation](#documentation)** | **[🤝 Contributing](#contributing)**

---

## 🎨 Features

### 🏗️ **Network Architecture Visualization**
- Interactive neuron and connection visualization
- Adjustable layer sizes and depth
- Real-time architecture updates
- Weight magnitude indicators

### 📈 **Training Progress Tracking**
- Real-time loss and accuracy curves
- Training vs validation comparison
- Learning rate impact analysis
- Epoch-by-epoch monitoring

### 🌍 **Decision Boundary Exploration**
- Visualize how networks classify regions
- Compare different dataset types (Moons, Circles)
- Understand non-linear decision boundaries
- Interactive plot exploration

### 📊 **Advanced Model Analysis**
- Compare multiple architectures
- Model performance metrics
- Confusion matrices
- Feature importance analysis

### 🎮 **Interactive Features**
- Customize network architecture in real-time
- Adjust training hyperparameters
- Zoom, pan, and save visualizations
- Export plots as images

---

## 🚀 Quick Start

### **Live Demo**
Visit the deployed app: [Neural Network Visualizer](https://neural-networks-visualizer-sinu.streamlit.app)

### **Local Installation**

**Prerequisites:**
- Python 3.9 or higher
- pip (Python package manager)

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/sinu-ops/neural-networks-visualizer.git
cd neural-networks-visualizer

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run Streamlit app
streamlit run app.py

# 6. Open browser
# App will open at: http://localhost:8501