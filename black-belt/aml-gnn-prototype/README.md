# AI-Native Compliance Officer: GNN Prototype

A sophisticated Graph Neural Network (GNN) implementation for detecting suspicious financial networks and money laundering patterns in Anti-Money Laundering (AML) systems.

## 🎯 Overview

This prototype demonstrates the core AI component of the AI-Native Compliance Officer system, focusing on:

- **Synthetic Financial Network Generation**: Creates realistic transaction networks with embedded suspicious patterns
- **Graph Neural Network Architecture**: Uses PyTorch Geometric with GAT and GCN layers
- **Suspicious Community Detection**: Identifies high-risk account clusters using graph algorithms
- **Compliance Reporting**: Generates regulatory-ready investigation reports
- **Interactive Visualization**: Creates comprehensive network visualizations

## 🏗️ Architecture

The prototype implements a hybrid GNN architecture:

```
Input Graph → GAT Layers → GCN Layer → Classification Head
     ↓
Node Features:
- Risk scores, transaction volumes, account age
- Network centrality measures
- Behavioral patterns

Edge Features:
- Transaction amounts, frequencies
- Temporal patterns
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- GPU recommended (but not required)

### Installation

1. **Clone or download the project files**:
   ```bash
   # Ensure you have these files:
   # - aml_gnn_prototype.py
   # - requirements.txt
   # - README.md
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: PyTorch Geometric installation may require specific versions depending on your PyTorch installation. If you encounter issues, visit [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

3. **Run the prototype**:
   ```bash
   python aml_gnn_prototype.py
   ```

### Basic Usage

```bash
# Run with default settings (1000 accounts, 8% suspicious ratio)
python aml_gnn_prototype.py

# Custom network size and suspicious ratio
python aml_gnn_prototype.py --accounts 2000 --suspicious_ratio 0.05

# Adjust training parameters
python aml_gnn_prototype.py --epochs 200 --hidden_dim 256

# Skip visualizations (faster execution)
python aml_gnn_prototype.py --no_viz

# Custom output directory
python aml_gnn_prototype.py --output_dir my_results
```

## 📊 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--accounts` | int | 1000 | Number of accounts in the synthetic network |
| `--suspicious_ratio` | float | 0.08 | Fraction of accounts involved in suspicious activities |
| `--epochs` | int | 150 | Number of training epochs for the GNN |
| `--hidden_dim` | int | 128 | Hidden layer dimensions in the neural network |
| `--output_dir` | str | 'aml_results' | Directory to save all output files |
| `--no_viz` | flag | False | Skip visualization generation (faster execution) |

## 📁 Output Files

The prototype generates several output files in the specified directory:

```
aml_results/
├── compliance_report_YYYYMMDD_HHMMSS.txt     # Regulatory compliance report
├── high_risk_accounts_YYYYMMDD_HHMMSS.csv    # CSV of flagged accounts
├── community_analysis_YYYYMMDD_HHMMSS.json   # Detailed community analysis
├── network_visualization.png                  # Network plots (if --no_viz not used)
└── training_history.png                      # Training metrics (if --no_viz not used)
```

### File Descriptions

1. **Compliance Report** (`compliance_report_*.txt`):
   - Executive summary of findings
   - High-priority investigation cases
   - Regulatory recommendations
   - Technical model details

2. **High-Risk Accounts** (`high_risk_accounts_*.csv`):
   - Account IDs flagged for investigation
   - Individual risk scores
   - Community associations

3. **Community Analysis** (`community_analysis_*.json`):
   - Detailed community detection results
   - Network topology metrics
   - Model performance statistics

4. **Visualizations** (`.png` files):
   - Network overview with risk-based coloring
   - Community detection results
   - Transaction volume heatmaps
   - Training loss and accuracy curves

## 🔍 Understanding the Results

### Risk Levels

- **HIGH RISK** (>70% confidence): Immediate investigation recommended, potential SAR filing
- **MEDIUM RISK** (40-70% confidence): Enhanced monitoring required
- **LOW RISK** (<40% confidence): Standard monitoring procedures

### Suspicious Patterns Detected

The system identifies several money laundering patterns:

1. **Circular Flows** (Layering): Money moving in circles to obscure origin
2. **Star Patterns** (Structuring): Central account with many small transactions
3. **Chain Patterns** (Placement): Sequential transfers through multiple accounts

### Community Metrics

- **Density**: How interconnected the community is
- **Centrality**: Importance of nodes within the network
- **Clustering**: Tendency of nodes to form tight groups
- **Transaction Volume**: Total monetary flow within the community

## 🛠️ Customization

### Modifying Network Generation

Edit the `FinancialNetworkGenerator` class to:
- Add new suspicious patterns
- Adjust transaction distributions
- Include additional node/edge features

### Enhancing the GNN Model

Modify the `AMLGraphNet` class to:
- Add more graph convolution layers
- Implement different attention mechanisms
- Include edge features in the model

### Custom Risk Scoring

Adjust the risk assessment logic in the `detect_suspicious_communities` method to incorporate domain-specific rules.

## 📈 Performance Expectations

### Default Configuration (1000 accounts):
- **Training Time**: 2-5 minutes (CPU), 30 seconds (GPU)
- **Memory Usage**: ~500MB RAM
- **Detection Accuracy**: >95% on synthetic data
- **False Positive Rate**: <5%

### Scaling Considerations:
- **10K accounts**: 10-20 minutes training
- **100K accounts**: 1-2 hours training, 8GB+ RAM recommended
- **1M+ accounts**: Consider distributed training or model optimization

## 🔧 Troubleshooting

### Common Issues

1. **PyTorch Geometric Installation**:
   ```bash
   # For CPU-only installation
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install torch-geometric
   ```

2. **Memory Issues with Large Networks**:
   - Reduce `--accounts` parameter
   - Use `--no_viz` flag
   - Increase system virtual memory

3. **Slow Training**:
   - Reduce `--epochs` parameter
   - Use GPU if available
   - Reduce `--hidden_dim` parameter

4. **Visualization Errors**:
   - Use `--no_viz` flag to skip plots
   - Ensure matplotlib backend is properly configured
   - Check display settings if running remotely

### Performance Optimization

For production deployments:

1. **Model Optimization**:
   - Use mixed precision training
   - Implement model quantization
   - Consider graph sampling for large networks

2. **Infrastructure**:
   - Deploy on GPU-enabled instances
   - Use distributed computing for very large graphs
   - Implement caching for feature computation

## 🧪 Example Use Cases

### Regulatory Compliance Testing
```bash
# Generate a large network for stress testing
python aml_gnn_prototype.py --accounts 5000 --suspicious_ratio 0.03 --epochs 100

# Focus on high-precision detection
python aml_gnn_prototype.py --suspicious_ratio 0.02 --epochs 250 --hidden_dim 256
```

### Research and Development
```bash
# Quick prototyping with smaller networks
python aml_gnn_prototype.py --accounts 500 --epochs 50 --no_viz

# Detailed analysis with comprehensive outputs
python aml_gnn_prototype.py --accounts 2000 --epochs 200 --output_dir detailed_analysis
```

## 📚 Technical Details

### Model Architecture

The GNN uses a hybrid approach:
- **Graph Attention Networks (GAT)**: Learns attention weights for different connections
- **Graph Convolutional Networks (GCN)**: Aggregates neighborhood information
- **Multi-layer Classification**: Maps graph features to risk scores

### Feature Engineering

**Node Features (11 dimensions)**:
- Financial: risk_score, account_age, transaction_volume, kyc_score
- Geographic: country_risk
- Behavioral: is_business, clustering_coefficient
- Network: degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality

**Edge Features (3 dimensions)**:
- amount (log-normalized)
- frequency (transaction count)
- days_since_last (temporal recency)

### Training Process

1. **Data Splitting**: 70% train, 10% validation, 20% test
2. **Loss Function**: Weighted cross-entropy (handles class imbalance)
3. **Optimization**: Adam optimizer with learning rate scheduling
4. **Regularization**: Dropout, batch normalization, early stopping
5. **Evaluation**: Accuracy, ROC-AUC, precision, recall, F1-score

## 🤝 Contributing

This is a prototype implementation. For production use, consider:

- Adding more sophisticated feature engineering
- Implementing real-time inference pipelines
- Adding model interpretability tools (SHAP, GradCAM)
- Integrating with actual banking data sources
- Adding privacy-preserving techniques

## 📄 License

This prototype is provided for educational and research purposes. Please ensure compliance with relevant financial regulations and data protection laws when adapting for production use.

## 🆘 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Consult the AI-Native Compliance Officer design document
4. Consider the theoretical background in graph neural networks and AML detection

---

**⚠️ Important Note**: This is a prototype using synthetic data. Real-world deployment requires:
- Regulatory approval and compliance validation
- Integration with actual banking systems
- Comprehensive testing with historical data
- Human oversight and validation processes
- Regular model updates and monitoring
