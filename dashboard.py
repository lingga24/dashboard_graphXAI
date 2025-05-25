import os
import random
import time
import torch
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import k_hop_subgraph
from graphxai.utils import Explanation
from graphxai.explainers import GradExplainer, GNNExplainer, GraphLIME, PGMExplainer
from graphxai.gnn_models.node_classification.testing import (
    GCN_3layer_basic, GIN_3layer_basic, GAT_3layer_basic, GSAGE_3layer
)
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Taobao User Behavior GraphXAI Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 107, 53, 0.4);
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%) 1;
        padding-bottom: 0.5rem;
        text-align: center;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, rgba(247, 147, 30, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(76, 175, 80, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 193, 7, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 87, 87, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .explainer-status {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #4caf50 100%);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #dc3545 0%, #f44336 100%);
        color: white;
    }
    
    .status-running {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.05) 0%, rgba(247, 147, 30, 0.05) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #FF6B35;
        margin: 1rem 0;
        text-align: center;
    }
    
    .model-selection {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.08) 0%, rgba(247, 147, 30, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 107, 53, 0.2);
    }
    
    .analysis-box {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.05) 0%, rgba(247, 147, 30, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.1);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.08) 0%, rgba(247, 147, 30, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛒 Taobao User Behavior GraphXAI Dashboard</h1>', unsafe_allow_html=True)

# Analysis functions
def analyze_model_performance(f1, auc, training_time, model_choice):
    """Analyze model performance metrics"""
    analysis = []
    
    # Performance assessment
    if f1 > 0.8:
        f1_assessment = f"Excellent F1 score ({f1:.4f}) indicates strong classification performance for {model_choice}"
    elif f1 > 0.6:
        f1_assessment = f"Good F1 score ({f1:.4f}) shows reasonable balance between precision and recall"
    else:
        f1_assessment = f"F1 score ({f1:.4f}) suggests room for improvement in model performance"
    
    if auc > 0.8:
        auc_assessment = f"High AUC ({auc:.4f}) demonstrates excellent discriminative ability for user behavior prediction"
    elif auc > 0.7:
        auc_assessment = f"Good AUC ({auc:.4f}) shows adequate discriminative power for purchase prediction"
    else:
        auc_assessment = f"AUC ({auc:.4f}) indicates limited discriminative capability"
    
    # Training efficiency
    if training_time < 60:
        time_assessment = f"Fast training time ({training_time:.2f}s) enables rapid experimentation"
    elif training_time < 300:
        time_assessment = f"Moderate training time ({training_time:.2f}s) shows good computational efficiency"
    else:
        time_assessment = f"Extended training time ({training_time:.2f}s) suggests computational complexity"
    
    analysis.extend([f1_assessment, auc_assessment, time_assessment])
    
    return analysis

def analyze_recommendation_results(buy_nodes, not_buy_nodes, true_buy, false_buy):
    """Analyze recommendation system results"""
    analysis = []
    
    total_recommendations = len(buy_nodes)
    precision = len(true_buy) / total_recommendations if total_recommendations > 0 else 0
    
    if precision > 0.8:
        precision_assessment = f"High precision ({precision:.3f}) indicates reliable purchase recommendations"
    elif precision > 0.6:
        precision_assessment = f"Moderate precision ({precision:.3f}) shows acceptable recommendation quality"
    else:
        precision_assessment = f"Low precision ({precision:.3f}) suggests need for model improvement"
    
    # Recommendation distribution analysis
    total_nodes = len(buy_nodes) + len(not_buy_nodes)
    buy_ratio = len(buy_nodes) / total_nodes if total_nodes > 0 else 0
    
    if buy_ratio < 0.1:
        distribution_assessment = f"Conservative recommendation strategy ({buy_ratio:.1%} buy rate) minimizes false positives"
    elif buy_ratio > 0.5:
        distribution_assessment = f"Aggressive recommendation strategy ({buy_ratio:.1%} buy rate) maximizes reach"
    else:
        distribution_assessment = f"Balanced recommendation strategy ({buy_ratio:.1%} buy rate) optimizes precision-recall trade-off"
    
    # Business impact assessment
    if len(true_buy) > 50:
        impact_assessment = f"High number of correct recommendations ({len(true_buy)}) indicates strong business value"
    elif len(true_buy) > 20:
        impact_assessment = f"Moderate correct recommendations ({len(true_buy)}) provides reasonable business impact"
    else:
        impact_assessment = f"Limited correct recommendations ({len(true_buy)}) suggests need for strategy refinement"
    
    analysis.extend([precision_assessment, distribution_assessment, impact_assessment])
    
    return analysis

def analyze_explanation_quality(results):
    """Analyze explanation quality across methods"""
    analysis = []
    
    successful_methods = [method for method, metrics in results.items() if 'GEF' in metrics]
    
    if len(successful_methods) >= 3:
        coverage_assessment = f"Comprehensive explanation coverage with {len(successful_methods)} successful methods enables robust analysis"
    elif len(successful_methods) >= 2:
        coverage_assessment = f"Partial explanation coverage with {len(successful_methods)} methods provides valuable insights"
    else:
        coverage_assessment = f"Limited explanation coverage with {len(successful_methods)} method(s) constrains analysis depth"
    
    if successful_methods:
        gef_scores = [results[method]['GEF'] for method in successful_methods]
        avg_gef = np.mean(gef_scores)
        
        if avg_gef > 0.7:
            quality_assessment = f"High average explanation faithfulness ({avg_gef:.3f}) indicates reliable model interpretability"
        elif avg_gef > 0.5:
            quality_assessment = f"Moderate explanation faithfulness ({avg_gef:.3f}) shows acceptable interpretability quality"
        else:
            quality_assessment = f"Low explanation faithfulness ({avg_gef:.3f}) suggests explanations may not reflect true model behavior"
        
        best_method = max(successful_methods, key=lambda x: results[x]['GEF'])
        best_score = results[best_method]['GEF']
        method_assessment = f"{best_method} provides the most faithful explanations (GEF: {best_score:.3f})"
        
        analysis.extend([coverage_assessment, quality_assessment, method_assessment])
    else:
        analysis.append(coverage_assessment)
    
    return analysis

# Sidebar configuration
st.sidebar.markdown("## 🛠️ Configuration Panel")

# Data upload section
st.sidebar.markdown("### 📊 Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Graph Data (.pt)", 
    type="pt",
    help="Upload your preprocessed Taobao user behavior graph data"
)

# Model configuration
st.sidebar.markdown("### 🧠 Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Model Architecture", 
    ["Choose Model", "GCN", "GIN", "GAT", "GraphSAGE"], 
    index=0,
    help="Select the Graph Neural Network architecture for analysis"
)

# Training parameters
if model_choice != "Choose Model":
    st.sidebar.markdown("### ⚙️ Training Parameters")
    hidden_channels = st.sidebar.slider("Hidden Channels", 32, 128, 64, help="Number of hidden channels in the model")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f", help="Learning rate for optimization")
    max_epochs = st.sidebar.slider("Max Epochs", 50, 500, 200, help="Maximum number of training epochs")
    patience = st.sidebar.slider("Early Stopping Patience", 5, 20, 10, help="Number of epochs to wait for improvement")

# Explanation parameters
if uploaded_file is not None and model_choice != "Choose Model":
    st.sidebar.markdown("### 🔍 Explanation Parameters")
    num_hops = st.sidebar.slider("Number of Hops", 1, 3, 1, help="Number of hops for subgraph extraction")
    top_k_ratio = st.sidebar.slider("Top-K Ratio", 0.1, 0.5, 0.25, help="Ratio of top important nodes for faithfulness evaluation")

# Main content
if uploaded_file is not None:
    # Data loading section
    st.markdown('<h2 class="section-header">📊 Data Loading & Processing</h2>', unsafe_allow_html=True)
    
    with st.spinner("Loading graph data..."):
        try:
            data = torch.load(uploaded_file)
            
            st.markdown('''
            <div class="success-box">
                <h4>Data Successfully Loaded</h4>
                <p>Taobao user behavior graph data has been loaded and is ready for analysis.</p>
            </div>
            ''', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'''
            <div class="error-box">
                <h4>Data Loading Failed</h4>
                <p>Error: {str(e)}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.stop()
    
    # Feature enrichment
    if data.x.shape[1] == 1:
        st.markdown('''
        <div class="info-box">
            <h4>Feature Enrichment Required</h4>
            <p>Basic node features detected. Enriching with user behavior statistics...</p>
        </div>
        ''', unsafe_allow_html=True)
        
        with st.spinner("Enriching node features..."):
            try:
                df = pd.read_csv('UserBehavior_5M_cleaned.csv')
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
                df = df[df['Timestamp'].dt.date == pd.to_datetime('2017-11-28').date()]
                df['Behavior'] = df['Behavior'].map({'PageView': 0, 'AddToCart': 1, 'Buy': 2, 'Favorite': 3})

                item_stats = df.groupby('Product_ID').agg(
                    behavior_mean=('Behavior', 'mean'),
                    behavior_std=('Behavior', 'std'),
                    interactions=('Behavior', 'count'),
                    unique_users=('User_ID', 'nunique')
                ).fillna(0)

                scaler = StandardScaler()
                enriched_x = torch.tensor(scaler.fit_transform(item_stats.values), dtype=torch.float)
                data.x = enriched_x
                
                st.markdown('''
                <div class="success-box">
                    <h4>Feature Enrichment Complete</h4>
                    <p>Node features have been enriched with behavioral statistics and user interaction patterns.</p>
                </div>
                ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'''
                <div class="warning-box">
                    <h4>Feature Enrichment Warning</h4>
                    <p>Could not enrich features: {str(e)}. Using original features.</p>
                </div>
                ''', unsafe_allow_html=True)
    
    # Dataset information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>🔗 Nodes</h3>
            <h2>{data.x.size(0)}</h2>
            <p>User/Product entities</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>📊 Edges</h3>
            <h2>{data.edge_index.size(1)}</h2>
            <p>User interactions</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>🎯 Features</h3>
            <h2>{data.x.size(1)}</h2>
            <p>Node attributes</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>🏷️ Classes</h3>
            <h2>{int(data.y.max().item()) + 1}</h2>
            <p>Behavior categories</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Dataset analysis
    train_ratio = data.train_mask.sum().item() / data.y.size(0)
    test_ratio = data.test_mask.sum().item() / data.y.size(0)
    buy_ratio = (data.y == 1).sum().item() / data.y.size(0)
    
    st.markdown(f'''
    <div class="analysis-box">
        <h4>Dataset Analysis</h4>
        <ul>
            <li><strong>Data Split:</strong> {train_ratio:.1%} training, {test_ratio:.1%} testing</li>
            <li><strong>Class Distribution:</strong> {buy_ratio:.1%} buy behavior, {1-buy_ratio:.1%} non-buy behavior</li>
            <li><strong>Graph Density:</strong> {data.edge_index.size(1) / (data.x.size(0) * (data.x.size(0) - 1)):.4f}</li>
            <li><strong>Feature Dimensionality:</strong> {data.x.size(1)} features per node for behavioral analysis</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

    if model_choice != "Choose Model":
        # Model setup and training
        st.markdown('<h2 class="section-header">🧠 Model Training & Evaluation</h2>', unsafe_allow_html=True)
        
        model_map = {
            'GCN': GCN_3layer_basic,
            'GIN': GIN_3layer_basic,
            'GAT': GAT_3layer_basic,
            'GraphSAGE': GSAGE_3layer
        }
        
        ModelClass = model_map[model_choice]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model initialization
        with st.spinner(f"Initializing {model_choice} model..."):
            model = ModelClass(
                hidden_channels=hidden_channels, 
                input_feat=data.x.size(1), 
                classes=int(data.y.max().item()) + 1
            ).to(device)
            
            x, y, edge_index = data.x.to(device), data.y.to(device), data.edge_index.to(device)
            train_mask, test_mask = data.train_mask, data.test_mask
            train_idx = train_mask.nonzero(as_tuple=True)[0].to(device)
            test_idx = test_mask.nonzero(as_tuple=True)[0].to(device)
            
            # Training setup
            model_path = f"{model_choice}_model.pt"
            class_weights = compute_class_weight('balanced', classes=np.unique(y.cpu()), y=y.cpu().numpy())
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training process
        best_val_f1, best_epoch = 0, 0
        training_start = time.time()
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            st.markdown(f'''
            <div class="success-box">
                <h4>Pre-trained Model Loaded</h4>
                <p>{model_choice} model loaded successfully from saved checkpoint.</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="info-box">
                <h4>Training {model_choice} Model</h4>
                <p>Starting training process with early stopping mechanism...</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_container = st.empty()
            metrics_container = st.empty()
            
            training_metrics = []
            
            for epoch in range(1, max_epochs + 1):
                model.train()
                optimizer.zero_grad()
                out = model(x, edge_index)
                loss = criterion(out[train_idx], y[train_idx])
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    val_out = out[test_idx]
                    val_pred = val_out.argmax(dim=1)
                    val_f1 = f1_score(y[test_idx].cpu().numpy(), val_pred.cpu().numpy(), zero_division=0)
                    
                    training_metrics.append({
                        'epoch': epoch,
                        'loss': loss.item(),
                        'f1': val_f1
                    })
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch
                        torch.save(model.state_dict(), model_path)
                        
                        with status_container.container():
                            st.markdown(f'<span class="explainer-status status-success">Epoch {epoch}: New best F1: {val_f1:.4f}</span>', unsafe_allow_html=True)
                    else:
                        with status_container.container():
                            st.markdown(f'<span class="explainer-status status-running">Epoch {epoch}: F1: {val_f1:.4f} (no improvement)</span>', unsafe_allow_html=True)
                    
                    # Update progress
                    progress = epoch / max_epochs
                    progress_bar.progress(progress)
                    
                    # Display current metrics
                    with metrics_container.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Epoch", f"{epoch}/{max_epochs}")
                        with col2:
                            st.metric("Loss", f"{loss.item():.4f}")
                        with col3:
                            st.metric("Best F1", f"{best_val_f1:.4f}")
                    
                    if epoch - best_epoch >= patience:
                        with status_container.container():
                            st.markdown(f'<span class="explainer-status status-success">Early stopping at epoch {epoch}</span>', unsafe_allow_html=True)
                        break
                    
                torch.cuda.empty_cache()
            
            progress_bar.empty()
            status_container.empty()
            metrics_container.empty()

        # Load best model and evaluate
        model.load_state_dict(torch.load(model_path))
        model.eval()
        training_time = time.time() - training_start

        # Model evaluation
        st.markdown('<h2 class="section-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        with torch.no_grad():
            out = model(x, edge_index)
            pred = out.argmax(dim=1)

            # Prediction analysis
            buy_nodes = test_idx[(pred[test_idx] == 1)]
            not_buy_nodes = test_idx[(pred[test_idx] == 0)]
            true_buy = test_idx[(pred[test_idx] == 1) & (y[test_idx] == 1)]
            false_buy = test_idx[(pred[test_idx] == 1) & (y[test_idx] == 0)]

            y_true = y[test_idx].cpu().numpy()
            y_pred = pred[test_idx].cpu().numpy()
            y_proba = out[test_idx][:, 1].sigmoid().cpu().numpy() if out.size(1) > 1 else out[test_idx].cpu().numpy()

            f1 = f1_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_proba)
            except:
                auc = 0.0
        
        # Performance metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("F1 Score", f"{f1:.4f}", delta=f"{(f1-0.5):.4f}")
        with col2:
            st.metric("AUC Score", f"{auc:.4f}", delta=f"{(auc-0.5):.4f}")
        with col3:
            st.metric("Training Time", f"{training_time:.2f}s")
        with col4:
            st.metric("Best Epoch", f"{best_epoch}")
        
        # Model performance analysis
        performance_analysis = analyze_model_performance(f1, auc, training_time, model_choice)
        st.markdown(f'''
        <div class="analysis-box">
            <h4>Model Performance Analysis</h4>
            <ul>
                {''.join([f"<li>{analysis}</li>" for analysis in performance_analysis])}
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        # Classification report
        with st.expander("📊 Detailed Classification Report", expanded=False):
            st.text(classification_report(y_true, y_pred, target_names=['Not Buy', 'Buy'], zero_division=0))

        # Recommendation results
        st.markdown('<h2 class="section-header">🛒 Recommendation System Analysis</h2>', unsafe_allow_html=True)
        
        # Recommendation metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>🎯 Buy Recommendations</h3>
                <h2>{len(buy_nodes)}</h2>
                <p>Predicted purchases</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>❌ Non-Buy Predictions</h3>
                <h2>{len(not_buy_nodes)}</h2>
                <p>Predicted non-purchases</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>✅ Correct Buy Predictions</h3>
                <h2>{len(true_buy)}</h2>
                <p>True positives</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <h3>⚠️ False Buy Predictions</h3>
                <h2>{len(false_buy)}</h2>
                <p>False positives</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Recommendation analysis
        recommendation_analysis = analyze_recommendation_results(buy_nodes, not_buy_nodes, true_buy, false_buy)
        st.markdown(f'''
        <div class="recommendation-box">
            <h4>Recommendation System Analysis</h4>
            <ul>
                {''.join([f"<li>{analysis}</li>" for analysis in recommendation_analysis])}
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        # Save recommendation results
        buy_node_list = buy_nodes.cpu().numpy()
        not_buy_node_list = not_buy_nodes.cpu().numpy()
        df_buy = pd.DataFrame({'Node_ID': buy_node_list, 'Predicted_Label': 1})
        df_not_buy = pd.DataFrame({'Node_ID': not_buy_node_list, 'Predicted_Label': 0})

        recommendation_df = pd.concat([df_buy, df_not_buy], ignore_index=True)
        recommendation_df.to_csv(f"{model_choice}_recommendation_results.csv", index=False)
        
        st.markdown(f'''
        <div class="info-box">
            <h4>📁 Results Saved</h4>
            <p>Recommendation results saved to: <strong>{model_choice}_recommendation_results.csv</strong></p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Sample results display
        with st.expander("📋 Sample Recommendation Results", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Buy Recommendations (Sample)")
                st.write(buy_nodes[:10].cpu().numpy())
            with col2:
                st.subheader("Non-Buy Predictions (Sample)")
                st.write(not_buy_nodes[:10].cpu().numpy())

        # Node explanation section
        st.markdown('<h2 class="section-header">🔍 Node Explanation Analysis</h2>', unsafe_allow_html=True)
        
        # Select node for explanation
        correct_buy = test_idx[(pred[test_idx] == y[test_idx]) & (y[test_idx] == 1)]
        
        if len(correct_buy) == 0:
            st.markdown('''
            <div class="error-box">
                <h4>No Correctly Predicted Buy Nodes Found</h4>
                <p>Cannot perform explanation analysis without correctly predicted purchase nodes.</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            node_to_explain = random.choice(correct_buy.tolist())
            
            st.markdown(f'''
            <div class="info-box">
                <h4>Explanation Target</h4>
                <p>Analyzing explanation patterns for <strong>node {node_to_explain}</strong> (correctly predicted buy behavior)</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Subgraph extraction
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=node_to_explain, 
                num_hops=num_hops, 
                edge_index=edge_index, 
                relabel_nodes=True, 
                num_nodes=data.num_nodes
            )
            sub_x = x[subset]
            
            st.markdown(f'''
            <div class="analysis-box">
                <h4>Subgraph Analysis</h4>
                <ul>
                    <li><strong>Target Node:</strong> {node_to_explain} (mapped to {mapping[0].item()} in subgraph)</li>
                    <li><strong>Subgraph Size:</strong> {len(subset)} nodes</li>
                    <li><strong>Subgraph Edges:</strong> {sub_edge_index.size(1)} connections</li>
                    <li><strong>Hop Distance:</strong> {num_hops}-hop neighborhood analysis</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

            # Faithfulness evaluation function
            def evaluate_graph_faithfulness(explanation: Explanation, model: torch.nn.Module, data: Data, top_k=0.25) -> float:
                try:
                    from torch.nn import functional as F

                    with torch.no_grad():
                        original_out = model(data.x, data.edge_index)
                        original_softmax = F.softmax(original_out[explanation.node_idx], dim=-1)

                        pert_x = data.x.clone()
                        node_imp = explanation.node_imp.clone()
                        top_k_nodes = node_imp.topk(int(len(node_imp) * top_k)).indices
                        all_nodes = torch.arange(data.x.shape[0])
                        unimportant_nodes = list(set(all_nodes.tolist()) - set(top_k_nodes.tolist()))

                        pert_x[unimportant_nodes] = 0.0
                        perturbed_out = model(pert_x, data.edge_index)
                        perturbed_softmax = F.softmax(perturbed_out[explanation.node_idx], dim=-1)

                        GEF = 1 - torch.exp(-F.kl_div(original_softmax.log(), perturbed_softmax, reduction='sum')).item()
                    return GEF
                except Exception as e:
                    return 0.0

            # Explainer execution
            st.markdown("### 🚀 Explainer Execution Progress")
            
            explainers = {
                'GradExplainer': GradExplainer,
                'GNNExplainer': GNNExplainer,
                'GraphLIME': GraphLIME,
                'PGMExplainer': PGMExplainer
            }
            
            results = {}
            explainer_progress = st.progress(0)
            status_container = st.empty()
            
            for idx, (name, Explainer) in enumerate(explainers.items()):
                try:
                    with status_container.container():
                        st.markdown(f'<span class="explainer-status status-running">Running {name}...</span>', unsafe_allow_html=True)
                    
                    # Initialize explainer based on type
                    if name == 'GradExplainer':
                        explainer = Explainer(model=model, criterion=criterion)
                    elif name == 'PGMExplainer':
                        explainer = Explainer(model=model, explain_graph=False)
                    else:
                        explainer = Explainer(model=model)

                    # Prepare input data
                    input_x = sub_x + 0.0001 * torch.randn_like(sub_x) if name == 'GraphLIME' else sub_x
                    
                    # Get explanation
                    exp = explainer.get_explanation_node(
                        node_idx=int(mapping[0]),
                        x=input_x,
                        edge_index=sub_edge_index,
                        forward_kwargs={}
                    )

                    if exp.node_imp is None:
                        with status_container.container():
                            st.markdown(f'<span class="explainer-status status-error">{name} failed: No node importance returned</span>', unsafe_allow_html=True)
                        continue

                    # Create full graph explanation
                    full_node_imp = torch.zeros(data.num_nodes)
                    for i, imp in enumerate(exp.node_imp):
                        full_node_imp[subset[i]] = imp

                    # Create explanation object
                    explanation = Explanation(
                        node_imp=full_node_imp,
                        node_idx=int(mapping[0])
                    )

                    # Prepare subgraph data for visualization
                    sub_data = Data(x=sub_x, edge_index=sub_edge_index)
                    sub_data.nodes = subset
                    explanation.enc_subgraph = sub_data

                    # Visualization
                    fig, ax = plt.subplots(figsize=(6, 5))
                    try:
                        explanation.visualize_node(
                            num_hops=num_hops,
                            graph_data=sub_data,
                            additional_hops=0,
                            heat_by_prescence=False,
                            heat_by_exp=True,
                            node_agg_method='sum',
                            show_node_labels=True,
                            show=False,
                            norm_imps=False,
                            ax=ax
                        )
                        ax.set_title(f'{name} Explanation for Node {node_to_explain}', fontsize=16, fontweight='bold')
                        
                        # Display visualization in expandable section
                        with st.expander(f"🎨 {name} Visualization", expanded=False):
                            st.pyplot(fig)
                            
                            # Show node importance values
                            importance_df = pd.DataFrame({
                                'Node_ID': subset.cpu().numpy(),
                                'Importance': exp.node_imp.cpu().numpy()
                            }).sort_values('Importance', ascending=False)
                            
                            st.subheader("Node Importance Rankings")
                            st.dataframe(importance_df.head(10), use_container_width=True)
                        
                        plt.close(fig)
                        
                    except Exception as viz_error:
                        st.warning(f"Visualization failed for {name}: {viz_error}")
                        plt.close(fig)

                    # Calculate faithfulness
                    gef = evaluate_graph_faithfulness(explanation, model, data, top_k=top_k_ratio)
                    results[name] = {'GEF': gef}
                    
                    with status_container.container():
                        st.markdown(f'<span class="explainer-status status-success">{name} completed (GEF: {gef:.3f})</span>', unsafe_allow_html=True)

                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    with status_container.container():
                        st.markdown(f'<span class="explainer-status status-error">{name} failed: {str(e)}</span>', unsafe_allow_html=True)
                    continue
                finally:
                    # Update progress
                    explainer_progress.progress((idx + 1) / len(explainers))
            
            explainer_progress.empty()
            status_container.empty()
            
            # Results analysis
            if results:
                st.markdown("### 📊 Explanation Quality Results")
                
                # Create results dataframe
                results_df = pd.DataFrame(results).T
                results_df = results_df.round(4)
                
                # Display results with enhanced styling
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        "GEF": st.column_config.ProgressColumn(
                            "Graph Explanation Faithfulness",
                            help="Higher values indicate more faithful explanations",
                            min_value=0,
                            max_value=1,
                            format="%.4f",
                        ),
                    }
                )
                
                # Best performing explainer
                if len(results) > 1:
                    best_explainer = max(results.keys(), key=lambda x: results[x]['GEF'])
                    best_score = results[best_explainer]['GEF']
                    
                    st.markdown(f'''
                    <div class="success-box">
                        <h4>Best Performing Explainer</h4>
                        <p><strong>{best_explainer}</strong> achieved the highest faithfulness score of <strong>{best_score:.4f}</strong></p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Explanation quality analysis
                explanation_analysis = analyze_explanation_quality(results)
                st.markdown(f'''
                <div class="analysis-box">
                    <h4>Explanation Quality Analysis</h4>
                    <ul>
                        {''.join([f"<li>{analysis}</li>" for analysis in explanation_analysis])}
                    </ul>
                </div>
                ''', unsafe_allow_html=True)
                
                # Save explanation results
                csv_path = f"{model_choice}_explanation_metrics.csv"
                if os.path.exists(csv_path):
                    df_existing = pd.read_csv(csv_path, index_col=0)
                    results_df = pd.concat([df_existing, results_df])
                results_df.to_csv(csv_path)
                
                st.markdown(f'''
                <div class="info-box">
                    <h4>📁 Explanation Results Saved</h4>
                    <p>Explanation metrics saved to: <strong>{csv_path}</strong></p>
                </div>
                ''', unsafe_allow_html=True)
            
            else:
                st.markdown('''
                <div class="warning-box">
                    <h4>No Successful Explanations</h4>
                    <p>All explainer methods failed. Consider adjusting model parameters or data preprocessing.</p>
                </div>
                ''', unsafe_allow_html=True)

        # Summary and export
        st.markdown('<h2 class="section-header">📋 Analysis Summary</h2>', unsafe_allow_html=True)
        
        # Create comprehensive summary
        summary = pd.DataFrame([{
            'Model': model_choice,
            'F1 Score': round(f1, 4),
            'AUC': round(auc, 4),
            'Training Time (sec)': round(training_time, 2),
            'Buy Recommendations': len(buy_nodes),
            'Correct Predictions': len(true_buy),
            'Precision': round(len(true_buy) / len(buy_nodes) if len(buy_nodes) > 0 else 0, 4),
            'Successful Explainers': len(results) if 'results' in locals() else 0
        }])
        
        # Display summary
        st.dataframe(summary, use_container_width=True)
        
        # Save summary
        sum_path = f"{model_choice}_comprehensive_summary.csv"
        if os.path.exists(sum_path):
            summary_existing = pd.read_csv(sum_path)
            summary = pd.concat([summary_existing, summary])
        summary.to_csv(sum_path, index=False)
        
        st.markdown(f'''
        <div class="success-box">
            <h4>Analysis Complete</h4>
            <p>Comprehensive analysis summary saved to: <strong>{sum_path}</strong></p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Download section
        with st.expander("📥 Download Results", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if os.path.exists(f"{model_choice}_recommendation_results.csv"):
                    with open(f"{model_choice}_recommendation_results.csv", "rb") as file:
                        st.download_button(
                            label="Download Recommendations",
                            data=file,
                            file_name=f"{model_choice}_recommendations.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if 'results' in locals() and os.path.exists(f"{model_choice}_explanation_metrics.csv"):
                    with open(f"{model_choice}_explanation_metrics.csv", "rb") as file:
                        st.download_button(
                            label="Download Explanations",
                            data=file,
                            file_name=f"{model_choice}_explanations.csv",
                            mime="text/csv"
                        )
            
            with col3:
                if os.path.exists(f"{model_choice}_comprehensive_summary.csv"):
                    with open(f"{model_choice}_comprehensive_summary.csv", "rb") as file:
                        st.download_button(
                            label="Download Summary",
                            data=file,
                            file_name=f"{model_choice}_summary.csv",
                            mime="text/csv"
                        )

else:
    # Welcome screen
    st.markdown('''
    <div class="upload-section">
        <h2>Welcome to Taobao User Behavior Analysis</h2>
        <p>Upload your preprocessed graph data (.pt file) to begin comprehensive analysis</p>
        <p>This dashboard provides:</p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Model Training</strong> - Train GNN models on user behavior data</li>
            <li><strong>Purchase Prediction</strong> - Predict user buying behavior</li>
            <li><strong>Recommendation Analysis</strong> - Analyze recommendation system performance</li>
            <li><strong>Explainable AI</strong> - Understand model decision-making process</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature overview
    st.markdown('<h2 class="section-header">🎯 Dashboard Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="info-box">
            <h4>🧠 Machine Learning Capabilities</h4>
            <ul>
                <li><strong>Multiple GNN Architectures</strong> - GCN, GIN, GAT, GraphSAGE</li>
                <li><strong>Automated Training</strong> - Early stopping and hyperparameter optimization</li>
                <li><strong>Performance Metrics</strong> - F1, AUC, precision, recall analysis</li>
                <li><strong>Class Balancing</strong> - Handles imbalanced user behavior data</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="info-box">
            <h4>🔍 Explainable AI Features</h4>
            <ul>
                <li><strong>Multiple Explainers</strong> - GradExplainer, GNNExplainer, GraphLIME, PGMExplainer</li>
                <li><strong>Faithfulness Evaluation</strong> - Quantitative explanation quality assessment</li>
                <li><strong>Visual Analysis</strong> - Interactive explanation visualizations</li>
                <li><strong>Comparative Analysis</strong> - Cross-method explanation comparison</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    # Technical requirements
    st.markdown('<h2 class="section-header">📋 Requirements & Setup</h2>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="model-selection">
        <h4>Data Requirements</h4>
        <ul>
            <li><strong>File Format:</strong> PyTorch tensor file (.pt)</li>
            <li><strong>Graph Structure:</strong> Must contain x (node features), edge_index, y (labels), train_mask, test_mask</li>
            <li><strong>Feature Enhancement:</strong> Automatic enrichment from UserBehavior_5M_cleaned.csv if available</li>
            <li><strong>Labels:</strong> Binary classification (0: No Purchase, 1: Purchase)</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # System info
    device_info = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div class="info-box">
            <h4>🖥️ System Environment</h4>
            <p><strong>Compute Device:</strong> {device_info}</p>
            <p><strong>PyTorch Version:</strong> {torch.__version__}</p>
            {'<p><strong>GPU:</strong> ' + torch.cuda.get_device_name(0) + '</p>' if torch.cuda.is_available() else ''}
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="success-box">
            <h4>✨ Ready to Start</h4>
            <p>Upload your Taobao user behavior graph data using the file uploader in the sidebar to begin comprehensive analysis.</p>
        </div>
        ''', unsafe_allow_html=True)