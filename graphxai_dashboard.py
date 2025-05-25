import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

# Import GraphXAI components
from graphxai.datasets import ShapeGGen
from graphxai.explainers import GradExplainer, GNNExplainer, PGMExplainer, GraphLIME
from graphxai.metrics.base import graph_exp_acc, graph_exp_faith
from graphxai.utils import Explanation, EnclosingSubgraph
from graphxai.gnn_models.node_classification.testing import (
    GCN_3layer_basic, GIN_3layer_basic, GAT_3layer_basic, GSAGE_3layer
)
from graphxai.gnn_models.node_classification import train, test

# Page config
st.set_page_config(
    page_title="GraphXAI Explainer Dashboard",
    page_icon="🔬",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%) 1;
        padding-bottom: 0.5rem;
        text-align: center;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
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
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 0.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .highlight-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .dataset-info {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .analysis-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">GraphXAI Explainer Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Configuration Panel")

# Dataset parameters
st.sidebar.markdown("### Dataset Parameters")
model_layers = st.sidebar.slider("Model Layers", 2, 5, 3, help="Number of layers in the GNN model")
num_subgraphs = st.sidebar.slider("Number of Subgraphs", 4, 16, 8, help="Number of subgraphs to generate")
subgraph_size = st.sidebar.slider("Subgraph Size", 3, 10, 5, help="Size of each subgraph")
prob_connection = st.sidebar.slider("Connection Probability", 0.1, 0.9, 0.3, help="Probability of edge connections")

# Model parameters
st.sidebar.markdown("### Model Parameters")
model_choice = st.sidebar.selectbox("Model Architecture", ['GCN', 'GIN', 'GAT', 'GSAGE'], index=0,
                                   help="Choose the GNN architecture")
hidden_channels = st.sidebar.slider("Hidden Channels", 16, 128, 32, help="Number of hidden channels")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f",
                                  help="Learning rate for training")
epochs = st.sidebar.slider("Training Epochs", 50, 500, 300, help="Number of training epochs")

# Explanation parameters
st.sidebar.markdown("### Explanation Parameters")
node_idx_input = st.sidebar.number_input("Node Index to Explain", 0, 100, 8, 
                                         help="Index of the node to generate explanations for")
num_hops = st.sidebar.slider("Number of Hops", 1, 4, 2, help="Number of hops for subgraph extraction")

# Run button
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("Run Complete Analysis", type="primary", 
                                 help="Start the full analysis pipeline")

# SafeGraphLIME class
class SafeGraphLIME(GraphLIME):
    def get_explanation_node(self, node_idx, x, edge_index, *args, **kwargs):
        self.model = self.model.to(x.device)
        
        try:
            result = super().get_explanation_node(node_idx, x, edge_index, *args, **kwargs)
            return result
        except Exception as e:
            try:
                result = super().get_explanation_node(node_idx, edge_index, x, *args, **kwargs)
                return result
            except Exception as e2:
                raise e2

# Convert to full explanation function
def convert_to_full_explanation(exp, node_idx, subset, sub_edge_index, mapping, edge_mask, data):
    full_node_imp = torch.zeros(data.num_nodes)
    
    if hasattr(exp, 'feature_imp') and exp.feature_imp is not None and exp.node_imp is None:
        if hasattr(exp, 'node_idx'):
            target_node = exp.node_idx
            if isinstance(target_node, torch.Tensor):
                target_node = target_node.item()
            mean_imp = exp.feature_imp.mean().item()
            full_node_imp[target_node] = mean_imp
    else:
        if exp.node_imp is not None:
            for i, imp in enumerate(exp.node_imp):
                if i < len(subset):
                    full_node_imp[subset[i]] = imp
    
    full_exp = Explanation(
        node_imp=full_node_imp,
        node_idx=node_idx
    )
    
    full_exp.enc_subgraph = EnclosingSubgraph(
        nodes=subset,
        edge_index=sub_edge_index,
        inv=mapping,
        edge_mask=edge_mask,
        directed=False
    )
    
    full_exp.node_reference = {}
    for sub_idx, full_idx in enumerate(subset):
        full_exp.node_reference[full_idx.item()] = sub_idx
    
    return full_exp

# ExplanationCompat class
class ExplanationCompat(Explanation):
    def __init__(self, enc_subgraph=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc_subgraph = enc_subgraph

# Helper function for visualization data
def get_data_from_explanation(exp, full_data):
    subset = exp.enc_subgraph.nodes
    edge_index = exp.enc_subgraph.edge_index
    return Data(
        x=full_data.x[subset],
        edge_index=edge_index,
        y=full_data.y[subset]
    )

# Analysis functions
def analyze_model_performance(metrics):
    """Analyze model performance metrics"""
    analysis = []
    
    # Performance assessment
    if metrics['f1'] > 0.8:
        f1_assessment = "Excellent F1 score indicates strong overall performance"
    elif metrics['f1'] > 0.6:
        f1_assessment = "Good F1 score shows reasonable balance between precision and recall"
    else:
        f1_assessment = "F1 score suggests room for improvement in model performance"
    
    if metrics['auroc'] > 0.8:
        auroc_assessment = "High AUROC demonstrates excellent discriminative ability"
    elif metrics['auroc'] > 0.7:
        auroc_assessment = "Good AUROC shows adequate discriminative power"
    else:
        auroc_assessment = "AUROC indicates limited discriminative capability"
    
    analysis.extend([f1_assessment, auroc_assessment])
    
    # Model reliability
    precision_recall_diff = abs(metrics['prec'] - metrics['rec'])
    if precision_recall_diff < 0.1:
        balance_assessment = "Well-balanced precision and recall indicate stable performance"
    else:
        if metrics['prec'] > metrics['rec']:
            balance_assessment = "Higher precision suggests conservative predictions with fewer false positives"
        else:
            balance_assessment = "Higher recall indicates comprehensive detection with more false positives"
    
    analysis.append(balance_assessment)
    
    return analysis

def analyze_explanation_metrics(combined_df):
    """Analyze explanation quality metrics"""
    analysis = []
    
    # GEA Analysis
    gea_scores = combined_df['Graph Explanation Accuracy (GEA)'].values
    avg_gea = np.mean(gea_scores)
    std_gea = np.std(gea_scores)
    
    if avg_gea > 0.7:
        gea_assessment = f"High average GEA ({avg_gea:.3f}) indicates explanations align well with ground truth"
    elif avg_gea > 0.5:
        gea_assessment = f"Moderate average GEA ({avg_gea:.3f}) shows reasonable explanation quality"
    else:
        gea_assessment = f"Low average GEA ({avg_gea:.3f}) suggests explanations may not accurately reflect true importance"
    
    if std_gea < 0.1:
        consistency_assessment = f"Low variance ({std_gea:.3f}) indicates consistent explanation quality across methods"
    else:
        consistency_assessment = f"High variance ({std_gea:.3f}) shows significant differences in explanation quality between methods"
    
    # GEF Analysis
    gef_scores = combined_df['Graph Explanation Faithfulness (GEF)'].values
    avg_gef = np.mean(gef_scores)
    
    if avg_gef > 0.7:
        gef_assessment = f"High average GEF ({avg_gef:.3f}) indicates explanations are faithful to model predictions"
    elif avg_gef > 0.5:
        gef_assessment = f"Moderate average GEF ({avg_gef:.3f}) shows acceptable faithfulness to model behavior"
    else:
        gef_assessment = f"Low average GEF ({avg_gef:.3f}) suggests explanations may not reflect actual model decision process"
    
    analysis.extend([gea_assessment, consistency_assessment, gef_assessment])
    
    # Method comparison
    best_gea_method = combined_df.loc[combined_df['Graph Explanation Accuracy (GEA)'].idxmax(), 'Explainer']
    best_gef_method = combined_df.loc[combined_df['Graph Explanation Faithfulness (GEF)'].idxmax(), 'Explainer']
    
    if best_gea_method == best_gef_method:
        method_assessment = f"{best_gea_method} provides the most reliable explanations with both highest accuracy and faithfulness"
    else:
        method_assessment = f"{best_gea_method} provides most accurate explanations while {best_gef_method} provides most faithful explanations"
    
    analysis.append(method_assessment)
    
    return analysis

def analyze_visualization_patterns(explanations, node_idx):
    """Analyze patterns in explanation visualizations"""
    analysis = []
    
    # Count successful explanations
    successful_methods = []
    if 'grad' in explanations:
        successful_methods.append("GradExplainer")
    if 'gnn' in explanations:
        successful_methods.append("GNNExplainer") 
    if 'pgm' in explanations:
        successful_methods.append("PGMExplainer")
    if 'lime' in explanations:
        successful_methods.append("GraphLIME")
    
    coverage_assessment = f"Successfully generated explanations using {len(successful_methods)} out of 4 methods: {', '.join(successful_methods)}"
    
    # Node importance analysis
    if 'grad' in explanations and 'gnn' in explanations:
        if len(successful_methods) >= 3:
            consensus_assessment = "Multiple explanation methods enable robust cross-validation of important node identification"
        else:
            consensus_assessment = "Limited explanation methods may reduce confidence in identified important nodes"
    else:
        consensus_assessment = "Insufficient explanation methods for comprehensive importance analysis"
    
    # Method diversity assessment
    if len(successful_methods) == 4:
        diversity_assessment = "Complete method coverage provides comprehensive perspective on node importance through gradient-based, perturbation-based, and probabilistic approaches"
    elif len(successful_methods) >= 2:
        diversity_assessment = "Partial method coverage provides limited but valuable insights into node importance patterns"
    else:
        diversity_assessment = "Single method provides isolated perspective on node importance"
    
    analysis.extend([coverage_assessment, consensus_assessment, diversity_assessment])
    
    return analysis

# Cache functions
@st.cache_data
def generate_dataset(model_layers, num_subgraphs, subgraph_size, prob_connection):
    try:
        dataset = ShapeGGen(
            model_layers=model_layers,
            num_subgraphs=num_subgraphs,
            subgraph_size=subgraph_size,
            prob_connection=prob_connection,
            add_sensitive_feature=False
        )
        
        data = dataset.graph
        explanation = dataset.explanations
        
        return dataset, data, explanation, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_resource
def train_model_and_init_explainers(_dataset, model_choice, hidden_channels, learning_rate, epochs):
    try:
        data = _dataset.get_graph(use_fixed_split=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(torch.unique(data.y))
        
        # Initialize model
        if model_choice == 'GCN':
            model = GCN_3layer_basic(hidden_channels=hidden_channels, input_feat=_dataset.n_features, classes=num_classes).to(device)
        elif model_choice == 'GIN':
            model = GIN_3layer_basic(hidden_channels=hidden_channels, input_feat=_dataset.n_features, classes=num_classes).to(device)
        elif model_choice == 'GAT':
            model = GAT_3layer_basic(hidden_channels=hidden_channels, input_feat=_dataset.n_features, classes=num_classes).to(device)
        elif model_choice == 'GSAGE':
            model = GSAGE_3layer(hidden_channels=hidden_channels, input_feat=_dataset.n_features, classes=num_classes).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(data.y.cpu()), y=data.y.cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop with progress
        progress_container = st.empty()
        
        for epoch in range(epochs):
            loss = train(model, optimizer, criterion, data)
            
            progress = (epoch + 1) / epochs
            with progress_container.container():
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                st.progress(progress)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Epoch", f"{epoch+1}/{epochs}")
                with col2:
                    st.metric("Loss", f"{loss:.4f}")
                with col3:
                    st.metric("Progress", f"{progress*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes=num_classes, get_auc=True)
        
        progress_container.empty()
        
        # Initialize explainers
        st.markdown('''
        <div class="info-box">
            <h4>Initializing Explainers</h4>
            <p>Setting up explanation algorithms for comprehensive analysis...</p>
        </div>
        ''', unsafe_allow_html=True)
        
        model.eval()
        
        grad_explainer = GradExplainer(model, criterion=F.nll_loss)
        gnn_explainer = GNNExplainer(model)
        pgm_explainer = PGMExplainer(model, explain_graph=False)
        
        try:
            lime_explainer = SafeGraphLIME(model)
            # st.success("SafeGraphLIME initialized successfully")
        except Exception as lime_init_error:
            st.warning(f"SafeGraphLIME initialization failed: {lime_init_error}")
            try:
                lime_explainer = GraphLIME(model)
                st.success("Regular GraphLIME initialized as fallback")
            except Exception as regular_lime_error:
                st.error(f"Both LIME initialization attempts failed: {regular_lime_error}")
                raise regular_lime_error
        
        st.markdown('''
        <div class="success-box">
            <h4>Explainer Initialization Complete</h4>
            <p>All explanation algorithms are ready for generating interpretable insights.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        return (model, data, {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec, 'auprc': auprc, 'auroc': auroc}, 
                grad_explainer, gnn_explainer, pgm_explainer, lime_explainer, None)
        
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)

# Main content
if run_analysis:
    # Generate dataset
    with st.spinner("Generating ShapeGGen dataset..."):
        dataset, data, explanation, dataset_error = generate_dataset(model_layers, num_subgraphs, subgraph_size, prob_connection)
        
        if dataset_error:
            st.markdown(f'''
            <div class="error-box">
                <h4>Dataset Generation Failed</h4>
                <p>{dataset_error}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.stop()
    
    # Dataset information
    st.markdown('<h2 class="section-header">Dataset Information</h2>', unsafe_allow_html=True)

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Nodes</h3>
            <h2>{data.num_nodes}</h2>
            <p>Graph vertices</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Edges</h3>
            <h2>{data.num_edges}</h2>
            <p>Graph connections</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{dataset.n_features}</h2>
            <p>Node attributes</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Classes</h3>
            <h2>{len(torch.unique(data.y))}</h2>
            <p>Label categories</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Model training
    st.markdown('<h2 class="section-header">Model Training & Explainer Initialization</h2>', unsafe_allow_html=True)
    
    with st.spinner("Training model and initializing explainers..."):
        result = train_model_and_init_explainers(dataset, model_choice, hidden_channels, learning_rate, epochs)
        
        if result[-1] is not None:
            st.markdown(f'''
            <div class="error-box">
                <h4>Training/Initialization Failed</h4>
                <p>{result[-1]}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.stop()
        
        model, data, metrics, grad_explainer, gnn_explainer, pgm_explainer, lime_explainer, _ = result
    
    # Training results with analysis
    st.markdown('''
    <div class="success-box">
        <h4>Model Training Complete</h4>
        <p>GNN model has been trained and all explainers are ready for analysis.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("F1 Score", f"{metrics['f1']:.4f}", delta=f"{(metrics['f1']-0.5):.4f}")
        st.metric("Accuracy", f"{metrics['acc']:.4f}", delta=f"{(metrics['acc']-0.5):.4f}")
    with col2:
        st.metric("Precision", f"{metrics['prec']:.4f}", delta=f"{(metrics['prec']-0.5):.4f}")
        st.metric("Recall", f"{metrics['rec']:.4f}", delta=f"{(metrics['rec']-0.5):.4f}")
    with col3:
        st.metric("AUROC", f"{metrics['auroc']:.4f}", delta=f"{(metrics['auroc']-0.5):.4f}")
        st.metric("AUPRC", f"{metrics['auprc']:.4f}", delta=f"{(metrics['auprc']-0.5):.4f}")
    
    # Model performance analysis
    performance_analysis = analyze_model_performance(metrics)
    st.markdown(f'''
    <div class="analysis-box">
        <h4>Model Performance Analysis</h4>
        <ul>
            {''.join([f"<li>{analysis}</li>" for analysis in performance_analysis])}
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # Explanation generation
    st.markdown('<h2 class="section-header">Explanation Generation</h2>', unsafe_allow_html=True)
    
    # Validate node index
    node_idx = min(node_idx_input, data.num_nodes - 1)
    if node_idx != node_idx_input:
        st.markdown(f'''
        <div class="warning-box">
            <h4>Node Index Adjustment</h4>
            <p>Node index adjusted from <strong>{node_idx_input}</strong> to <strong>{node_idx}</strong> (maximum available: {data.num_nodes - 1})</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with st.spinner("Generating explanations..."):
        try:
            # Data preparation
            x, edge_index = data.x, data.edge_index
            
            st.markdown(f'''
            <div class="info-box">
                <h4>Explanation Target</h4>
                <p>Analyzing importance patterns for <strong>node {node_idx}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Subgraph extraction
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, num_hops, data.edge_index, relabel_nodes=True, 
                num_nodes=data.num_nodes, flow='source_to_target'
            )
            sub_x = data.x[subset]
            sub_node_idx = mapping.item()
            
            st.markdown(f'''
            <div class="info-box">
                <h4>Subgraph Analysis</h4>
                <p><strong>Subgraph Size:</strong> {len(subset)} nodes | <strong>Target Node in Subgraph:</strong> {sub_node_idx}</p>
                <p><strong>Original Target:</strong> {node_idx} | <strong>Subgraph Connectivity:</strong> {sub_edge_index.shape[1]} edges</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Progress tracking
            st.markdown("### Explainer Execution Progress")
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            explanations = {}
            
            # GradExplainer
            progress_bar.progress(0.1)
            with status_container.container():
                st.markdown('<span class="explainer-status status-running">Running GradExplainer...</span>', unsafe_allow_html=True)
            grad_explanation = grad_explainer.get_explanation_node(
                sub_node_idx, sub_x, sub_edge_index
            )
            explanations['grad'] = grad_explanation
            with status_container.container():
                st.markdown('<span class="explainer-status status-success">GradExplainer Completed</span>', unsafe_allow_html=True)
            
            progress_bar.progress(0.3)
            with status_container.container():
                st.markdown('<span class="explainer-status status-running">Running GNNExplainer...</span>', unsafe_allow_html=True)
            gnn_explanation = gnn_explainer.get_explanation_node(
                sub_node_idx, sub_x, sub_edge_index
            )
            explanations['gnn'] = gnn_explanation
            with status_container.container():
                st.markdown('<span class="explainer-status status-success">GNNExplainer Completed</span>', unsafe_allow_html=True)
            
            # PGM and GraphLIME use full graph
            progress_bar.progress(0.5)
            with status_container.container():
                st.markdown('<span class="explainer-status status-running">Running PGMExplainer...</span>', unsafe_allow_html=True)
            pgm_explanation = pgm_explainer.get_explanation_node(
                node_idx, data.x, data.edge_index
            )
            explanations['pgm'] = pgm_explanation
            with status_container.container():
                st.markdown('<span class="explainer-status status-success">PGMExplainer Completed</span>', unsafe_allow_html=True)
            
            progress_bar.progress(0.7)
            with status_container.container():
                st.markdown('<span class="explainer-status status-running">Running SafeGraphLIME...</span>', unsafe_allow_html=True)
            
            # SafeGraphLIME with multiple fallback approaches
            try:
                lime_explanation = lime_explainer.get_explanation_node(
                    node_idx, data.x, data.edge_index
                )
                explanations['lime'] = lime_explanation
                with status_container.container():
                    st.markdown('<span class="explainer-status status-success">SafeGraphLIME Completed</span>', unsafe_allow_html=True)
            except Exception as e1:
                st.markdown(f'''
                <div class="warning-box">
                    <p>Primary approach failed: {e1}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                try:
                    with status_container.container():
                        st.markdown('<span class="explainer-status status-running">Trying alternative parameter order...</span>', unsafe_allow_html=True)
                    lime_explanation = lime_explainer.get_explanation_node(
                        node_idx, data.edge_index, data.x
                    )
                    explanations['lime'] = lime_explanation
                    with status_container.container():
                        st.markdown('<span class="explainer-status status-success">SafeGraphLIME Completed (Alternative approach)</span>', unsafe_allow_html=True)
                except Exception as e2:
                    try:
                        with status_container.container():
                            st.markdown('<span class="explainer-status status-running">Trying fresh initialization...</span>', unsafe_allow_html=True)
                        fresh_lime_explainer = SafeGraphLIME(model)
                        lime_explanation = fresh_lime_explainer.get_explanation_node(
                            node_idx, data.x, data.edge_index
                        )
                        explanations['lime'] = lime_explanation
                        with status_container.container():
                            st.markdown('<span class="explainer-status status-success">SafeGraphLIME Completed (Fresh initialization)</span>', unsafe_allow_html=True)
                    except Exception as e3:
                        try:
                            with status_container.container():
                                st.markdown('<span class="explainer-status status-running">Trying regular GraphLIME...</span>', unsafe_allow_html=True)
                            regular_lime = GraphLIME(model)
                            lime_explanation = regular_lime.get_explanation_node(
                                node_idx, data.x, data.edge_index
                            )
                            explanations['lime'] = lime_explanation
                            with status_container.container():
                                st.markdown('<span class="explainer-status status-success">Regular GraphLIME Completed</span>', unsafe_allow_html=True)
                        except Exception as e4:
                            st.markdown(f'''
                            <div class="error-box">
                                <h4>All LIME Approaches Failed</h4>
                                <p>Final error: {e4}</p>
                            </div>
                            ''', unsafe_allow_html=True)
            
            # Convert to uniform format
            progress_bar.progress(0.9)
            with status_container.container():
                st.markdown('<span class="explainer-status status-running">Converting explanations...</span>', unsafe_allow_html=True)
            
            grad_full_exp = convert_to_full_explanation(
                grad_explanation, node_idx, subset, sub_edge_index, mapping, edge_mask, data
            )
            gnn_full_exp = convert_to_full_explanation(
                gnn_explanation, node_idx, subset, sub_edge_index, mapping, edge_mask, data
            )
            pgm_full_exp = convert_to_full_explanation(
                pgm_explanation, node_idx, subset, sub_edge_index, mapping, edge_mask, data
            )
            if 'lime' in explanations:
                lime_full_exp = convert_to_full_explanation(
                    lime_explanation, node_idx, subset, sub_edge_index, mapping, edge_mask, data
                )
            
            progress_bar.progress(1.0)
            progress_bar.empty()
            status_container.empty()
            
        except Exception as e:
            st.markdown(f'''
            <div class="error-box">
                <h4>Explanation Generation Failed</h4>
                <p>{str(e)}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.stop()
    
    st.markdown('''
    <div class="success-box">
        <h4>Explanation Generation Complete</h4>
        <p>Successfully generated explanations for comprehensive analysis.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Visualization pattern analysis
    visualization_analysis = analyze_visualization_patterns(explanations, node_idx)
    st.markdown(f'''
    <div class="analysis-box">
        <h4>Explanation Coverage Analysis</h4>
        <ul>
            {''.join([f"<li>{analysis}</li>" for analysis in visualization_analysis])}
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # Quantitative evaluation
    st.markdown('<h2 class="section-header">Quantitative Evaluation Metrics</h2>', unsafe_allow_html=True)
    
    try:
        # Ground truth
        ground_truth = dataset.explanations[node_idx][0]
        
        st.markdown('''
        <div class="success-box">
            <h4>Ground Truth Loaded</h4>
            <p>Reference explanation available for quantitative evaluation.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Calculate metrics
        gea_grad = graph_exp_acc(ground_truth, grad_explanation)
        gea_gnn = graph_exp_acc(ground_truth, gnn_explanation)
        gea_pgm = graph_exp_acc(ground_truth, pgm_explanation)
        
        # Special LIME evaluation
        if 'lime' in explanations:
            st.markdown('''
            <div class="info-box">
                <h4>LIME Evaluation Processing</h4>
                <p>Converting LIME feature importance to node importance for evaluation...</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Get LIME explanation again
            lime_explanation = lime_explainer.get_explanation_node(node_idx, x, edge_index)
            
            # Convert feature importance to node importance
            node_importance = lime_explanation.feature_imp.mean().item()
            lime_node_importance = torch.zeros(data.num_nodes)
            lime_node_importance[node_idx] = node_importance
            
            # Create node reference
            node_reference = {i: i for i in range(data.num_nodes)}
            
            # Create edge mask
            edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
            for edge_id, (src, dst) in enumerate(data.edge_index.T):
                if src in subset and dst in subset:
                    edge_mask[edge_id] = True
            
            # Create ExplanationCompat
            lime_exp = ExplanationCompat(
                node_imp=lime_node_importance,
                node_idx=node_idx,
                enc_subgraph=EnclosingSubgraph(
                    nodes=subset,
                    edge_index=sub_edge_index,
                    inv=mapping,
                    edge_mask=edge_mask,
                    directed=False
                )
            )
            
            lime_exp.node_reference = node_reference
            gea_lime = graph_exp_acc(ground_truth, lime_exp)
            
            # Calculate faithfulness
            gef_grad = graph_exp_faith(grad_explanation, dataset, model)
            gef_gnn = graph_exp_faith(gnn_explanation, dataset, model)
            gef_pgm = graph_exp_faith(pgm_explanation, dataset, model)
            gef_lime = graph_exp_faith(lime_exp, dataset, model)
            
            # Combined results
            combined_results = [
                ['GradExplainer', gea_grad, gef_grad],
                ['GNNExplainer', gea_gnn, gef_gnn],
                ['PGMExplainer', gea_pgm, gef_pgm],
                ['GraphLIME', gea_lime, gef_lime]
            ]
        else:
            # Without LIME
            gef_grad = graph_exp_faith(grad_explanation, dataset, model)
            gef_gnn = graph_exp_faith(gnn_explanation, dataset, model)
            gef_pgm = graph_exp_faith(pgm_explanation, dataset, model)
            
            combined_results = [
                ['GradExplainer', gea_grad, gef_grad],
                ['GNNExplainer', gea_gnn, gef_gnn],
                ['PGMExplainer', gea_pgm, gef_pgm]
            ]
        
        # Display results
        st.markdown("### Quantitative Results Dashboard")
        
        combined_df = pd.DataFrame(combined_results, columns=['Explainer', 'Graph Explanation Accuracy (GEA)', 'Graph Explanation Faithfulness (GEF)'])
        
        st.dataframe(
            combined_df, 
            use_container_width=True,
            column_config={
                "Graph Explanation Accuracy (GEA)": st.column_config.ProgressColumn(
                    "GEA Score",
                    help="Higher is better",
                    min_value=0,
                    max_value=1,
                    format="%.4f",
                ),
                "Graph Explanation Faithfulness (GEF)": st.column_config.ProgressColumn(
                    "GEF Score", 
                    help="Higher is better",
                    min_value=0,
                    max_value=1,
                    format="%.4f",
                ),
            }
        )
        
        # Best performance insights
        best_gea = combined_df.loc[combined_df['Graph Explanation Accuracy (GEA)'].idxmax()]
        best_gef = combined_df.loc[combined_df['Graph Explanation Faithfulness (GEF)'].idxmax()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'''
            <div class="success-box">
                <h4>Highest Accuracy</h4>
                <p><strong>{best_gea['Explainer']}</strong> achieved the best Graph Explanation Accuracy</p>
                <p><strong>Score:</strong> {best_gea['Graph Explanation Accuracy (GEA)']:.4f}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="success-box">
                <h4>Highest Faithfulness</h4>
                <p><strong>{best_gef['Explainer']}</strong> achieved the best Graph Explanation Faithfulness</p>
                <p><strong>Score:</strong> {best_gef['Graph Explanation Faithfulness (GEF)']:.4f}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Comprehensive metric analysis
        metric_analysis = analyze_explanation_metrics(combined_df)
        st.markdown(f'''
        <div class="analysis-box">
            <h4>Explanation Quality Analysis</h4>
            <ul>
                {''.join([f"<li>{analysis}</li>" for analysis in metric_analysis])}
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'''
        <div class="error-box">
            <h4>Evaluation Failed</h4>
            <p>{str(e)}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Interactive visualization
    st.markdown('<h2 class="section-header">Explanation Visualizations</h2>', unsafe_allow_html=True)
    
    try:
        # Ensure node_idx is int
        if isinstance(node_idx, torch.Tensor):
            node_idx = node_idx.item()
        
        st.markdown(f'''
        <div class="info-box">
            <h4>Visualization Overview</h4>
            <p>Comparative analysis of explanation patterns for <strong>node {node_idx}</strong></p>
            <p><strong>Layout:</strong> Comprehensive subplot arrangement | <strong>Visualization Scope:</strong> {num_hops}-hop neighborhood</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Create visualization
        available_explanations = []
        if ground_truth:
            available_explanations.append(('Ground Truth', ground_truth))
        if 'grad' in explanations:
            available_explanations.append(('GradExplainer', grad_full_exp))
        if 'gnn' in explanations:
            available_explanations.append(('GNNExplainer', gnn_full_exp))
        if 'lime' in explanations:
            available_explanations.append(('GraphLIME', lime_explanation))
        if 'pgm' in explanations:
            available_explanations.append(('PGMExplainer', pgm_explanation))
        
        num_plots = len(available_explanations)
        fig, ax = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
        fig.suptitle(f'Explanation Analysis for Node {node_idx}', 
                   fontsize=18, fontweight='bold', y=0.98)
        
        if num_plots == 1:
            ax = [ax]
        
        # Generate visualizations
        for idx, (name, exp) in enumerate(available_explanations):
            if hasattr(exp, 'node_idx') and isinstance(exp.node_idx, torch.Tensor):
                exp.node_idx = exp.node_idx.item()
            
            exp.visualize_node(
                num_hops=2,
                graph_data=data,
                ax=ax[idx],
                show_node_labels=True
            )
            ax[idx].set_title(name, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualization insights
        st.markdown(f'''
        <div class="analysis-box">
            <h4>Visualization Pattern Analysis</h4>
            <ul>
                <li><strong>Method Coverage:</strong> Successfully visualized {num_plots} explanation methods for comprehensive comparison</li>
                <li><strong>Node Importance Patterns:</strong> Each visualization highlights different aspects of node importance based on the underlying algorithm</li>
                <li><strong>Consensus Analysis:</strong> Areas of agreement across multiple methods indicate robust importance assignments</li>
                <li><strong>Method Divergence:</strong> Differences between methods reveal distinct perspectives on feature importance and decision boundaries</li>
                <li><strong>Ground Truth Comparison:</strong> {'Visual comparison with ground truth enables validation of explanation accuracy' if ground_truth else 'Ground truth not available for direct visual validation'}</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'''
        <div class="error-box">
            <h4>Visualization Generation Failed</h4>
            <p>{str(e)}</p>
        </div>
        ''', unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown('''
    <div class="info-box">
        <h2 class="highlight-text">Welcome to the GraphXAI Explainer Dashboard</h2>
        <p>This comprehensive dashboard provides advanced graph neural network explanation capabilities with detailed analysis:</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="success-box">
            <h4>Core Features</h4>
            <ul>
                <li><strong>Interactive Configuration</strong> - Customize all parameters</li>
                <li><strong>Multiple GNN Models</strong> - GCN, GIN, GAT, GSAGE support</li>
                <li><strong>Four Explainer Methods</strong> - Comprehensive explanation coverage</li>
                <li><strong>Quantitative Evaluation</strong> - Accuracy and faithfulness metrics</li>
                <li><strong>Rich Visualizations</strong> - Interactive explanation displays</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="info-box">
            <h4>Supported Explainers</h4>
            <ul>
                <li><strong>GradExplainer</strong> - Gradient-based attribution</li>
                <li><strong>GNNExplainer</strong> - Mutual information approach</li>
                <li><strong>PGMExplainer</strong> - Probabilistic graphical models</li>
                <li><strong>GraphLIME</strong> - Local surrogate explanations</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    # Configuration suggestions
    st.markdown('<h2 class="section-header">Recommended Configurations</h2>', unsafe_allow_html=True)
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown('''
        <div class="metric-card">
            <h3>Quick Start</h3>
            <p><strong>Stable Configuration</strong></p>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>Model: GCN</li>
                <li>Layers: 3</li>
                <li>Subgraphs: 8</li>
                <li>Size: 5</li>
                <li>Epochs: 300</li>
                <li>Node: 8</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with config_col2:
        st.markdown('''
        <div class="metric-card">
            <h3>High Performance</h3>
            <p><strong>Advanced Analysis</strong></p>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>Model: GAT</li>
                <li>Layers: 4</li>
                <li>Subgraphs: 10</li>
                <li>Size: 6</li>
                <li>Epochs: 400</li>
                <li>Node: 15</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with config_col3:
        st.markdown('''
        <div class="metric-card">
            <h3>Fast Testing</h3>
            <p><strong>Quick Experiments</strong></p>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>Model: GIN</li>
                <li>Layers: 2</li>
                <li>Subgraphs: 4</li>
                <li>Size: 5</li>
                <li>Epochs: 100</li>
                <li>Node: 5</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    # Technical implementation
    st.markdown('<h2 class="section-header">Technical Implementation</h2>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="dataset-info">
        <h4 class="highlight-text">Advanced Features</h4>
        <ul>
            <li><strong>Automated Data Validation:</strong> Comprehensive compatibility checking</li>
            <li><strong>Intelligent Error Handling:</strong> Graceful degradation and detailed diagnostics</li>
            <li><strong>Real-time Progress Tracking:</strong> Live updates during training and explanation</li>
            <li><strong>Interactive Visualizations:</strong> Rich, comparative explanation displays</li>
            <li><strong>Quantitative Metrics:</strong> Standardized evaluation with GEA and GEF scores</li>
            <li><strong>Comprehensive Analysis:</strong> Automated interpretation of results and patterns</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # System information
    st.markdown('<h2 class="section-header">System Environment</h2>', unsafe_allow_html=True)
    
    device_info = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    
    sys_col1, sys_col2 = st.columns(2)
    
    with sys_col1:
        st.markdown(f'''
        <div class="info-box">
            <h4>Compute Environment</h4>
            <p><strong>Device:</strong> {device_info}</p>
            <p><strong>PyTorch Version:</strong> {torch.__version__}</p>
            {'<p><strong>GPU:</strong> ' + torch.cuda.get_device_name(0) + '</p>' if torch.cuda.is_available() else ''}
            {'<p><strong>CUDA Version:</strong> ' + str(torch.version.cuda) + '</p>' if torch.cuda.is_available() else ''}
        </div>
        ''', unsafe_allow_html=True)
    
    # Call to action
    st.markdown('''
    <div class="success-box">
        <h4>Ready to Start?</h4>
        <p>Configure your parameters in the sidebar and click <strong>"Run Complete Analysis"</strong> to begin!</p>
        <p>The dashboard will guide you through dataset generation, model training, explanation generation, and comprehensive evaluation with detailed analysis.</p>
    </div>
    ''', unsafe_allow_html=True)