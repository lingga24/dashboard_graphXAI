import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 1rem 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 GraphXAI Explainer Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.markdown("## 🛠️ Configuration")

# Dataset parameters
st.sidebar.markdown("### Dataset Parameters")
model_layers = st.sidebar.slider("Model Layers", 2, 5, 3)
num_subgraphs = st.sidebar.slider("Number of Subgraphs", 4, 16, 8)
subgraph_size = st.sidebar.slider("Subgraph Size", 3, 10, 5)
prob_connection = st.sidebar.slider("Connection Probability", 0.1, 0.9, 0.3)

# Model parameters
st.sidebar.markdown("### Model Parameters")
model_choice = st.sidebar.selectbox("Model Type", ['GCN', 'GIN', 'GAT', 'GSAGE'])
hidden_channels = st.sidebar.slider("Hidden Channels", 16, 128, 32)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.slider("Training Epochs", 50, 500, 300)

# Explanation parameters
st.sidebar.markdown("### Explanation Parameters")
node_idx_input = st.sidebar.number_input("Node Index to Explain", 0, 100, 8)
num_hops = st.sidebar.slider("Number of Hops", 1, 4, 2)

# Explainer selection
st.sidebar.markdown("### Explainer Selection")
use_grad = st.sidebar.checkbox("GradExplainer", value=True)
use_gnn = st.sidebar.checkbox("GNNExplainer", value=True)
use_pgm = st.sidebar.checkbox("PGMExplainer", value=True)
use_lime = st.sidebar.checkbox("GraphLIME", value=True)

# Generate/Run button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# Initialize session state
if 'dataset_generated' not in st.session_state:
    st.session_state.dataset_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Helper functions
@st.cache_data
def generate_dataset(model_layers, num_subgraphs, subgraph_size, prob_connection):
    """Generate ShapeGGen dataset"""
    try:
        dataset = ShapeGGen(
            model_layers=model_layers,
            num_subgraphs=num_subgraphs,
            subgraph_size=subgraph_size,
            prob_connection=prob_connection,
            add_sensitive_feature=False
        )
        return dataset, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def train_model(_dataset, model_choice, hidden_channels, learning_rate, epochs):
    """Train the GNN model"""
    try:
        data = _dataset.get_graph(use_fixed_split=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(torch.unique(data.y))
        
        # Initialize model
        if model_choice == 'GCN':
            model = GCN_3layer_basic(
                hidden_channels=hidden_channels, 
                input_feat=_dataset.n_features, 
                classes=num_classes
            ).to(device)
        elif model_choice == 'GIN':
            model = GIN_3layer_basic(
                hidden_channels=hidden_channels, 
                input_feat=_dataset.n_features, 
                classes=num_classes
            ).to(device)
        elif model_choice == 'GAT':
            model = GAT_3layer_basic(
                hidden_channels=hidden_channels, 
                input_feat=_dataset.n_features, 
                classes=num_classes
            ).to(device)
        elif model_choice == 'GSAGE':
            model = GSAGE_3layer(
                hidden_channels=hidden_channels, 
                input_feat=_dataset.n_features, 
                classes=num_classes
            ).to(device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        
        # Class weights for balanced training
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(data.y.cpu()), 
            y=data.y.cpu().numpy()
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs):
            loss = train(model, optimizer, criterion, data)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Training Progress: {epoch+1}/{epochs} epochs - Loss: {loss:.4f}')
        
        # Evaluate model
        f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes=num_classes, get_auc=True)
        
        progress_bar.empty()
        status_text.empty()
        
        return model, data, {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec, 'auprc': auprc, 'auroc': auroc}, None
        
    except Exception as e:
        return None, None, None, str(e)

def convert_to_full_explanation(exp, node_idx, subset, sub_edge_index, mapping, edge_mask, data):
    """Convert explanation from subgraph to full graph"""
    try:
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
        
        return full_exp, None
    except Exception as e:
        return None, str(e)

def create_lime_compatible_explanation(lime_exp, node_idx, subset, sub_edge_index, mapping, edge_mask, data):
    """Create LIME-compatible explanation for evaluation"""
    try:
        # Handle feature importance conversion
        if hasattr(lime_exp, 'feature_imp') and lime_exp.feature_imp is not None:
            node_importance = lime_exp.feature_imp.mean().item()
        else:
            node_importance = 1.0  # Default importance
        
        lime_node_importance = torch.zeros(data.num_nodes)
        lime_node_importance[node_idx] = node_importance
        
        # Create node reference
        node_reference = {i: i for i in range(data.num_nodes)}
        
        # Create edge mask
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        for edge_id, (src, dst) in enumerate(data.edge_index.T):
            if src in subset and dst in subset:
                edge_mask[edge_id] = True
        
        class ExplanationCompat(Explanation):
            def __init__(self, enc_subgraph=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.enc_subgraph = enc_subgraph
        
        lime_compatible = ExplanationCompat(
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
        lime_compatible.node_reference = node_reference
        
        return lime_compatible, None
    except Exception as e:
        return None, str(e)

def validate_data_compatibility(data, model, node_idx):
    """Validate data compatibility before running explainers"""
    issues = []
    
    # Check node index validity
    if node_idx >= data.num_nodes:
        issues.append(f"Node index {node_idx} exceeds available nodes ({data.num_nodes})")
    
    # Check data types
    if data.x.dtype != torch.float32:
        issues.append(f"Node features have dtype {data.x.dtype}, expected torch.float32")
    
    if data.edge_index.dtype != torch.long:
        issues.append(f"Edge index has dtype {data.edge_index.dtype}, expected torch.long")
    
    # Check for NaN or infinite values
    if torch.isnan(data.x).any():
        issues.append("Node features contain NaN values")
    
    if torch.isinf(data.x).any():
        issues.append("Node features contain infinite values")
    
    # Check graph connectivity
    if data.edge_index.shape[1] == 0:
        issues.append("Graph has no edges")
    
    return issues

# Main content
if run_analysis:
    # Generate dataset
    with st.spinner("🔄 Generating dataset..."):
        dataset, dataset_error = generate_dataset(model_layers, num_subgraphs, subgraph_size, prob_connection)
        
        if dataset_error:
            st.error(f"❌ Dataset generation failed: {dataset_error}")
            st.stop()
        
        st.session_state.dataset_generated = True
    
    # Display dataset info
    st.markdown('<div class="section-header">📊 Dataset Information</div>', unsafe_allow_html=True)
    
    data = dataset.graph
    explanation = dataset.explanations
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Nodes</h3>
            <h2>{data.num_nodes}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Edges</h3>
            <h2>{data.num_edges}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{dataset.n_features}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Classes</h3>
            <h2>{len(torch.unique(data.y))}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Train model
    st.markdown('<div class="section-header">🧠 Model Training</div>', unsafe_allow_html=True)
    
    with st.spinner("🏃‍♂️ Training model..."):
        model, data, metrics, train_error = train_model(dataset, model_choice, hidden_channels, learning_rate, epochs)
        
        if train_error:
            st.error(f"❌ Model training failed: {train_error}")
            st.stop()
        
        st.session_state.model_trained = True
    
    # Display training results
    st.success("✅ Model training completed!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("F1 Score", f"{metrics['f1']:.4f}")
        st.metric("Accuracy", f"{metrics['acc']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['prec']:.4f}")
        st.metric("Recall", f"{metrics['rec']:.4f}")
    with col3:
        st.metric("AUROC", f"{metrics['auroc']:.4f}")
        st.metric("AUPRC", f"{metrics['auprc']:.4f}")
    
    # Generate explanations
    st.markdown('<div class="section-header">🔍 Explanation Generation</div>', unsafe_allow_html=True)
    
    # Validate node index
    node_idx = min(node_idx_input, data.num_nodes - 1)
    if node_idx != node_idx_input:
        st.warning(f"⚠️ Node index adjusted to {node_idx} (max available: {data.num_nodes - 1})")
    
    # Validate data compatibility
    data_issues = validate_data_compatibility(data, model, node_idx)
    if data_issues:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ **Data compatibility issues detected:**")
        for issue in data_issues:
            st.markdown(f"- {issue}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.spinner("🔬 Generating explanations..."):
        try:
            # Initialize device and prepare data
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            model = model.to(device)
            
            # Ensure proper data types
            if data.edge_index.dtype != torch.long:
                data.edge_index = data.edge_index.long()
            if data.x.dtype != torch.float32:
                data.x = data.x.float()
            
            # Clean data (remove NaN/inf values)
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                st.warning("⚠️ Cleaning NaN/infinite values from node features")
                data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            edge_index = data.edge_index
            x = data.x
            
            # Set model to evaluation mode (critical for GraphLIME)
            model.eval()
            
            st.info(f"📍 Explaining node {node_idx} in graph with {data.num_nodes} nodes and {data.num_edges} edges")
            st.info(f"🧠 Model type: {type(model).__name__}, Device: {next(model.parameters()).device}")
            
            # Test model output to ensure compatibility
            try:
                with torch.no_grad():
                    test_output = model(x, edge_index)
                    st.info(f"✅ Model test successful - Output shape: {test_output.shape}")
            except Exception as model_error:
                st.error(f"❌ Model compatibility test failed: {model_error}")
                st.stop()
            
            # Get subgraph for gradient-based methods
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, num_hops, edge_index, relabel_nodes=True, 
                num_nodes=data.num_nodes, flow='source_to_target'
            )
            sub_x = x[subset]
            sub_node_idx = mapping.item()
            
            st.info(f"🔍 Subgraph contains {len(subset)} nodes, explaining node {sub_node_idx} (original: {node_idx})")
            
            # Initialize explanations dictionary
            explanations = {}
            explanation_errors = {}
            
            # Initialize explainers with error handling
            explainers = {}
            
            # GradExplainer
            if use_grad:
                try:
                    explainers['grad'] = GradExplainer(model, criterion=torch.nn.CrossEntropyLoss())
                    st.success("✅ GradExplainer initialized")
                except Exception as e:
                    explanation_errors['grad'] = f"Initialization failed: {str(e)}"
                    st.error(f"❌ GradExplainer failed to initialize: {e}")
            
            # GNNExplainer
            if use_gnn:
                try:
                    explainers['gnn'] = GNNExplainer(model)
                    st.success("✅ GNNExplainer initialized")
                except Exception as e:
                    explanation_errors['gnn'] = f"Initialization failed: {str(e)}"
                    st.error(f"❌ GNNExplainer failed to initialize: {e}")
            
            # PGMExplainer
            if use_pgm:
                try:
                    explainers['pgm'] = PGMExplainer(model, explain_graph=False, p_threshold=0.1)
                    st.success("✅ PGMExplainer initialized")
                except Exception as e:
                    explanation_errors['pgm'] = f"Initialization failed: {str(e)}"
                    st.error(f"❌ PGMExplainer failed to initialize: {e}")
            
            # GraphLIME with enhanced error handling
            if use_lime:
                try:
                    # Initialize GraphLIME exactly like in notebook
                    explainers['lime'] = GraphLIME(model)
                    st.success("✅ GraphLIME initialized")
                except Exception as e:
                    explanation_errors['lime'] = f"Initialization failed: {str(e)}"
                    st.error(f"❌ GraphLIME failed to initialize: {e}")
            
            # Generate explanations with individual error handling
            progress_bar = st.progress(0)
            total_explainers = len(explainers)
            current_progress = 0
            
            # GradExplainer
            if 'grad' in explainers:
                try:
                    progress_bar.progress(current_progress / total_explainers if total_explainers > 0 else 0)
                    st.info("🔄 Running GradExplainer...")
                    explanations['grad'] = explainers['grad'].get_explanation_node(sub_node_idx, sub_x, sub_edge_index)
                    st.success("✅ GradExplainer completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['grad'] = f"Explanation generation failed: {str(e)}"
                    st.error(f"❌ GradExplainer failed: {e}")
            
            # GNNExplainer
            if 'gnn' in explainers:
                try:
                    progress_bar.progress(current_progress / total_explainers if total_explainers > 0 else 0)
                    st.info("🔄 Running GNNExplainer...")
                    explanations['gnn'] = explainers['gnn'].get_explanation_node(sub_node_idx, sub_x, sub_edge_index)
                    st.success("✅ GNNExplainer completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['gnn'] = f"Explanation generation failed: {str(e)}"
                    st.error(f"❌ GNNExplainer failed: {e}")
            
            # PGMExplainer
            if 'pgm' in explainers:
                try:
                    progress_bar.progress(current_progress / total_explainers if total_explainers > 0 else 0)
                    st.info("🔄 Running PGMExplainer...")
                    explanations['pgm'] = explainers['pgm'].get_explanation_node(node_idx, x, edge_index)
                    st.success("✅ PGMExplainer completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['pgm'] = f"Explanation generation failed: {str(e)}"
                    st.error(f"❌ PGMExplainer failed: {e}")
            
            # GraphLIME - exactly matching notebook implementation
            if 'lime' in explainers:
                try:
                    progress_bar.progress(current_progress / total_explainers if total_explainers > 0 else 0)
                    st.info("🔄 Running GraphLIME...")
                    
                    # Use the exact same approach as notebook - full graph with original node_idx
                    explanations['lime'] = explainers['lime'].get_explanation_node(node_idx, x, edge_index)
                    st.success("✅ GraphLIME completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['lime'] = f"Explanation generation failed: {str(e)}"
                    st.error(f"❌ GraphLIME failed: {e}")
                    
                    # Additional debug info for GraphLIME
                    st.error(f"Debug info - Model type: {type(model)}")
                    st.error(f"Debug info - Model device: {next(model.parameters()).device}")
                    st.error(f"Debug info - Data device: {x.device}")
                    st.error(f"Debug info - Edge index device: {edge_index.device}")
                    st.error(f"Debug info - Feature shape: {x.shape}")
                    st.error(f"Debug info - Edge index shape: {edge_index.shape}")
                    st.error(f"Debug info - Target node: {node_idx}")
                    
                    # Try to get more specific error info
                    try:
                        with torch.no_grad():
                            model.eval()
                            test_output = model(x, edge_index)
                            st.info(f"Model output shape: {test_output.shape}")
                            st.info(f"Model output sample: {test_output[0][:5]}")
                    except Exception as model_test_error:
                        st.error(f"Model test failed: {model_test_error}")
            
            progress_bar.progress(1.0)
            progress_bar.empty()
            
            # Convert successful explanations to full format
            if 'grad' in explanations:
                grad_full, grad_error = convert_to_full_explanation(
                    explanations['grad'], node_idx, subset, sub_edge_index, mapping, edge_mask, data
                )
                if grad_error:
                    st.warning(f"⚠️ GradExplainer conversion failed: {grad_error}")
                else:
                    explanations['grad_full'] = grad_full
            
            if 'gnn' in explanations:
                gnn_full, gnn_error = convert_to_full_explanation(
                    explanations['gnn'], node_idx, subset, sub_edge_index, mapping, edge_mask, data
                )
                if gnn_error:
                    st.warning(f"⚠️ GNNExplainer conversion failed: {gnn_error}")
                else:
                    explanations['gnn_full'] = gnn_full
            
            # Report results
            successful_explainers = len([k for k in explanations.keys() if not k.endswith('_full')])
            failed_explainers = len(explanation_errors)
            
            if successful_explainers > 0:
                st.success(f"✅ Successfully generated {successful_explainers} explanations!")
            
            if failed_explainers > 0:
                st.warning(f"⚠️ {failed_explainers} explainer(s) failed. See details above.")
                
        except Exception as e:
            st.error(f"❌ Critical error in explanation generation: {e}")
            st.info("💡 Try reducing dataset complexity or changing model parameters")
            st.stop()
    
    # Only proceed with evaluation if we have successful explanations
    if len([k for k in explanations.keys() if not k.endswith('_full')]) > 0:
        # Evaluation metrics
        st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
        
        # Ground truth
        try:
            ground_truth = dataset.explanations[node_idx][0]
            st.info("✅ Ground truth explanation loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load ground truth: {e}")
            ground_truth = None
        
        # Prepare evaluation data
        eval_results = []
        
        # Calculate metrics for successful explainers (only if ground truth is available)
        if ground_truth is not None:
            if 'grad' in explanations:
                try:
                    gea_grad = graph_exp_acc(ground_truth, explanations['grad'])
                    gef_grad = graph_exp_faith(explanations['grad'], dataset, model)
                    eval_results.append(['GradExplainer', gea_grad, gef_grad])
                except Exception as e:
                    st.warning(f"⚠️ GradExplainer evaluation failed: {e}")
            
            if 'gnn' in explanations:
                try:
                    gea_gnn = graph_exp_acc(ground_truth, explanations['gnn'])
                    gef_gnn = graph_exp_faith(explanations['gnn'], dataset, model)
                    eval_results.append(['GNNExplainer', gea_gnn, gef_gnn])
                except Exception as e:
                    st.warning(f"⚠️ GNNExplainer evaluation failed: {e}")
            
            if 'pgm' in explanations:
                try:
                    gea_pgm = graph_exp_acc(ground_truth, explanations['pgm'])
                    gef_pgm = graph_exp_faith(explanations['pgm'], dataset, model)
                    eval_results.append(['PGMExplainer', gea_pgm, gef_pgm])
                except Exception as e:
                    st.warning(f"⚠️ PGMExplainer evaluation failed: {e}")
            
            if 'lime' in explanations:
                try:
                    # Match the exact notebook implementation for LIME evaluation
                    lime_explanation = explanations['lime']
                    
                    # Convert feature importance GraphLIME to node importance (exactly like notebook)
                    node_importance = lime_explanation.feature_imp.mean().item()
                    lime_node_importance = torch.zeros(data.num_nodes)  # For full graph
                    lime_node_importance[node_idx] = node_importance
                    
                    # Create node_reference lengkap untuk full graph (exactly like notebook)
                    node_reference = {i: i for i in range(data.num_nodes)}
                    
                    # Create edge_mask manual (exactly like notebook)
                    edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
                    for edge_id, (src, dst) in enumerate(data.edge_index.T):
                        if src in subset and dst in subset:
                            edge_mask[edge_id] = True
                    
                    # Use the exact same class as in notebook
                    class ExplanationCompat(Explanation):
                        def __init__(self, enc_subgraph=None, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.enc_subgraph = enc_subgraph
                    
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
                    
                    lime_exp.node_reference = node_reference  # This is the key from notebook!
                    
                    gea_lime = graph_exp_acc(ground_truth, lime_exp)
                    gef_lime = graph_exp_faith(lime_exp, dataset, model)
                    eval_results.append(['GraphLIME', gea_lime, gef_lime])
                except Exception as e:
                    st.warning(f"⚠️ GraphLIME evaluation failed: {e}")
        
        # Display metrics table
        if eval_results:
            metrics_df = pd.DataFrame(eval_results, columns=[
                'Explainer', 'Graph Explanation Accuracy', 'Graph Explanation Faithfulness'
            ])
            st.dataframe(metrics_df, use_container_width=True)
            
            # Summary insights
            if len(eval_results) > 1:
                best_gea = metrics_df.loc[metrics_df['Graph Explanation Accuracy'].idxmax()]
                best_gef = metrics_df.loc[metrics_df['Graph Explanation Faithfulness'].idxmax()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'''
                    <div class="info-box">
                        <h4>🎯 Best Accuracy</h4>
                        <p><strong>{best_gea['Explainer']}</strong> achieved the highest Graph Explanation Accuracy with a score of <strong>{best_gea['Graph Explanation Accuracy']:.4f}</strong></p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="info-box">
                        <h4>🔍 Best Faithfulness</h4>
                        <p><strong>{best_gef['Explainer']}</strong> achieved the highest Graph Explanation Faithfulness with a score of <strong>{best_gef['Graph Explanation Faithfulness']:.4f}</strong></p>
                    </div>
                    ''', unsafe_allow_html=True)
        elif ground_truth is not None:
            st.warning("⚠️ No successful evaluations to display")
        else:
            st.info("ℹ️ Evaluation skipped due to missing ground truth")
        
        # Visualizations
        st.markdown('<div class="section-header">🎨 Explanation Visualizations</div>', unsafe_allow_html=True)
        
        # Count available visualizations
        viz_explanations = []
        
        # Check which explanations we can visualize
        if ground_truth is not None:
            viz_explanations.append(('Ground Truth', ground_truth))
        
        if 'grad_full' in explanations:
            viz_explanations.append(('GradExplainer', explanations['grad_full']))
        elif 'grad' in explanations:
            viz_explanations.append(('GradExplainer', explanations['grad']))
        
        if 'gnn_full' in explanations:
            viz_explanations.append(('GNNExplainer', explanations['gnn_full']))
        elif 'gnn' in explanations:
            viz_explanations.append(('GNNExplainer', explanations['gnn']))
        
        if 'lime' in explanations:
            viz_explanations.append(('GraphLIME', explanations['lime']))
        
        if 'pgm' in explanations:
            viz_explanations.append(('PGMExplainer', explanations['pgm']))
        
        if len(viz_explanations) > 0:
            try:
                num_plots = len(viz_explanations)
                cols = min(num_plots, 5)  # Maximum 5 columns
                rows = (num_plots + cols - 1) // cols  # Calculate required rows
                
                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
                
                # Ensure axes is always a 2D array
                if rows == 1 and cols == 1:
                    axes = [[axes]]
                elif rows == 1:
                    axes = [axes]
                elif cols == 1:
                    axes = [[ax] for ax in axes]
                
                # Create visualizations
                for idx, (name, exp) in enumerate(viz_explanations):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row][col]
                    
                    try:
                        # Ensure node_idx is an integer
                        if hasattr(exp, 'node_idx') and isinstance(exp.node_idx, torch.Tensor):
                            exp.node_idx = exp.node_idx.item()
                        
                        # Create visualization
                        exp.visualize_node(
                            num_hops=num_hops,
                            graph_data=data,
                            ax=ax,
                            show_node_labels=True
                        )
                        ax.set_title(name, fontsize=14, fontweight='bold')
                        
                    except Exception as viz_error:
                        ax.text(0.5, 0.5, f'{name}\nVisualization Failed\n{str(viz_error)}', 
                               ha='center', va='center', transform=ax.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        st.warning(f"⚠️ {name} visualization failed: {viz_error}")
                
                # Hide unused subplots
                for idx in range(len(viz_explanations), rows * cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row][col].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Visualization creation failed: {e}")
                st.info("💡 Try reducing the number of hops or using a different node index")
        else:
            st.warning("⚠️ No explanations available for visualization")
    
    else:
        st.error("❌ No successful explanations generated. Please check the error messages above and try different parameters.")
        
        # Show troubleshooting tips
        st.markdown('''
        ### 🔧 Troubleshooting Tips:
        
        **For GraphLIME tensor dimension errors:**
        - The error suggests a mismatch between expected and actual tensor dimensions
        - Try using a simpler model (GCN instead of GAT/GSAGE)
        - Reduce the number of features or graph complexity
        - Try a different node index (some nodes may have unusual local structure)
        
        **General tips:**
        - Try reducing the number of subgraphs or subgraph size
        - Use a different model type (GCN is usually most stable)
        - Reduce the number of hops (try 1 instead of 2)
        - Try a different node index
        - Reduce training epochs if memory is an issue
        - Disable problematic explainers using the checkboxes in the sidebar
        ''')
    
    # Show error summary if any
    if explanation_errors:
        st.markdown('<div class="section-header">⚠️ Error Summary</div>', unsafe_allow_html=True)
        error_df = pd.DataFrame([
            {'Explainer': explainer, 'Error': error} 
            for explainer, error in explanation_errors.items()
        ])
        st.dataframe(error_df, use_container_width=True)
        
        # Specific help for GraphLIME errors
        if 'lime' in explanation_errors and 'Sizes of tensors must match' in explanation_errors['lime']:
            st.markdown('''
            <div class="warning-box">
                <h4>🔍 GraphLIME Tensor Dimension Error - Specific Help</h4>
                <p>The "Sizes of tensors must match except in dimension 1" error in GraphLIME typically occurs due to:</p>
                <ul>
                    <li><strong>Feature dimension mismatch:</strong> The model expects different feature dimensions than provided</li>
                    <li><strong>Model architecture incompatibility:</strong> Some model types (GAT, GSAGE) may have different tensor handling</li>
                    <li><strong>Graph structure issues:</strong> Isolated nodes or unusual connectivity patterns</li>
                </ul>
                <p><strong>Solutions to try:</strong></p>
                <ul>
                    <li>Use GCN model (most compatible with GraphLIME)</li>
                    <li>Try a different node index (avoid isolated or edge nodes)</li>
                    <li>Reduce graph complexity (fewer subgraphs, smaller subgraph size)</li>
                    <li>Ensure the selected node has sufficient neighbors</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown('''
    <div class="info-box">
        <h2>Welcome to GraphXAI Explainer Dashboard! 🚀</h2>
        <p>This interactive dashboard allows you to:</p>
        <ul>
            <li>🔧 Configure dataset and model parameters</li>
            <li>🤖 Train different GNN models (GCN, GIN, GAT, GSAGE)</li>
            <li>🔍 Generate explanations using multiple explainer methods</li>
            <li>📊 Compare explanation quality metrics</li>
            <li>🎨 Visualize and compare explanations</li>
        </ul>
        <p><strong>Configure your parameters in the sidebar and click "Run Analysis" to get started!</strong></p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Show some example configurations
    st.markdown("### 💡 Suggested Configurations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        **🚀 Quick Start (Most Stable)**
        - Model Layers: 3
        - Subgraphs: 6
        - Subgraph Size: 4
        - Model: GCN
        - Epochs: 200
        - Node Index: 8
        ''')
    
    with col2:
        st.markdown('''
        **🎯 High Performance**
        - Model Layers: 4
        - Subgraphs: 10
        - Subgraph Size: 6
        - Model: GAT
        - Epochs: 350
        - Node Index: 15
        ''')
    
    with col3:
        st.markdown('''
        **⚡ Fast Testing**
        - Model Layers: 2
        - Subgraphs: 4
        - Subgraph Size: 3
        - Model: GIN
        - Epochs: 100
        - Node Index: 5
        ''')
    
    # Important notes about GraphLIME
    st.markdown('''
    ### 🔍 Important Notes for GraphLIME
    
    GraphLIME can be sensitive to certain configurations. If you encounter tensor dimension errors:
    
    - **Use GCN model** - Most compatible with GraphLIME
    - **Start with smaller graphs** - Fewer subgraphs and smaller subgraph size
    - **Choose central nodes** - Avoid nodes at the edge of the graph
    - **Use the explainer checkboxes** - Disable problematic explainers if needed
    
    The dashboard includes multiple fallback strategies for GraphLIME and will attempt different approaches automatically.
    ''')
    
    # System information
    st.markdown("### 🖥️ System Information")
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    st.info(f"Running on: **{device_info}**")
    
    if torch.cuda.is_available():
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"CUDA Version: {torch.version.cuda}")
    
    st.info(f"PyTorch Version: {torch.__version__}")
    st.info("📝 This dashboard automatically handles data validation, tensor compatibility, and provides detailed error reporting.")