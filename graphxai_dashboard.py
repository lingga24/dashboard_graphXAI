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
    dataset = ShapeGGen(
        model_layers=model_layers,
        num_subgraphs=num_subgraphs,
        subgraph_size=subgraph_size,
        prob_connection=prob_connection,
        add_sensitive_feature=False
    )
    return dataset

@st.cache_resource
def train_model(_dataset, model_choice, hidden_channels, learning_rate, epochs):
    """Train the GNN model"""
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
    
    return model, data, {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec, 'auprc': auprc, 'auroc': auroc}

def convert_to_full_explanation(exp, node_idx, subset, sub_edge_index, mapping, edge_mask, data):
    """Convert explanation from subgraph to full graph"""
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

# Main content
if run_analysis:
    # Generate dataset
    with st.spinner("🔄 Generating dataset..."):
        dataset = generate_dataset(model_layers, num_subgraphs, subgraph_size, prob_connection)
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
        model, data, metrics = train_model(dataset, model_choice, hidden_channels, learning_rate, epochs)
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
    
    with st.spinner("🔬 Generating explanations..."):
        try:
            # Initialize device and move data
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            model = model.to(device)
            
            # Ensure edge_index is properly formatted
            edge_index = data.edge_index
            if edge_index.dtype != torch.long:
                edge_index = edge_index.long()
            
            # Ensure node features are properly formatted
            x = data.x
            if x.dtype != torch.float32:
                x = x.float()
            
            st.info(f"📍 Explaining node {node_idx} in graph with {data.num_nodes} nodes and {data.num_edges} edges")
            
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
            
            try:
                explainers['grad'] = GradExplainer(model, criterion=torch.nn.CrossEntropyLoss())
                st.success("✅ GradExplainer initialized")
            except Exception as e:
                explanation_errors['grad'] = str(e)
                st.error(f"❌ GradExplainer failed to initialize: {e}")
            
            try:
                explainers['gnn'] = GNNExplainer(model)
                st.success("✅ GNNExplainer initialized")
            except Exception as e:
                explanation_errors['gnn'] = str(e)
                st.error(f"❌ GNNExplainer failed to initialize: {e}")
            
            try:
                explainers['pgm'] = PGMExplainer(model, explain_graph=False, p_threshold=0.1)
                st.success("✅ PGMExplainer initialized")
            except Exception as e:
                explanation_errors['pgm'] = str(e)
                st.error(f"❌ PGMExplainer failed to initialize: {e}")
            
            try:
                explainers['lime'] = GraphLIME(model)
                st.success("✅ GraphLIME initialized")
            except Exception as e:
                explanation_errors['lime'] = str(e)
                st.error(f"❌ GraphLIME failed to initialize: {e}")
            
            # Generate explanations with individual error handling
            progress_bar = st.progress(0)
            total_explainers = len(explainers)
            current_progress = 0
            
            # GradExplainer
            if 'grad' in explainers:
                try:
                    progress_bar.progress(current_progress / (total_explainers * 2))
                    st.info("🔄 Running GradExplainer...")
                    explanations['grad'] = explainers['grad'].get_explanation_node(sub_node_idx, sub_x, sub_edge_index)
                    st.success("✅ GradExplainer completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['grad'] = str(e)
                    st.error(f"❌ GradExplainer failed: {e}")
            
            # GNNExplainer
            if 'gnn' in explainers:
                try:
                    progress_bar.progress(current_progress / (total_explainers * 2))
                    st.info("🔄 Running GNNExplainer...")
                    explanations['gnn'] = explainers['gnn'].get_explanation_node(sub_node_idx, sub_x, sub_edge_index)
                    st.success("✅ GNNExplainer completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['gnn'] = str(e)
                    st.error(f"❌ GNNExplainer failed: {e}")
            
            # PGMExplainer
            if 'pgm' in explainers:
                try:
                    progress_bar.progress(current_progress / (total_explainers * 2))
                    st.info("🔄 Running PGMExplainer...")
                    explanations['pgm'] = explainers['pgm'].get_explanation_node(node_idx, x, edge_index)
                    st.success("✅ PGMExplainer completed")
                    current_progress += 1
                except Exception as e:
                    explanation_errors['pgm'] = str(e)
                    st.error(f"❌ PGMExplainer failed: {e}")
            
            # GraphLIME with special handling
            if 'lime' in explainers:
                try:
                    progress_bar.progress(current_progress / (total_explainers * 2))
                    st.info("🔄 Running GraphLIME...")
                    
                    # Try with full graph first
                    try:
                        explanations['lime'] = explainers['lime'].get_explanation_node(node_idx, x, edge_index)
                        st.success("✅ GraphLIME completed (full graph)")
                    except Exception as full_e:
                        st.warning(f"⚠️ GraphLIME failed on full graph: {full_e}")
                        # Try with subgraph
                        try:
                            explanations['lime'] = explainers['lime'].get_explanation_node(sub_node_idx, sub_x, sub_edge_index)
                            st.success("✅ GraphLIME completed (subgraph)")
                        except Exception as sub_e:
                            raise Exception(f"Both full graph ({full_e}) and subgraph ({sub_e}) failed")
                    
                    current_progress += 1
                except Exception as e:
                    explanation_errors['lime'] = str(e)
                    st.error(f"❌ GraphLIME failed: {e}")
            
            progress_bar.progress(1.0)
            progress_bar.empty()
            
            # Convert successful explanations to full format
            if 'grad' in explanations:
                explanations['grad_full'] = convert_to_full_explanation(
                    explanations['grad'], node_idx, subset, sub_edge_index, mapping, edge_mask, data
                )
            
            if 'gnn' in explanations:
                explanations['gnn_full'] = convert_to_full_explanation(
                    explanations['gnn'], node_idx, subset, sub_edge_index, mapping, edge_mask, data
                )
            
            # Ground truth
            ground_truth = dataset.explanations[node_idx][0]
            
            # Report results
            successful_explainers = len(explanations)
            failed_explainers = len(explanation_errors)
            
            if successful_explainers > 0:
                st.success(f"✅ Successfully generated {successful_explainers} explanations!")
            
            if failed_explainers > 0:
                st.warning(f"⚠️ {failed_explainers} explainer(s) failed. See details above.")
                
        except Exception as e:
            st.error(f"❌ Critical error in explanation generation: {e}")
            st.stop()
    
    
    st.success("✅ Explanations generated successfully!")
    
    # Only proceed with evaluation if we have successful explanations
    if len(explanations) > 0:
        # Evaluation metrics
        st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
        
        # Prepare evaluation data
        eval_results = []
        
        # Calculate metrics for successful explainers
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
                # Special handling for LIME
                if hasattr(explanations['lime'], 'feature_imp') and explanations['lime'].feature_imp is not None:
                    node_importance = explanations['lime'].feature_imp.mean().item()
                    lime_node_importance = torch.zeros(data.num_nodes)
                    lime_node_importance[node_idx] = node_importance
                    
                    node_reference = {i: i for i in range(data.num_nodes)}
                    edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
                    for edge_id, (src, dst) in enumerate(data.edge_index.T):
                        if src in subset and dst in subset:
                            edge_mask[edge_id] = True
                    
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
                    lime_exp.node_reference = node_reference
                    
                    gea_lime = graph_exp_acc(ground_truth, lime_exp)
                    gef_lime = graph_exp_faith(lime_exp, dataset, model)
                    eval_results.append(['GraphLIME', gea_lime, gef_lime])
                else:
                    st.warning("⚠️ GraphLIME explanation format not compatible for evaluation")
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
        else:
            st.error("❌ No successful evaluations to display")
        
        # Visualizations
        st.markdown('<div class="section-header">🎨 Explanation Visualizations</div>', unsafe_allow_html=True)
        
        # Determine number of successful explanations for subplot layout
        viz_count = 1 + len(explanations)  # 1 for ground truth + successful explanations
        
        try:
            fig, axes = plt.subplots(1, min(viz_count, 5), figsize=(5*min(viz_count, 5), 5))
            
            # Ensure axes is always a list
            if viz_count == 1:
                axes = [axes]
            elif viz_count == 2:
                axes = [axes] if not hasattr(axes, '__len__') else axes
            
            current_ax = 0
            
            # Ground Truth
            try:
                dataset.explanations[node_idx][0].visualize_node(
                    num_hops=num_hops,
                    graph_data=data,
                    ax=axes[current_ax],
                    show_node_labels=True
                )
                axes[current_ax].set_title("Ground Truth", fontsize=14, fontweight='bold')
                current_ax += 1
            except Exception as e:
                st.warning(f"⚠️ Ground truth visualization failed: {e}")
            
            # Successful explanations
            if 'grad_full' in explanations:
                try:
                    if isinstance(explanations['grad_full'].node_idx, torch.Tensor):
                        explanations['grad_full'].node_idx = explanations['grad_full'].node_idx.item()
                    explanations['grad_full'].visualize_node(
                        num_hops=num_hops,
                        graph_data=data,
                        ax=axes[current_ax],
                        show_node_labels=True
                    )
                    axes[current_ax].set_title("GradExplainer", fontsize=14, fontweight='bold')
                    current_ax += 1
                except Exception as e:
                    st.warning(f"⚠️ GradExplainer visualization failed: {e}")
            
            if 'gnn_full' in explanations:
                try:
                    if isinstance(explanations['gnn_full'].node_idx, torch.Tensor):
                        explanations['gnn_full'].node_idx = explanations['gnn_full'].node_idx.item()
                    explanations['gnn_full'].visualize_node(
                        num_hops=num_hops,
                        graph_data=data,
                        ax=axes[current_ax],
                        show_node_labels=True
                    )
                    axes[current_ax].set_title("GNNExplainer", fontsize=14, fontweight='bold')
                    current_ax += 1
                except Exception as e:
                    st.warning(f"⚠️ GNNExplainer visualization failed: {e}")
            
            if 'lime' in explanations:
                try:
                    if isinstance(explanations['lime'].node_idx, torch.Tensor):
                        explanations['lime'].node_idx = explanations['lime'].node_idx.item()
                    explanations['lime'].visualize_node(
                        num_hops=num_hops,
                        graph_data=data,
                        ax=axes[current_ax],
                        show_node_labels=True
                    )
                    axes[current_ax].set_title("GraphLIME", fontsize=14, fontweight='bold')
                    current_ax += 1
                except Exception as e:
                    st.warning(f"⚠️ GraphLIME visualization failed: {e}")
            
            if 'pgm' in explanations:
                try:
                    if isinstance(explanations['pgm'].node_idx, torch.Tensor):
                        explanations['pgm'].node_idx = explanations['pgm'].node_idx.item()
                    explanations['pgm'].visualize_node(
                        num_hops=num_hops,
                        graph_data=data,
                        ax=axes[current_ax],
                        show_node_labels=True
                    )
                    axes[current_ax].set_title("PGMExplainer", fontsize=14, fontweight='bold')
                    current_ax += 1
                except Exception as e:
                    st.warning(f"⚠️ PGMExplainer visualization failed: {e}")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Visualization failed: {e}")
            st.info("💡 Try reducing the number of hops or changing the node index")
    
    else:
        st.error("❌ No successful explanations generated. Please check the error messages above and try different parameters.")
        
        # Show troubleshooting tips
        st.markdown('''
        ### 🔧 Troubleshooting Tips:
        - Try reducing the number of subgraphs or subgraph size
        - Use a different model type (GCN is usually most stable)
        - Reduce the number of hops
        - Try a different node index
        - Reduce training epochs if memory is an issue
        ''')
    
    # Show error summary if any
    if explanation_errors:
        st.markdown('<div class="section-header">⚠️ Error Summary</div>', unsafe_allow_html=True)
        error_df = pd.DataFrame([
            {'Explainer': explainer, 'Error': error} 
            for explainer, error in explanation_errors.items()
        ])
        st.dataframe(error_df, use_container_width=True)

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
        **🚀 Quick Start**
        - Model Layers: 3
        - Subgraphs: 8
        - Subgraph Size: 5
        - Model: GCN
        - Epochs: 300
        ''')
    
    with col2:
        st.markdown('''
        **🎯 High Performance**
        - Model Layers: 4
        - Subgraphs: 12
        - Subgraph Size: 7
        - Model: GAT
        - Epochs: 400
        ''')
    
    with col3:
        st.markdown('''
        **⚡ Fast Testing**
        - Model Layers: 2
        - Subgraphs: 6
        - Subgraph Size: 4
        - Model: GIN
        - Epochs: 150
        ''')