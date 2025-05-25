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

# === Streamlit Setup ===
st.title('Graph Neural Network Model Dashboard')

# File Upload
uploaded_file = st.file_uploader("Upload Graph Data (.pt)", type="pt")
if uploaded_file is not None:
    data = torch.load(uploaded_file)

    # === Enrich Features if needed ===
    if data.x.shape[1] == 1:
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
    # === Choose Model ===
    model_choice = st.selectbox(
        "Choose Model", 
        ["Choose Model", "GCN", "GIN", "GAT", "GraphSAGE"], 
        index=0  # Default is "Choose Model"
    )

    if model_choice != "Choose Model":
        model_map = {
            'GCN': GCN_3layer_basic,
            'GIN': GIN_3layer_basic,
            'GAT': GAT_3layer_basic,
            'GraphSAGE': GSAGE_3layer
        }

        # === Model Setup ===
        ModelClass = model_map[model_choice]

        # === Load Model ===
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ModelClass(hidden_channels=64, input_feat=data.x.size(1), classes=int(data.y.max().item()) + 1).to(device)

        # === Prepare Data ===
        x, y, edge_index = data.x.to(device), data.y.to(device), data.edge_index.to(device)
        train_mask, test_mask = data.train_mask, data.test_mask
        train_idx = train_mask.nonzero(as_tuple=True)[0].to(device)
        test_idx = test_mask.nonzero(as_tuple=True)[0].to(device)

        # === Training or Loading Model ===
        model_path = f"{model_choice}_model.pt"
        class_weights = compute_class_weight('balanced', classes=np.unique(y.cpu()), y=y.cpu().numpy())
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_val_f1, best_epoch, patience = 0, 0, 10
        training_start = time.time()

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            st.write(f"Model {model_choice} loaded successfully.")
        else:
            for epoch in range(1, 201):
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
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch
                        torch.save(model.state_dict(), model_path)
                        st.write(f"[Epoch {epoch}] ✅ New best F1: {val_f1:.4f}, Loss: {loss.item():.4f}")
                    else:
                        st.write(f"[Epoch {epoch}] F1: {val_f1:.4f}, Loss: {loss.item():.4f} (no improvement)")
                    if epoch - best_epoch >= patience:
                        st.write(f"🛑 Early stopping at epoch {epoch}. Best was at epoch {best_epoch} with F1 {best_val_f1:.4f}")
                        break
                torch.cuda.empty_cache()

        model.load_state_dict(torch.load(model_path))
        model.eval()
        training_time = time.time() - training_start

        # === Evaluation ===
        with torch.no_grad():
            out = model(x, edge_index)
            pred = out.argmax(dim=1)

            # === Menyimpan node hasil prediksi
            buy_nodes = test_idx[(pred[test_idx] == 1)]
            not_buy_nodes = test_idx[(pred[test_idx] == 0)]

            # Tampilkan jumlah masing-masing
            st.write(f"🔍 Jumlah node yang direkomendasikan untuk beli (Buy): {len(buy_nodes)}")
            st.write(f"🧩 Jumlah node yang tidak direkomendasikan untuk beli (Not Buy): {len(not_buy_nodes)}")

            # Jika ingin melihat ID node-nya:
            st.write("Contoh node rekomendasi (Buy):", buy_nodes[:10].cpu().numpy())
            st.write("Contoh node bukan rekomendasi (Not Buy):", not_buy_nodes[:10].cpu().numpy())

            # Simpan ke CSV
            buy_node_list = buy_nodes.cpu().numpy()
            not_buy_node_list = not_buy_nodes.cpu().numpy()
            df_buy = pd.DataFrame({'Node_ID': buy_node_list, 'Predicted_Label': 1})
            df_not_buy = pd.DataFrame({'Node_ID': not_buy_node_list, 'Predicted_Label': 0})

            recommendation_df = pd.concat([df_buy, df_not_buy], ignore_index=True)
            recommendation_df.to_csv(f"{model_choice}_recommendation_results.csv", index=False)
            st.write("📁 Hasil prediksi per node disimpan di:", f"{model_choice}_recommendation_results.csv")

            true_buy = test_idx[(pred[test_idx] == 1) & (y[test_idx] == 1)]
            false_buy = test_idx[(pred[test_idx] == 1) & (y[test_idx] == 0)]

            st.write(f"✅ Benar direkomendasikan Buy: {len(true_buy)}")
            st.write(f"❌ Salah direkomendasikan Buy: {len(false_buy)}")

            y_true = y[test_idx].cpu().numpy()
            y_pred = pred[test_idx].cpu().numpy()
            y_proba = out[test_idx][:, 1].sigmoid().cpu().numpy() if out.size(1) > 1 else out[test_idx].cpu().numpy()

            st.write("📊 Classification Report on Test Set:")
            st.text(classification_report(y_true, y_pred, target_names=['Not Buy', 'Buy'], zero_division=0))
            f1 = f1_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_proba)
            except:
                auc = 0.0
            st.write(f"🧪 {model_choice} - F1: {f1:.4f}, AUC: {auc:.4f}, Time: {training_time:.2f}s")

        # === Node Explanation ===
        correct_buy = test_idx[(pred[test_idx] == y[test_idx]) & (y[test_idx] == 1)]
        if len(correct_buy) == 0:
            raise Exception("No correctly predicted Buy node found.")
        node_to_explain = random.choice(correct_buy.tolist())
        st.write(f"Explaining node {node_to_explain}...")

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(node_idx=node_to_explain, num_hops=1, edge_index=edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        sub_x = x[subset]

        def evaluate_graph_faithfulness(explanation: Explanation, model: torch.nn.Module, data: Data, top_k=0.25) -> float:
            try:
                # Untuk node-level explainer
                from torch.nn import functional as F

                # Hitung softmax output awal
                with torch.no_grad():
                    original_out = model(data.x, data.edge_index)
                    original_softmax = F.softmax(original_out[explanation.node_idx], dim=-1)

                    # Buat salinan fitur, lalu mask node yang tidak penting
                    pert_x = data.x.clone()
                    node_imp = explanation.node_imp.clone()
                    top_k_nodes = node_imp.topk(int(len(node_imp) * top_k)).indices
                    all_nodes = torch.arange(data.x.shape[0])
                    unimportant_nodes = list(set(all_nodes.tolist()) - set(top_k_nodes.tolist()))

                    pert_x[unimportant_nodes] = 0.0  # masking unimportant nodes
                    perturbed_out = model(pert_x, data.edge_index)
                    perturbed_softmax = F.softmax(perturbed_out[explanation.node_idx], dim=-1)

                    # KL-divergence sebagai metrik fidelity
                    GEF = 1 - torch.exp(-F.kl_div(original_softmax.log(), perturbed_softmax, reduction='sum')).item()
                return GEF
            except Exception as e:
                return 0.0

        explainers = {
            'GradExplainer': GradExplainer,
            'GNNExplainer': GNNExplainer,
            'GraphLIME': GraphLIME,
            'PGMExplainer': PGMExplainer
        }
        results = {}

        for name, Explainer in explainers.items():
            st.write(f"Running {name}...")
            try:
                explainer = Explainer(model=model, criterion=criterion) if name == 'GradExplainer' \
                    else Explainer(model=model, explain_graph=False) if name == 'PGMExplainer' \
                    else Explainer(model=model)

                input_x = sub_x + 0.0001 * torch.randn_like(sub_x) if name == 'GraphLIME' else sub_x
                exp = explainer.get_explanation_node(
                    node_idx=int(mapping[0]),
                    x=input_x,
                    edge_index=sub_edge_index,
                    forward_kwargs={}
                )

                if exp.node_imp is None:
                    st.write(f"{name} failed: No node importance returned.")
                    continue

                full_node_imp = torch.zeros(data.num_nodes)
                for i, imp in enumerate(exp.node_imp):
                    full_node_imp[subset[i]] = imp

                explanation = Explanation(
                    node_imp=full_node_imp,
                    node_idx=int(mapping[0])
                )

                # Prepare subgraph data for visualization
                sub_data = Data(x=sub_x, edge_index=sub_edge_index)
                sub_data.nodes = subset  # required by GraphXAI visualization

                # Create Explanation and attach subgraph context
                explanation = Explanation(node_imp=full_node_imp, node_idx=int(mapping[0]))
                explanation.enc_subgraph = sub_data  # ✅ attach subgraph to Explanation

                # Correcting the visualize_node call
                fig, ax = plt.subplots(figsize=(8, 8))
                st.write(f"{name} node importance values:")
                st.write(exp.node_imp)
                explanation.visualize_node(
                    num_hops=1,  # specify the number of hops for visualization
                    graph_data=sub_data,
                    additional_hops=0,
                    heat_by_prescence=False,
                    heat_by_exp=True,
                    node_agg_method='sum',
                    show_node_labels=True,
                    show=True,
                    norm_imps=False
                )
                st.pyplot(fig)  # Display the figure in Streamlit

                gef = evaluate_graph_faithfulness(explanation, model, data)
                results[name] = {'GEF': gef}
                torch.cuda.empty_cache()
            except Exception as e:
                st.write(f"{name} failed: {e}")
                continue

        # === Save Results ===
        df = pd.DataFrame(results).T
        csv_path = f"{model_choice}_explanation_metrics.csv"
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path, index_col=0)
            df = pd.concat([df_existing, df])
        df.to_csv(csv_path)
        st.write("\nSaved to CSV. Explanation Metrics:")
        st.write(df)

        summary = pd.DataFrame([{
            'Model': model_choice,
            'F1 Score': round(f1, 4),
            'AUC': round(auc, 4),
            'Training Time (sec)': round(training_time, 2)
        }])
        sum_path = f"{model_choice}_runtime_metrics.csv"
        if os.path.exists(sum_path):
            summary_existing = pd.read_csv(sum_path)
            summary = pd.concat([summary_existing, summary])
        summary.to_csv(sum_path, index=False)
        st.write("\nTraining Summary:")
        st.write(summary)
