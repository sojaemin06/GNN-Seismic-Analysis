import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.train import train_model, PushoverDataset
import random

def run_comparative_study():
    print("--- Starting Comparative Study: GAT vs. GCN vs. MLP ---")
    
    # 1. Prepare Fixed Data Splits (Same for all models)
    # This ensures fair comparison by using identical Train/Val/Test sets
    print("\n[Setup] Preparing Fixed Train/Val/Test Splits...")
    dataset_dir_name = 'processed'
    dataset_path = Path(project_root) / 'data' / dataset_dir_name
    
    # Load dataset to get unique IDs
    dataset = PushoverDataset(root=str(dataset_path))
    all_ids = list(set([d.structure_id for d in dataset if hasattr(d, 'structure_id')]))
    
    random.seed(42) # Global seed for reproducibility
    num_total = len(all_ids)
    num_test = int(num_total * 0.2)
    fixed_test_ids = random.sample(all_ids, num_test)
    
    print(f"  -> Total Unique Structures: {num_total}")
    print(f"  -> Fixed Test Set: {len(fixed_test_ids)} structures")
    print("--------------------------------------------")
    
    models = ['gnn', 'gcn', 'mlp']
    # 충분한 학습을 위해 에포크 설정 (실제 논문용은 200~300 권장)
    epochs = 200 
    sample_count = None 
    
    results = []
    all_predictions = {}
    
    output_dir = Path(project_root) / 'results' / 'experiments'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type in models:
        print(f"\n[Experiment] Training {model_type.upper()} model...")
        
        start_time = time.time()
        
        try:
            model_filename = f"best_model_{model_type}.pt"
            
            metrics = train_model(
                dataset_dir_name=dataset_dir_name,
                model_name=model_filename,
                model_type=model_type,
                epochs=epochs,
                sample_count=sample_count,
                silent=False,
                fixed_test_ids=fixed_test_ids 
            )
            
            elapsed_time = time.time() - start_time
            
            # --- Additional Engineering Metrics Calculation ---
            y_true = metrics['y_true'].numpy() # [N, 100]
            y_pred = metrics['y_pred'].numpy() # [N, 100]
            
            # 1. Initial Stiffness (K_ini)
            # Use first 10 points (0.04 drift) or similar small range
            k_idx = 5 
            k_true = y_true[:, k_idx] / (k_idx + 1) # Simple slope proxy
            k_pred = y_pred[:, k_idx] / (k_idx + 1)
            r2_k = r2_score(k_true, k_pred)
            
            # 2. Energy Dissipation (Area under curve)
            # Assuming uniform spacing dx=1 for index integration (relative comparison)
            e_true = np.sum(y_true, axis=1)
            e_pred = np.sum(y_pred, axis=1)
            r2_e = r2_score(e_true, e_pred)
            
            # 3. Peak Strength (V_max)
            v_true = np.max(y_true, axis=1)
            v_pred = np.max(y_pred, axis=1)
            r2_v = r2_score(v_true, v_pred)
            
            print(f"  -> Done. Test R2: {metrics['test_r2']:.4f}")
            print(f"     Metrics R2 -> Stiffness: {r2_k:.4f}, Energy: {r2_e:.4f}, Peak: {r2_v:.4f}")
            
            display_name = 'GAT' if model_type == 'gnn' else model_type.upper()
            
            results.append({
                'model': display_name,
                'test_r2': metrics['test_r2'], # Global Curve R2
                'r2_stiffness': r2_k,
                'r2_energy': r2_e,
                'r2_peak': r2_v,
                'test_loss': metrics['test_loss'],
                'inference_time_ms': metrics.get('inference_time_per_sample_ms', 0),
                'training_time': elapsed_time
            })
            
            if metrics.get('y_true') is not None and metrics.get('y_pred') is not None:
                all_predictions[display_name] = {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save CSV
    df = pd.DataFrame(results)
    print("\n--- Comparative Results ---")
    print(df)
    csv_path = output_dir / 'comparative_study_results.csv'
    df.to_csv(csv_path, index=False)
    
    # Plot 1: Detailed Metrics Bar Chart
    plot_detailed_metrics(df, output_dir)
    
    # Plot 2 & 3: Scatter & Curves
    if all_predictions:
        plot_scatter_and_curves(all_predictions, output_dir, df)

def plot_detailed_metrics(df, output_dir):
    # Plot R2 for: Global, Stiffness, Energy, Peak
    metrics = ['test_r2', 'r2_stiffness', 'r2_energy', 'r2_peak']
    titles = ['Global Curve $R^2$', 'Initial Stiffness $R^2$', 'Energy Dissipation $R^2$', 'Peak Strength $R^2$']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns: continue
        ax = axes[i]
        ax.bar(df['model'], df[metric], color=colors[:len(df)])
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for j, v in enumerate(df[metric]):
            ax.text(j, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_study_metrics.png', dpi=300)
    print(f"Detailed metrics plot saved to {output_dir / 'comparative_study_metrics.png'}")

def plot_scatter_and_curves(predictions, output_dir, metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = list(predictions.keys())
    
    for i, model_name in enumerate(models):
        data = predictions[model_name]
        peak_true = np.max(data['y_true'], axis=1)
        peak_pred = np.max(data['y_pred'], axis=1)
        
        ax = axes[i]
        ax.scatter(peak_true, peak_pred, alpha=0.5, s=10, c='blue')
        
        min_val = min(peak_true.min(), peak_pred.min())
        max_val = max(peak_true.max(), peak_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f"{model_name}: Peak Shear Prediction", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Normalized Peak Shear")
        ax.set_ylabel("Predicted Normalized Peak Shear")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Display Multiple R2s
        row = metrics_df[metrics_df['model'] == model_name].iloc[0]
        text_str = (f"Global $R^2$ = {row['test_r2']:.3f}\n"
                    f"Peak $R^2$ = {row['r2_peak']:.3f}\n"
                    f"Energy $R^2$ = {row['r2_energy']:.3f}")
        
        ax.text(0.05, 0.80, text_str, transform=ax.transAxes, fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_scatter_plots.png', dpi=300)
    print(f"Scatter plots saved to {output_dir / 'comparative_scatter_plots.png'}")

    # Curve plotting logic remains same (omitted for brevity, assume previous implementation or re-include)
    # ... (Re-including curve plotting for completeness)
    
    if 'GAT' in predictions:
        gat_data = predictions['GAT']
        mse_per_sample = np.mean((gat_data['y_true'] - gat_data['y_pred'])**2, axis=1)
        sorted_indices = np.argsort(mse_per_sample)
        
        sample_indices = [sorted_indices[0], sorted_indices[len(sorted_indices)//2], sorted_indices[-1]]
        titles = ['Best Prediction', 'Median Prediction', 'Worst Prediction']
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        x_axis = np.linspace(0, 1, 100)
        
        for i, idx in enumerate(sample_indices):
            ax = axes2[i]
            ax.plot(x_axis, gat_data['y_true'][idx], 'k-', lw=2.5, label='OpenSees (Target)')
            colors = {'GAT': 'blue', 'GCN': 'green', 'MLP': 'red'}
            styles = {'GAT': '-', 'GCN': '--', 'MLP': ':'}
            for model_name in models:
                pred = predictions[model_name]['y_pred'][idx]
                ax.plot(x_axis, pred, color=colors.get(model_name, 'gray'), linestyle=styles.get(model_name, '-'), lw=2, label=model_name)
            ax.set_title(f"{titles[i]} (Sample #{idx})", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            if i==0: ax.legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / 'comparative_curve_samples.png', dpi=300)

if __name__ == '__main__':
    run_comparative_study()
