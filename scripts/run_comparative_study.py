import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.train import train_model

def run_comparative_study():
    print("--- Starting Comparative Study: GAT vs. GCN vs. MLP ---")
    
    # 실험 설정
    models = ['gnn', 'gcn', 'mlp']
    
    # 충분한 학습을 위해 에포크 설정 (실험용으로 200회 정도가 적당)
    # 로컬에서 사용자가 직접 실행할 때 시간이 걸리더라도 결과를 볼 수 있도록 silent=False 권장
    epochs = 50 
    
    sample_count = None 
    
    results = []
    # Store predictions for visualization
    all_predictions = {} 
    
    output_dir = Path(project_root) / 'results' / 'experiments'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type in models:
        print(f"\n[Experiment] Training {model_type.upper()} model...")
        
        start_time = time.time()
        
        try:
            model_filename = f"best_model_{model_type}.pt"
            
            metrics = train_model(
                dataset_dir_name='processed',
                model_name=model_filename,
                model_type=model_type,
                epochs=epochs,
                sample_count=sample_count,
                silent=False 
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"  -> Done. Test R2: {metrics['test_r2']:.4f}, Test Loss: {metrics['test_loss']:.4f}")
            print(f"     Inference Time: {metrics.get('inference_time_per_sample_ms', 0):.4f} ms/sample")
            
            display_name = 'GAT' if model_type == 'gnn' else model_type.upper()
            
            results.append({
                'model': display_name,
                'test_r2': metrics['test_r2'],
                'test_loss': metrics['test_loss'],
                'inference_time_ms': metrics.get('inference_time_per_sample_ms', 0),
                'train_loss': metrics['train_loss'],
                'val_loss': metrics['val_loss'],
                'training_time': elapsed_time
            })
            
            # Store predictions for plotting (convert tensor to numpy)
            if metrics.get('y_true') is not None and metrics.get('y_pred') is not None:
                all_predictions[display_name] = {
                    'y_true': metrics['y_true'].numpy(),
                    'y_pred': metrics['y_pred'].numpy()
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
    print(f"\nResults saved to {csv_path}")

    # Plot 1: Bar Charts (Metrics)
    plot_comparison_bars(df, output_dir)
    
    # Plot 2 & 3: Scatter Plots and Representative Curves (Paper Figures)
    if all_predictions:
        plot_scatter_and_curves(all_predictions, output_dir)

def plot_comparison_bars(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    metrics = [
        ('test_r2', 'Test R2 Score (Higher is Better)', 0, 1.0, 0.01),
        ('test_loss', 'Test MSE Loss (Lower is Better)', None, None, 0.001),
        ('inference_time_ms', 'Inference Time (ms/sample)', None, None, 0.01)
    ]
    
    for i, (col, title, ylim_min, ylim_max, text_offset) in enumerate(metrics):
        if col not in df.columns: continue
        axes[i].bar(df['model'], df[col], color=colors[:len(df)])
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        if ylim_min is not None: axes[i].set_ylim(ylim_min, ylim_max)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        for j, v in enumerate(df[col]):
            axes[i].text(j, v + text_offset, f"{v:.4f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_study_bars.png', dpi=300)
    print(f"Bar plots saved to {output_dir / 'comparative_study_bars.png'}")

def plot_scatter_and_curves(predictions, output_dir):
    # 1. Scatter Plot (Predicted vs Actual for all test samples)
    # To avoid clutter, we flatten the curve points or take the mean/max shear
    # Let's plot "Peak Base Shear" (Max V) comparison
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = list(predictions.keys())
    
    for i, model_name in enumerate(models):
        data = predictions[model_name]
        # Calculate Peak Base Shear (Max value in each curve)
        # y_true shape: [Num_Samples, 100]
        peak_true = np.max(data['y_true'], axis=1)
        peak_pred = np.max(data['y_pred'], axis=1)
        
        ax = axes[i]
        ax.scatter(peak_true, peak_pred, alpha=0.5, s=10, c='blue')
        
        # Perfect fit line
        min_val = min(peak_true.min(), peak_pred.min())
        max_val = max(peak_true.max(), peak_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f"{model_name}: Peak Shear Prediction", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Normalized Peak Shear")
        ax.set_ylabel("Predicted Normalized Peak Shear")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Calculate R2 for peak shear specifically
        ss_res = np.sum((peak_true - peak_pred) ** 2)
        ss_tot = np.sum((peak_true - np.mean(peak_true)) ** 2)
        r2_peak = 1 - ss_res / ss_tot
        ax.text(0.05, 0.9, f"$R^2={r2_peak:.3f}$", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_scatter_plots.png', dpi=300)
    print(f"Scatter plots saved to {output_dir / 'comparative_scatter_plots.png'}")

    # 2. Representative Curves Plot
    # Select 3 samples: Best, Median, Worst (based on MSE of GAT)
    if 'GAT' in predictions:
        gat_data = predictions['GAT']
        mse_per_sample = np.mean((gat_data['y_true'] - gat_data['y_pred'])**2, axis=1)
        
        sorted_indices = np.argsort(mse_per_sample)
        best_idx = sorted_indices[0]
        median_idx = sorted_indices[len(sorted_indices)//2]
        worst_idx = sorted_indices[-1]
        
        sample_indices = [best_idx, median_idx, worst_idx]
        titles = ['Best Prediction', 'Median Prediction', 'Worst Prediction']
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        x_axis = np.linspace(0, 1, 100) # Normalized Drift
        
        for i, idx in enumerate(sample_indices):
            ax = axes2[i]
            
            # Ground Truth
            ax.plot(x_axis, gat_data['y_true'][idx], 'k-', lw=2.5, label='OpenSees (Target)')
            
            # Predictions from all models
            colors = {'GAT': 'blue', 'GCN': 'green', 'MLP': 'red'}
            styles = {'GAT': '-', 'GCN': '--', 'MLP': ':'}
            
            for model_name in models:
                pred_curve = predictions[model_name]['y_pred'][idx]
                ax.plot(x_axis, pred_curve, color=colors.get(model_name, 'gray'), 
                        linestyle=styles.get(model_name, '-'), lw=2, label=f'{model_name}')
            
            ax.set_title(f"{titles[i]} (Sample #{idx})", fontsize=12, fontweight='bold')
            ax.set_xlabel("Normalized Roof Drift")
            ax.set_ylabel("Normalized Base Shear")
            ax.grid(True, linestyle='--', alpha=0.5)
            if i == 0: ax.legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / 'comparative_curve_samples.png', dpi=300)
        print(f"Curve sample plots saved to {output_dir / 'comparative_curve_samples.png'}")

if __name__ == '__main__':
    run_comparative_study()