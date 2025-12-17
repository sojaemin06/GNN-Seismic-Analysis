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

from src.gnn1.train import train_model

def run_ood_test():
    print("--- Starting OOD Generalization Test ---")
    
    models = ['gnn', 'gcn', 'mlp']
    
    # In-Distribution (ID) Performance (Load from previous results)
    # We will fetch this from 'comparative_study_results.csv' if available, or re-evaluate.
    # To keep it consistent, let's load the best models and re-evaluate on ID Test Set first.
    # Actually, we can just use the OOD evaluation function.
    
    ood_dataset_dir = 'ood' # data/ood
    id_dataset_dir = 'processed' # data/processed
    
    results = []
    
    output_dir = Path(project_root) / 'results' / 'experiments'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Evaluate on OOD Data
    print(f"\n[Evaluation] Testing models on OOD Dataset ('{ood_dataset_dir}')...")
    
    for model_type in models:
        model_filename = f"best_model_{model_type}.pt"
        display_name = 'GAT' if model_type == 'gnn' else model_type.upper()
        
        try:
            # We use train_model in evaluation mode by setting epochs=0 (or 1) but we need a proper eval function.
            # train_model is designed for training. Let's use it but just load weights and test.
            # We can trick it by setting epochs=0 if the logic supports it, or just 1 epoch.
            # Actually train_model loads weights if they exist.
            # Let's modify train_model slightly or just use it as is with epochs=0 if possible.
            # train_model code runs training loop for epochs range. range(0) does nothing.
            # Then it runs test. Perfect.
            
            # Note: We need to make sure we don't overwrite the best model.
            # But train_model saves model if val loss improves. With 0 epochs, val loss won't be calculated/improved?
            # Actually, train_model logic:
            # for epoch in range(epochs): ...
            # if val_loader: ...
            # Test & R2 Calculation...
            # So if epochs=0, it skips training loop, skips validation loop (unless we force it), and goes to test.
            # BUT, it loads the model from `model_path` if it exists.
            
            # Caution: The `dataset_dir` argument changes the dataset.
            # We want to test on 'ood'.
            
            metrics = train_model(
                dataset_dir_name=ood_dataset_dir,
                model_name=model_filename, # Load THIS model
                model_type=model_type,
                epochs=0, # Skip training
                sample_count=None,
                silent=True
            )
            
            results.append({
                'model': display_name,
                'dataset': 'OOD (6-Story)',
                'r2': metrics['test_r2'],
                'loss': metrics['test_loss']
            })
            print(f"  -> {display_name}: R2={metrics['test_r2']:.4f}, Loss={metrics['test_loss']:.4f}")
            
        except Exception as e:
            print(f"  -> {display_name} Failed: {e}")
            import traceback
            traceback.print_exc()

    # 2. Evaluate on ID Data (Reference)
    # We can load previous comparative results or re-run on ID.
    # Loading from CSV is faster.
    comp_results_path = output_dir / 'comparative_study_results.csv'
    if comp_results_path.exists():
        print(f"\n[Reference] Loading In-Distribution results from {comp_results_path}...")
        df_id = pd.read_csv(comp_results_path)
        for _, row in df_id.iterrows():
            results.append({
                'model': row['model'],
                'dataset': 'In-Distribution (3-5 Story)',
                'r2': row['test_r2'],
                'loss': row['test_loss']
            })
    else:
        print("\n[Warning] Comparative study results not found. Skipping ID reference.")

    # 3. Visualization
    df_res = pd.DataFrame(results)
    print("\n--- Generalization Test Results ---")
    print(df_res)
    
    df_res.to_csv(output_dir / 'ood_test_results.csv', index=False)
    plot_ood_comparison(df_res, output_dir)

def plot_ood_comparison(df, output_dir):
    # Bar Chart: ID vs OOD R2 for each model
    models = df['model'].unique()
    datasets = df['dataset'].unique()
    
    if len(datasets) < 2:
        print("Not enough datasets for comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Filter data
    id_data = df[df['dataset'].str.contains('In-Distribution')]
    ood_data = df[df['dataset'].str.contains('OOD')]
    
    # Align data with models order
    id_scores = [id_data[id_data['model'] == m]['r2'].values[0] if not id_data[id_data['model'] == m].empty else 0 for m in models]
    ood_scores = [ood_data[ood_data['model'] == m]['r2'].values[0] if not ood_data[ood_data['model'] == m].empty else 0 for m in models]
    
    rects1 = ax.bar(x - width/2, id_scores, width, label='In-Distribution (3-5 Story)', color='skyblue')
    rects2 = ax.bar(x + width/2, ood_scores, width, label='OOD (6 Story)', color='salmon')
    
    ax.set_ylabel('$R^2$ Score')
    ax.set_title('Generalization Performance: In-Distribution vs Out-of-Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = output_dir / 'ood_generalization_plot.png'
    plt.savefig(plot_path)
    print(f"OOD Comparison plot saved to {plot_path}")

if __name__ == '__main__':
    run_ood_test()
