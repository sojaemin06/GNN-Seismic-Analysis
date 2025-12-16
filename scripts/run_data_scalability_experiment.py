import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from pathlib import Path

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.train import train_model, PushoverDataset

def run_experiment():
    # --- 실험 설정 ---
    # 현재 확보된 데이터 최대 개수 확인 (약 750개)
    # 실험 단계 설정 (데이터 개수)
    sample_counts = [100, 200, 300, 400, 500, 600, 700]
    
    # 생성 시간 추정치 (샘플당 초, 이전 로그 기반 평균)
    avg_gen_time_per_sample = 60.0 
    
    # 학습 에포크 (빠른 실험을 위해 50, 실제 논문용은 200~300 권장)
    epochs = 50 
    
    print(f"--- Starting Data Scalability Experiment (Fixed Test Set) ---")
    print(f"Sample Counts: {sample_counts}")
    print(f"Training Epochs: {epochs}")
    
    output_dir = Path(project_root) / 'results' / 'experiments'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Fixed Test Set
    print("\n[Setup] Preparing Fixed Test Set...")
    dataset_dir_name = 'processed'
    dataset_path = Path(project_root) / 'data' / dataset_dir_name
    
    # Load dataset to get IDs (Using the class from train.py logic or reusing it)
    # We need to instantiate PushoverDataset to access processed data easily
    dataset = PushoverDataset(root=str(dataset_path))
    
    all_ids = list(set([d.structure_id for d in dataset if hasattr(d, 'structure_id')]))
    num_total = len(all_ids)
    num_test = int(num_total * 0.2)
    
    random.seed(42) # Fixed seed for test set selection
    fixed_test_ids = random.sample(all_ids, num_test)
    
    print(f"  -> Total Unique Structures: {num_total}")
    print(f"  -> Fixed Test Structures: {len(fixed_test_ids)} (20%)")
    print("--------------------------------------------")

    results = []

    for count in sample_counts:
        print(f"\n[Experiment] Training with {count} samples (plus fixed test set)...")
        
        # 1. 학습 시간 측정
        start_train = time.time()
        
        # 학습 실행 (Metrics 반환)
        try:
            # We pass sample_count for TRAINING data size. 
            # train_model logic with fixed_test_ids:
            # 1. Removes test_ids from pool.
            # 2. Samples `sample_count` from the remaining pool for Train+Val.
            metrics = train_model(
                sample_count=count, 
                epochs=epochs, 
                silent=True, 
                dataset_dir_name='processed',
                fixed_test_ids=fixed_test_ids,
                model_type='gnn' # Using Proposed GAT Model
            )
        except Exception as e:
            print(f"Error during training with {count} samples: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        end_train = time.time()
        train_time = end_train - start_train
        
        # 2. 데이터 생성 시간 (추정)
        gen_time = count * avg_gen_time_per_sample
        total_time = gen_time + train_time
        
        # 3. 결과 기록
        result_entry = {
            'sample_count': count,
            'gen_time_est': gen_time,
            'train_time': train_time,
            'total_time_est': total_time,
            'test_loss': metrics['test_loss'],
            'test_r2': metrics['test_r2'],
            'train_loss_final': metrics['train_loss'],
            'val_loss_best': metrics['val_loss']
        }
        results.append(result_entry)
        
        print(f"  -> Done. Test R2: {metrics['test_r2']:.4f}, Train Time: {train_time:.1f}s")

    # --- 결과 저장 ---
    df = pd.DataFrame(results)
    csv_path = output_dir / 'data_scalability_experiment_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nExperiment finished. Results saved to {csv_path}")
    
    # --- 시각화 ---
    plot_results(df, output_dir)

def plot_results(df, output_dir):
    plt.figure(figsize=(12, 5))

    # 1. 데이터 수 vs 성능 (R2)
    plt.subplot(1, 2, 1)
    plt.plot(df['sample_count'], df['test_r2'], marker='o', linestyle='-', color='b', label='Test R2')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('R2 Score (Fixed Test Set)')
    plt.title('GAT Performance vs Data Size')
    plt.grid(True)
    plt.legend()

    # 2. 데이터 수 vs 시간 (생성/학습)
    plt.subplot(1, 2, 2)
    plt.plot(df['sample_count'], df['gen_time_est'] / 60, marker='s', linestyle='--', color='g', label='Gen Time (Est.)')
    plt.plot(df['sample_count'], df['train_time'] / 60, marker='^', linestyle='-', color='r', label='Train Time')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Time (minutes)')
    plt.title('Cost (Time) vs Data Size')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = output_dir / 'data_scalability_plots.png'
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")

if __name__ == '__main__':
    run_experiment()