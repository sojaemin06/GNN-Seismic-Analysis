import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.train import train_model

def run_experiment():
    # --- 실험 설정 ---
    # 현재 확보된 데이터 최대 개수 확인 (약 363개)
    # 실험 단계 설정 (데이터 개수)
    sample_counts = [50, 100, 150, 200, 250, 300]
    
    # 생성 시간 추정치 (샘플당 초, 이전 로그 기반 평균)
    avg_gen_time_per_sample = 60.0 
    
    # 학습 에포크 (빠른 실험을 위해 100, 실제 논문용은 200~300 권장)
    epochs = 100 
    
    results = []
    
    print(f"--- Starting Data Scalability Experiment ---")
    print(f"Sample Counts: {sample_counts}")
    print(f"Estimated Gen Time per Sample: {avg_gen_time_per_sample}s")
    print(f"Training Epochs: {epochs}")
    print("--------------------------------------------")

    output_dir = Path(project_root) / 'results' / 'experiments'
    output_dir.mkdir(parents=True, exist_ok=True)

    for count in sample_counts:
        print(f"\n[Experiment] Training with {count} samples...")
        
        # 1. 학습 시간 측정
        start_train = time.time()
        
        # 학습 실행 (Metrics 반환)
        try:
            metrics = train_model(sample_count=count, epochs=epochs, silent=True, dataset_dir_name='processed')
        except Exception as e:
            print(f"Error during training with {count} samples: {e}")
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
    plt.xlabel('Number of Samples')
    plt.ylabel('R2 Score')
    plt.title('GNN Performance vs Data Size')
    plt.grid(True)
    plt.legend()

    # 2. 데이터 수 vs 시간 (생성/학습)
    plt.subplot(1, 2, 2)
    plt.plot(df['sample_count'], df['gen_time_est'] / 60, marker='s', linestyle='--', color='g', label='Gen Time (Est.)')
    plt.plot(df['sample_count'], df['train_time'] / 60, marker='^', linestyle='-', color='r', label='Train Time')
    plt.xlabel('Number of Samples')
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
