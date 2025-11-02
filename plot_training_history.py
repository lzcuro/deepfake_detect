#!/usr/bin/env python3
"""
학습 히스토리를 시각화하는 스크립트
weight 디렉토리에 저장된 .pkl 파일을 읽어서 plot으로 표시
"""

import os
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def load_training_history(pkl_file):
    """Pickle 파일에서 학습 히스토리 로드"""
    if not os.path.exists(pkl_file):
        print(f"[ERROR] 파일을 찾을 수 없습니다 - {pkl_file}")
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            history = pickle.load(f)
        
        train_loss, train_acc, valid_loss, valid_acc = history
        
        
        
        
        
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc
        }
    except Exception as e:
        print(f"[ERROR] 파일을 읽는 중 에러 발생 - {e}")
        return None


def plot_training_history(history, output_dir='.', save_plots=True, show_plot=True, filename='training_history'):
    """학습 히스토리를 플롯으로 시각화"""
    
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    valid_loss = history['valid_loss']
    valid_acc = history['valid_acc']
    
    # 데이터 길이가 다를 수 있으므로 별도 처리
    train_batches = range(1, len(train_loss) + 1)
    valid_epochs = range(1, len(valid_loss) + 1)
    
    # 2x2 subplot 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 1. Training & Validation Loss
    axes[0, 0].plot(train_batches, train_loss, 'b-', label='Training Loss', linewidth=1, alpha=0.5)
    axes[0, 0].plot(valid_epochs, valid_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Batch (Train) / Epoch (Valid)', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training & Validation Accuracy
    axes[0, 1].plot(train_batches, train_acc, 'b-', label='Training Accuracy', linewidth=1, alpha=0.5)
    axes[0, 1].plot(valid_epochs, valid_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Batch (Train) / Epoch (Valid)', fontsize=10)
    axes[0, 1].set_ylabel('Accuracy', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss Comparison (separate)
    axes[1, 0].plot(train_batches, train_loss, 'b-', label='Training Loss', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Batch', fontsize=10)
    axes[1, 0].set_ylabel('Loss', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Validation Accuracy
    axes[1, 1].plot(valid_epochs, valid_acc, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('Accuracy', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] 플롯 저장 완료: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="학습 히스토리 시각화")
    parser.add_argument("-f", "--file", help="분석할 .pkl 파일 경로")
    parser.add_argument("-d", "--directory", default="weight", help="디렉토리에서 .pkl 파일 검색")
    parser.add_argument("-m", "--model", choices=["ed", "vae"], help="모델 타입 필터 (ed 또는 vae)")
    parser.add_argument("--no-show", action="store_true", help="플롯 표시 안 함")
    parser.add_argument("--no-save", action="store_true", help="플롯 저장 안 함")
    
    args = parser.parse_args()
    
    # 단일 파일 지정
    if args.file:
        pkl_file = args.file
        if not pkl_file.endswith('.pkl'):
            print("[ERROR] .pkl 파일을 지정해주세요")
            return
        
        history = load_training_history(pkl_file)
        if history:
            base_name = os.path.basename(pkl_file).replace('.pkl', '')
            plot_training_history(history, output_dir='.', filename=base_name, save_plots=not args.no_save, show_plot=not args.no_show)
    
    # 디렉토리에서 자동 검색
    elif args.directory:
        pkl_files = []
        
        # 디렉토리에서 .pkl 파일 찾기
        for file in os.listdir(args.directory):
            if file.endswith('.pkl') and 'genconvit_' in file:
                # 모델 타입 필터링
                if args.model and args.model in file:
                    pkl_files.append(os.path.join(args.directory, file))
                elif not args.model:
                    pkl_files.append(os.path.join(args.directory, file))
        
        if not pkl_files:
            print(f"[ERROR] {args.directory} 디렉토리에서 .pkl 파일을 찾을 수 없습니다")
            return
        
        print(f"[INFO] {len(pkl_files)}개의 학습 파일을 찾았습니다:\n")
        
        # 각 파일에 대해 플롯 생성
        for i, pkl_file in enumerate(pkl_files, 1):
            print(f"[{i}/{len(pkl_files)}] {os.path.basename(pkl_file)}")
            history = load_training_history(pkl_file)
            
            if history:
                model_type = "ed" if "ed" in pkl_file else "vae"
                output_dir = os.path.dirname(pkl_file)
                filename = os.path.basename(pkl_file).replace('.pkl', '')
                
                # 저장 경로 지정
                plot_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                
                plot_training_history(
                    history, 
                    output_dir=plot_dir,
                    filename=filename,
                    save_plots=not args.no_save, 
                    show_plot=not args.no_show
                )
                
                # 최종 성능 출력
                print(f"   최종 성능:")
                print(f"   - Train Loss: {history['train_loss'][-1]:.4f}")
                print(f"   - Train Acc:  {history['train_acc'][-1]*100:.2f}%")
                print(f"   - Valid Loss: {history['valid_loss'][-1]:.4f}")
                print(f"   - Valid Acc:  {history['valid_acc'][-1]*100:.2f}%")
                print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

