#!/usr/bin/env python3


import sys, os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from model.config import load_config
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE
from dataset.loader import load_data, load_checkpoint
import argparse

config = load_config()

# RTX 5090 GPU 최적화 설정 (호환성 문제로 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("⚠️  RTX 5090 호환성 문제로 CPU로 실행됩니다.")
    print("💡 GPU 사용을 원한다면 최신 PyTorch를 설치하세요.")
    print("="*60)

def train_single_model_parallel(model_config):
    """단일 모델을 병렬로 학습하는 함수"""
    model_type, dataloaders, num_epochs, pretrained_model_filename, batch_size, model_name, device_id = model_config
    
    # 각 프로세스마다 다른 GPU 사용 (GPU가 여러 개인 경우)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device(f"cuda:{device_id % torch.cuda.device_count()}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"🤖 {model_name} 모델 학습 시작 (Device: {device})")
    print(f"{'='*60}")
    
    # 모델 생성
    if model_type == "ed":
        model = GenConViTED(config)
    else:
        model = GenConViTVAE(config)
    
    # 파인튜닝을 위한 낮은 학습률 설정
    if pretrained_model_filename:
        learning_rate = float(config["learning_rate"]) * 0.1
        print(f"🔧 파인튜닝 모드: 학습률 {learning_rate:.6f} (기본값의 10%)")
    else:
        learning_rate = float(config["learning_rate"])
        print(f"🔧 처음부터 학습: 학습률 {learning_rate:.6f}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    
    # 클래스 불균형 해결을 위한 가중치 설정 (Real:Fake ≈ 1:5.27)
    # Real 클래스(0)에 더 높은 가중치를 부여
    weights = torch.tensor([5.27, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion.to(device)
    mse = nn.MSELoss()
    
    # 검증 손실 기반으로 학습률 동적 조정 (verbose 인자 제거)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    if pretrained_model_filename:
        print(f"🔄 사전 훈련된 모델 로딩: {pretrained_model_filename}")
        model, optimizer, start_epoch, min_loss = load_pretrained(
            model, optimizer, pretrained_model_filename
        )
    else:
        start_epoch = 0
        min_loss = float('inf')
    
    model.to(device)
    torch.manual_seed(1)
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    since = time.time()
    
    # Early Stopping 설정
    patience = 10
    best_epoch = 0
    no_improve_count = 0
    
    print(f"\n학습 설정:")
    print(f"  총 에포크: {num_epochs}")
    print(f"  Early Stopping: {patience} 에포크 동안 개선 없으면 중단")
    print(f"  배치 크기: {batch_size}")
    print(f"  목표: 검증 손실 최소화")
    print("="*60)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # 학습 함수 import
        if model_type == "ed":
            from train.train_ed import train, valid
        else:
            from train.train_vae import train, valid
        
        # 학습
        train_loss, train_acc, epoch_train_loss = train(
            model,
            device,
            dataloaders["train"],
            criterion,
            optimizer,
            epoch,
            train_loss,
            train_acc,
            mse,
        )
        
        # 검증
        valid_loss, valid_acc = valid(
            model,
            device,
            dataloaders["valid"],
            criterion,
            epoch,
            valid_loss,
            valid_acc,
            mse,
        )
        
        # 스케줄러 step 호출 (ReduceLROnPlateau는 validation loss를 인자로 받음)
        scheduler.step(valid_loss[-1])
        
        # 시각적 로깅
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 진행률 바 계산
        progress = (epoch + 1) / num_epochs
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 성능 개선 표시
        if epoch > 0:
            loss_improvement = valid_loss[-2] - valid_loss[-1] if len(valid_loss) > 1 else 0
            acc_improvement = valid_acc[-1] - valid_acc[-2] if len(valid_acc) > 1 else 0
            
            loss_arrow = "↑" if loss_improvement > 0 else "↓" if loss_improvement < 0 else "→"
            acc_arrow = "↑" if acc_improvement > 0 else "↓" if acc_improvement < 0 else "→"
        else:
            loss_arrow = "NEW"
            acc_arrow = "NEW"
        
        print(f"\n[{model_name}] EPOCH {epoch+1:2d}/{num_epochs} | {bar} | {progress*100:5.1f}%")
        print(f"[{model_name}] 시간: {epoch_time:6.2f}초 | 학습률: {current_lr:.2e}")
        print(f"[{model_name}] 학습 손실: {epoch_train_loss:8.4f} | {loss_arrow} 검증 손실: {valid_loss[-1]:8.4f}")
        print(f"[{model_name}] 검증 정확도: {valid_acc[-1]*100:6.2f}% | {acc_arrow} 개선: {acc_improvement*100:+.2f}%" if epoch > 0 else f"[{model_name}] 검증 정확도: {valid_acc[-1]*100:6.2f}% | {acc_arrow}")
        print("-" * 60)
        
        # Early Stopping 및 최고 성능 모델 저장
        if valid_loss[-1] < min_loss:
            min_loss = valid_loss[-1]
            best_epoch = epoch
            no_improve_count = 0
            best_model_path = os.path.join("weight", f"best_genconvit_{model_type}.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "min_loss": min_loss,
                "valid_acc": valid_acc[-1],
                "model_type": model_type
            }, best_model_path)
            print(f"[{model_name}] ★ 새로운 최고 성능! 모델 저장: {best_model_path}")
        else:
            no_improve_count += 1
            print(f"[{model_name}] ⚠️  {no_improve_count}/{patience} 에포크 동안 개선 없음")
            
            # Early Stopping 체크
            if no_improve_count >= patience:
                print(f"\n[{model_name}] 🛑 Early Stopping! {patience} 에포크 동안 개선 없어서 학습 중단")
                print(f"[{model_name}] ★ 최고 성능: 에포크 {best_epoch}, 손실 {min_loss:.4f}")
                break
    
    time_elapsed = time.time() - since
    
    print(f"\n{'='*60}")
    print(f"✓ {model_name} 학습 완료!")
    print(f"{'='*60}")
    print(f"총 소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초")
    print(f"최종 검증 정확도: {valid_acc[-1]*100:.2f}%")
    print(f"최종 검증 손실: {valid_loss[-1]:.4f}")
    print(f"최고 성능: 에포크 {best_epoch}, 손실 {min_loss:.4f}")
    print(f"{'='*60}")
    
    # 모델 저장
    file_path = os.path.join(
        "weight",
        f'genconvit_{model_type}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )
    
    # 학습 히스토리 저장
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)
    
    # 최종 모델 저장
    state = {
        "epoch": num_epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": valid_loss[-1],
        "final_acc": valid_acc[-1],
        "config": config,
        "model_type": model_type
    }
    
    weight = f"{file_path}.pth"
    torch.save(state, weight)
    
    print(f"[{model_name}] 모델 저장 완료: {weight}")
    print("="*60)
    
    return {
        "model_type": model_type,
        "model_name": model_name,
        "weight_path": weight,
        "final_acc": valid_acc[-1],
        "final_loss": valid_loss[-1],
        "best_epoch": best_epoch,
        "min_loss": min_loss,
        "training_time": time_elapsed
    }


def train_parallel_models(data_path, num_epochs, batch_size, pretrained_ed=None, pretrained_vae=None):
    """두 모델(ED, VAE)을 병렬로 학습"""
    print(f"\n{'='*80}")
    print("🚀 GenConViT 병렬 학습 시작!")
    print(f"{'='*80}")
    print(f"데이터 경로: {data_path}")
    print(f"에포크 수: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"사전 훈련 ED 모델: {pretrained_ed}")
    print(f"사전 훈련 VAE 모델: {pretrained_vae}")
    print(f"{'='*80}")
    
    start_time = perf_counter()
    
    # 데이터 로딩
    print("데이터 로딩 중...")
    dataloaders, dataset_sizes = load_data(data_path, batch_size)
    print("데이터 로딩 완료!")
    
    print(f"데이터셋 크기:")
    for split, size in dataset_sizes.items():
        print(f"  {split}: {size:,}개")
    
    # 병렬 학습을 위한 설정
    model_configs = [
        ("ed", dataloaders, num_epochs, pretrained_ed, batch_size, "ED (Autoencoder)", 0),
        ("vae", dataloaders, num_epochs, pretrained_vae, batch_size, "VAE (Variational Autoencoder)", 1)
    ]
    
    # 병렬 학습 실행
    print(f"\n🔄 두 모델을 병렬로 학습 시작...")
    
    # ThreadPoolExecutor를 사용하여 병렬 실행 (GPU 메모리 공유를 위해)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(train_single_model_parallel, config) for config in model_configs]
        results = [future.result() for future in futures]
    
    end_time = perf_counter()
    
    # 결과 종합
    print(f"\n{'='*80}")
    print("🎉 병렬 학습 완료!")
    print(f"{'='*80}")
    print(f"전체 실행 시간: {(end_time - start_time) / 60:.1f}분")
    
    print(f"\n📊 최종 결과 비교:")
    print(f"{'='*80}")
    print(f"{'모델':<20} {'정확도':<10} {'손실':<10} {'최고 에포크':<12} {'학습시간':<10}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['model_name']:<20} {result['final_acc']*100:>8.2f}% {result['final_loss']:>8.4f} {result['best_epoch']:>10} {result['training_time']/60:>8.1f}분")
    
    # 최고 성능 모델 찾기
    best_model = max(results, key=lambda x: x['final_acc'])
    print(f"\n🏆 최고 성능 모델: {best_model['model_name']}")
    print(f"   정확도: {best_model['final_acc']*100:.2f}%")
    print(f"   손실: {best_model['final_loss']:.4f}")
    print(f"   모델 파일: {best_model['weight_path']}")
    print(f"{'='*80}")
    
    return results


def load_pretrained(model, optimizer, pretrained_model_filename):
    """사전 훈련된 모델 로드"""
    assert os.path.isfile(
        pretrained_model_filename
    ), "Saved model file does not exist. Exiting."

    model, optimizer, start_epoch, min_loss = load_checkpoint(
        model, optimizer, filename=pretrained_model_filename
    )
    # optimizer 상태를 GPU로 이동
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model, optimizer, start_epoch, min_loss


def train_simple_model(
    model, mod, dataloaders, num_epochs, pretrained_model_filename, batch_size, model_name
):
    """WandB 없이 간단한 모델 학습"""
    print(f"\n{'='*60}")
    print(f"🤖 {model_name} 모델 학습 시작")
    print(f"{'='*60}")

    # 파인튜닝을 위한 낮은 학습률 설정
    if pretrained_model_filename:
        learning_rate = float(config["learning_rate"]) * 0.1
        print(f"🔧 파인튜닝 모드: 학습률 {learning_rate:.6f} (기본값의 10%)")
    else:
        learning_rate = float(config["learning_rate"])
        print(f"🔧 처음부터 학습: 학습률 {learning_rate:.6f}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    
    # Gradient Clipping 추가 (overfitting 방지) - 이 부분은 train_ed/vae.py로 이동
    # max_grad_norm = 1.0
    # 클래스 불균형 해결을 위한 가중치 설정 (Real:Fake ≈ 1:5.27)
    weights = torch.tensor([5.27, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion.to(device)
    mse = nn.MSELoss()

    # 검증 손실 기반으로 학습률 동적 조정 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    if pretrained_model_filename:
        print(f"🔄 사전 훈련된 모델 로딩: {pretrained_model_filename}")
        model, optimizer, start_epoch, min_loss = load_pretrained(
            model, optimizer, pretrained_model_filename
        )
    else:
        start_epoch = 0
        min_loss = float('inf')

    model.to(device)
    torch.manual_seed(1)
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    since = time.time()
    
    # Early Stopping 설정
    patience = 10  # 검증 손실이 개선되지 않는 최대 에포크 수
    best_epoch = 0
    no_improve_count = 0
    
    print(f"\n학습 설정:")
    print(f"  총 에포크: {num_epochs}")
    print(f"  Early Stopping: {patience} 에포크 동안 개선 없으면 중단")
    print(f"  배치 크기: {batch_size}")
    print(f"  목표: 검증 손실 최소화")
    print("="*60)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # 학습 함수 import
        if mod == "ed":
            from train.train_ed import train, valid
        else:
            from train.train_vae import train, valid
        
        # 학습
        train_loss, train_acc, epoch_train_loss = train(
            model,
            device,
            dataloaders["train"],
            criterion,
            optimizer,
            epoch,
            train_loss,
            train_acc,
            mse,
        )
        
        # 검증
        valid_loss, valid_acc = valid(
            model,
            device,
            dataloaders["valid"],
            criterion,
            epoch,
            valid_loss,
            valid_acc,
            mse,
        )
        
        # 스케줄러 step 호출 (ReduceLROnPlateau는 validation loss를 인자로 받음)
        scheduler.step(valid_loss[-1])
        
        # 시각적 로깅 (깔끔한 버전)
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 진행률 바 계산
        progress = (epoch + 1) / num_epochs
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 성능 개선 표시
        if epoch > 0:
            loss_improvement = valid_loss[-2] - valid_loss[-1] if len(valid_loss) > 1 else 0
            acc_improvement = valid_acc[-1] - valid_acc[-2] if len(valid_acc) > 1 else 0
            
            loss_arrow = "↑" if loss_improvement > 0 else "↓" if loss_improvement < 0 else "→"
            acc_arrow = "↑" if acc_improvement > 0 else "↓" if acc_improvement < 0 else "→"
        else:
            loss_arrow = "NEW"
            acc_arrow = "NEW"
        
        print(f"\nEPOCH {epoch+1:2d}/{num_epochs} | {bar} | {progress*100:5.1f}%")
        print(f"시간: {epoch_time:6.2f}초 | 학습률: {current_lr:.2e}")
        print(f"학습 손실: {epoch_train_loss:8.4f} | {loss_arrow} 검증 손실: {valid_loss[-1]:8.4f}")
        print(f"검증 정확도: {valid_acc[-1]*100:6.2f}% | {acc_arrow} 개선: {acc_improvement*100:+.2f}%" if epoch > 0 else f"검증 정확도: {valid_acc[-1]*100:6.2f}% | {acc_arrow}")
        print("-" * 60)
        
        # Early Stopping 및 최고 성능 모델 저장
        if valid_loss[-1] < min_loss:
            min_loss = valid_loss[-1]
            best_epoch = epoch
            no_improve_count = 0
            best_model_path = os.path.join("weight", f"best_genconvit_{mod}.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "min_loss": min_loss,
                "valid_acc": valid_acc[-1],
                "model_type": mod
            }, best_model_path)
            print(f"★ 새로운 최고 성능! 모델 저장: {best_model_path}")
        else:
            no_improve_count += 1
            print(f"⚠️  {no_improve_count}/{patience} 에포크 동안 개선 없음")
            
            # Early Stopping 체크
            if no_improve_count >= patience:
                print(f"\n🛑 Early Stopping! {patience} 에포크 동안 개선 없어서 학습 중단")
                print(f"★ 최고 성능: 에포크 {best_epoch}, 손실 {min_loss:.4f}")
                break

    time_elapsed = time.time() - since

    print(f"\n{'='*60}")
    print(f"✓ {model_name} 학습 완료!")
    print(f"{'='*60}")
    print(f"총 소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초")
    print(f"최종 검증 정확도: {valid_acc[-1]*100:.2f}%")
    print(f"최종 검증 손실: {valid_loss[-1]:.4f}")
    print(f"최고 성능: 에포크 {best_epoch}, 손실 {min_loss:.4f}")
    print(f"{'='*60}")

    # 모델 저장
    file_path = os.path.join(
        "weight",
        f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )

    # 학습 히스토리 저장
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)

    # 최종 모델 저장
    state = {
        "epoch": num_epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": valid_loss[-1],
        "final_acc": valid_acc[-1],
        "config": config,
        "model_type": mod
    }

    weight = f"{file_path}.pth"
    torch.save(state, weight)

    print(f"모델 저장 완료: {weight}")
    print("="*60)
    
    return weight, valid_acc[-1], valid_loss[-1]


def main():
    parser = argparse.ArgumentParser(description="GenConViT 학습 (단일/병렬 모드 지원)")
    parser.add_argument("-d", "--data", required=True, help="학습 데이터 경로")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="학습 에포크 수")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="배치 크기 (기본값: 32)")
    parser.add_argument("-m", "--model", choices=["ed", "vae", "both"], help="모델 타입 (both: 병렬 학습)")
    parser.add_argument("-p", "--pretrained", help="사전 훈련된 모델 파일 (단일 모델용)")
    parser.add_argument("--pretrained-ed", help="사전 훈련된 ED 모델 파일 (병렬 학습용)")
    parser.add_argument("--pretrained-vae", help="사전 훈련된 VAE 모델 파일 (병렬 학습용)")
    parser.add_argument("--parallel", action="store_true", help="병렬 학습 모드 강제 활성화")
    
    args = parser.parse_args()
    
    # 병렬 학습 모드 결정
    if args.model == "both" or args.parallel:
        # 병렬 학습 모드
        print(f"\n{'='*80}")
        print("🚀 GenConViT 병렬 학습 모드!")
        print(f"{'='*80}")
        
        # 병렬 학습 실행
        results = train_parallel_models(
            data_path=args.data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            pretrained_ed=args.pretrained_ed,
            pretrained_vae=args.pretrained_vae
        )
        
    else:
        # 단일 모델 학습 모드
        if not args.model:
            print("❌ 오류: 단일 모델 학습 시 --model 옵션을 지정해주세요 (ed 또는 vae)")
            return
            
        print(f"\n{'='*60}")
        print("GenConViT 단일 모델 학습!")
        print(f"{'='*60}")
        print(f"데이터 경로: {args.data}")
        print(f"에포크 수: {args.epochs}")
        print(f"배치 크기: {args.batch_size}")
        print(f"모델 타입: {args.model}")
        if args.pretrained:
            print(f"사전 훈련 모델: {args.pretrained}")
        print(f"{'='*60}")
        
        start_time = perf_counter()
        
        # 데이터 로딩
        print("데이터 로딩 중...")
        dataloaders, dataset_sizes = load_data(args.data, args.batch_size)
        print("데이터 로딩 완료!")
        
        print(f"데이터셋 크기:")
        for split, size in dataset_sizes.items():
            print(f"  {split}: {size:,}개")

        # 모델 생성
        if args.model == "ed":
            model = GenConViTED(config)
            model_name = "ED (Autoencoder)"
        else:
            model = GenConViTVAE(config)
            model_name = "VAE (Variational Autoencoder)"
        
        # 학습 실행
        weight, acc, loss = train_simple_model(
            model, args.model, dataloaders, args.epochs, args.pretrained, args.batch_size, model_name
        )
        
        end_time = perf_counter()
        
        print(f"\n{'='*60}")
        print("전체 학습 완료!")
        print(f"{'='*60}")
        print(f"전체 실행 시간: {(end_time - start_time) / 60:.1f}분")
        print(f"최종 결과: 정확도 {acc*100:.2f}%, 손실 {loss:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # 사용법 예제 출력
        main()
