# GenConViT GradCAM Tool

이 도구는 GenConViT 모델이 딥페이크 탐지 시 어떤 이미지 영역에 집중하는지 시각적으로 확인할 수 있게 해주는 GradCAM 구현체입니다.

## 🚀 주요 기능

- **비디오 프레임별 GradCAM 생성**: 비디오의 각 프레임에서 모델의 주의 영역 시각화
- **단일 이미지 GradCAM**: 개별 이미지에 대한 모델의 집중 영역 분석
- **예측 결과 표시**: FAKE/REAL 분류 결과와 신뢰도 점수 함께 표시
- **Feature Map PCA 시각화**: 모델의 특징 맵 분석 (선택사항)

## 📋 요구사항

- Python 3.7+
- PyTorch
- OpenCV
- PIL (Pillow)
- matplotlib
- scikit-learn
- GenConViT 모델 가중치 파일들

## 🔧 설치 및 설정

1. **의존성 설치**:
```bash
pip install torch torchvision opencv-python pillow matplotlib scikit-learn
```

2. **모델 가중치 확인**:
```
weight/
├── genconvit_ed_inference.pth
└── genconvit_vae_inference.pth
```

## 📖 사용법

### 1. 기본 실행

```python
from gradcam import process_video, process_single_image

# 비디오 처리 (30프레임마다)
process_video('path/to/video.mp4', frame_interval=30)

# 단일 이미지 처리
process_single_image('path/to/image.jpg')
```

### 2. 직접 실행

```bash
# GradCAM 도구 상태 확인
python gradcam/gradcam.py
```

### 3. 함수별 사용법

#### `process_video(video_path, frame_interval=15, output_dir="gradcam_outputs")`
- `video_path`: 처리할 비디오 파일 경로
- `frame_interval`: 처리할 프레임 간격 (기본값: 15)
- `output_dir`: 결과 저장 폴더

#### `process_single_image(image_path, output_dir="gradcam_outputs")`
- `image_path`: 처리할 이미지 파일 경로
- `output_dir`: 결과 저장 폴더

## 📁 출력 결과

### 저장 위치
- `gradcam_outputs/` 폴더에 결과 저장

### 파일 형식
- `frame_{번호}_gradcam.jpg`: 비디오 프레임별 GradCAM 결과
- `{이미지명}_gradcam.jpg`: 단일 이미지 GradCAM 결과

### 결과 내용
- 원본 이미지 + GradCAM 히트맵 오버레이
- 예측 결과 (FAKE/REAL)와 신뢰도 점수
- 모델이 집중한 영역을 빨간색으로 표시

## ⚙️ 고급 설정

### 1. 프레임 간격 조정
```python
# 더 빠른 처리 (적은 프레임)
process_video('video.mp4', frame_interval=60)

# 더 상세한 분석 (많은 프레임)
process_video('video.mp4', frame_interval=5)
```

### 2. 출력 폴더 변경
```python
process_video('video.mp4', output_dir='my_custom_outputs')
```

### 3. Feature Map PCA 활성화
```python
# gradcam.py에서 주석 해제
fmap_path = os.path.join(output_dir, f"frame_{frame_idx}_fmap.jpg")
visualize_featuremap_pca(fmap_tensor, pil_img, fmap_path)
```

## 🔍 결과 해석

### GradCAM 히트맵
- **빨간색 영역**: 모델이 가장 집중하는 부분
- **파란색 영역**: 모델이 덜 주의하는 부분
- **중간 색상**: 중간 정도의 주의도

### 예측 결과
- **FAKE**: 딥페이크로 판별 (신뢰도 점수와 함께)
- **REAL**: 진짜 영상으로 판별 (신뢰도 점수와 함께)

## 🚨 주의사항

1. **모델 가중치**: 반드시 사전 훈련된 가중치 파일이 필요
2. **메모리 사용량**: 긴 비디오 처리 시 충분한 RAM 필요
3. **GPU 권장**: CPU보다 GPU 사용 시 훨씬 빠른 처리 가능
4. **파일 형식**: MP4, AVI, MOV 등 일반적인 비디오 형식 지원

## 🐛 문제 해결

### 모델 로딩 실패
- `weight/` 폴더에 가중치 파일이 있는지 확인
- PyTorch 버전 호환성 확인

### 메모리 부족
- `frame_interval` 값을 늘려서 처리할 프레임 수 줄이기
- 더 작은 해상도의 비디오 사용

### 출력 폴더 권한 오류
- `gradcam_outputs/` 폴더 생성 권한 확인
- 다른 출력 경로 지정

## 📞 지원

문제가 발생하거나 개선 사항이 있다면 이슈를 등록해주세요.

---

**GenConViT GradCAM Tool** - 딥페이크 탐지 모델의 의사결정 과정을 투명하게 분석하세요! 🎯
