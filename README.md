모델 상태
✅ 이미 훈련 완료: genconvit_vae_inference.pth 가중치 파일 사용
✅ 추론 모드: 새로운 데이터로 예측만 수행
✅ 가중치 고정: 더 이상 학습할 필요 없음

sample_1.mp4 (입력 비디오)
    ↓
extract_frames() → 10개 프레임 추출
    ↓
face_rec() → MediaPipe로 각 프레임에서 얼굴 검출
    ↓
preprocess_frame() → 224x224 크기로 정규화
    ↓
GenConViT 모델 → 프레임별 로짓 예측
    ↓
결과 출력: 각 프레임의 FAKE/REAL 확률


2025/08/20
현재 dlib는 의존성 문제가 심해서 얼굴 검출 로직을 
mediapipe으로 교체!!

py310 환경에서 실행할 수 있도록.
"conda activate py310"

###########################################################################################################
2025/08/27
## 🆕 새로운 기능: 프레임별 로짓 분석

### 단일 비디오 프레임별 분석

#### **영상 분석**
```bash
python prediction.py --p sample_prediction_data/sample_1.mp4 --f 10
```

#### **영상 분석 + GradCAM 시각화**
```bash
python prediction.py --p sample_prediction_data/sample_1.mp4 --f 10 --gradcam
```

### 🖼️ 단일 이미지 분석 (NEW!)

#### **이미지 분석**
```bash
python prediction.py --p sample_prediction_data/image.jpg
```

#### **이미지 + GradCAM 시각화**
```bash
python prediction.py --p sample_prediction_data/image.jpg --gradcam
```

#### 폴더 전체 처리 (배치 처리)
```bash
python prediction.py --p sample_prediction_data --v --f 10
```

**옵션 설명:**
- `--p file_path`: 분석할 비디오/이미지 파일 경로 또는 폴더 경로
  - **파일 경로**를 주면: 자동으로 단일 파일 분석 모드
  - **폴더 경로**를 주면: 자동으로 배치 처리 모드 (폴더 내 모든 파일 분석)
- `--f 10`: 추출할 프레임 수 (비디오만, 기본값: 15)
- `--gradcam`: GradCAM 시각화 활성화 (단일 파일 분석 시에만 사용 가능)
- **지원 형식**: `.mp4`, `.avi`, `.mov`, `.jpg`, `.jpeg`, `.png`


### GradCAM 시각화 결과
- **비디오**: 정확도 0.8 이상인 프레임에만 GradCAM 히트맵 생성
- **이미지**: 모든 이미지에 GradCAM 히트맵 생성
- **저장 위치**: `result/gradcam_outputs/` 폴더
- **파일 형식**: 
  - 비디오: `frame_XX_gradcam.jpg` (XX = 프레임 번호)
  - 이미지: `image_gradcam.jpg`
- **시각화**: 모델이 집중한 영역을 빨간색 히트맵으로 표시


###########################################################################################################
2025/10/26
## 🎯 모델 정밀도 평가 (NEW!)

### **전체 데이터셋 평가**
```bash
# 기본 폴더 : sample_prediction_data
python prediction.py --evaluate

# 대상 폴더 데이터로 평가 (경로를 바로 지정)
python prediction.py --evaluate my_data_folder
```

**옵션 설명:**
- `--evaluate`: sample_prediction_data의 real/fake 이미지 100장씩으로 모델 정밀도 평가
- `--e`: ED 모델만 사용하여 평가
- `--v`: VAE 모델만 사용하여 평가
- `--fp16`: 반정밀도 사용


### **특정 모델로 평가**
```bash
# ED 모델만 평가
python prediction.py --evaluate --e

# VAE 모델만 평가  
python prediction.py --evaluate --v

# 반정밀도로 평가
python prediction.py --evaluate --fp16
```

### **학습 가중치 성능 평가**
```bash
# ED 모델의 특정 가중치로 평가
python prediction.py --evaluate sample_prediction_data_diffusion --e "genconvit_ed_Oct_28_2025_02_06_15"

# VAE 모델의 특정 가중치로 평가
python prediction.py --evaluate sample_prediction_data_diffusion --v "genconvit_vae_Oct_28_2025_02_08_15"
```

**핵심 옵션:**
- `--evaluate [폴더 경로]`: 평가 모드 활성화, 폴더 경로를 지정하면 해당 폴더를 평가 (경로 생략 시 sample_prediction_data 기본값 사용)
- `--e [파일 경로]`: ED 모델을 사용하며, 특정 가중치 파일명을 지정합니다. (경로 생략 시 기본 추론 가중치 사용)
- `--v [파일 경로]`: VAE 모델을 사용하며, 특정 가중치 파일명을 지정합니다. (경로 생략 시 기본 추론 가중치 사용)

###########################################################################################################
2025/10/27

### **가중치의 학습 로그 확인하기**
```bash
python plot_training_history.py -d weight --no-show # 전체 로그
python plot_training_history.py -d weight -m ed --no-show # ed 모델
python plot_training_history.py -d weight -m vae --no-show # vae 모델
```