import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from model.config import load_config
from model.genconvit import GenConViT
from decord import VideoReader, cpu
import glob
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import imageio
# GPU 설정 개선 - RTX 5090 최적화 + CUDA 호환성 체크
def get_device():
    if torch.cuda.is_available():
        try:
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            
            # CUDA 호환성 체크
            print("🔍 CUDA 호환성 체크 중...")
            print(f"PyTorch CUDA: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")
            
            # 간단한 CUDA 연산 테스트
            test_tensor = torch.randn(1, 3, 224, 224).cuda()
            test_result = torch.nn.functional.relu(test_tensor)
            print("✅ CUDA 연산 테스트 성공!")
            del test_tensor, test_result
            torch.cuda.empty_cache()
            
            # RTX 5090 최적화 설정
            if "RTX 5090" in torch.cuda.get_device_name(0):
                print("🚀 RTX 5090 감지! 최적화 설정 적용 중...")
                # CUDA 그래프 최적화 활성화
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                # 메모리 할당 최적화
                torch.cuda.set_per_process_memory_fraction(0.95)  # GPU 메모리의 95% 사용
            
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # GPU 성능 정보 출력
            props = torch.cuda.get_device_properties(0)
            print(f"Compute Capability: {props.major}.{props.minor}")
            print(f"Multi-Processor Count: {props.multi_processor_count}")
            
        except Exception as e:
            print(f"❌ CUDA 초기화 실패: {e}")
            print("🔄 CPU 모드로 전환합니다.")
            device = torch.device("cpu")
            
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_device()


def load_genconvit(config, net, ed_weight, vae_weight, fp16):    
    try:
        model = GenConViT(
            config,
            ed= ed_weight,
            vae= vae_weight, 
            net=net,
            fp16=fp16
        )

        # 전역 device 변수 사용
        global device
        
        # CUDA 사용 가능 여부 확인 후 모델 이동
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                model.to(device)
                print("✅ 모델을 GPU로 이동 성공!")
            except Exception as e:
                print(f"❌ GPU 이동 실패: {e}")
                print("🔄 CPU 모드로 전환합니다.")
                device = torch.device("cpu")
                model.to(device)
        else:
            model.to(device)
        
        model.eval()
        
        # RTX 5090 최적화: 메모리 사용량 모니터링
        if torch.cuda.is_available() and device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            print(f"🚀 Model loaded on RTX 5090!")
            print(f"📊 Memory Usage:")
            print(f"   - Allocated: {allocated:.1f} MB")
            print(f"   - Reserved: {reserved:.1f} MB")
            print(f"   - Total GPU: {total:.1f} MB")
            print(f"   - Usage: {allocated/total*100:.1f}%")
            
            # RTX 5090에서 fp16 사용 권장
            if fp16 and "RTX 5090" in torch.cuda.get_device_name(0):
                print("⚡ RTX 5090에서 fp16 모드 활성화 - 성능 향상!")
        else:
            print(f"🖥️ Model loaded on {device}")
        
        if fp16 and device.type == "cuda":
            model.half()

        return model
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        raise e


# MediaPipe으로 대체
def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    
    # MediaPipe 얼굴 검출 초기화
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    ) as face_detection:
        
        for _, frame in tqdm(enumerate(frames), total=len(frames)):
            # MediaPipe로 얼굴 검출
            results = face_detection.process(frame)
            
            if results.detections:
                for detection in results.detections:
                    if count < len(frames):
                        # 얼굴 영역 추출
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                            int(bbox.width * w), int(bbox.height * h)
                        
                        face_image = frame[y:y+height, x:x+width]
                        face_image = cv2.resize(
                            face_image, (224, 224), interpolation=cv2.INTER_AREA
                        )
                        
                        temp_face[count] = face_image
                        count += 1
                    else:
                        break

    return ([], 0) if count == 0 else (temp_face[:count], count)

def face_rec_with_original(frames, p=None, klass=None):
    """원본 프레임과 얼굴 바운딩 박스 정보를 함께 반환하는 함수"""
    # 요청된 프레임 수만큼만 정확히 할당 (이전 데이터 간섭 방지)
    max_frames = len(frames)
    temp_face = np.zeros((max_frames, 224, 224, 3), dtype=np.uint8)
    original_frames = []  # 원본 프레임 저장 (새로 초기화)
    face_bboxes = []  # 얼굴 바운딩 박스 저장 (새로 초기화)
    count = 0
    
    # MediaPipe 얼굴 검출 초기화
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    ) as face_detection:
        
        for frame_idx, frame in tqdm(enumerate(frames), total=len(frames)):
            # 최대 프레임 수 제한 (명확한 경계)
            if count >= max_frames:
                print(f"📏 최대 프레임 수 ({max_frames})에 도달하여 처리 중단")
                break
                
            # MediaPipe로 얼굴 검출
            results = face_detection.process(frame)
            
            if results.detections:
                for detection in results.detections:
                    # 이중 체크: count와 max_frames 모두 확인
                    if count < max_frames and count < len(temp_face):
                        # 얼굴 영역 추출
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                            int(bbox.width * w), int(bbox.height * h)
                        
                        # 모델 입력용: 224x224로 리사이즈
                        face_image = frame[y:y+height, x:x+width]
                        face_image_resized = cv2.resize(
                            face_image, (224, 224), interpolation=cv2.INTER_AREA
                        )
                        
                        temp_face[count] = face_image_resized
                        
                        # 원본 프레임과 얼굴 바운딩 박스 저장
                        original_frames.append(frame.copy())  # 명시적 복사
                        face_bboxes.append((x, y, width, height))
                        # print(f"프레임 {count+1}: 얼굴 검출 - 바운딩 박스: ({x}, {y}, {width}, {height})")
                        count += 1
                        break  # 첫 번째 얼굴만 사용
                    else:
                        break

    return ([], 0, [], []) if count == 0 else (temp_face[:count], count, original_frames, face_bboxes)

def preprocess_frame_with_original(frame, original_frames, face_bboxes):
    """원본 프레임 정보를 포함한 전처리 함수"""
    # 모델 입력용 텐서 (224x224)
    df_tensor = torch.tensor(frame).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)
    
    # GPU로 이동
    df_tensor = df_tensor.to(device)
    
    # 원본 프레임과 바운딩 박스 정보도 함께 반환
    return {
        'tensor': df_tensor,
        'original_frames': original_frames,
        'face_bboxes': face_bboxes
    }

# GradCAM hook용 저장 공간
gradcam_features = []
gradcam_gradients = []

def gradcam_forward_hook(module, input, output):
    """GradCAM을 위한 forward hook"""
    gradcam_features.append(output)

def gradcam_backward_hook(module, grad_input, grad_output):
    """GradCAM을 위한 backward hook"""
    gradcam_gradients.append(grad_output[0])

def generate_gradcam_full_frame(model, original_frame, target_class=None):
    """전체 프레임 기반 GradCAM 생성 - 올바른 접근법"""
    try:
        # 전체 프레임을 모델 입력 크기로 리사이즈
        frame_resized = cv2.resize(original_frame, (224, 224), interpolation=cv2.INTER_AREA)
        
        # 텐서로 변환 및 정규화
        frame_tensor = torch.tensor(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = normalize_data()["vid"](frame_tensor / 255.0)
        frame_tensor = frame_tensor.to(device)
        
        # print(f"GradCAM: 전체 프레임 입력 형태: {frame_tensor.shape}")
        
        # 기존 GradCAM 함수 호출
        return generate_gradcam_with_hooks_improved(model, frame_tensor, target_class)
        
    except Exception as e:
        print(f"전체 프레임 GradCAM 생성 중 오류: {e}")
        return None

def generate_gradcam_with_hooks_improved(model, input_tensor, target_class=None):
    """개선된 Hook 기반 GradCAM 생성 함수 - 로컬 변수 사용"""
    try:
        # GradCAM을 위해 모델을 train mode로 설정 (gradient 계산 활성화)
        model.train()
        
        # 로컬 변수로 특징맵과 gradient 저장 (프레임 간 간섭 방지)
        local_features = []
        local_gradients = []
        
        # 로컬 Hook 함수 정의
        def local_forward_hook(module, input, output):
            local_features.append(output)
        
        def local_backward_hook(module, grad_input, grad_output):
            local_gradients.append(grad_output[0])
        
        # 적절한 Target Layer 찾기
        target_layer, layer_name = find_optimal_target_layer_for_gradcam(model)
        
        if target_layer is None:
            print("GradCAM: 적절한 Target Layer를 찾을 수 없음")
            return None
        
        # print(f"GradCAM: {layer_name} 레이어에 Hook 등록")
        
        # Target layer가 gradient를 계산하도록 설정
        for param in target_layer.parameters():
            param.requires_grad_(True)
        
        # Hook 등록 (로컬 함수 사용)
        forward_handle = target_layer.register_forward_hook(local_forward_hook)
        backward_handle = target_layer.register_backward_hook(local_backward_hook)
        
        try:
            # Gradient 계산을 위한 설정
            input_tensor.requires_grad_(True)
            
            # 모델의 모든 파라미터가 gradient를 계산하도록 설정
            for param in model.parameters():
                param.requires_grad_(True)
            
            # 이전 gradient 완전 제거
            model.zero_grad()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 모델 예측
            output = model(input_tensor)
            
            # GenConViT 모델 출력 처리 (튜플 형태 고려)
            if isinstance(output, tuple):
                output = output[0]  # 분류 결과만 사용
                # print(f"GradCAM: GenConViT 튜플 출력 감지, 분류 결과 사용")
            
            # print(f"GradCAM: 모델 출력 형태 = {output.shape}")
            # print(f"GradCAM: 모델 출력 값 = {output.detach().cpu().numpy()}")
            
            # 배치 처리: 첫 번째 프레임만 사용
            if output.size(0) > 1:
                # print(f"GradCAM: 배치 크기 {output.size(0)} 감지, 첫 번째 프레임만 사용")
                output = output[0:1]  # 첫 번째 프레임만 유지
                # print(f"GradCAM: 수정된 출력 형태 = {output.shape}")
                # print(f"GradCAM: 수정된 출력 값 = {output.detach().cpu().numpy()}")
            
            # 타겟 클래스 결정
            if target_class is None:
                target_class = torch.argmax(output, dim=1)
            
            # print(f"GradCAM: 타겟 클래스 = {target_class.item()}")
            
            # Gradient 계산을 위해 입력 텐서의 gradient 활성화 확인
            # print(f"GradCAM: 입력 텐서 requires_grad = {input_tensor.requires_grad}")
            
            # 타겟 클래스에 대한 gradient 계산
            model.zero_grad()
            target_output = output[0, target_class.item()]
            # print(f"GradCAM: 타겟 출력 값 = {target_output.item()}")
            # print(f"GradCAM: 타겟 출력 requires_grad = {target_output.requires_grad}")
            
            # Gradient 계산
            target_output.backward(retain_graph=True)
            
            # Hook을 통한 특징 맵과 gradient 추출
            if len(local_features) == 0 or len(local_gradients) == 0:
                # print("GradCAM: Hook을 통한 특징 맵 또는 gradient 획득 실패")
                return None
            
            fmap = local_features[0].detach()
            grads = local_gradients[0].detach()
            
            # print(f"GradCAM: 특징 맵 형태 = {fmap.shape}")
            # print(f"GradCAM: gradient 형태 = {grads.shape}")
            
            # 표준 Grad-CAM 계산 로직
            # 가중치 계산: gradient의 공간 차원에 대한 평균
            weights = grads.mean(dim=(2, 3), keepdim=True)
            # print(f"GradCAM: 가중치 형태 = {weights.shape}")
            # print(f"GradCAM: 가중치 범위 = [{weights.min():.6f}, {weights.max():.6f}]")
            
            # CAM 생성: 가중치와 특징 맵의 곱의 합
            cam = (weights * fmap).sum(dim=1, keepdim=True)
            # print(f"GradCAM: CAM 생성 후 형태 = {cam.shape}")
            # print(f"GradCAM: CAM 원본 범위 = [{cam.min():.6f}, {cam.max():.6f}]")
            
            # ReLU 적용
            cam = torch.relu(cam)
            # print(f"GradCAM: ReLU 후 범위 = [{cam.min():.6f}, {cam.max():.6f}]")
            
            # 원본 크기로 보간 (bilinear)
            cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            # print(f"GradCAM: 보간 후 형태 = {cam.shape}")
            
            # 정규화 개선 - 더 강한 대비
            cam = np.maximum(cam, 0)
            cam_min, cam_max = cam.min(), cam.max()
            # print(f"GradCAM: 정규화 전 범위 = [{cam_min:.6f}, {cam_max:.6f}]")
            
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
                # 대비 강화를 위한 감마 조정
                cam = np.power(cam, 0.7)  # 감마 < 1로 밝은 영역 강조
            else:
                # print("GradCAM: 경고 - 모든 값이 동일함!")
                cam = np.zeros_like(cam)
            
            # print(f"GradCAM: 최종 CAM 형태 = {cam.shape}, 범위 = [{cam.min():.6f}, {cam.max():.6f}]")
            # print(f"GradCAM: 0이 아닌 픽셀 수 = {np.count_nonzero(cam)}")
            
            return cam
            
        finally:
            # Hook 제거
            forward_handle.remove()
            backward_handle.remove()
            
            # 로컬 변수 명시적 정리
            local_features.clear()
            local_gradients.clear()
            
            # Gradient 정리
            model.zero_grad()
            
            # 모델을 다시 eval 모드로 복원
            model.eval()
            
            # GPU 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        print(f"개선된 Hook 기반 GradCAM 생성 중 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return None

def find_optimal_target_layer_for_gradcam(model):
    """GradCAM을 위한 최적 Target Layer 찾기 - Backbone 네트워크 우선"""
    try:
        # print("GradCAM: 최적 Target Layer 탐색 중...")
        
        if hasattr(model, 'model_ed'):
            # print("   ED 모델 감지됨")
            
            # 1. 우선순위: backbone 네트워크 (실제 분류에 사용되는 네트워크)
            if hasattr(model.model_ed, 'backbone'):
                backbone = model.model_ed.backbone
                # print(f"   ✅ Backbone 네트워크 발견: {type(backbone).__name__}")
                
                # Backbone의 특징 추출 레이어 찾기
                if hasattr(backbone, 'stages'):  # ConvNeXt 구조
                    last_stage = backbone.stages[-1]
                    # print(f"   ✅ Backbone 마지막 stage 선택: {last_stage}")
                    return last_stage, "ED Backbone 마지막 Stage"
                
                elif hasattr(backbone, 'features'):  # 일반적인 CNN 구조
                    features = backbone.features
                    conv_layers = []
                    for i, layer in enumerate(features):
                        if isinstance(layer, torch.nn.Conv2d):
                            conv_layers.append((i, layer, layer.out_channels))
                            # print(f"   Backbone Conv2d 레이어 {i}: {layer.out_channels}채널")
                    
                    if conv_layers:
                        best_idx, best_layer, best_channels = conv_layers[-1]
                        # print(f"   ✅ Backbone 최적 레이어 선택: 인덱스 {best_idx}, {best_channels}채널")
                        return best_layer, f"ED Backbone Conv2d-{best_idx} ({best_channels}채널)"
                
                elif hasattr(backbone, 'layer4'):  # ResNet 구조
                    last_layer = backbone.layer4
                    # print(f"   ✅ Backbone layer4 선택: {last_layer}")
                    return last_layer, "ED Backbone Layer4"
            
            # 2. 백업: encoder 네트워크 (기존 방식)
            if hasattr(model.model_ed, 'encoder') and hasattr(model.model_ed.encoder, 'features'):
                features = model.model_ed.encoder.features
                # print(f"   백업: Encoder에서 {len(features)}개 레이어 발견")
                
                # 모든 Conv2d 레이어 출력
                conv_layers = []
                for i, layer in enumerate(features):
                    if isinstance(layer, torch.nn.Conv2d):
                        conv_layers.append((i, layer, layer.out_channels))
                        print(f"   Conv2d 레이어 {i}: {layer.out_channels}채널")
                
                if conv_layers:
                    # 가장 마지막 Conv2d 레이어 사용 (가장 고수준 특징)
                    best_idx, best_layer, best_channels = conv_layers[-1]
                    print(f"   ✅ Encoder 최적 레이어 선택: 인덱스 {best_idx}, {best_channels}채널")
                    return best_layer, f"ED Encoder Conv2d-{best_idx} ({best_channels}채널)"
                
                # Conv2d가 없으면 마지막 레이어
                last_layer = features[-1]
                print(f"   ✅ Encoder 마지막 레이어 사용: {last_layer}")
                return last_layer, "ED Encoder 마지막 레이어"
        
        elif hasattr(model, 'model_vae'):
            print("   VAE 모델 감지됨")
            
            if hasattr(model.model_vae, 'encoder') and hasattr(model.model_vae.encoder, 'features'):
                features = model.model_vae.encoder.features
                print(f"   총 {len(features)}개 레이어 발견")
                
                # 모든 Conv2d 레이어 출력
                conv_layers = []
                for i, layer in enumerate(features):
                    if isinstance(layer, torch.nn.Conv2d):
                        conv_layers.append((i, layer, layer.out_channels))
                        print(f"   Conv2d 레이어 {i}: {layer.out_channels}채널")
                
                if conv_layers:
                    # 가장 마지막 Conv2d 레이어 사용
                    best_idx, best_layer, best_channels = conv_layers[-1]
                    print(f"   ✅ 최적 레이어 선택: 인덱스 {best_idx}, {best_channels}채널")
                    return best_layer, f"VAE Encoder Conv2d-{best_idx} ({best_channels}채널)"
                
                # Conv2d가 없으면 마지막 레이어
                last_layer = features[-1]
                print(f"   ✅ 마지막 레이어 사용: {last_layer}")
                return last_layer, "VAE Encoder 마지막 레이어"
        
        print("   ❌ 적절한 Target Layer를 찾을 수 없음")
        return None, None
        
    except Exception as e:
        print(f"   ❌ Target Layer 탐색 중 오류: {e}")
        import traceback
        print(f"   상세 오류: {traceback.format_exc()}")
        return None, None

def create_gradcam_visualization_improved(original_frame, gradcam, prediction, confidence, face_bbox=None):
    """개선된 GradCAM 시각화 함수 - 전체 프레임에 자연스러운 매핑"""
    try:
        # 원본 프레임을 numpy 배열로 변환
        if isinstance(original_frame, np.ndarray):
            pass  # 이미 numpy 배열
        elif isinstance(original_frame, torch.Tensor):
            original_frame = original_frame.cpu().numpy()
        else:
            print(f"GradCAM: 지원하지 않는 프레임 타입: {type(original_frame)}")
            return None
        
        # 원본 프레임 정규화 (0-255)
        if original_frame.max() <= 1.0:
            original_frame = (original_frame * 255).astype(np.uint8)
        
        # 원본 프레임 크기 확인
        original_h, original_w = original_frame.shape[:2]
        # print(f"GradCAM: 원본 프레임 크기: {original_h}x{original_w}")
        # print(f"GradCAM: 입력 GradCAM 크기: {gradcam.shape}")
        
        # GradCAM 정규화 - 개선된 대비
        cam = np.maximum(gradcam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        # print(f"GradCAM 시각화: 입력 범위 = [{cam_min:.6f}, {cam_max:.6f}]")
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            # 대비 강화 - 감마 조정으로 특징 강조
            cam = np.power(cam, 0.5)  # 감마 < 1로 밝은 영역 더 강조
        
        cam = np.uint8(255 * cam)
        # print(f"GradCAM 시각화: 정규화 후 범위 = [{cam.min()}, {cam.max()}]")
        # print(f"GradCAM 시각화: 0이 아닌 픽셀 수 = {np.count_nonzero(cam)}")
        
        # 전체 프레임으로 GradCAM 확장 (자연스러운 매핑)
        # print(f"GradCAM: 전체 프레임으로 자연스럽게 매핑 - {cam.shape} -> ({original_h}, {original_w})")
        cam_resized = cv2.resize(cam, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        
        # 히트맵 생성 (JET 컬러맵)
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 원본과 히트맵 합성 - 히트맵 강조
        alpha = 0.6  # 히트맵 비중 높임 (원본 40%, 히트맵 60%)
        overlay = (1 - alpha) * original_frame + alpha * heatmap
        overlay = np.uint8(np.clip(overlay, 0, 255))
        # print(f"GradCAM 시각화: 합성 비율 = 원본({1-alpha:.1f}) + 히트맵({alpha:.1f})")
        
        # 예측 결과 텍스트 추가
        prediction_text = f"{prediction}: {confidence:.3f} (Full-Frame)"
        
        # PIL Image로 변환하여 텍스트 추가
        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)
        
        try:
            # Windows 폰트 경로 시도
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            try:
                # Linux 폰트 경로 시도
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
        
        # 텍스트 위치 설정 (좌측 상단)
        text_position = (10, 10)
        text_color = (255, 255, 255)  # 흰색
        
        draw.text(text_position, prediction_text, font=font, fill=text_color)
        
        # numpy 배열로 다시 변환하여 반환
        final_visualization = np.array(overlay_pil)
        
        # print(f"GradCAM: 시각화 생성 완료 - 최종 크기: {final_visualization.shape}")
        return final_visualization
        
    except Exception as e:
        print(f"개선된 GradCAM 시각화 생성 중 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return original_frame

def create_debug_visualization(original_frame, gradcam, face_bbox=None):
    """디버깅을 위한 시각화 - Face bbox와 GradCAM 위치 확인"""
    try:
        debug_frame = original_frame.copy()
        
        # Face bounding box 그리기 (녹색)
        if face_bbox is not None:
            x, y, width, height = face_bbox
            cv2.rectangle(debug_frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            cv2.putText(debug_frame, f"Face: ({x},{y},{width},{height})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # GradCAM의 최대 활성화 지점 찾기
        if gradcam is not None:
            # 원본 크기로 리사이즈
            gradcam_resized = cv2.resize(gradcam, (debug_frame.shape[1], debug_frame.shape[0]))
            max_y, max_x = np.unravel_index(np.argmax(gradcam_resized), gradcam_resized.shape)
            
            # 최대 활성화 지점 표시 (빨간색 원)
            cv2.circle(debug_frame, (max_x, max_y), 10, (255, 0, 0), -1)
            cv2.putText(debug_frame, f"Max: ({max_x},{max_y})", 
                       (max_x + 15, max_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # print(f"디버그: Face bbox = {face_bbox}")
            # print(f"디버그: GradCAM 최대 활성화 지점 = ({max_x}, {max_y})")
            
            # Face 중심점과 GradCAM 최대점 거리 계산
            if face_bbox is not None:
                face_center_x = x + width // 2
                face_center_y = y + height // 2
                distance = np.sqrt((max_x - face_center_x)**2 + (max_y - face_center_y)**2)
                # print(f"디버그: Face 중심 = ({face_center_x}, {face_center_y})")
                # print(f"디버그: 중심점과의 거리 = {distance:.1f} 픽셀")
                
                # 중심점도 표시 (파란색)
                cv2.circle(debug_frame, (face_center_x, face_center_y), 8, (0, 0, 255), -1)
                cv2.putText(debug_frame, "Face Center", 
                           (face_center_x + 15, face_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return debug_frame
        
    except Exception as e:
        print(f"디버그 시각화 생성 중 오류: {e}")
        return original_frame

def create_gradcam_gif(gradcam_dir, output_dir, video_name, fps=2, duration=400):
    """GradCAM 프레임들을 GIF로 변환하는 함수"""
    try:
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # GradCAM 이미지 파일들 찾기
        gradcam_files = sorted([f for f in os.listdir(gradcam_dir) if f.endswith('_gradcam.jpg')])
        
        if len(gradcam_files) == 0:
            print(f"GIF: {gradcam_dir}에서 GradCAM 이미지를 찾을 수 없습니다.")
            return None
        
        # GradCAM GIF 생성
        gradcam_images = []
        target_size = None
        
        # 첫 번째 이미지 크기를 기준으로 설정
        for file in gradcam_files:
            img_path = os.path.join(gradcam_dir, file)
            img = imageio.imread(img_path)
            
            if target_size is None:
                target_size = (img.shape[1], img.shape[0])  # (width, height)
            
            # 모든 이미지를 동일한 크기로 리사이즈
            if img.shape[:2] != target_size[::-1]:  # (height, width) vs (width, height)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            
            gradcam_images.append(img)
        
        gradcam_gif_path = None
        if gradcam_images:
            gradcam_gif_path = os.path.join(output_dir, f"{video_name}_gradcam.gif")
            imageio.mimsave(gradcam_gif_path, gradcam_images, duration=duration)
        
        # 통합 GIF 생성 (GradCAM 두 개를 나란히 배치)
        combined_gif_path = None
        if gradcam_images:
            combined_images = []
            combined_target_size = None
            
            for gradcam_file in gradcam_files:
                gradcam_img = imageio.imread(os.path.join(gradcam_dir, gradcam_file))
                
                # 같은 이미지를 두 번 나란히 배치
                combined = np.hstack([gradcam_img, gradcam_img])
                
                # 통합 이미지도 동일한 크기로 맞추기
                if combined_target_size is None:
                    combined_target_size = (combined.shape[1], combined.shape[0])  # (width, height)
                
                if combined.shape[:2] != combined_target_size[::-1]:
                    combined = cv2.resize(combined, combined_target_size, interpolation=cv2.INTER_CUBIC)
                
                combined_images.append(combined)
            
            combined_gif_path = os.path.join(output_dir, f"{video_name}_combined.gif")
            imageio.mimsave(combined_gif_path, combined_images, duration=duration)
        
        return {
            'gradcam_gif': gradcam_gif_path,
            'combined_gif': combined_gif_path
        }
        
    except Exception as e:
        print(f"GIF 생성 중 오류: {e}")
        print(f"💡 일부 이미지의 크기가 다를 수 있습니다. 이미지 크기 통일을 시도 중...")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return None


def preprocess_frame(frame):
    # CPU에서 전처리 후 GPU로 이동
    df_tensor = torch.tensor(frame).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)
    
    # GPU로 이동
    df_tensor = df_tensor.to(device)
    
    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def pred_vid_frame_by_frame(df, model, max_frames=None):
    """프레임별 로짓을 반환하는 함수"""
    with torch.no_grad():
        # 각 프레임별로 예측 수행
        frame_logits = []
        frame_predictions = []
        
        # 처리할 프레임 수 결정 (명시적 제한)
        total_frames = len(df)
        process_frames = min(total_frames, max_frames) if max_frames else total_frames
        
        for i in range(process_frames):
            try:
                # 단일 프레임을 배치 차원으로 확장
                single_frame = df[i:i+1]
                logit = model(single_frame).squeeze()
                prediction = torch.sigmoid(logit)
                
                # 로짓값 처리 - 완전히 안전한 변환
                if logit.dim() == 0:  # 스칼라 텐서
                    frame_logits.append(float(logit.cpu().item()))
                else:  # 벡터 텐서
                    logit_np = logit.cpu().numpy()
                    if logit_np.size == 1:  # 단일 요소 배열
                        frame_logits.append(float(logit_np.item()))
                    else:  # 다중 요소 배열
                        frame_logits.append([float(x) for x in logit_np.flatten()])
                
                # 예측값 처리 - 완전히 안전한 변환
                if prediction.dim() == 0:  # 스칼라 텐서
                    frame_predictions.append(float(prediction.cpu().item()))
                else:  # 벡터 텐서
                    pred_np = prediction.cpu().numpy()
                    if pred_np.size == 1:  # 단일 요소 배열
                        frame_predictions.append(float(pred_np.item()))
                    else:  # 다중 요소 배열
                        frame_predictions.append([float(x) for x in pred_np.flatten()])
                        
            except Exception as e:
                print(f"프레임 {i+1} 처리 중 오류: {e}")
                frame_logits.append(0.0)
                frame_predictions.append(0.5)
        
        # 전체 프레임에 대한 평균 예측값도 계산
        try:
            all_frames = model(df).squeeze()
            overall_prediction = torch.sigmoid(all_frames)
            overall_result = max_prediction_value(overall_prediction)
        except Exception as e:
            print(f"전체 프레임 처리 중 오류: {e}")
            overall_result = (0, 0.5)
        
        return {
            'frame_logits': frame_logits,
            'frame_predictions': frame_predictions,
            'overall_result': overall_result
        }


def max_prediction_value(y_pred):
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]

def cleanup_gpu_memory():
    """GPU 메모리 정리 - 프레임 간 간섭 방지"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()

# 영상 파일에 대해 프레임을 추출한다.
def extract_frames(video_file, num_frames=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frames = len(vr)

    if num_frames == -1: 
        # if -1, get all frames
        indices = np.arange(total_frames).astype(int)
    else:
        # 요청된 프레임 수만큼 정확히 추출
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = vr.get_batch(indices).asnumpy()
    
    # 요청된 프레임 수와 실제 반환 프레임 수가 일치하는지 확인
    if len(frames) != num_frames and num_frames != -1:
        print(f"⚠️  프레임 수 불일치: 요청 {num_frames}, 실제 {len(frames)}")
    
    return frames[:num_frames] if num_frames != -1 else frames  # 정확한 수만 반환


def df_face_from_folder(vid, num_frames):
    img_list = glob.glob(vid+"/*")
    img = []
    for f in img_list:
        try:
            im = Image.open(f).convert('RGB')
            img.append(np.asarray(im))
        except:
            pass
 
    face, count = face_rec(img[:num_frames])
    return preprocess_frame(face) if count > 0 else []

def df_face_from_image(img_path):
    """단일 이미지에서 얼굴을 추출하는 함수"""
    try:
        # 이미지 로드
        im = Image.open(img_path).convert('RGB')
        img_array = np.asarray(im)
        
        # 얼굴 검출
        face, count = face_rec([img_array])
        
        if count > 0:
            return preprocess_frame(face)
        else:
            print(f"❌ 이미지에서 얼굴을 검출할 수 없습니다: {img_path}")
            return []
    except Exception as e:
        print(f"❌ 이미지 처리 중 오류 발생: {e}")
        return []

def df_face_from_image_with_original(img_path):
    """단일 이미지에서 얼굴을 추출하는 함수 (원본 정보 포함)"""
    try:
        # 이미지 로드
        im = Image.open(img_path).convert('RGB')
        img_array = np.asarray(im)
        
        # 얼굴 검출 (원본 정보 포함)
        face, count, original_frames, face_bboxes = face_rec_with_original([img_array])
        
        if count > 0:
            processed_tensor = preprocess_frame(face)
            return {
                'tensor': processed_tensor,
                'original_frames': original_frames,
                'face_bboxes': face_bboxes
            }
        else:
            print(f"❌ 이미지에서 얼굴을 검출할 수 없습니다: {img_path}")
            return None
    except Exception as e:
        print(f"❌ 이미지 처리 중 오류 발생: {e}")
        return None

def df_face(vid, num_frames):
    img = extract_frames(vid, num_frames)
    face, count = face_rec(img)
    return preprocess_frame(face) if count > 0 else []

def df_face_with_original(vid, num_frames):
    """원본 프레임 정보를 포함한 얼굴 추출 함수"""
    print(f"🎬 비디오에서 {num_frames}개 프레임 추출 시작...")
    img = extract_frames(vid, num_frames)
    print(f"📹 추출된 프레임 수: {len(img)}")
    
    face, count, original_frames, face_bboxes = face_rec_with_original(img)
    print(f"👤 얼굴 검출된 프레임 수: {count}")
    
    if count > 0:
        # 요청된 프레임 수와 실제 검출된 프레임 수 확인
        if count != num_frames:
            print(f"⚠️  프레임 수 불일치: 요청 {num_frames}, 검출 {count}")
        
        return preprocess_frame_with_original(face, original_frames, face_bboxes)
    else:
        print("❌ 얼굴이 검출된 프레임이 없습니다.")
        return []


def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )

def is_image(img_path):
    """이미지 파일인지 확인하는 함수"""
    return os.path.isfile(img_path) and img_path.lower().endswith(
        tuple([".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"])
    )

def is_video_folder(vid_folder):
    img_list = glob.glob(vid_folder+"/*")
    return len(img_list)>=1 and img_list[0].endswith(tuple(["png", "jpeg","jpg"]))


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result

print("pred_func.py 동작 성공")