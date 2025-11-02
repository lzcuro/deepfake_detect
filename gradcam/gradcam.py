import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

# GenConViT 모델 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.pred_func import load_genconvit, df_face, df_face_from_folder
from model.config import load_config

# 설정 로드
config = load_config()

# 저장 폴더 설정 (현재 작업 디렉토리 기준)
save_path = "result/gradcam_outputs"
os.makedirs(save_path, exist_ok=True)

# 전처리 (GenConViT 모델에 맞게 수정)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# GenConViT 모델 로드
def load_model():
    """GenConViT 모델을 로드합니다."""
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        try:
            # CUDA 호환성 테스트
            test_tensor = torch.randn(1, 3, 224, 224).cuda()
            test_result = torch.nn.functional.relu(test_tensor)
            del test_tensor, test_result
            torch.cuda.empty_cache()
            device = "cuda"
            print(f"🖥️ Using device: {device}")
        except Exception as e:
            print(f"⚠️ CUDA test failed: {e}")
            print("🔄 Falling back to CPU mode")
            device = "cpu"
    else:
        device = "cpu"
    
    print(f"🖥️ Using device: {device}")
    
    # 모델 가중치 파일 경로 설정 (확장자 제거 - genconvit.py에서 자동 추가)
    ed_weight = "genconvit_ed_inference"
    vae_weight = "genconvit_vae_inference"
    net = "genconvit"
    fp16 = False
    
    try:
        print("🔧 Loading GenConViT model...")
        print(f"🔍 Loading weights from:")
        print(f"   - ED: {ed_weight} (will be loaded as weight/{ed_weight}.pth)")
        print(f"   - VAE: {vae_weight} (will be loaded as weight/{vae_weight}.pth)")
        
        # 가중치 파일 존재 확인 (확장자 포함)
        ed_weight_path = f"weight/{ed_weight}.pth"
        vae_weight_path = f"weight/{vae_weight}.pth"
        
        if not os.path.exists(ed_weight_path):
            print(f"❌ ED weight file not found: {ed_weight_path}")
            return None, device
        if not os.path.exists(vae_weight_path):
            print(f"❌ VAE weight file not found: {vae_weight_path}")
            return None, device
            
        print(f"✅ Weight files found:")
        print(f"   - ED: {ed_weight_path}")
        print(f"   - VAE: {vae_weight_path}")
            
        model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        return model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, device

# 모델과 디바이스 로드
model, device = load_model()

# GradCAM hook용 저장 공간
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_input, grad_output):
    if grad_output and len(grad_output) > 0:
        gradients.append(grad_output[0])

# GenConViT 모델의 타겟 레이어 설정
def setup_hooks():
    """GenConViT 모델에 hook을 설정합니다."""
    global target_layer
    
    # 모델이 로드되지 않은 경우
    if model is None:
        print("❌ Error: Model not loaded. Cannot setup hooks.")
        return False
    
    # GenConViT 모델의 적절한 레이어를 타겟으로 설정
    target_layer = None
    
    # 1. ED 모델의 중간 특징 맵 레이어 찾기 (GradCAM에 최적)
    if hasattr(model, 'model_ed'):
        weight_layers = []
        for name, module in model.model_ed.named_modules():
            if hasattr(module, 'weight'):
                weight_layers.append((name, module))
        
        if weight_layers:
            target_idx = -2 if len(weight_layers) > 1 else -1
            target_name, target_layer = weight_layers[target_idx]
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
        else:
            target_layer = list(model.model_ed.modules())[-1]
    
    # 2. VAE 모델의 classifier 레이어 찾기
    if target_layer is None and hasattr(model, 'model_vae'):
        if hasattr(model.model_vae, 'classifier'):
            target_layer = model.model_vae.classifier
        elif hasattr(model.model_vae, 'fc'):
            target_layer = model.model_vae.fc
    
    # 3. 기본 모델의 마지막 레이어 사용
    if target_layer is None:
        target_layer = list(model.modules())[-1]
    
    try:
        if target_layer is not None:
            return True
        else:
            return False
    except Exception as e:
        return False

# Hook 설정
if not setup_hooks():
    # 폴백: 여러 레이어에 hook 등록 시도
    try:
        global target_layer
        
        # 1. ED 모델의 중간 레이어에 hook 등록
        if hasattr(model, 'model_ed'):
            ed_modules = list(model.model_ed.modules())
            target_idx = -3 if len(ed_modules) >= 3 else -2
            target_layer = ed_modules[target_idx]
            
            if hasattr(target_layer, 'weight'):
                target_layer.register_forward_hook(forward_hook)
                target_layer.register_backward_hook(backward_hook)
        else:
            # 2. 전체 모델의 중간 레이어에 hook 등록
            all_modules = list(model.modules())
            target_idx = len(all_modules) // 2
            target_layer = all_modules[target_idx]
            
            if hasattr(target_layer, 'weight'):
                target_layer.register_forward_hook(forward_hook)
                target_layer.register_backward_hook(backward_hook)
    except Exception as e:
        pass

# GenConViT 모델용 GradCAM 계산
def compute_gradcam(input_tensor):
    """GenConViT 모델에 맞는 GradCAM을 계산합니다."""
    features.clear()
    gradients.clear()

    try:
        input_tensor.requires_grad_(True)
        input_tensor = input_tensor.to(device)

        # GenConViT 모델의 forward pass 실행
        model_output = model(input_tensor)
        
        # 모델 출력 처리 (GenConViT는 분류 결과를 반환)
        if isinstance(model_output, tuple):
            output = model_output[0]  # 첫 번째 요소가 분류 결과
        else:
            output = model_output
        
        # 클래스 인덱스 결정 (prediction.py와 동일한 방식)
        if output.dim() == 2:
            # [batch_size, num_classes] 형태
            # sigmoid 적용 후 확률이 높은 클래스 선택
            probs = torch.sigmoid(output)
            # 직접 인덱싱으로 안전하게 처리
            fake_prob = probs[0, 0].item()
            real_prob = probs[0, 1].item()
            class_idx = 0 if fake_prob > real_prob else 1
            target_output = output[0, class_idx]
        elif output.dim() == 1:
            # [num_classes] 형태
            probs = torch.sigmoid(output)
            class_idx = 0 if probs[0].item() > probs[1].item() else 1
            target_output = output[class_idx]
        else:
            class_idx = 0
            target_output = output.flatten()[0]
        
        model.zero_grad()
        
        if target_output.dim() > 0:
            target_output = target_output.squeeze()
        
        target_output.backward()
        
        if len(features) == 1 and len(gradients) == 1:
            try:
                fmap = features[0].detach()
                grads = gradients[0].detach()
                
                if fmap.shape == grads.shape:
                    
                    # 표준 Grad-CAM 계산 로직
                    if grads.dim() >= 4 and fmap.dim() >= 4:
                        # 4D 텐서: [batch, channels, height, width]
                        weights = grads.mean(dim=(2, 3), keepdim=True)
                        cam = (weights * fmap).sum(dim=1, keepdim=True)
                    elif grads.dim() == 2 and fmap.dim() == 2:
                        # 2D 텐서: [batch, features] - 1D 특징을 2D로 변환
                        batch_size, num_features = fmap.shape
                        spatial_size = int(np.sqrt(num_features))
                        if spatial_size * spatial_size != num_features:
                            spatial_size = int(np.sqrt(num_features)) + 1
                            target_size = spatial_size * spatial_size
                            fmap_padded = torch.zeros(batch_size, target_size, device=fmap.device, dtype=fmap.dtype)
                            grads_padded = torch.zeros(batch_size, target_size, device=grads.device, dtype=grads.dtype)
                            fmap_padded[:, :num_features] = fmap
                            grads_padded[:, :num_features] = grads
                            fmap = fmap_padded
                            grads = grads_padded
                        
                        fmap_2d = fmap.view(batch_size, 1, spatial_size, spatial_size)
                        grads_2d = grads.view(batch_size, 1, spatial_size, spatial_size)
                        weights = grads_2d.mean(dim=(2, 3), keepdim=True)
                        cam = (weights * fmap_2d).sum(dim=1, keepdim=True)
                    else:
                        weights = grads.mean(dim=tuple(range(2, grads.dim())), keepdim=True)
                        cam = (weights * fmap).sum(dim=1, keepdim=True)
                else:
                    return None, output, None

                cam = F.relu(cam)
                
                # CAM 차원 확인 및 안전한 보간
                if cam.dim() == 4:
                    cam = cam.squeeze(1)
                elif cam.dim() == 2:
                    cam = cam.unsqueeze(0)
                
                if cam.dim() == 3 and cam.shape[1] > 1 and cam.shape[2] > 1:
                    cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
                    cam = cam.squeeze(1)
                else:
                    return None, output, None
                
                cam = cam.squeeze().cpu().numpy()

                cam -= cam.min()
                cam /= (cam.max() + 1e-8)
                
                return cam, output, fmap
            except Exception as e:
                return None, output, None
        else:
            return None, output, None
            
    except Exception as e:
        return None, None, None

# GradCAM 시각화 (예측 결과 텍스트 추가)
def visualize_gradcam(cam, image_pil, save_path, prediction_text):
    """GradCAM 결과를 시각화하고 저장합니다."""
    if cam is None:
        print("⚠️ Warning: CAM is None, skipping visualization")
        return
        
    # 원본 이미지 크기 획득
    W, H = image_pil.size

    # cam 정규화
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = np.uint8(255 * cam)

    # 고해상도 보간
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 이미지와 혼합
    image = np.array(image_pil)
    overlay = 0.5 * image + 0.5 * heatmap
    overlay = np.uint8(np.clip(overlay, 0, 255))

    # PIL Image로 변환하여 텍스트 추가
    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)
    
    try:
        # Windows 환경에 맞는 폰트 설정
        if os.name == 'nt':  # Windows
            font = ImageFont.truetype("arial.ttf", 30)
        else:  # Linux/Mac
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # 텍스트 위치 설정 (예: 좌측 상단)
    text_position = (10, 10)
    text_color = (255, 255, 255)  # 흰색

    draw.text(text_position, prediction_text, font=font, fill=text_color)
    overlay_pil.save(save_path)

# Feature Map PCA 시각화
def visualize_featuremap_pca(feature_map, image_pil, save_path):
    """특징 맵의 PCA 결과를 시각화합니다."""
    if feature_map is None:
        print("⚠️ Warning: Feature map is None, skipping PCA visualization")
        return
        
    fmap = feature_map.squeeze().cpu().numpy()  # shape: [C, H, W]
    C, H, W = fmap.shape

    # Flatten and apply PCA
    fmap_flat = fmap.reshape(C, -1).T  # shape: [H*W, C]
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(fmap_flat).reshape(H, W)

    pc1 -= pc1.min()
    pc1 /= (pc1.max() + 1e-8)
    pc1 = np.uint8(255 * pc1)

    # Upsample to original image size
    cam_resized = cv2.resize(pc1, image_pil.size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * np.array(image_pil.resize(image_pil.size)) + 0.5 * heatmap
    overlay = np.uint8(np.clip(overlay, 0, 255))
    Image.fromarray(overlay).save(save_path)

# 얼굴 기반 처리 함수
def process_video(video_path, frame_interval=15, output_dir=save_path):
    """비디오를 처리하여 GradCAM을 생성합니다."""
    if model is None:
        print("❌ Error: Model not loaded. Cannot process video.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        face_data = df_face(video_path, frame_interval)
        
        if len(face_data) == 0:
            return
        
        for frame_idx, face_tensor in enumerate(face_data):
            try:
                # face_tensor는 이미 전처리된 텐서
                # prediction.py와 동일한 방식으로 모델에 입력
                with torch.no_grad():
                    model_output = model(face_tensor.unsqueeze(0).to(device))
                
                # 모델 출력 처리
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
                
                if output.dim() == 2:
                    probs = torch.sigmoid(output)
                    fake_prob = probs[0, 0].item()
                    real_prob = probs[0, 1].item()
                else:
                    probs = torch.sigmoid(output)
                    fake_prob = probs[0].item()
                    real_prob = probs[1].item()
                
                pred_label = "FAKE" if fake_prob > real_prob else "REAL"
                prediction_text = f"Pred: {pred_label} (Fake: {fake_prob:.3f}, Real: {real_prob:.3f})"
                
                cam, _, fmap_tensor = compute_gradcam(face_tensor.unsqueeze(0))
                
                if cam is not None:
                    face_img = face_tensor.cpu().numpy()
                    face_img = np.transpose(face_img, (1, 2, 0))
                    face_img = (face_img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(face_img)
                    
                    grad_path = os.path.join(output_dir, f"face_frame_{frame_idx}_gradcam.jpg")
                    visualize_gradcam(cam, pil_img, grad_path, prediction_text)
            except Exception as e:
                continue
        
    except Exception as e:
        pass

# 얼굴 기반 단일 이미지 처리 함수
def process_single_image(image_path, output_dir=save_path):
    """단일 이미지에 대해 GradCAM을 생성합니다."""
    if model is None:
        print("❌ Error: Model not loaded. Cannot process image.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 이미지를 폴더로 간주하여 df_face_from_folder 사용
        image_folder = os.path.dirname(image_path)
        if not image_folder:
            image_folder = "."
            
        face_data = df_face_from_folder(image_folder, 1)
        
        if len(face_data) == 0:
            return
        
        face_tensor = face_data[0]
        
        with torch.no_grad():
            model_output = model(face_tensor.unsqueeze(0).to(device))
        
        if isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
        if output.dim() == 2:
            # [batch_size, num_classes] 형태
            probs = torch.sigmoid(output)
            # 직접 인덱싱으로 안전하게 처리
            fake_prob = probs[0, 0].item()
            real_prob = probs[0, 1].item()
        else:
            # [num_classes] 형태
            probs = torch.sigmoid(output)
            fake_prob = probs[0].item()
            real_prob = probs[1].item()
        
        pred_label = "FAKE" if fake_prob > real_prob else "REAL"
        prediction_text = f"Pred: {pred_label} (Fake: {fake_prob:.3f}, Real: {real_prob:.3f})"
        
        # GradCAM 계산 (얼굴 텐서 사용)
        cam, _, fmap_tensor = compute_gradcam(face_tensor.unsqueeze(0))
        
        if cam is not None:
            face_img = face_tensor.cpu().numpy()
            face_img = np.transpose(face_img, (1, 2, 0))
            face_img = (face_img * 255).astype(np.uint8)
            pil_img = Image.fromarray(face_img)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            grad_path = os.path.join(output_dir, f"{base_name}_gradcam.jpg")
            visualize_gradcam(cam, pil_img, grad_path, prediction_text)

    except Exception as e:
        pass

# 실행 예시
if __name__ == "__main__":
    if model is not None:
        print("✅ GenConViT GradCAM Tool ready")
        print(f"   Device: {device}")
        print(f"   Output: {save_path}")
        print("\nUsage:")
        print("   from gradcam import process_video, process_single_image")
        print("   process_video('path/to/video.mp4')")
        print("   process_single_image('path/to/image.jpg')")
    else:
        print("❌ Model loading failed. Check weight files and configuration.")