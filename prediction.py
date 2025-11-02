import os
import argparse
import json
from time import perf_counter
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from model.pred_func import *
from model.config import load_config

config = load_config()

def parse_prediction_results(y, y_val):
    """예측 결과(y, y_val)를 파싱하여 확률과 로짓값을 반환"""
    # 예측값 처리
    if isinstance(y_val, list):
        fake_prob = y_val[0] if len(y_val) > 0 else 0.0
        real_prob = 1 - fake_prob
    else:
        fake_prob = y_val
        real_prob = 1 - fake_prob
    
    # 로짓값 처리
    if isinstance(y, list):
        fake_logit = y[0] if len(y) > 0 else 0.0
        real_logit = y[1] if len(y) > 1 else -fake_logit
    else:
        fake_logit = y
        real_logit = -y
    
    # 예측 결과 결정
    prediction = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if fake_prob > 0.5 else real_prob
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'fake_prob': fake_prob,
        'real_prob': real_prob,
        'fake_logit': fake_logit,
        'real_logit': real_logit,
        'is_fake': fake_prob > 0.5
    }

def safe_execute(func, error_msg="오류 발생", show_traceback=False):
    """안전한 함수 실행 헬퍼"""
    try:
        return func()
    except Exception as e:
        if show_traceback:
            import traceback
            print(f"{error_msg}: {e}")
            print(traceback.format_exc())
        else:
            print(f"{error_msg}: {e}")
        return None

def generate_gradcam_with_fallback(model, original_frame, df_tensor, target_class, is_video_frame=False):
    """GradCAM 생성 (전체 프레임 → 크롭 기반 폴백)"""
    prefix = "           " if is_video_frame else ""
    
    # 방법 1: 전체 프레임 기반 GradCAM
    gradcam = safe_execute(
        lambda: generate_gradcam_full_frame(model, original_frame, target_class),
        f"{prefix}전체 프레임 GradCAM 생성 중 오류"
    )
    
    if gradcam is not None:
        if not is_video_frame:
            print(f"✅ 전체 프레임 기반 GradCAM 생성 성공!")
        return gradcam
    
    # 방법 2: 크롭 기반 GradCAM (폴백)
    if is_video_frame:
        print(f"{prefix}GradCAM: 크롭 기반 방법 시도...")
    
    gradcam = safe_execute(
        lambda: generate_gradcam_with_hooks_improved(model, df_tensor, target_class),
        f"{prefix}크롭 기반 GradCAM 생성 중 오류"
    )
    
    if gradcam is not None:
        if is_video_frame:
            print(f"{prefix}GradCAM: ✅ 크롭 기반 생성 성공!")
        else:
            print(f"✅ 크롭 기반 GradCAM 생성 성공!")
    
    return gradcam

def save_gradcam_visualization(original_frame, gradcam, pred_label, confidence, face_bbox, output_path, is_video_frame=False):
    """GradCAM 시각화 저장"""
    prefix = "           " if is_video_frame else ""
    
    visualization = safe_execute(
        lambda: create_gradcam_visualization_improved(original_frame, gradcam, pred_label, confidence, face_bbox),
        f"{prefix}GradCAM 시각화 중 오류",
        show_traceback=True
    )
    
    if visualization is not None:
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        return True
    return False

def cleanup_gradcam_frames(gradcam_output_dir):
    """이전 영상의 GradCAM 프레임 이미지들을 삭제하는 함수"""
    import glob
    
    # GradCAM 이미지 삭제
    gradcam_files = glob.glob(os.path.join(gradcam_output_dir, "frame_*_gradcam.jpg"))
    
    for old_file in gradcam_files:
        try:
            os.remove(old_file)
        except:
            pass

def vids(
    ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, num_frames=15, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            is_vid_folder = is_video_folder(curr_vid)
            if is_video(curr_vid) or is_vid_folder:
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                    vid_folder=is_vid_folder
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            elif is_image(curr_vid):
                # 이미지 파일 처리
                result, accuracy, count, pred = predict_image(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    net,
                    "uncategorized",
                    count
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid file: {curr_vid}. Please provide a valid video or image file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def analyze_single_image(
    ed_weight, vae_weight, image_path, net=None, fp16=False, enable_gradcam=False
):
    """단일 이미지의 로짓 분석 함수 (GradCAM 옵션 지원)"""
    try:
        print(f"🖼️  이미지 분석: {os.path.basename(image_path)}")
        
        model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
        
        if enable_gradcam:
            processed_data = df_face_from_image_with_original(image_path)
            if not processed_data:
                print("❌ 이미지에서 얼굴을 검출할 수 없습니다.")
                return None
            
            df_tensor = processed_data['tensor']
            original_frame = processed_data['original_frames'][0]
            face_bbox = processed_data['face_bboxes'][0] if len(processed_data['face_bboxes']) > 0 else None
            print(f"✅ 이미지 처리 완료\n")
        else:
            df = df_face_from_image(image_path)
            if len(df) == 0:
                print("❌ 이미지에서 얼굴을 추출할 수 없습니다.")
                return None
            df_tensor = df
            original_frame = None
            face_bbox = None
            print(f"📊 얼굴 추출 완료\n")
        
        if fp16:
            df_tensor = df_tensor.half()
        
        y, y_val = pred_vid(df_tensor, model)
        results = parse_prediction_results(y, y_val)
        
        print(f"🎯 예측: {results['prediction']} (신뢰도: {results['confidence']:.4f})")
        print(f"📊 [FAKE: {results['fake_prob']:.4f}, REAL: {results['real_prob']:.4f}] | 로짓 [{results['fake_logit']:.4f}, {results['real_logit']:.4f}]")
        
        if enable_gradcam and original_frame is not None:
            gradcam_output_dir = os.path.join("result", "gradcam_outputs")
            os.makedirs(gradcam_output_dir, exist_ok=True)
            cleanup_gradcam_frames(gradcam_output_dir)
            
            target_class = torch.tensor(0 if results['is_fake'] else 1).to(df_tensor.device)
            gradcam = generate_gradcam_with_fallback(model, original_frame, df_tensor, target_class, is_video_frame=False)
            
            if gradcam is not None:
                output_path = os.path.join(gradcam_output_dir, f"image_gradcam.jpg")
                save_gradcam_visualization(original_frame, gradcam, results['prediction'], results['confidence'], face_bbox, output_path)
        
        return results
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        return None
    finally:
        cleanup_gpu_memory()


def evaluate_single_class(files, expected_label, class_name, model, data_dir, fp16, results):
    """단일 클래스(Real/Fake)에 대한 평가 수행"""
    print(f"🔍 {class_name} 이미지 평가 중...")
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(data_dir, filename)
        try:
            df = df_face_from_image(file_path)
            
            if len(df) == 0:
                print(f"   {i:3d}/{len(files)}: {filename} ❌ 얼굴 검출 실패")
                results['failed_detection'] += 1
                results['total'] += 1
                continue
            
            results['successful_detection'] += 1
            
            if fp16:
                df.half()
            
            y, y_val = pred_vid(df, model)
            parsed = parse_prediction_results(y, y_val)
            is_correct = parsed['prediction'] == expected_label
            
            results['predictions'].append({
                'filename': filename,
                'prediction': parsed['prediction'],
                'fake_prob': parsed['fake_prob'],
                'real_prob': parsed['real_prob'],
                'correct': is_correct
            })
            
            if is_correct:
                results['correct'] += 1
                print(f"   {i:3d}/{len(files)}: {filename} ✅ {expected_label} (확률: {parsed['confidence']:.3f})")
            else:
                print(f"   {i:3d}/{len(files)}: {filename} ❌ {parsed['prediction']} (확률: {parsed['confidence']:.3f}) - 오분류!")
            
            results['total'] += 1
        except Exception as e:
            print(f"   {i:3d}/{len(files)}: {filename} ❌ 오류: {e}")
            results['total'] += 1

def evaluate_model_precision(
    ed_weight, vae_weight, data_dir="sample_prediction_data", net=None, fp16=False
):
    """모델의 정밀도를 평가하는 함수"""
    print("🎯 GenConViT 모델 정밀도 평가 시작")
    print("=" * 60)
    
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    
    results = {
        'real': {'correct': 0, 'total': 0, 'predictions': [], 'failed_detection': 0, 'successful_detection': 0},
        'fake': {'correct': 0, 'total': 0, 'predictions': [], 'failed_detection': 0, 'successful_detection': 0}
    }
    
    files = os.listdir(data_dir)
    real_files = sorted([f for f in files if f.startswith('real_') and f.endswith('.png')])
    fake_files = sorted([f for f in files if f.startswith('fake_') and f.endswith('.png')])
    
    print(f"📊 데이터셋 정보:")
    print(f"   - Real 이미지: {len(real_files)}개")
    print(f"   - Fake 이미지: {len(fake_files)}개")
    print(f"   - 총 이미지: {len(real_files) + len(fake_files)}개\n")
    
    evaluate_single_class(real_files, "REAL", "Real", model, data_dir, fp16, results['real'])
    print()
    evaluate_single_class(fake_files, "FAKE", "Fake", model, data_dir, fp16, results['fake'])
    
    # 결과 분석 및 출력
    print("\n" + "=" * 60)
    print("📊 평가 결과 요약")
    print("=" * 60)
    
    # Real 클래스 성능
    real_accuracy = results['real']['correct'] / max(results['real']['successful_detection'], 1) * 100
    real_detection_rate = results['real']['successful_detection'] / max(results['real']['total'], 1) * 100
    
    print(f"🎭 REAL 클래스:")
    print(f"   - 정확도: {results['real']['correct']}/{results['real']['successful_detection']} ({real_accuracy:.1f}%)")
    print(f"   - 얼굴 검출률: {real_detection_rate:.1f}% ({results['real']['successful_detection']}/{results['real']['total']})")
    print(f"   - 얼굴 검출 실패: {results['real']['failed_detection']}개")
    
    # Fake 클래스 성능
    fake_accuracy = results['fake']['correct'] / max(results['fake']['successful_detection'], 1) * 100
    fake_detection_rate = results['fake']['successful_detection'] / max(results['fake']['total'], 1) * 100
    
    print(f"🎭 FAKE 클래스:")
    print(f"   - 정확도: {results['fake']['correct']}/{results['fake']['successful_detection']} ({fake_accuracy:.1f}%)")
    print(f"   - 얼굴 검출률: {fake_detection_rate:.1f}% ({results['fake']['successful_detection']}/{results['fake']['total']})")
    print(f"   - 얼굴 검출 실패: {results['fake']['failed_detection']}개")
    
    # 전체 성능 (얼굴 검출 성공한 이미지만으로 계산)
    total_correct = results['real']['correct'] + results['fake']['correct']
    total_successful = results['real']['successful_detection'] + results['fake']['successful_detection']
    total_samples = results['real']['total'] + results['fake']['total']
    overall_accuracy = total_correct / max(total_successful, 1) * 100
    
    print(f"\n🎯 전체 성능:")
    print(f"   - 전체 정확도: {total_correct}/{total_successful} ({overall_accuracy:.1f}%)")
    print(f"   - 평균 얼굴 검출률: {(real_detection_rate + fake_detection_rate) / 2:.1f}%")
    print(f"   - 총 얼굴 검출 실패: {results['real']['failed_detection'] + results['fake']['failed_detection']}개")
    
    # 오분류 사례 분석
    print(f"\n🔍 오분류 분석:")
    real_misclassified = [p for p in results['real']['predictions'] if not p['correct']]
    fake_misclassified = [p for p in results['fake']['predictions'] if not p['correct']]
    
    print(f"   - REAL → FAKE 오분류: {len(real_misclassified)}개")
    if real_misclassified:
        print(f"     가장 확신도 높은 오분류: {max(real_misclassified, key=lambda x: x['fake_prob'])['filename']} (FAKE 확률: {max(real_misclassified, key=lambda x: x['fake_prob'])['fake_prob']:.3f})")
    
    print(f"   - FAKE → REAL 오분류: {len(fake_misclassified)}개")
    if fake_misclassified:
        print(f"     가장 확신도 높은 오분류: {max(fake_misclassified, key=lambda x: x['real_prob'])['filename']} (REAL 확률: {max(fake_misclassified, key=lambda x: x['real_prob'])['real_prob']:.3f})")
    
    # 결과를 JSON으로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join("result", f"precision_evaluation_{net}_{timestamp}.json")
    
    evaluation_summary = {
        'timestamp': timestamp,
        'model': net,
        'dataset': 'sample_prediction_data',
        'total_samples': total_samples,
        'total_successful_detection': total_successful,
        'overall_accuracy': overall_accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'real_detection_rate': real_detection_rate,
        'fake_detection_rate': fake_detection_rate,
        'real_correct': results['real']['correct'],
        'real_total': results['real']['total'],
        'real_successful_detection': results['real']['successful_detection'],
        'fake_correct': results['fake']['correct'],
        'fake_total': results['fake']['total'],
        'fake_successful_detection': results['fake']['successful_detection'],
        'real_failed_detection': results['real']['failed_detection'],
        'fake_failed_detection': results['fake']['failed_detection'],
        'total_failed_detection': results['real']['failed_detection'] + results['fake']['failed_detection'],
        'misclassified_real': len(real_misclassified),
        'misclassified_fake': len(fake_misclassified),
        'detailed_results': results
    }
    
    os.makedirs("result", exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 상세 결과가 저장되었습니다: {result_file}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    return evaluation_summary


def analyze_single_video_frame_by_frame(
    ed_weight, vae_weight, video_path, num_frames=15, net=None, fp16=False, enable_gradcam=False
):
    """단일 비디오의 프레임별 로짓 분석 함수 (GradCAM 옵션 지원)"""
    try:
        print(f"🎬 비디오 분석: {os.path.basename(video_path)} ({num_frames}프레임)")
        
        model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
        
        if not is_video(video_path) and not is_video_folder(video_path):
            print(f"❌ 유효하지 않은 비디오 파일: {video_path}")
            return None
        
        if enable_gradcam:
            processed_data = df_face_with_original(video_path, num_frames)
            if not processed_data:
                print("❌ 프레임에서 얼굴을 검출할 수 없습니다.")
                return None
            
            df_tensor = processed_data['tensor']
            original_frames = processed_data['original_frames']
            face_bboxes = processed_data['face_bboxes']
            print(f"✅ {len(df_tensor)}개 프레임 추출 완료")
        else:
            if is_video_folder(video_path):
                df = df_face_from_folder(video_path, num_frames)
            else:
                df = df_face(video_path, num_frames)
            
            if len(df) == 0:
                print("❌ 얼굴을 추출할 수 없습니다.")
                return None
            
            df_tensor = df
            original_frames = None
            face_bboxes = None
            print(f"📊 추출된 프레임 수: {len(df)}\n")
        
        if fp16:
            df_tensor = df_tensor.half()
        
        pred_results = pred_vid_frame_by_frame(df_tensor, model, num_frames)
        
        if enable_gradcam:
            gradcam_output_dir = os.path.join("result", "gradcam_outputs")
            os.makedirs(gradcam_output_dir, exist_ok=True)
            cleanup_gradcam_frames(gradcam_output_dir)
        
        # 각 프레임별 결과 출력
        for i, (logit, prediction) in enumerate(zip(pred_results['frame_logits'], pred_results['frame_predictions'])):
            y_val = prediction if isinstance(prediction, list) else [prediction, 1-prediction]
            y = logit if isinstance(logit, list) else [logit, -logit]
            
            parsed = parse_prediction_results(y, y_val)
            print(f"프레임 {i+1}: {parsed['prediction']} ({parsed['confidence']:.4f})")
            
            # GradCAM 생성 및 시각화
            if enable_gradcam and original_frames and i < len(original_frames):
                single_frame = df_tensor[i:i+1]
                original_frame = original_frames[i]
                face_bbox = face_bboxes[i] if i < len(face_bboxes) else None
                
                target_class = torch.tensor(0 if parsed['is_fake'] else 1).to(single_frame.device)
                gradcam = generate_gradcam_with_fallback(model, original_frame, single_frame, target_class, is_video_frame=True)
                
                if gradcam is not None:
                    output_path = os.path.join(gradcam_output_dir, f"frame_{i+1:02d}_gradcam.jpg")
                    save_gradcam_visualization(original_frame, gradcam, parsed['prediction'], parsed['confidence'], face_bbox, output_path, is_video_frame=True)
        
        # 전체 요약
        frame_preds = pred_results['frame_predictions']
        avg_fake_prob = sum(p if not isinstance(p, list) else p[0] for p in frame_preds) / len(frame_preds)
        overall_pred = "FAKE" if avg_fake_prob > 0.5 else "REAL"
        overall_conf = avg_fake_prob if avg_fake_prob > 0.5 else (1 - avg_fake_prob)
        fake_frames = sum(1 for p in frame_preds if (p if not isinstance(p, list) else p[0]) > 0.5)
        real_frames = len(frame_preds) - fake_frames
        
        print(f"\n🎯 전체 예측: {overall_pred} (신뢰도: {overall_conf:.4f}) | 프레임: {len(frame_preds)}개 (FAKE: {fake_frames}, REAL: {real_frames})")
        
        if enable_gradcam:
            gif_output_dir = os.path.join("result", "gradcam_gif")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            gif_results = create_gradcam_gif(gradcam_dir=gradcam_output_dir, output_dir=gif_output_dir, video_name=video_name)
            
            if gif_results:
                combined_gif = gif_results.get('combined_gif', '')
                gradcam_gif = gif_results.get('gradcam_gif', '')
                if combined_gif:
                    print(f"\n✅ GIF 생성: {os.path.basename(combined_gif)}")
                elif gradcam_gif:
                    print(f"\n✅ GIF 생성: {os.path.basename(gradcam_gif)}")
        
        return pred_results
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        return None
    finally:
        cleanup_gpu_memory()

def faceforensics(
    ed_weight,
    vae_weight,
    root_dir="FaceForensics++",
    dataset=None,
    num_frames=15,
    net=None,
    fp16=False,
):
    keywords = ["original_sequences/youtube/c40/videos/"]

    compression = ["c40", "c23"]
    folders = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for k in compression:
        keywords = [
            f"original_sequences/youtube/{k}/videos/",
            f"manipulated_sequences/Deepfakes/{k}/videos/",
            f"manipulated_sequences/Face2Face/{k}/videos/",
            f"manipulated_sequences/FaceSwap/{k}/videos/",
            f"manipulated_sequences/NeuralTextures/{k}/videos/",
        ]

        for kw, folder in zip(keywords, folders):
            result = set_result()
            count = 0
            accuracy = 0

            if os.path.isfile(os.path.join("json_file", "ff_file_list.json")):
                with open(os.path.join("json_file", "ff_file_list.json")) as data_file:
                    ff_data = json.load(data_file)

            for ff_file in ff_data:
                curr_vid = os.path.join(root_dir, kw + ff_file + ".mp4")
                klass = "REAL" if folder == "original" else "FAKE"
                label = "FAKE" if folder == "original" else "REAL"
                try:
                    if is_video(curr_vid):
                        result, accuracy, count, _ = predict(
                            curr_vid,
                            model,
                            fp16,
                            result,
                            num_frames,
                            net,
                            klass,
                            count,
                            accuracy,
                            label,
                            compression,
                        )
                    else:
                        print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")

    return result


def timit(ed_weight, vae_weight, root_dir="DeepfakeTIMIT", dataset=None, num_frames=15, net=None, fp16=False):
    keywords = ["higher_quality", "lower_quality"]
    result = set_result()
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    count = 0
    accuracy = 0
    i = 0
    for keyword in keywords:
        keyword_folder_path = os.path.join(root_dir, keyword)
        for subfolder_name in os.listdir(keyword_folder_path):
            subfolder_path = os.path.join(keyword_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Loop through the AVI files in the subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".avi"):
                        curr_vid = os.path.join(subfolder_path, filename)
                        try:
                            if is_video(curr_vid):
                                result, accuracy, count, _ = predict(
                                    curr_vid,
                                    model,
                                    fp16,
                                    result,
                                    num_frames,
                                    net,
                                    "DeepfakeTIMIT",
                                    count,
                                    accuracy,
                                    "FAKE",
                                )
                            else:
                                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

                        except Exception as e:
                            print(f"An error occurred: {str(e)}")

    return result


def dfdc(
    ed_weight,
    vae_weight,
    root_dir="deepfake-detection-challenge\\train_sample_videos",
    dataset=None,
    num_frames=15,
    net=None,
    fp16=False,
):
    result = set_result()
    if os.path.isfile(os.path.join("json_file", "dfdc_files.json")):
        with open(os.path.join("json_file", "dfdc_files.json")) as data_file:
            dfdc_data = json.load(data_file)

    if os.path.isfile(os.path.join(root_dir, "metadata.json")):
        with open(os.path.join(root_dir, "metadata.json")) as data_file:
            dfdc_meta = json.load(data_file)
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    count = 0
    accuracy = 0
    for dfdc in dfdc_data:
        dfdc_file = os.path.join(root_dir, dfdc)

        try:
            if is_video(dfdc_file):
                result, accuracy, count, _ = predict(
                    dfdc_file,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "dfdc",
                    count,
                    accuracy,
                    dfdc_meta[dfdc]["label"],
                )
            else:
                print(f"Invalid video file: {dfdc_file}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def celeb(ed_weight, vae_weight, root_dir="Celeb-DF-v2", dataset=None, num_frames=15, net=None, fp16=False):
    with open(os.path.join("json_file", "celeb_test.json"), "r") as f:
        cfl = json.load(f)
    result = set_result()
    ky = ["Celeb-real", "Celeb-synthesis"]
    count = 0
    accuracy = 0
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0]
        filename = ck_[1]
        correct_label = "FAKE" if klass == "Celeb-synthesis" else "REAL"
        vid = os.path.join(root_dir, ck)

        try:
            if is_video(vid):
                result, accuracy, count, _ = predict(
                    vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    klass,
                    count,
                    accuracy,
                    correct_label,
                )
            else:
                print(f"Invalid video file: {vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred x: {str(e)}")

    return result

# 이미지 예측 함수 : 단일 이미지에 대한 로짓을 출력
def predict_image(
    img_path,
    model,
    fp16,
    result,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None
):
    count += 1
    print(f"\n\n{str(count)} Loading... {img_path}")

    start_time = perf_counter()

    # 이미지에서 얼굴 추출
    df = df_face_from_image(img_path)

    if len(df) == 0:
        print(f"❌ 이미지에서 얼굴을 검출할 수 없습니다: {img_path}")
        return result, accuracy, count, [0, 0.5]

    if fp16:
        df.half()
    
    y, y_val = pred_vid(df, model)
    
    result = store_result(
        result, os.path.basename(img_path), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    end_time = perf_counter()
    print("\n\n only one image--- %s seconds ---" % (end_time - start_time))
    
    return result, accuracy, count, [y, y_val]

# 영상 예측 함수 : 폴더 속 모든 영상에 대한 로짓을 출력
def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
    vid_folder=None
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    start_time = perf_counter()

    # locate the extracted frames of the video if provided.
    if vid_folder:
        df = df_face_from_folder(vid, num_frames)
    else:
        df = df_face(vid, num_frames)  # extract face from the frames

    if fp16:
        df.half()
    
    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    end_time = perf_counter()
    print("\n\n only one video--- %s seconds ---" % (end_time - start_time))
    
    return result, accuracy, count, [y, y_val]

# 명령어 인자자
def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large.",
    )
    parser.add_argument(
        "--e", nargs='?', const='genconvit_ed_inference', default='genconvit_ed_inference', help="weight for ed.",
    )
    parser.add_argument(
        "--v", '--value', nargs='?', const='genconvit_vae_inference', default='genconvit_vae_inference', help="weight for vae.",
    )
    
    parser.add_argument("--fp16", type=str, help="half precision support")
    parser.add_argument("--gradcam", action="store_true", help="GradCAM 시각화 활성화 (단일 파일 분석 시에만 사용)")
    parser.add_argument("--evaluate", nargs='?', const="sample_prediction_data", default=None, help="모델 정밀도 평가 (기본값: sample_prediction_data, 경로를 지정하면 해당 폴더 평가)")

    args = parser.parse_args()
    path = args.p if args.p else "sample_prediction_data"
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False
    # 단일 파일인지 자동 감지 (--single 플래그 제거)
    single_analysis = os.path.isfile(path) and (is_video(path) or is_image(path)) if args.p else False
    enable_gradcam = args.gradcam
    evaluate_model = args.evaluate is not None
    eval_data_dir = args.evaluate if args.evaluate else "sample_prediction_data"

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v
    
        
    print(f'\nUsing {net}\n')  
    

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s
    
    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight, single_analysis, enable_gradcam, evaluate_model, eval_data_dir


def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16, ed_weight, vae_weight, single_analysis, enable_gradcam, evaluate_model, eval_data_dir = gen_parser()
    
    if evaluate_model:
        # 모델 정밀도 평가
        print("🎯 모델 정밀도 평가 모드")
        print(f"📂 평가 대상 폴더: {eval_data_dir}")
        result = evaluate_model_precision(
            ed_weight, vae_weight, data_dir=eval_data_dir, net=net, fp16=fp16
        )
    # 단일 파일인 경우 자동으로 단일 파일 분석 모드로 전환
    elif single_analysis:
        # 단일 파일 프레임별 로짓 분석 (비디오 또는 이미지)
        if os.path.isfile(path):
            if is_image(path):
                result = analyze_single_image(ed_weight, vae_weight, path, net, fp16, enable_gradcam)
            elif is_video(path):
                result = analyze_single_video_frame_by_frame(ed_weight, vae_weight, path, num_frames, net, fp16, enable_gradcam)
            else:
                print(f"❌ 지원하지 않는 파일 형식입니다: {path}")
                print("💡 지원 형식: .mp4, .avi, .mov, .jpg, .jpeg, .png")
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {path}")
            print("💡 사용법:")
            print("   비디오 로짓만: python prediction.py --p video_path.mp4 --f 10")
            print("   비디오 로짓+GradCAM: python prediction.py --p video_path.mp4 --f 10 --gradcam")
            print("   이미지 로짓만: python prediction.py --p image_path.jpg")
            print("   이미지 로짓+GradCAM: python prediction.py --p image_path.jpg --gradcam")
    else:
        # 기존 배치 처리 로직
        result = (
            globals()[dataset](ed_weight, vae_weight, path, dataset, num_frames, net, fp16)
            if dataset in ["dfdc", "faceforensics", "timit", "celeb"]
            else vids(ed_weight, vae_weight, root_dir=path, dataset=dataset, num_frames=num_frames, net=net, fp16=fp16)
        )

        curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
        file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

        with open(file_path, "w") as f:
            json.dump(result, f)
    
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()