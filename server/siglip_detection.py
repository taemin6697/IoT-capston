# -*- coding: utf-8 -*-
"""
SIGLIP과 Gradio를 이용한 웹캠 이벤트 감지 및 녹화 스크립트 (v13 - UI 레이블 수정 기능)

이 스크립트는 Gradio를 사용하여 사용자의 웹캠 스트림에서 SIGLIP 모델을 통해
사용자가 UI에서 입력한 레이블에 따라 객체/상황을 감지하고, 이벤트 발생 시 전후 영상을 녹화합니다.
추론 빈도를 조절하고 UI를 간소화하여 감지 결과 및 상세 정보를 텍스트로 표시합니다.
"""

import os
import threading

import gradio as gr
import torch
import time
import cv2
import numpy as np
import uuid
import logging
import re
from datetime import datetime
from PIL import Image
from transformers import pipeline
from collections import deque
import argparse

# --- 로깅 설정 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "./logs_v13_siglip_labels")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"siglip_gradio_labels_v13_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIGLIPGradioLabelsTestV13")

# --- 설정값 ---
OUTPUT_DIR = os.path.join(script_dir, "./record_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-384"
CLASSIFIER = None
# UI에서 사용자가 직접 입력하도록 변경, 아래는 초기 기본값
DEFAULT_CANDIDATE_LABELS = ["A man wearing a black short sleeve"]
SIGLIP_THRESHOLD = 0.18

desired_fps = 3 # 추론 빈도 FPS (SIGLIP 모델의 부하 고려)
processing_interval = 1.0 / desired_fps if desired_fps > 0 else 0.2
pre_record_time = 5.0
buffer_size = 1#int(pre_record_time / processing_interval) if processing_interval > 0 else int(pre_record_time * (1 / 0.2))
stop_threshold_count = int(desired_fps * 5) # 예: 3fps * 5초 = 15회

DEVICE = None
processing_active = False
processing_lock = threading.Lock()
last_inference_time = 0.0
recording = False
video_writer = None
frame_buffer = deque(maxlen=buffer_size)
false_detection_count = 0
last_siglip_result_bool = False
last_siglip_detected_label_str = None
last_siglip_score_float = 0.0

# --- 모델 초기화 함수 ---
def load_model(model_name=SIGLIP_MODEL_NAME, gpu_idx=0):
    global CLASSIFIER, DEVICE
    if CLASSIFIER is not None:
        logger.info("SIGLIP 모델이 이미 로드되었습니다.")
        return CLASSIFIER, DEVICE
    try:
        logger.info(f"SIGLIP 모델 로딩 시작: {model_name}")
        if gpu_idx >= 0 and torch.cuda.is_available():
            DEVICE = torch.device(f"cuda:{gpu_idx}")
            logger.info(f"사용 장치: {DEVICE} (GPU)")
        else:
            DEVICE = torch.device("cpu")
            logger.info(f"사용 장치: {DEVICE} (CPU)")
        CLASSIFIER = pipeline(task="zero-shot-image-classification", model=model_name, device=DEVICE)
        logger.info(f"SIGLIP 모델 로드 성공. 장치: {CLASSIFIER.device}")
        return CLASSIFIER, DEVICE
    except Exception as e:
        logger.error(f"SIGLIP 모델 ({model_name}) 로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise gr.Error(f"SIGLIP 모델 로드 실패! 로그를 확인하세요: {log_file}") from e

# --- SIGLIP 모델 분류 함수 ---
def classify_image_with_siglip(pil_image, classifier_pipeline, labels_to_check, confidence_threshold):
    logger.debug(f"SIGLIP 분류 시작 - 레이블: {labels_to_check}, 이미지 크기: {pil_image.size if pil_image else 'N/A'}")
    if not labels_to_check: # 레이블 리스트가 비어있으면
        logger.warning("분류할 후보 레이블이 없습니다.")
        return False, None, 0.0, "후보 레이블 없음"
    try:
        outputs = classifier_pipeline(pil_image, candidate_labels=labels_to_check)
        print(outputs) # 디버깅용 출력
        if not outputs:
            logger.warning("SIGLIP 결과가 비어있습니다.")
            return False, None, 0.0, "결과 없음"
        
        # outputs가 단일 딕셔너리를 반환하는 경우가 있으므로, 리스트가 아니면 리스트로 감싸줌
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        # outputs의 각 요소가 딕셔너리 리스트인 경우 (일부 모델 파이프라인)
        # 예: [[{'score': 0.9, 'label': 'cat'}, {'score': 0.1, 'label': 'dog'}]]
        # 여기서는 파이프라인이 [{'score':s, 'label':l}, ...] 형태를 반환한다고 가정.
        # 만약 [{'score':s, 'label':l}] 형태의 리스트를 포함하는 리스트라면 첫번째 요소를 사용
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]

        if not outputs or not all(isinstance(item, dict) and 'score' in item and 'label' in item for item in outputs):
            logger.error(f"SIGLIP 출력 형식이 예상과 다릅니다: {outputs}")
            return False, None, 0.0, "출력 형식 오류"

        best_output = max(outputs, key=lambda x: x['score'])
        score = best_output['score']
        detected_label = best_output['label']
        result = score >= confidence_threshold
        raw_response_for_log = f"Label: {detected_label}, Score: {score:.4f}, Threshold: {confidence_threshold}"
        logger.info(f"SIGLIP 결과: {result}, {raw_response_for_log}")
        return result, detected_label, score, raw_response_for_log
    except Exception as e:
        logger.error(f"SIGLIP 분류 중 오류: {e}", exc_info=True)
        return False, None, 0.0, f"분류 오류: {str(e)}"

# --- 프레임 처리 및 UI 업데이트 함수 ---
def process_frame_and_update(frame_np_rgb, current_labels_from_ui_str, model_path_state_ignored, gpu_idx_state_ignored):
    global processing_active, CLASSIFIER
    global last_inference_time, processing_interval
    global recording, video_writer, frame_buffer, false_detection_count
    global last_siglip_result_bool, last_siglip_detected_label_str, last_siglip_score_float

    current_time = time.time()
    status_text_for_ui = "오류 발생"
    output_frame_for_gradio = frame_np_rgb.copy() if frame_np_rgb is not None else np.zeros((100, 100, 3), dtype=np.uint8)
    saved_file_path_for_gradio = None

    # UI에서 받은 레이블 문자열 파싱
    if current_labels_from_ui_str and current_labels_from_ui_str.strip():
        candidate_labels_for_inference = [label.strip() for label in current_labels_from_ui_str.split(',') if label.strip()]
    else: # 입력이 없거나 비어있으면 기본값 또는 빈 리스트 (여기선 빈 리스트로 처리하여 경고 유도)
        candidate_labels_for_inference = [] 
    
    # UI 표시용 활성 레이블 (너무 길면 일부만)
    active_labels_display = ", ".join(candidate_labels_for_inference)
    if len(active_labels_display) > 50:
        active_labels_display = active_labels_display[:47] + "..."


    with processing_lock:
        if not processing_active:
            status_text_for_ui = "중지됨. '시작' 버튼을 누르세요."
            return output_frame_for_gradio, status_text_for_ui, saved_file_path_for_gradio

    if frame_np_rgb is None:
        return output_frame_for_gradio, "웹캠 프레임 없음", saved_file_path_for_gradio

    if CLASSIFIER is None:
        logger.error("SIGLIP 모델(CLASSIFIER)이 준비되지 않았습니다.")
        return output_frame_for_gradio, "오류: 모델 초기화 안됨!", saved_file_path_for_gradio

    if frame_np_rgb is not None:
        frame_buffer.append(frame_np_rgb.copy())

    perform_inference_this_frame = False
    if current_time - last_inference_time >= processing_interval:
        perform_inference_this_frame = True
        last_inference_time = current_time

    current_frame_detection_result = False

    if perform_inference_this_frame:
        logger.info(f"수행: SIGLIP 추론 (레이블: {active_labels_display})")
        if not candidate_labels_for_inference:
            logger.warning("추론할 후보 레이블이 없습니다. UI에서 레이블을 입력하세요.")
            last_siglip_result_bool = False
            last_siglip_detected_label_str = None
            last_siglip_score_float = 0.0
        else:
            try:
                pil_image = Image.fromarray(frame_np_rgb)
                result, label, score, raw_log = classify_image_with_siglip(
                    pil_image, CLASSIFIER, candidate_labels_for_inference, SIGLIP_THRESHOLD
                )
                last_siglip_result_bool = result
                last_siglip_detected_label_str = label if result else None
                last_siglip_score_float = score
                current_frame_detection_result = result
            except Exception as e:
                logger.error(f"SIGLIP 추론 중 오류 발생: {e}", exc_info=True)
                last_siglip_result_bool = False
    
    if not recording:
        if current_frame_detection_result and perform_inference_this_frame:
            try:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                abs_output_dir = os.path.abspath(OUTPUT_DIR)
                # 파일명에 감지된 레이블 포함 (공백 등 안전하게 처리)
                safe_detected_label = re.sub(r'[^\w_.)( -]', '', last_siglip_detected_label_str if last_siglip_detected_label_str else "event").strip().replace(" ", "_")
                video_filename = os.path.join(abs_output_dir, f"record_siglip_{timestamp_str}_{safe_detected_label}_{uuid.uuid4().hex[:4]}.mp4")
                
                height, width, _ = frame_np_rgb.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_filename, fourcc, desired_fps, (width, height))
                
                if not video_writer.isOpened(): raise IOError(f"VideoWriter를 열 수 없습니다: {video_filename}")

                logger.info(f"감지 ({last_siglip_detected_label_str})! 녹화 시작: {video_filename}")
                logger.info(f"버퍼 ({len(frame_buffer)} 프레임) 기록 중...")
                for pre_frame_rgb in list(frame_buffer):
                    pre_frame_bgr = cv2.cvtColor(pre_frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(pre_frame_bgr)
                
                recording = True; false_detection_count = 0; saved_file_path_for_gradio = video_filename
            except (IOError, cv2.error) as e:
                logger.error(f"VideoWriter 생성/쓰기 오류: {e}")
                if video_writer is not None and video_writer.isOpened(): video_writer.release()
                video_writer = None; recording = False
    else: # 녹화 중
        if frame_np_rgb is not None and video_writer is not None and video_writer.isOpened():
            try:
                video_writer.write(cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR))
            except cv2.error as e:
                logger.error(f"녹화 중 프레임 쓰기 오류: {e}")
                if video_writer is not None and video_writer.isOpened(): video_writer.release()
                video_writer = None; recording = False; false_detection_count = 0

        if perform_inference_this_frame:
            if not current_frame_detection_result: false_detection_count += 1
            else: false_detection_count = 0
            
            if false_detection_count >= stop_threshold_count:
                logger.info(f"미감지 {stop_threshold_count}회 연속: 녹화 종료")
                if video_writer is not None and video_writer.isOpened(): video_writer.release()
                video_writer = None; recording = False; false_detection_count = 0

    status_text_for_ui = f"감지 결과 ({active_labels_display}): {last_siglip_result_bool}"
    if recording: status_text_for_ui += " (녹화 중)"
    if last_siglip_result_bool and last_siglip_detected_label_str:
        status_text_for_ui += f"\n레이블: {last_siglip_detected_label_str} (점수: {last_siglip_score_float:.2f})"
            
    return output_frame_for_gradio, status_text_for_ui, saved_file_path_for_gradio

# --- Gradio 인터페이스 ---
def create_gradio_interface(model_path_ignored, gpu_device_idx):
    logger.info("Gradio 인터페이스 생성 시작")
    abs_output_dir_ui = os.path.abspath(OUTPUT_DIR)

    device_info_str = "정보 없음 (모델 로드 전)"
    if DEVICE:
        device_info_str = f"{DEVICE.type.upper()}"
        if DEVICE.type == "cuda" and torch.cuda.is_available() and gpu_device_idx >= 0:
            try: device_info_str += f":{gpu_device_idx} ({torch.cuda.get_device_name(gpu_device_idx)})"
            except Exception as e: logger.warning(f"CUDA 장치 이름 가져오기 실패: {e}"); device_info_str += f":{gpu_device_idx}"
    elif gpu_device_idx >=0 and torch.cuda.is_available(): device_info_str = f"SIGLIP GPU:{gpu_device_idx} (모델 로드 시)"
    elif gpu_device_idx < 0: device_info_str = "SIGLIP CPU (모델 로드 시)"

    with gr.Blocks(title=f"SIGLIP 이벤트 감지 (추론 FPS: {desired_fps})", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# SIGLIP 웹캠 실시간 이벤트 감지 및 녹화 (목표 추론 FPS: {desired_fps})")
        gr.Markdown(f"아래 텍스트박스에 감지할 레이블들을 쉼표로 구분하여 입력하세요 (예: a person, a car). 현재 임계값: {SIGLIP_THRESHOLD}")
        
        with gr.Row():
            with gr.Column(scale=1):
                webcam_input = gr.Image(label="웹캠 입력", sources=["webcam"], streaming=True, type="numpy", height=300)
                with gr.Row():
                    start_button = gr.Button("▶ 감지 시작", variant="primary")
                    stop_button = gr.Button("■ 감지 중지", interactive=False)
                
                siglip_labels_input_ui = gr.Textbox(
                    label="SIGLIP 감지 레이블 (쉼표로 구분)", 
                    value=",".join(DEFAULT_CANDIDATE_LABELS), 
                    lines=2, 
                    placeholder="예: a cat, a dog playing, a book on a table"
                )
                gr.Markdown(f"**SIGLIP 모델:** `{SIGLIP_MODEL_NAME}`\n**사용 장치:** `{device_info_str}`")

            with gr.Column(scale=1):
                output_image_dummy = gr.Image(label="처리 결과 (숨김)", type="numpy", height=1, interactive=False, visible=False)
                latest_saved_file_dummy = gr.File(label="최근 저장된 파일 (숨김)", interactive=False, visible=False)
                status_text_output = gr.Textbox(label="감지 상태 및 정보", value="대기 중...", lines=3, interactive=False)
                gr.Markdown(f"녹화 파일 저장 위치: `{abs_output_dir_ui}`")
        
        with gr.Accordion("세부 설정 정보 (코드 내 수정)", open=False):
            gr.Markdown(f"""
            - **SIGLIP 임계값:** `{SIGLIP_THRESHOLD}`
            - **추론 간격 (목표 FPS):** `{processing_interval:.2f}초 ({desired_fps} FPS)`
            - **사전 녹화 시간:** `{pre_record_time}초` (버퍼 크기: {buffer_size} 프레임)
            - **녹화 중지 조건:** 미감지 연속 `{stop_threshold_count}`회 (약 `{stop_threshold_count * processing_interval:.1f}`초)
            """)

        def start_processing_wrapper():
            global processing_active, last_inference_time, frame_buffer, recording, video_writer, false_detection_count
            global last_siglip_result_bool, last_siglip_detected_label_str, last_siglip_score_float
            logger.info("감지/녹화 처리 시작 요청됨")
            with processing_lock: processing_active = True
            last_inference_time = 0.0; frame_buffer.clear()
            if video_writer is not None and video_writer.isOpened(): video_writer.release()
            video_writer = None; recording = False; false_detection_count = 0
            last_siglip_result_bool = False; last_siglip_detected_label_str = None; last_siglip_score_float = 0.0
            return "감지/녹화 처리 중...", gr.update(interactive=False), gr.update(interactive=True)

        def stop_processing_wrapper():
            global processing_active, recording, video_writer
            logger.info("감지/녹화 처리 중지 요청됨")
            with processing_lock: processing_active = False
            status_msg = "감지/녹화 중지됨."
            if recording:
                if video_writer is not None and video_writer.isOpened():
                    video_writer.release(); logger.info("진행 중이던 녹화 파일 저장 완료."); status_msg += " (녹화 파일 저장됨)"
                video_writer = None; recording = False
            return status_msg, gr.update(interactive=True), gr.update(interactive=False)

        start_button.click(fn=start_processing_wrapper, inputs=[], outputs=[status_text_output, start_button, stop_button])
        stop_button.click(fn=stop_processing_wrapper, inputs=[], outputs=[status_text_output, start_button, stop_button])
        
        # inputs 리스트에 siglip_labels_input_ui 추가
        webcam_input.stream(
            fn=process_frame_and_update,
            inputs=[webcam_input, siglip_labels_input_ui, gr.State(SIGLIP_MODEL_NAME), gr.State(0)], 
            outputs=[output_image_dummy, status_text_output, latest_saved_file_dummy], #stream_every=1,
        )
        with gr.Accordion("로그 및 디버깅 정보", open=False):
            gr.Markdown(f"- **로그 파일:** `{os.path.abspath(log_file)}`")
    logger.info("Gradio 인터페이스 생성 완료")
    return demo

# --- 메인 실행 ---
if __name__ == "__main__":
    DEFAULT_GPU_DEVICE_IDX = 0 if torch.cuda.is_available() else -1
    logger.info(f"SIGLIP 웹캠 감지 및 녹화 애플리케이션 (v13 - UI 레이블 수정) 시작")
    # (로그 설정 정보는 생략, 필요시 추가)

    try:
        load_model(gpu_idx=DEFAULT_GPU_DEVICE_IDX)
    except gr.Error as e: logger.critical(f"Gradio 초기 모델 로드 오류: {e}")
    except Exception as e: logger.critical(f"모델 로드 중 치명적 오류 발생: {e}", exc_info=True); exit()

    app_interface = create_gradio_interface(None, DEFAULT_GPU_DEVICE_IDX)
    
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    logger.info(f"Gradio 애플리케이션 실행. 허용된 파일 경로: {abs_output_dir}")
    try:
        app_interface.launch(share=True, allowed_paths=[abs_output_dir])
    except Exception as e: logger.critical(f"Gradio 앱 실행 중 오류: {e}", exc_info=True)

    logger.info("애플리케이션 종료됨.")
    if video_writer is not None and video_writer.isOpened():
        logger.info("애플리케이션 종료 전, 열려있는 녹화 파일 저장 시도...")
        video_writer.release()