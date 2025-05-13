# -*- coding: utf-8 -*-
"""
SmolVLM과 Gradio를 이용한 웹캠 이벤트 감지 및 녹화 스크립트 (v14 - UI 프롬프트 수정 기능)

이 스크립트는 Gradio를 사용하여 사용자의 웹캠 스트림에서 SmolVLM 모델을 통해
사용자가 UI에서 입력한 프롬프트에 따라 이미지를 판단하고, 이벤트 발생 시 전후 영상을 녹화합니다.
추론 빈도를 조절하고 UI를 간소화하여 감지 결과(True/False)만 텍스트로 표시합니다.
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
from transformers import AutoProcessor, AutoModelForImageTextToText
from collections import deque
import argparse

# --- 로깅 설정 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "./logs_v14_smolvlm_prompt")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"smolvlm_gradio_prompt_v14_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SmolVLMPromptGradioTestV14")

# --- 설정값 ---
OUTPUT_DIR = os.path.join(script_dir, "./record_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SmolVLM 모델 및 관련 설정
SMOLVLM_MODEL_PATH = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
MODEL = None
PROCESSOR = None
PARAM_DTYPE = None

# 기본 프롬프트 (UI에서 변경 가능)
DEFAULT_SMOLVLM_PROMPT = "Is there a A man wearing a black short sleeve in this image? Answer yes or no."

desired_fps = 3
processing_interval = 1.0 / desired_fps if desired_fps > 0 else 0.2
pre_record_time = 5.0
buffer_size = int(pre_record_time / processing_interval) if processing_interval > 0 else int(pre_record_time * (1 / 0.2))
stop_threshold_count = int(desired_fps * 5)

DEVICE = None
processing_active = False
processing_lock = threading.Lock()
last_inference_time = 0.0
recording = False
video_writer = None
frame_buffer = deque(maxlen=buffer_size)
false_detection_count = 0
last_smolvlm_result_bool = False
last_smolvlm_raw_response = "대기 중..."
# last_smolvlm_detected_label_str은 프롬프트가 동적이므로 특정 레이블을 지칭하기 어려움

# --- 모델 응답 파싱 헬퍼 함수 ---
def _extract_yes_no(raw_text):
    if raw_text is None: return False
    raw = raw_text.lower().strip()
    if ":" in raw: raw = raw.split(":")[-1].strip()
    raw = re.sub(r"^[ \n\t.:;!-]+", "", raw)
    match = re.match(r"^(yes|no|true|false)", raw, re.IGNORECASE)
    if match:
        answer = match.group(1)
        logger.debug(f"Extracted answer: '{answer}' from raw: '{raw_text}'")
        return answer in ("yes", "true")
    logger.debug(f"Could not extract yes/no from: '{raw_text}'")
    return False

# --- 모델 초기화 함수 ---
def load_model(model_path=SMOLVLM_MODEL_PATH, gpu_idx=0):
    global MODEL, PROCESSOR, DEVICE, PARAM_DTYPE
    if MODEL is not None and PROCESSOR is not None:
        logger.info("SmolVLM 모델과 프로세서가 이미 로드되었습니다.")
        return MODEL, PROCESSOR, DEVICE, PARAM_DTYPE
    try:
        logger.info(f"SmolVLM 모델 로딩 시작: {model_path}")
        if gpu_idx >= 0 and torch.cuda.is_available():
            DEVICE = torch.device(f"cuda:{gpu_idx}")
            logger.info(f"사용 장치: {DEVICE} (GPU)")
        else:
            DEVICE = torch.device("cpu")
            logger.info(f"사용 장치: {DEVICE} (CPU)")
        PARAM_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
        PROCESSOR = AutoProcessor.from_pretrained(model_path)
        MODEL = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=PARAM_DTYPE).to(DEVICE)
        MODEL.eval()
        actual_param_dtype = next(MODEL.parameters()).dtype
        if actual_param_dtype != PARAM_DTYPE:
            logger.warning(f"Requested dtype {PARAM_DTYPE} but model loaded with {actual_param_dtype}. Using {actual_param_dtype}.")
            PARAM_DTYPE = actual_param_dtype
        logger.info(f"SmolVLM 모델 로드 성공. 최종 파라미터 dtype: {PARAM_DTYPE}, 사용 장치: {MODEL.device}")
        return MODEL, PROCESSOR, DEVICE, PARAM_DTYPE
    except Exception as e:
        logger.error(f"SmolVLM 모델 ({model_path}) 로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise gr.Error(f"SmolVLM 모델 로드 실패! 로그를 확인하세요: {log_file}") from e

# --- SmolVLM 모델 분류 함수 ---
def classify_image_with_smolvlm(pil_image, model, processor, prompt_text):
    logger.debug(f"SmolVLM 분류 시작 - 프롬프트: '{prompt_text}', 이미지 크기: {pil_image.size if pil_image else 'N/A'}")
    msgs = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt_text}]}]
    try:
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(
                **inputs, do_sample=False, max_new_tokens=20, pad_token_id=processor.tokenizer.pad_token_id # 답변 길이 약간 늘림
            )
        raw_answer = processor.batch_decode(ids, skip_special_tokens=True)[0]
        logger.debug(f"SmolVLM 원시 응답: '{raw_answer}'")
        detected = _extract_yes_no(raw_answer)
        generated_text_for_log = raw_answer.split("assistant:")[-1].strip() if "assistant:" in raw_answer.lower() else raw_answer

        if detected:
            logger.info(f"SmolVLM 프롬프트 '{prompt_text[:30]}...' 결과: True (응답: '{generated_text_for_log}')")
            # detected_label은 프롬프트에 따라 달라지므로 여기서는 True/False에 집중
            return True, generated_text_for_log
        else:
            logger.info(f"SmolVLM 프롬프트 '{prompt_text[:30]}...' 결과: False (응답: '{generated_text_for_log}')")
            return False, generated_text_for_log
    except Exception as e:
        logger.error(f"SmolVLM 분류 중 오류 (프롬프트: '{prompt_text}'): {e}", exc_info=True)
        return False, f"분류 오류: {str(e)}"

# --- 프레임 처리 및 UI 업데이트 함수 ---
def process_frame_and_update(frame_np_rgb, current_prompt_from_ui, model_path_state_ignored, gpu_idx_state_ignored):
    global processing_active, MODEL, PROCESSOR
    global last_inference_time, processing_interval
    global recording, video_writer, frame_buffer, false_detection_count
    global last_smolvlm_result_bool, last_smolvlm_raw_response

    current_time = time.time()
    status_text_for_ui = "오류 발생"
    output_frame_for_gradio = frame_np_rgb.copy() if frame_np_rgb is not None else np.zeros((100, 100, 3), dtype=np.uint8)
    saved_file_path_for_gradio = None

    with processing_lock:
        if not processing_active:
            status_text_for_ui = "중지됨. '시작' 버튼을 누르세요."
            return output_frame_for_gradio, status_text_for_ui, saved_file_path_for_gradio

    if frame_np_rgb is None:
        return output_frame_for_gradio, "웹캠 프레임 없음", saved_file_path_for_gradio

    if MODEL is None or PROCESSOR is None:
        logger.error("SmolVLM 모델 또는 프로세서가 준비되지 않았습니다.")
        return output_frame_for_gradio, "오류: 모델 초기화 안됨!", saved_file_path_for_gradio

    if frame_np_rgb is not None:
        frame_buffer.append(frame_np_rgb.copy())

    perform_inference_this_frame = False
    if current_time - last_inference_time >= processing_interval:
        perform_inference_this_frame = True
        last_inference_time = current_time

    current_frame_detection_result = False

    if perform_inference_this_frame:
        logger.info(f"수행: SmolVLM 추론 (프롬프트: {current_prompt_from_ui[:50]}...)")
        try:
            pil_image = Image.fromarray(frame_np_rgb)
            result, raw_response = classify_image_with_smolvlm( # detected_label 반환 안함
                pil_image, MODEL, PROCESSOR, current_prompt_from_ui
            )
            last_smolvlm_result_bool = result
            last_smolvlm_raw_response = raw_response
            current_frame_detection_result = result
        except Exception as e:
            logger.error(f"SmolVLM 추론 중 오류 발생: {e}", exc_info=True)
            last_smolvlm_result_bool = False
            last_smolvlm_raw_response = f"추론 오류: {str(e)}"

    if not recording:
        if current_frame_detection_result and perform_inference_this_frame:
            try:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                abs_output_dir = os.path.abspath(OUTPUT_DIR)
                # 파일명에 프롬프트 일부 대신 "event" 사용
                video_filename = os.path.join(abs_output_dir, f"record_smolvlm_{timestamp_str}_event_{uuid.uuid4().hex[:4]}.mp4")

                height, width, _ = frame_np_rgb.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_filename, fourcc, desired_fps, (width, height))

                if not video_writer.isOpened():
                    raise IOError(f"VideoWriter를 열 수 없습니다: {video_filename}")

                logger.info(f"감지! 녹화 시작: {video_filename}")
                logger.info(f"버퍼 ({len(frame_buffer)} 프레임, 약 {len(frame_buffer) * processing_interval:.1f}초) 기록 중...")
                for pre_frame_rgb in list(frame_buffer):
                    pre_frame_bgr = cv2.cvtColor(pre_frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(pre_frame_bgr)

                recording = True
                false_detection_count = 0
                saved_file_path_for_gradio = video_filename
            except (IOError, cv2.error) as e:
                logger.error(f"VideoWriter 생성 또는 쓰기 오류: {e}")
                if video_writer is not None and video_writer.isOpened(): video_writer.release()
                video_writer = None; recording = False
    else: # 녹화 중
        if frame_np_rgb is not None and video_writer is not None and video_writer.isOpened():
            try:
                frame_bgr_for_write = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr_for_write)
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

    # UI 텍스트 생성 시 현재 프롬프트 일부 포함
    active_prompt_display = current_prompt_from_ui if current_prompt_from_ui else DEFAULT_SMOLVLM_PROMPT
    status_text_for_ui = f"프롬프트 응답 ('{active_prompt_display[:20]}...'): {last_smolvlm_result_bool}"
    if recording: status_text_for_ui += " (녹화 중)"

    return output_frame_for_gradio, status_text_for_ui, saved_file_path_for_gradio

# --- Gradio 인터페이스 ---
def create_gradio_interface(model_path_ignored, gpu_device_idx):
    logger.info("Gradio 인터페이스 생성 시작")
    abs_output_dir_ui = os.path.abspath(OUTPUT_DIR)

    device_info_str = "정보 없음 (모델 로드 전)"
    if DEVICE: # 모델 로드 후 DEVICE가 설정되었다면
        device_info_str = f"{DEVICE.type.upper()}"
        if DEVICE.type == "cuda" and torch.cuda.is_available() and gpu_device_idx >= 0:
            try: device_info_str += f":{gpu_device_idx} ({torch.cuda.get_device_name(gpu_device_idx)})"
            except Exception as e: logger.warning(f"CUDA 장치 이름 가져오기 실패: {e}"); device_info_str += f":{gpu_device_idx}"
    elif gpu_device_idx >=0 and torch.cuda.is_available(): device_info_str = f"SmolVLM GPU:{gpu_device_idx} (모델 로드 시 확인)"
    elif gpu_device_idx < 0 : device_info_str = "SmolVLM CPU (모델 로드 시 확인)"

    with gr.Blocks(title=f"SmolVLM 프롬프트 기반 감지 및 녹화 (추론 FPS: {desired_fps})", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# SmolVLM 웹캠 실시간 프롬프트 기반 감지 및 녹화 (목표 추론 FPS: {desired_fps})")

        with gr.Row():
            with gr.Column(scale=1):
                webcam_input = gr.Image(label="웹캠 입력", sources=["webcam"], streaming=True, type="numpy", height=300)
                with gr.Row():
                    start_button = gr.Button("▶ 감지 시작", variant="primary")
                    stop_button = gr.Button("■ 감지 중지", interactive=False)

                # 사용자 프롬프트 입력 UI
                prompt_input_ui = gr.Textbox(
                    label="SmolVLM 프롬프트 입력",
                    value=DEFAULT_SMOLVLM_PROMPT,
                    lines=3,
                    placeholder="예: Is there a cat in this image? Answer with only yes or no."
                )
                gr.Markdown(f"**SmolVLM 모델:** `{SMOLVLM_MODEL_PATH}`\n**사용 장치:** `{device_info_str}`")

            with gr.Column(scale=1):
                output_image_dummy = gr.Image(label="처리 결과 (숨김)", type="numpy", height=1, interactive=False, visible=False)
                latest_saved_file_dummy = gr.File(label="최근 저장된 파일 (숨김)", interactive=False, visible=False)
                status_text_output = gr.Textbox(label="감지 상태 및 프롬프트 응답", value="대기 중...", lines=3, interactive=False)
                gr.Markdown(f"녹화 파일 저장 위치: `{abs_output_dir_ui}`")

        with gr.Accordion("세부 설정 정보 (코드 내 수정)", open=False):
            gr.Markdown(f"""
            - **추론 간격 (목표 FPS):** `{processing_interval:.2f}초 ({desired_fps} FPS)`
            - **사전 녹화 시간:** `{pre_record_time}초` (버퍼 크기: {buffer_size} 프레임)
            - **녹화 중지 조건:** 미감지 연속 `{stop_threshold_count}`회 (약 `{stop_threshold_count * processing_interval:.1f}`초)
            """)

        def start_processing_wrapper():
            global processing_active, last_inference_time, frame_buffer, recording, video_writer, false_detection_count
            global last_smolvlm_result_bool, last_smolvlm_raw_response
            logger.info("감지/녹화 처리 시작 요청됨")
            with processing_lock: processing_active = True
            last_inference_time = 0.0
            frame_buffer.clear()
            if video_writer is not None and video_writer.isOpened(): video_writer.release()
            video_writer = None; recording = False; false_detection_count = 0
            last_smolvlm_result_bool = False; last_smolvlm_raw_response = "처리 시작됨..."
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

        webcam_input.stream(
            fn=process_frame_and_update,
            # inputs 리스트에 prompt_input_ui 추가, 다른 더미 입력 제거 또는 적절히 수정
            inputs=[webcam_input, prompt_input_ui, gr.State(SMOLVLM_MODEL_PATH), gr.State(0)],
            outputs=[output_image_dummy, status_text_output, latest_saved_file_dummy]
        )
        with gr.Accordion("로그 및 디버깅 정보", open=False):
            gr.Markdown(f"- **로그 파일:** `{os.path.abspath(log_file)}`")
    logger.info("Gradio 인터페이스 생성 완료")
    return demo

# --- 메인 실행 ---
if __name__ == "__main__":
    DEFAULT_GPU_DEVICE_IDX = 0 if torch.cuda.is_available() else -1
    logger.info(f"SmolVLM 웹캠 프롬프트 기반 감지 및 녹화 애플리케이션 (v14) 시작")
    logger.info(f"--- 주요 설정 ---")
    logger.info(f"로그 저장 폴더: {os.path.abspath(log_dir)}")
    logger.info(f"녹화 저장 폴더: {os.path.abspath(OUTPUT_DIR)}")
    logger.info(f"초기 프롬프트: {DEFAULT_SMOLVLM_PROMPT}")
    # (다른 설정값 로그는 생략, 필요시 추가)
    logger.info(f"-----------------")

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