# -*- coding: utf-8 -*-
"""
SmolVLM과 Gradio를 이용한 웹캠 이벤트 감지 및 녹화 스크립트 (v17_smolvlm_threaded_writer - 비디오 쓰기 스레드 분리)

이 스크립트는 Gradio를 사용하여 사용자의 웹캠 스트림에서 SmolVLM 모델을 통해
사용자가 UI에서 입력한 프롬프트에 따라 이미지를 판단하고, 이벤트 발생 시 영상을 녹화합니다.
VLM 추론 및 비디오 쓰기 작업은 각각 별도 스레드에서 실행되며, 사전 녹화 기능은 없습니다.
"""

import os
import threading
import queue  # 스레드 안전 큐
import time
import cv2
import numpy as np
import uuid
import logging
import re
import argparse
from datetime import datetime
from PIL import Image

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- 로깅 설정 ---
logger = logging.getLogger("SmolVLMThreadedWriter_v17")  # 로거 이름 변경

# --- 기본 설정값 (argparse를 통해 오버라이드 가능) ---
DEFAULT_MODEL_PATH = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_OUTPUT_DIR_BASE = "./record_videos"
DEFAULT_PROMPT = "Is there a person in this image? Answer with only yes or no."
DEFAULT_FRAME_QUEUE_SIZE = 2  # VLM 입력용 큐 크기
DEFAULT_RECORDING_BUFFER_SECONDS = 5.0  # 녹화 프레임 큐 버퍼링 시간(초)
DEFAULT_STOP_THRESHOLD = 5
DEFAULT_RECORD_FPS = 4

# --- 전역 변수 및 공유 객체 ---
MODEL = None
PROCESSOR = None
DEVICE = None
PARAM_DTYPE = None

LATEST_FRAMES_Q = None  # 웹캠 -> VLM 스레드 (frame_np_rgb)
RECORDING_FRAMES_Q = None  # Gradio 스레드 -> 녹화 스레드 (명령 및 BGR 프레임)

STOP_PROCESSING_EVENT = threading.Event()  # 모든 작업 중지용
RECORDING_ACTIVE_FLAG = threading.Event()  # 현재 녹화 세션 활성화 여부 (Gradio 스레드가 설정/해제)

# VIDEO_WRITER는 이제 recording_writer_worker 스레드의 지역 변수로 관리됨
CURRENT_PROMPT_REF = [DEFAULT_PROMPT]
PROMPT_LOCK = threading.Lock()

LAST_VLM_BOOL_RESULT = False
LAST_VLM_RAW_RESPONSE = "시스템 준비 중..."
LAST_VLM_RESULT_LOCK = threading.Lock()

FALSE_DETECTION_COUNT = 0
FALSE_DETECTION_COUNT_LOCK = threading.Lock()


# --- SmolVLM 모델 응답 파싱 헬퍼 함수 ---
def _extract_yes_no(raw_text):
    if raw_text is None: return False
    raw = raw_text.lower().strip()
    if ":" in raw: raw = raw.split(":")[-1].strip()
    raw = re.sub(r"^[ \n\t.:;!-]+", "", raw)
    match = re.match(r"^(yes|no|true|false)", raw, re.IGNORECASE)
    if match:
        answer = match.group(1)
        return answer in ("yes", "true")
    logger.debug(f"SmolVLM 응답에서 yes/no 추출 불가: '{raw_text}'")
    return False


# --- 모델 초기화 함수 ---
def load_smolvlm_model(model_path, gpu_idx):
    global MODEL, PROCESSOR, DEVICE, PARAM_DTYPE
    if MODEL is not None:
        logger.info("SmolVLM 모델이 이미 로드되었습니다.")
        return
    try:
        logger.info(f"SmolVLM 모델 로딩 시작: {model_path}")
        if gpu_idx >= 0 and torch.cuda.is_available():
            DEVICE = torch.device(f"cuda:{gpu_idx}")
            logger.info(f"사용 장치: {DEVICE} (GPU: {torch.cuda.get_device_name(gpu_idx)})")
        else:
            DEVICE = torch.device("cpu")
            logger.info(f"사용 장치: {DEVICE} (CPU)")

        PARAM_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
        PROCESSOR = AutoProcessor.from_pretrained(model_path)
        MODEL = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=PARAM_DTYPE).to(DEVICE)
        MODEL.eval()

        actual_param_dtype = next(MODEL.parameters()).dtype
        if actual_param_dtype != PARAM_DTYPE:
            logger.warning(
                f"요청된 dtype {PARAM_DTYPE}과 실제 로드된 모델 dtype {actual_param_dtype}이 다릅니다. {actual_param_dtype} 사용.")
            PARAM_DTYPE = actual_param_dtype
        logger.info(f"SmolVLM 모델 로드 성공. 최종 파라미터 dtype: {PARAM_DTYPE}, 사용 장치: {MODEL.device}")
    except Exception as e:
        logger.error(f"SmolVLM 모델 ({model_path}) 로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise gr.Error(f"SmolVLM 모델 로드 실패! 로그를 확인하세요.") from e


# --- SmolVLM 모델 분류 함수 ---
def classify_image_with_smolvlm_core(pil_image, text_prompt, model, processor):
    msgs = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": text_prompt}]}]
    try:
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(
                **inputs, do_sample=False, max_new_tokens=20, pad_token_id=processor.tokenizer.pad_token_id
            )
        raw_answer = processor.batch_decode(ids, skip_special_tokens=True)[0]
        detected = _extract_yes_no(raw_answer)
        generated_text_for_log = raw_answer.split("assistant:")[
            -1].strip() if "assistant:" in raw_answer.lower() else raw_answer
        # 로그 레벨 조정: 감지될 때만 INFO, 아니면 DEBUG
        log_level = logging.INFO if detected else logging.DEBUG
        logger.log(log_level,
                   f"SmolVLM 결과: {detected}, 프롬프트: '{text_prompt[:30]}...', 응답: '{generated_text_for_log[:30]}'")
        return detected, generated_text_for_log
    except Exception as e:
        logger.error(f"SmolVLM 분류 중 오류 (프롬프트: '{text_prompt}'): {e}", exc_info=True)
        return False, f"분류 오류: {str(e)}"


# --- VLM 처리 스레드 워커 함수 ---
def vlm_worker(args_namespace):
    global LAST_VLM_BOOL_RESULT, LAST_VLM_RAW_RESPONSE, CURRENT_PROMPT_REF
    logger.info("VLM 처리 스레드 시작됨.")
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not STOP_PROCESSING_EVENT.is_set():
        try:
            frame_np_rgb = LATEST_FRAMES_Q.get(timeout=0.1)
            with PROMPT_LOCK:
                current_prompt = CURRENT_PROMPT_REF[0]
            pil_image = Image.fromarray(frame_np_rgb)
            detected, raw_text = classify_image_with_smolvlm_core(pil_image, current_prompt, MODEL, PROCESSOR)
            with LAST_VLM_RESULT_LOCK:
                LAST_VLM_BOOL_RESULT = detected
                LAST_VLM_RAW_RESPONSE = raw_text
            LATEST_FRAMES_Q.task_done()
            consecutive_errors = 0
        except queue.Empty:
            time.sleep(0.01); continue
        except Exception as e:
            logger.error(f"VLM 워커 루프 내 오류: {e}", exc_info=True)
            consecutive_errors += 1
            with LAST_VLM_RESULT_LOCK:
                LAST_VLM_BOOL_RESULT = False;
                LAST_VLM_RAW_RESPONSE = f"VLM 처리 오류: {str(e)}"
            if LATEST_FRAMES_Q.unfinished_tasks > 0:
                try:
                    LATEST_FRAMES_Q.task_done()
                except ValueError:
                    pass
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"VLM 워커 연속 오류 {max_consecutive_errors}회, 스레드 중지.");
                STOP_PROCESSING_EVENT.set();
                break
            time.sleep(0.5)
    logger.info("VLM 처리 스레드 종료됨.")


# --- 녹화 전용 스레드 워커 함수 ---
def recording_writer_worker(args_namespace):
    logger.info("녹화 쓰기 스레드 시작됨.")
    local_video_writer = None
    active_filename = None  # 현재 녹화중인 파일명 로깅용

    while True:  # 이 스레드는 'SHUTDOWN' 명령으로 종료
        try:
            item = RECORDING_FRAMES_Q.get(timeout=1.0)  # 명령 대기, 타임아웃으로 STOP_PROCESSING_EVENT도 간헐적 체크

            if STOP_PROCESSING_EVENT.is_set() and RECORDING_FRAMES_Q.empty():  # 외부 종료 신호 및 큐 비었으면 종료
                logger.info("외부 종료 신호 및 녹화 큐 비어있음. 녹화 쓰기 스레드 종료 시도.")
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"종료 전 마지막 비디오 파일 저장: {active_filename}")
                    local_video_writer.release()
                local_video_writer = None
                break

            command = item.get('type')
            data = item.get('data')

            if command == 'START':
                if local_video_writer and local_video_writer.isOpened():  # 이전 녹화가 비정상 종료된 경우 대비
                    logger.warning(f"새 녹화 시작 전 이전 VideoWriter 강제 해제: {active_filename}")
                    local_video_writer.release()

                filename = data['filename']
                width = data['width']
                height = data['height']
                fps = data['fps']
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                local_video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                if not local_video_writer.isOpened():
                    logger.error(f"VideoWriter 열기 실패 (녹화 스레드): {filename}")
                    local_video_writer = None  # 실패 시 None으로 설정
                else:
                    active_filename = filename
                    logger.info(f"녹화 시작됨 (녹화 스레드): {active_filename}")

            elif command == 'FRAME':
                if local_video_writer and local_video_writer.isOpened():
                    frame_bgr = data  # 이미 BGR로 변환되어 전달받음
                    local_video_writer.write(frame_bgr)
                # else: logger.warning("녹화 활성화 안됨 또는 VideoWriter 없음. 프레임 무시.") # 너무 빈번할 수 있어 주석 처리

            elif command == 'STOP':
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"녹화 중지 명령 수신, 파일 저장 완료: {active_filename}")
                    local_video_writer.release()
                local_video_writer = None
                active_filename = None

            elif command == 'SHUTDOWN':
                logger.info("녹화 쓰기 스레드 SHUTDOWN 명령 수신.")
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"최종 종료 전 비디오 파일 저장: {active_filename}")
                    local_video_writer.release()
                local_video_writer = None
                RECORDING_FRAMES_Q.task_done()  # 마지막 명령 처리 완료
                break  # 루프 탈출하여 스레드 종료

            RECORDING_FRAMES_Q.task_done()

        except queue.Empty:  # get 타임아웃 발생
            if STOP_PROCESSING_EVENT.is_set():  # 타임아웃 동안 중지 신호가 왔는지 확인
                logger.info("외부 종료 신호 및 타임아웃. 녹화 쓰기 스레드 종료 시도.")
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"종료 전 마지막 비디오 파일 저장 (타임아웃): {active_filename}")
                    local_video_writer.release()
                local_video_writer = None
                break
            continue  # 큐가 비었으면 다시 대기
        except Exception as e:
            logger.error(f"녹화 쓰기 스레드 루프 내 오류: {e}", exc_info=True)
            if local_video_writer and local_video_writer.isOpened():  # 오류 발생 시 현재 파일 닫기 시도
                try:
                    local_video_writer.release()
                except:
                    pass
            local_video_writer = None
            active_filename = None
            if RECORDING_FRAMES_Q.unfinished_tasks > 0:  # 큐 작업 카운터 정리
                try:
                    RECORDING_FRAMES_Q.task_done()
                except ValueError:
                    pass
            time.sleep(0.1)  # 오류 후 잠시 대기

    logger.info("녹화 쓰기 스레드 종료됨.")


# --- Gradio 프레임 처리 및 UI 업데이트 함수 ---
def process_gradio_stream(frame_np_rgb, current_ui_prompt, args_namespace):
    global FALSE_DETECTION_COUNT, RECORDING_ACTIVE_FLAG  # VIDEO_WRITER는 이제 직접 사용 안함
    global LAST_VLM_BOOL_RESULT, LAST_VLM_RAW_RESPONSE, CURRENT_PROMPT_REF
    global LATEST_FRAMES_Q, RECORDING_FRAMES_Q  # 두 큐 모두 사용

    output_frame_for_gradio = frame_np_rgb.copy() if frame_np_rgb is not None else np.zeros((100, 100, 3),
                                                                                            dtype=np.uint8)
    status_text_for_ui = "오류 발생"
    # saved_file_path_for_gradio = None # 녹화 파일명은 이제 녹화 스레드가 관리, UI 표시는 필요시 다른 방식

    if STOP_PROCESSING_EVENT.is_set():  # Gradio 스트림 루프도 중지 신호에 반응
        if RECORDING_ACTIVE_FLAG.is_set():  # 녹화 중이었다면 중지 명령 전송
            logger.info("처리 중지 신호 (스트림), 녹화 중지 명령 전송...")
            if RECORDING_FRAMES_Q:
                try:
                    RECORDING_FRAMES_Q.put_nowait({'type': 'STOP'})
                except queue.Full:
                    logger.error("녹화 중지 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
            RECORDING_ACTIVE_FLAG.clear()  # 즉시 플래그 해제
        status_text_for_ui = "중지됨. '시작' 버튼을 눌러주세요."
        return output_frame_for_gradio, status_text_for_ui, None  # saved_file_path_for_gradio는 None

    if frame_np_rgb is None:
        status_text_for_ui = "웹캠 프레임 없음"
        return output_frame_for_gradio, status_text_for_ui, None

    with PROMPT_LOCK:
        CURRENT_PROMPT_REF[0] = current_ui_prompt

    try:  # VLM 처리용 큐에 프레임 추가
        if LATEST_FRAMES_Q: LATEST_FRAMES_Q.put_nowait(frame_np_rgb.copy())
    except queue.Full:
        logger.warning("VLM 처리 큐 가득 참. 프레임 누락 가능.")
    except AttributeError:
        logger.debug("LATEST_FRAMES_Q 초기화 전 (감지 시작 전).")

    with LAST_VLM_RESULT_LOCK:
        current_vlm_detected = LAST_VLM_BOOL_RESULT

    # 녹화 로직: 명령을 RECORDING_FRAMES_Q에 전달
    if not RECORDING_ACTIVE_FLAG.is_set():
        if current_vlm_detected:  # 이벤트 감지됨
            record_videos_dir = args_namespace.output_dir#os.path.join(args_namespace.output_dir, "videos")  # 녹화 파일명은 여기서 생성
            os.makedirs(record_videos_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(record_videos_dir, f"rec_smolvlm_{timestamp_str}_{uuid.uuid4().hex[:4]}.mp4")
            height, width, _ = frame_np_rgb.shape

            start_command = {
                'type': 'START',
                'data': {'filename': video_filename, 'width': width, 'height': height, 'fps': args_namespace.record_fps}
            }
            if RECORDING_FRAMES_Q:
                try:
                    RECORDING_FRAMES_Q.put_nowait(start_command)
                    RECORDING_ACTIVE_FLAG.set()  # START 명령 성공 시 플래그 설정
                    with FALSE_DETECTION_COUNT_LOCK:
                        FALSE_DETECTION_COUNT = 0
                    logger.info(f"녹화 시작 명령 전송: {video_filename}")
                except queue.Full:
                    logger.error("녹화 시작 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
            else:
                logger.warning("RECORDING_FRAMES_Q 초기화 전, 녹화 시작 불가.")

    if RECORDING_ACTIVE_FLAG.is_set():  # 녹화 플래그가 설정되어 있다면 프레임 전송
        try:
            if RECORDING_FRAMES_Q:
                frame_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                RECORDING_FRAMES_Q.put_nowait({'type': 'FRAME', 'data': frame_bgr})
        except queue.Full:
            logger.warning("녹화 프레임 전송 실패: RECORDING_FRAMES_Q 가득 참. 프레임 누락 가능.")

        # 녹화 중지 조건 판단 (VLM 결과 기반)
        with FALSE_DETECTION_COUNT_LOCK:
            if not current_vlm_detected:
                FALSE_DETECTION_COUNT += 1
            else:
                FALSE_DETECTION_COUNT = 0
            current_false_count = FALSE_DETECTION_COUNT

        if current_false_count >= args_namespace.stop_threshold:
            logger.info(f"미감지 {args_namespace.stop_threshold}회 연속, 녹화 중지 명령 전송.")
            if RECORDING_FRAMES_Q:
                try:
                    RECORDING_FRAMES_Q.put_nowait({'type': 'STOP'})
                except queue.Full:
                    logger.error("녹화 중지 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
            RECORDING_ACTIVE_FLAG.clear()  # 중지 명령 보냈으므로 플래그 해제
            with FALSE_DETECTION_COUNT_LOCK:
                FALSE_DETECTION_COUNT = 0

    with LAST_VLM_RESULT_LOCK:
        active_prompt_display = CURRENT_PROMPT_REF[0]
        status_text_for_ui = f"프롬프트 ('{active_prompt_display[:20]}...'): {LAST_VLM_BOOL_RESULT} (응답: {LAST_VLM_RAW_RESPONSE[:30]})"
    if RECORDING_ACTIVE_FLAG.is_set(): status_text_for_ui += " (녹화 중)"

    return output_frame_for_gradio, status_text_for_ui, None  # saved_file_path는 이제 직접 관리 안함


# --- Gradio 인터페이스 ---
def create_gradio_app(args_namespace):
    global LATEST_FRAMES_Q, RECORDING_FRAMES_Q, STOP_PROCESSING_EVENT, CURRENT_PROMPT_REF
    global FALSE_DETECTION_COUNT, RECORDING_ACTIVE_FLAG
    global LAST_VLM_BOOL_RESULT, LAST_VLM_RAW_RESPONSE, DEVICE  # DEVICE는 여기서 정보 표시용

    vlm_thread_instance = None
    recording_thread_instance = None  # 녹화 스레드 인스턴스용 변수

    logger.info("Gradio 인터페이스 생성 시작")
    # videos_abs_path = os.path.abspath(os.path.join(args_namespace.output_dir, "videos"))
    # os.makedirs(videos_abs_path, exist_ok=True)

    device_info_str = f"{DEVICE.type.upper()}" if DEVICE else "장치 정보 없음"
    if DEVICE and DEVICE.type == "cuda":
        try:
            device_info_str += f":{args_namespace.gpu_idx} ({torch.cuda.get_device_name(args_namespace.gpu_idx)})"
        except:
            device_info_str += f":{args_namespace.gpu_idx}"
    CURRENT_PROMPT_REF[0] = args_namespace.default_prompt

    with gr.Blocks(title="SmolVLM 감지 및 녹화 (v17)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SmolVLM 웹캠 실시간 감지 및 녹화 (v17 - 비디오 쓰기 스레드 분리)")
        gr.Markdown(f"SmolVLM 모델: `{args_namespace.model_path}` | 사용 장치: `{device_info_str}`")
        with gr.Row():  # UI 레이아웃 등은 v15.1과 동일하게 유지
            with gr.Column(scale=2):
                webcam_input = gr.Image(label="웹캠 입력", sources=["webcam"], streaming=True, type="numpy", height=480,
                                        width=480)
            with gr.Column(scale=1):
                prompt_input_ui = gr.Textbox(label="SmolVLM 프롬프트 입력", value=args_namespace.default_prompt, lines=3,
                                             interactive=True)
                status_text_output = gr.Textbox(label="감지 상태 및 VLM 응답", value="대기 중...", lines=4, interactive=False)
                latest_saved_file_ui = gr.File(label="최근 저장된 녹화 파일 (표시용)", interactive=False, visible=False)
                #gr.Markdown(f"녹화 파일 저장 위치: `{videos_abs_path}`")
        with gr.Row():
            start_button = gr.Button("▶ 감지 시작", variant="primary", interactive=True)
            stop_button = gr.Button("■ 감지 중지", interactive=False)

        with gr.Accordion("설정 정보 (실행 시 인자로 전달됨)", open=False):
            gr.Markdown(f"""
            - **VLM 최대 호출**: 활성화 (별도 스레드에서 가능한 빠르게 처리)
            - **프레임 큐 크기 (VLM 입력용)**: `{args_namespace.frame_queue_size}`
            - **녹화 프레임 버퍼링 시간**: `{args_namespace.recording_buffer_seconds}`초 (추정 FPS: `{args_namespace.record_fps}`)
            - **녹화 중지 조건**: 미감지 연속 `{args_namespace.stop_threshold}`회 (VLM 추론 기준)
            - **녹화 비디오 FPS**: `{args_namespace.record_fps}`
            """)

        def start_processing():
            nonlocal vlm_thread_instance, recording_thread_instance
            logger.info("감지/녹화 처리 시작 요청됨")
            STOP_PROCESSING_EVENT.clear()
            RECORDING_ACTIVE_FLAG.clear()

            # VIDEO_WRITER는 이제 녹화 스레드 내부에서 관리되므로 여기서 직접 릴리즈 안함
            with LAST_VLM_RESULT_LOCK:
                LAST_VLM_BOOL_RESULT = False;
                LAST_VLM_RAW_RESPONSE = "처리 시작됨..."
            with FALSE_DETECTION_COUNT_LOCK:
                FALSE_DETECTION_COUNT = 0

            global LATEST_FRAMES_Q, RECORDING_FRAMES_Q
            LATEST_FRAMES_Q = queue.Queue(maxsize=args_namespace.frame_queue_size)
            recording_q_maxsize = int(args_namespace.recording_buffer_seconds * args_namespace.record_fps)
            if recording_q_maxsize <= 0: recording_q_maxsize = args_namespace.record_fps  # 최소 1초 분량
            RECORDING_FRAMES_Q = queue.Queue(maxsize=recording_q_maxsize)
            logger.info(f"녹화 프레임 큐 초기화됨 (최대 크기: {recording_q_maxsize} 프레임)")

            # 이전 스레드들이 있다면 종료 시도
            if vlm_thread_instance and vlm_thread_instance.is_alive():
                logger.warning("이전 VLM 스레드 중지 시도...")
                # STOP_PROCESSING_EVENT는 이미 설정된 상태일 수 있으므로, join만 시도
                vlm_thread_instance.join(timeout=1)
            if recording_thread_instance and recording_thread_instance.is_alive():
                logger.warning("이전 녹화 스레드 중지 시도...")
                if RECORDING_FRAMES_Q: RECORDING_FRAMES_Q.put({'type': 'SHUTDOWN'})  # 종료 명령
                recording_thread_instance.join(timeout=2)  # 파일 I/O가 있을 수 있으므로 좀 더 대기

            STOP_PROCESSING_EVENT.clear()  # 새 스레드들을 위해 다시 명확히 클리어

            vlm_thread_instance = threading.Thread(target=vlm_worker, args=(args_namespace,), daemon=True)
            vlm_thread_instance.start()

            recording_thread_instance = threading.Thread(target=recording_writer_worker, args=(args_namespace,),
                                                         daemon=True)
            recording_thread_instance.start()

            logger.info("VLM 및 녹화 스레드 시작됨. 감지 처리 활성화.")
            return "감지 처리 중...", gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)

        def stop_processing():
            nonlocal vlm_thread_instance, recording_thread_instance
            logger.info("감지/녹화 처리 중지 요청됨")
            STOP_PROCESSING_EVENT.set()
            status_msg = "처리 중지 중..."

            if RECORDING_ACTIVE_FLAG.is_set():  # 녹화 중이었다면 명시적으로 중지 명령
                logger.info("활성 녹화 중지 명령 전송...")
                if RECORDING_FRAMES_Q:
                    try:
                        RECORDING_FRAMES_Q.put_nowait({'type': 'STOP'})
                    except queue.Full:
                        logger.error("STOP 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
                RECORDING_ACTIVE_FLAG.clear()  # 즉시 플래그 해제

            # VLM 스레드 종료
            if vlm_thread_instance and vlm_thread_instance.is_alive():
                logger.info("VLM 스레드 종료 대기 중...")
                vlm_thread_instance.join(timeout=3)
                if vlm_thread_instance.is_alive():
                    logger.warning("VLM 스레드가 타임아웃 내에 종료되지 않음.")
                else:
                    logger.info("VLM 스레드 성공적으로 종료됨.")
            vlm_thread_instance = None

            # 녹화 스레드 종료 (큐에 SHUTDOWN 명령 보내고 대기)
            if recording_thread_instance and recording_thread_instance.is_alive():
                logger.info("녹화 쓰기 스레드에 SHUTDOWN 명령 전송...")
                if RECORDING_FRAMES_Q: RECORDING_FRAMES_Q.put({'type': 'SHUTDOWN'})  # 정상 종료 유도
                recording_thread_instance.join(
                    timeout=max(3, args_namespace.recording_buffer_seconds + 1))  # 버퍼 비울 시간 고려
                if recording_thread_instance.is_alive():
                    logger.warning("녹화 쓰기 스레드가 타임아웃 내에 종료되지 않음.")
                else:
                    logger.info("녹화 쓰기 스레드 성공적으로 종료됨.")
            recording_thread_instance = None

            # VIDEO_WRITER는 녹화 스레드에서 관리하므로 여기서 직접 release하지 않음
            if RECORDING_ACTIVE_FLAG.is_set():  # 이 시점에서는 이미 clear 되었어야 함
                logger.warning("stop_processing 진입 시 RECORDING_ACTIVE_FLAG가 여전히 설정되어 있음.")
                RECORDING_ACTIVE_FLAG.clear()  # 확실히 클리어

            # 남은 큐 정리 (선택적)
            if LATEST_FRAMES_Q:
                while not LATEST_FRAMES_Q.empty():
                    try:
                        LATEST_FRAMES_Q.get_nowait()
                    except queue.Empty:
                        break
            if RECORDING_FRAMES_Q:  # 이미 비워졌거나 SHUTDOWN 처리되었을 가능성 높음
                while not RECORDING_FRAMES_Q.empty():
                    try:
                        RECORDING_FRAMES_Q.get_nowait()
                    except queue.Empty:
                        break

            status_msg = "모든 처리 중지 완료."
            return status_msg, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)

        start_button.click(fn=start_processing, inputs=[],
                           outputs=[status_text_output, start_button, stop_button, prompt_input_ui])
        stop_button.click(fn=stop_processing, inputs=[],
                          outputs=[status_text_output, start_button, stop_button, prompt_input_ui])
        webcam_input.stream(fn=lambda frame, prompt: process_gradio_stream(frame, prompt, args_namespace),
                            inputs=[webcam_input, prompt_input_ui],
                            outputs=[webcam_input, status_text_output, latest_saved_file_ui])
    logger.info("Gradio 인터페이스 생성 완료")
    return demo


# --- 메인 실행 ---
def main():
    parser = argparse.ArgumentParser(description="SmolVLM 웹캠 감지 및 녹화 (v17 - 쓰기 스레드 분리)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="SmolVLM 모델 경로")
    parser.add_argument("--gpu_idx", type=int, default=(0 if torch.cuda.is_available() else -1),
                        help="GPU 인덱스 (-1은 CPU)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR_BASE, help="로그 및 비디오 저장 디렉토리")
    parser.add_argument("--frame_queue_size", type=int, default=DEFAULT_FRAME_QUEUE_SIZE, help="VLM 처리용 프레임 큐 크기")
    parser.add_argument("--recording_buffer_seconds", type=float, default=DEFAULT_RECORDING_BUFFER_SECONDS,
                        help="녹화 프레임 큐 버퍼링 시간(초)")
    parser.add_argument("--default_prompt", type=str, default=DEFAULT_PROMPT, help="기본 SmolVLM 프롬프트")
    parser.add_argument("--stop_threshold", type=int, default=DEFAULT_STOP_THRESHOLD, help="연속 미감지 시 녹화 중지 카운트")
    parser.add_argument("--record_fps", type=int, default=DEFAULT_RECORD_FPS, help="녹화 비디오 FPS")
    args = parser.parse_args()

    # 로그 디렉토리는 output_dir 하위에 생성
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"log_v17_smolvlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler()]
    )
    logger.info(f"애플리케이션 시작 (v17 - SmolVLM, 쓰기 스레드 분리)")
    logger.info(f"--- 실행 인자 ---");
    [logger.info(f"{n}: {v}") for n, v in vars(args).items()]
    logger.info(f"로그 파일: {os.path.abspath(log_file_path)}");
    logger.info(f"-----------------")

    videos_output_dir = args.output_dir #os.path.join(args.output_dir, "videos")
    os.makedirs(videos_output_dir, exist_ok=True)
    logger.info(f"녹화 비디오 저장 경로: {os.path.abspath(videos_output_dir)}")

    try:
        load_smolvlm_model(args.model_path, args.gpu_idx)
    except Exception as e:
        logger.critical(f"모델 로드 실패로 프로그램 종료: {e}"); return

    app_interface = create_gradio_app(args)
    try:
        logger.info("Gradio 애플리케이션 실행...")
        app_interface.launch(share=True, max_threads=32)  # Gradio 내부 스레드 풀 증가
    except Exception as e:
        logger.critical(f"Gradio 앱 실행 중 치명적 오류: {e}", exc_info=True)
    finally:
        logger.info("애플리케이션 종료 절차 시작...")
        STOP_PROCESSING_EVENT.set()

        # 주 스레드 종료 시점에서 백그라운드 스레드들이 완전히 종료되도록 추가 처리 가능
        # 예를 들어, create_gradio_app에서 관리하는 스레드 객체들을 main에서도 참조하여 join 호출
        # 현재는 start/stop 버튼 핸들러에서 스레드 관리가 주로 이루어짐.
        # Gradio 앱 종료 시 stop_processing이 호출되지 않을 수 있으므로,
        # 녹화 스레드에 SHUTDOWN 명령을 보내는 로직이 필요할 수 있음.
        # 여기서는 STOP_PROCESSING_EVENT와 daemon=True 스레드에 의존.
        # 가장 확실한 방법은 stop_processing 함수를 여기서 호출하거나,
        # 해당 함수에서 사용하는 스레드 인스턴스를 main 스코프에서 관리하는 것.
        # 현재는 stop_button 핸들러와 STOP_PROCESSING_EVENT에 의존.

        if RECORDING_FRAMES_Q:  # 녹화 큐가 존재하면 SHUTDOWN 명령 시도
            logger.info("애플리케이션 종료 시 녹화 스레드에 SHUTDOWN 명령 전달 시도")
            try:
                RECORDING_FRAMES_Q.put({'type': 'SHUTDOWN'}, timeout=0.5)
            except queue.Full:
                logger.warning("SHUTDOWN 명령 전달 실패: RECORDING_FRAMES_Q 가득 참")

        # VIDEO_WRITER는 녹화 스레드가 관리하므로 여기서 직접 release 하지 않음.
        # 녹화 스레드가 SHUTDOWN 명령을 받고 자체적으로 정리해야 함.
        if RECORDING_ACTIVE_FLAG.is_set():
            logger.warning("애플리케이션 종료 시 RECORDING_ACTIVE_FLAG가 아직 설정되어 있습니다.")
            # 이 경우 녹화 스레드가 정상적으로 파일을 닫지 못했을 수 있음.

        logger.info("애플리케이션 종료됨.")


if __name__ == "__main__":
    main()