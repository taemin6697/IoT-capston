# -*- coding: utf-8 -*-
"""
SIGLIP과 Gradio를 이용한 웹캠 객체 감지 및 녹화 스크립트 (v17.2_siglip_single_label - 단일 레이블 및 임계값 수정)

이 스크립트는 Gradio를 사용하여 사용자의 웹캠 스트림에서 SIGLIP 모델을 통해
사용자가 UI에서 입력한 단일 레이블에 따라 이미지를 판단하고, 이벤트 발생 시 영상을 녹화합니다.
SIGLIP 추론 및 비디오 쓰기 작업은 각각 별도 스레드에서 실행되며, 사전 녹화 기능은 없습니다.
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
from transformers import pipeline

# --- 로깅 설정 ---
logger = logging.getLogger("SIGLIPSingleLabel_v17_2")  # 로거 이름 변경

# --- 기본 설정값 (argparse를 통해 오버라이드 가능) ---
DEFAULT_SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-384"
DEFAULT_OUTPUT_DIR_BASE = "./record_videos"
DEFAULT_PROMPT = "person"  # 단일 레이블로 변경
DEFAULT_FRAME_QUEUE_SIZE = 2
DEFAULT_RECORDING_BUFFER_SECONDS = 5.0
DEFAULT_STOP_THRESHOLD = 5
DEFAULT_RECORD_FPS = 4
DEFAULT_SIGLIP_CONFIDENCE_THRESHOLD = 0.25  # 임계값 0.25로 변경

# --- 전역 변수 및 공유 객체 ---
SIGLIP_CLASSIFIER_PIPELINE = None
DEVICE = None

LATEST_FRAMES_Q = None
RECORDING_FRAMES_Q = None

STOP_PROCESSING_EVENT = threading.Event()
RECORDING_ACTIVE_FLAG = threading.Event()

CURRENT_PROMPT_REF = [DEFAULT_PROMPT]
PROMPT_LOCK = threading.Lock()

LAST_SIGLIP_BOOL_RESULT = False
LAST_SIGLIP_RAW_RESPONSE = "시스템 준비 중..."
LAST_SIGLIP_RESULT_LOCK = threading.Lock()

FALSE_DETECTION_COUNT = 0
FALSE_DETECTION_COUNT_LOCK = threading.Lock()


# --- 모델 초기화 함수 (SIGLIP 파이프라인 로드) ---
def load_siglip_pipeline(model_name, gpu_idx_param):
    global SIGLIP_CLASSIFIER_PIPELINE, DEVICE
    if SIGLIP_CLASSIFIER_PIPELINE is not None:
        logger.info("SIGLIP 파이프라인이 이미 로드되었습니다.")
        return
    try:
        logger.info(f"SIGLIP 파이프라인 로딩 시작: {model_name}")
        pipeline_device_arg = -1
        if gpu_idx_param >= 0 and torch.cuda.is_available():
            DEVICE = torch.device(f"cuda:{gpu_idx_param}")
            pipeline_device_arg = gpu_idx_param
            logger.info(f"사용 장치: {DEVICE} (GPU: {torch.cuda.get_device_name(gpu_idx_param)})")
        else:
            DEVICE = torch.device("cpu")
            logger.info(f"사용 장치: {DEVICE} (CPU)")

        SIGLIP_CLASSIFIER_PIPELINE = pipeline(
            task="zero-shot-image-classification",
            model=model_name,
            device=pipeline_device_arg
        )
        if hasattr(SIGLIP_CLASSIFIER_PIPELINE.model, 'device'):
            logger.info(f"SIGLIP 모델이 로드된 실제 장치: {SIGLIP_CLASSIFIER_PIPELINE.model.device}")
        logger.info(f"SIGLIP 파이프라인 로드 성공. (모델: {model_name})")
    except Exception as e:
        logger.error(f"SIGLIP 파이프라인 ({model_name}) 로드 중 오류: {e}", exc_info=True)
        raise gr.Error(f"SIGLIP 파이프라인 로드 실패! 로그를 확인하세요.") from e


# --- SIGLIP 모델 분류 함수 ---
def classify_image_with_siglip(pil_image, classifier_pipeline, labels_to_check, confidence_threshold):
    # labels_to_check는 이제 단일 레이블을 포함하는 리스트 (예: ["person"])
    logger.debug(f"SIGLIP 분류 시작 - 레이블: {labels_to_check}, 이미지 크기: {pil_image.size if pil_image else 'N/A'}")
    if not labels_to_check or not labels_to_check[0]:  # 단일 레이블이 비었는지 확인
        logger.warning("분류할 후보 레이블이 없습니다 (단일 레이블 모드).")
        return False, "후보 레이블 없음", 0.0, "후보 레이블 없음"

    try:
        # candidate_labels는 항상 리스트여야 함
        outputs = classifier_pipeline(pil_image, candidate_labels=labels_to_check)
        logger.debug(f"SIGLIP Raw Outputs: {outputs}")

        if not outputs:
            logger.warning("SIGLIP 결과가 비어있습니다.")
            return False, "결과 없음", 0.0, "결과 없음"

        if not isinstance(outputs, list): outputs = [outputs]
        if outputs and isinstance(outputs[0], list): outputs = outputs[0]

        if not outputs or not all(isinstance(item, dict) and 'score' in item and 'label' in item for item in outputs):
            logger.error(f"SIGLIP 출력 형식이 예상과 다릅니다: {outputs}")
            return False, "출력 형식 오류", 0.0, "출력 형식 오류"

        # 단일 레이블이므로, outputs 리스트에는 해당 레이블에 대한 결과만 있거나,
        # 파이프라인이 여전히 여러 레이블을 가정하고 하나의 결과만 반환할 수 있음.
        # 첫번째 결과를 사용 (단일 레이블이므로 결과도 하나일 것으로 예상)
        best_output = outputs[0]
        score = best_output['score']
        # detected_label은 입력된 labels_to_check[0]과 동일해야 함
        detected_label = best_output['label']

        result = score >= confidence_threshold
        raw_response_for_log = f"Label: {detected_label}, Score: {score:.4f}"

        log_level = logging.INFO if result else logging.DEBUG
        logger.log(log_level,
                   f"SIGLIP 감지: {result}, {raw_response_for_log} (InputLabel: {labels_to_check[0]}, Threshold: {confidence_threshold})")
        return result, detected_label, score, raw_response_for_log
    except Exception as e:
        logger.error(f"SIGLIP 분류 중 오류 (레이블: {labels_to_check}): {e}", exc_info=True)
        return False, f"분류 오류: {str(e)}", 0.0, f"분류 오류: {str(e)}"


# --- SIGLIP 처리 스레드 워커 함수 ---
def siglip_classification_worker(args_namespace):
    global LAST_SIGLIP_BOOL_RESULT, LAST_SIGLIP_RAW_RESPONSE, CURRENT_PROMPT_REF, SIGLIP_CLASSIFIER_PIPELINE
    logger.info("SIGLIP 처리 스레드 시작됨.")
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not STOP_PROCESSING_EVENT.is_set():
        try:
            frame_np_rgb = LATEST_FRAMES_Q.get(timeout=0.1)
            with PROMPT_LOCK:
                current_single_label_str = CURRENT_PROMPT_REF[0].strip()

            if not current_single_label_str:  # 입력된 단일 레이블이 비어있으면
                with LAST_SIGLIP_RESULT_LOCK:
                    LAST_SIGLIP_BOOL_RESULT = False
                    LAST_SIGLIP_RAW_RESPONSE = "입력된 단일 레이블이 없습니다."
                LATEST_FRAMES_Q.task_done();
                time.sleep(0.1);
                continue

            # 단일 레이블을 리스트로 감싸서 전달
            labels_to_check = [current_single_label_str]

            pil_image = Image.fromarray(frame_np_rgb)
            detected_bool, _, _, raw_response_str = classify_image_with_siglip(
                pil_image,
                SIGLIP_CLASSIFIER_PIPELINE,
                labels_to_check,
                args_namespace.siglip_confidence_threshold
            )
            with LAST_SIGLIP_RESULT_LOCK:
                LAST_SIGLIP_BOOL_RESULT = detected_bool
                LAST_SIGLIP_RAW_RESPONSE = raw_response_str
            LATEST_FRAMES_Q.task_done();
            consecutive_errors = 0
        except queue.Empty:
            time.sleep(0.01); continue
        except Exception as e:
            logger.error(f"SIGLIP 워커 루프 내 오류: {e}", exc_info=True)
            consecutive_errors += 1
            with LAST_SIGLIP_RESULT_LOCK:
                LAST_SIGLIP_BOOL_RESULT = False;
                LAST_SIGLIP_RAW_RESPONSE = f"SIGLIP 처리 오류: {str(e)}"
            if LATEST_FRAMES_Q.unfinished_tasks > 0:
                try:
                    LATEST_FRAMES_Q.task_done()
                except ValueError:
                    pass
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"SIGLIP 워커 연속 오류 {max_consecutive_errors}회, 스레드 중지.");
                STOP_PROCESSING_EVENT.set();
                break
            time.sleep(0.5)
    logger.info("SIGLIP 처리 스레드 종료됨.")


# --- 녹화 전용 스레드 워커 함수 ---
def recording_writer_worker(args_namespace):
    logger.info("녹화 쓰기 스레드 시작됨.")
    local_video_writer = None
    active_filename = None

    while True:
        try:
            item = RECORDING_FRAMES_Q.get(timeout=1.0)

            if STOP_PROCESSING_EVENT.is_set() and RECORDING_FRAMES_Q.empty():
                logger.info("외부 종료 신호 및 녹화 큐 비어있음. 녹화 쓰기 스레드 종료 시도.")
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"종료 전 마지막 비디오 파일 저장: {active_filename}")
                    local_video_writer.release()
                local_video_writer = None;
                active_filename = None
                break

            command = item.get('type')
            data = item.get('data')

            if command == 'START':
                if local_video_writer and local_video_writer.isOpened():
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
                    local_video_writer = None
                else:
                    active_filename = filename
                    logger.info(f"녹화 시작됨 (녹화 스레드): {active_filename}")

            elif command == 'FRAME':
                if local_video_writer and local_video_writer.isOpened():
                    frame_bgr = data
                    local_video_writer.write(frame_bgr)

            elif command == 'STOP':
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"녹화 중지 명령 수신, 파일 저장 완료: {active_filename}")
                    local_video_writer.release()
                local_video_writer = None;
                active_filename = None

            elif command == 'SHUTDOWN':
                logger.info("녹화 쓰기 스레드 SHUTDOWN 명령 수신.")
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"최종 종료 전 비디오 파일 저장: {active_filename}")
                    local_video_writer.release()
                local_video_writer = None;
                active_filename = None
                if RECORDING_FRAMES_Q.unfinished_tasks > 0: RECORDING_FRAMES_Q.task_done()
                break

            RECORDING_FRAMES_Q.task_done()

        except queue.Empty:
            if STOP_PROCESSING_EVENT.is_set():
                logger.info("외부 종료 신호 및 타임아웃. 녹화 쓰기 스레드 종료 시도.")
                if local_video_writer and local_video_writer.isOpened():
                    logger.info(f"종료 전 마지막 비디오 파일 저장 (타임아웃): {active_filename}")
                    local_video_writer.release()
                local_video_writer = None;
                active_filename = None
                break
            continue
        except Exception as e:
            logger.error(f"녹화 쓰기 스레드 루프 내 오류: {e}", exc_info=True)
            if local_video_writer and local_video_writer.isOpened():
                try:
                    local_video_writer.release()
                except:
                    pass
            local_video_writer = None;
            active_filename = None
            if RECORDING_FRAMES_Q.unfinished_tasks > 0:
                try:
                    RECORDING_FRAMES_Q.task_done()
                except ValueError:
                    pass
            time.sleep(0.1)
    logger.info("녹화 쓰기 스레드 종료됨.")


# --- Gradio 프레임 처리 및 UI 업데이트 함수 ---
def process_gradio_stream(frame_np_rgb, current_ui_prompt, args_namespace):
    global FALSE_DETECTION_COUNT, RECORDING_ACTIVE_FLAG
    global LAST_SIGLIP_BOOL_RESULT, LAST_SIGLIP_RAW_RESPONSE, CURRENT_PROMPT_REF
    global LATEST_FRAMES_Q, RECORDING_FRAMES_Q

    output_frame_for_gradio = frame_np_rgb.copy() if frame_np_rgb is not None else np.zeros((100, 100, 3),
                                                                                            dtype=np.uint8)
    status_text_for_ui = "오류 발생"

    if STOP_PROCESSING_EVENT.is_set():
        if RECORDING_ACTIVE_FLAG.is_set():
            logger.info("처리 중지 신호 (스트림), 녹화 중지 명령 전송...")
            if RECORDING_FRAMES_Q:
                try:
                    RECORDING_FRAMES_Q.put_nowait({'type': 'STOP'})
                except queue.Full:
                    logger.error("녹화 중지 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
            RECORDING_ACTIVE_FLAG.clear()
        status_text_for_ui = "중지됨. '시작' 버튼을 눌러주세요."
        return output_frame_for_gradio, status_text_for_ui, None

    if frame_np_rgb is None:
        status_text_for_ui = "웹캠 프레임 없음"
        return output_frame_for_gradio, status_text_for_ui, None

    with PROMPT_LOCK:
        CURRENT_PROMPT_REF[0] = current_ui_prompt

    try:
        if LATEST_FRAMES_Q: LATEST_FRAMES_Q.put_nowait(frame_np_rgb.copy())
    except queue.Full:
        logger.warning("SIGLIP 처리 큐 가득 참. 프레임 누락 가능.")
    except AttributeError:
        logger.debug("LATEST_FRAMES_Q 초기화 전 (감지 시작 전).")

    with LAST_SIGLIP_RESULT_LOCK:
        current_siglip_detected = LAST_SIGLIP_BOOL_RESULT

    if not RECORDING_ACTIVE_FLAG.is_set():
        if current_siglip_detected:
            record_videos_dir = args_namespace.output_dir #os.path.join(args_namespace.output_dir, "videos")
            os.makedirs(record_videos_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            detected_label_in_filename = "event"
            with LAST_SIGLIP_RESULT_LOCK:
                temp_raw_response = LAST_SIGLIP_RAW_RESPONSE
            if "Label:" in temp_raw_response:
                try:
                    label_part = temp_raw_response.split("Label:")[1].split(",")[0].strip()
                    detected_label_in_filename = re.sub(r'\W+', '', label_part)[:10]
                except:
                    pass

            video_filename = os.path.join(record_videos_dir,
                                          f"rec_siglip_{timestamp_str}_{detected_label_in_filename}_{uuid.uuid4().hex[:4]}.mp4")
            height, width, _ = frame_np_rgb.shape

            start_command = {
                'type': 'START',
                'data': {'filename': video_filename, 'width': width, 'height': height, 'fps': args_namespace.record_fps}
            }
            if RECORDING_FRAMES_Q:
                try:
                    RECORDING_FRAMES_Q.put_nowait(start_command)
                    RECORDING_ACTIVE_FLAG.set()
                    with FALSE_DETECTION_COUNT_LOCK:
                        FALSE_DETECTION_COUNT = 0
                    logger.info(f"녹화 시작 명령 전송: {video_filename}")
                except queue.Full:
                    logger.error("녹화 시작 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
            else:
                logger.warning("RECORDING_FRAMES_Q 초기화 전, 녹화 시작 불가.")

    if RECORDING_ACTIVE_FLAG.is_set():
        try:
            if RECORDING_FRAMES_Q:
                frame_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                RECORDING_FRAMES_Q.put_nowait({'type': 'FRAME', 'data': frame_bgr})
        except queue.Full:
            logger.warning("녹화 프레임 전송 실패: RECORDING_FRAMES_Q 가득 참. 프레임 누락 가능.")

        with FALSE_DETECTION_COUNT_LOCK:
            if not current_siglip_detected:
                FALSE_DETECTION_COUNT += 1
            else:
                FALSE_DETECTION_COUNT = 0
            current_false_count = FALSE_DETECTION_COUNT

        if current_false_count >= args_namespace.stop_threshold:
            logger.info(f"미감지 {args_namespace.stop_threshold}회 연속, 녹화 중지 명령 전송 (SIGLIP).")
            if RECORDING_FRAMES_Q:
                try:
                    RECORDING_FRAMES_Q.put_nowait({'type': 'STOP'})
                except queue.Full:
                    logger.error("녹화 중지 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
            RECORDING_ACTIVE_FLAG.clear()
            with FALSE_DETECTION_COUNT_LOCK:
                FALSE_DETECTION_COUNT = 0

    with LAST_SIGLIP_RESULT_LOCK:
        active_prompt_display = CURRENT_PROMPT_REF[0]
        status_text_for_ui = f"레이블 ('{active_prompt_display[:20]}...'): {LAST_SIGLIP_BOOL_RESULT} ({LAST_SIGLIP_RAW_RESPONSE})"
    if RECORDING_ACTIVE_FLAG.is_set(): status_text_for_ui += " (녹화 중)"

    return output_frame_for_gradio, status_text_for_ui, None


# --- Gradio 인터페이스 ---
def create_gradio_app(args_namespace):
    global LATEST_FRAMES_Q, RECORDING_FRAMES_Q, STOP_PROCESSING_EVENT, CURRENT_PROMPT_REF
    global FALSE_DETECTION_COUNT, RECORDING_ACTIVE_FLAG
    global LAST_SIGLIP_BOOL_RESULT, LAST_SIGLIP_RAW_RESPONSE, DEVICE

    siglip_thread_instance = None
    recording_thread_instance = None

    logger.info("Gradio 인터페이스 생성 시작 (SIGLIP - 단일 레이블 모드)")  # 모드 명시
    # videos_abs_path = os.path.abspath(os.path.join(args_namespace.output_dir, "videos"))
    # os.makedirs(videos_abs_path, exist_ok=True)

    device_info_str = f"{DEVICE.type.upper()}" if DEVICE else "장치 정보 없음"
    if DEVICE and DEVICE.type == "cuda":
        try:
            device_info_str += f":{args_namespace.gpu_idx} ({torch.cuda.get_device_name(args_namespace.gpu_idx)})"
        except:
            device_info_str += f":{args_namespace.gpu_idx}"
    CURRENT_PROMPT_REF[0] = args_namespace.default_prompt

    with gr.Blocks(title="SIGLIP 단일 레이블 감지 및 녹화 (v17.2)", theme=gr.themes.Soft()) as demo:  # 타이틀 변경
        gr.Markdown("# SIGLIP 웹캠 실시간 단일 레이블 기반 감지 및 녹화 (v17.2)")  # Markdown 변경
        gr.Markdown(
            f"SIGLIP 모델: `{args_namespace.siglip_model_name}` | 사용 장치: `{device_info_str}` | 기본 신뢰도: `{args_namespace.siglip_confidence_threshold}`")  # 신뢰도 정보 추가

        with gr.Row():
            with gr.Column(scale=2):
                webcam_input = gr.Image(label="웹캠 입력", sources=["webcam"], streaming=True, type="numpy")#, height=480,width=480)
            with gr.Column(scale=1):
                prompt_input_ui = gr.Textbox(
                    label="SIGLIP 단일 감지 레이블 입력",  # UI 텍스트 변경
                    value=args_namespace.default_prompt,
                    lines=1,  # 단일 레이블이므로 줄 수 1로 변경
                    placeholder="예: person",  # 플레이스홀더 변경
                    interactive=True
                )
                status_text_output = gr.Textbox(
                    label="감지 상태 및 SIGLIP 최고 점수 정보",
                    value="대기 중...",
                    lines=4,
                    interactive=False
                )
                latest_saved_file_ui = gr.File(label="최근 저장된 녹화 파일 (표시용)", interactive=False, visible=False)
                #gr.Markdown(f"녹화 파일 저장 위치: `{videos_abs_path}`")
        with gr.Row():
            start_button = gr.Button("▶ 감지 시작", variant="primary", interactive=True)
            stop_button = gr.Button("■ 감지 중지", interactive=False)

        with gr.Accordion("설정 정보 (실행 시 인자로 전달됨)", open=False):
            gr.Markdown(f"""
            - **SIGLIP 최대 호출**: 활성화 (별도 스레드에서 가능한 빠르게 처리)
            - **SIGLIP 신뢰도 임계값**: `{args_namespace.siglip_confidence_threshold}`
            - **프레임 큐 크기 (SIGLIP 입력용)**: `{args_namespace.frame_queue_size}`
            - **녹화 프레임 버퍼링 시간**: `{args_namespace.recording_buffer_seconds}`초 (추정 FPS: `{args_namespace.record_fps}`)
            - **녹화 중지 조건**: 미감지 연속 `{args_namespace.stop_threshold}`회 (SIGLIP 추론 기준)
            - **녹화 비디오 FPS**: `{args_namespace.record_fps}`
            """)

        def start_processing():
            nonlocal siglip_thread_instance, recording_thread_instance
            logger.info("감지/녹화 처리 시작 요청됨 (SIGLIP - 단일 레이블 모드)")
            STOP_PROCESSING_EVENT.clear()
            RECORDING_ACTIVE_FLAG.clear()

            with LAST_SIGLIP_RESULT_LOCK:
                LAST_SIGLIP_BOOL_RESULT = False;
                LAST_SIGLIP_RAW_RESPONSE = "처리 시작됨 (SIGLIP)..."
            with FALSE_DETECTION_COUNT_LOCK:
                FALSE_DETECTION_COUNT = 0

            global LATEST_FRAMES_Q, RECORDING_FRAMES_Q
            LATEST_FRAMES_Q = queue.Queue(maxsize=args_namespace.frame_queue_size)
            recording_q_maxsize = int(args_namespace.recording_buffer_seconds * args_namespace.record_fps)
            if recording_q_maxsize <= 0: recording_q_maxsize = max(1, args_namespace.record_fps)
            RECORDING_FRAMES_Q = queue.Queue(maxsize=recording_q_maxsize)
            logger.info(f"녹화 프레임 큐 초기화됨 (최대 크기: {recording_q_maxsize} 프레임)")

            if siglip_thread_instance and siglip_thread_instance.is_alive():
                logger.warning("이전 SIGLIP 스레드 중지 시도...")
                siglip_thread_instance.join(timeout=1)
            if recording_thread_instance and recording_thread_instance.is_alive():
                logger.warning("이전 녹화 스레드 중지 시도...")
                if RECORDING_FRAMES_Q: RECORDING_FRAMES_Q.put({'type': 'SHUTDOWN'})
                recording_thread_instance.join(timeout=2)

            STOP_PROCESSING_EVENT.clear()

            siglip_thread_instance = threading.Thread(target=siglip_classification_worker, args=(args_namespace,),
                                                      daemon=True)
            siglip_thread_instance.start()

            recording_thread_instance = threading.Thread(target=recording_writer_worker, args=(args_namespace,),
                                                         daemon=True)
            recording_thread_instance.start()

            logger.info("SIGLIP 및 녹화 스레드 시작됨. 감지 처리 활성화.")
            return "감지 처리 중 (SIGLIP)...", gr.update(interactive=False), gr.update(interactive=True), gr.update(
                interactive=False)

        def stop_processing():
            nonlocal siglip_thread_instance, recording_thread_instance
            logger.info("감지/녹화 처리 중지 요청됨 (SIGLIP - 단일 레이블 모드)")
            STOP_PROCESSING_EVENT.set()
            status_msg = "처리 중지 중 (SIGLIP)..."

            if RECORDING_ACTIVE_FLAG.is_set():
                logger.info("활성 녹화 중지 명령 전송 (SIGLIP)...")
                if RECORDING_FRAMES_Q:
                    try:
                        RECORDING_FRAMES_Q.put_nowait({'type': 'STOP'})
                    except queue.Full:
                        logger.error("STOP 명령 전송 실패: RECORDING_FRAMES_Q 가득 참")
                RECORDING_ACTIVE_FLAG.clear()

            if siglip_thread_instance and siglip_thread_instance.is_alive():
                logger.info("SIGLIP 스레드 종료 대기 중...")
                siglip_thread_instance.join(timeout=3)
                if siglip_thread_instance.is_alive():
                    logger.warning("SIGLIP 스레드가 타임아웃 내에 종료되지 않음.")
                else:
                    logger.info("SIGLIP 스레드 성공적으로 종료됨.")
            siglip_thread_instance = None

            if recording_thread_instance and recording_thread_instance.is_alive():
                logger.info("녹화 쓰기 스레드에 SHUTDOWN 명령 전송...")
                if RECORDING_FRAMES_Q: RECORDING_FRAMES_Q.put({'type': 'SHUTDOWN'})
                recording_thread_instance.join(timeout=max(3, int(args_namespace.recording_buffer_seconds) + 1))
                if recording_thread_instance.is_alive():
                    logger.warning("녹화 쓰기 스레드가 타임아웃 내에 종료되지 않음.")
                else:
                    logger.info("녹화 쓰기 스레드 성공적으로 종료됨.")
            recording_thread_instance = None

            if RECORDING_ACTIVE_FLAG.is_set():
                logger.warning("stop_processing 진입 시 RECORDING_ACTIVE_FLAG가 여전히 설정되어 있음.")
                RECORDING_ACTIVE_FLAG.clear()

            if LATEST_FRAMES_Q:
                while not LATEST_FRAMES_Q.empty():
                    try:
                        LATEST_FRAMES_Q.get_nowait()
                    except queue.Empty:
                        break
            if RECORDING_FRAMES_Q:
                while not RECORDING_FRAMES_Q.empty():
                    try:
                        RECORDING_FRAMES_Q.get_nowait()
                    except queue.Empty:
                        break

            status_msg = "모든 처리 중지 완료 (SIGLIP)."
            return status_msg, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)

        start_button.click(fn=start_processing, inputs=[],
                           outputs=[status_text_output, start_button, stop_button, prompt_input_ui])
        stop_button.click(fn=stop_processing, inputs=[],
                          outputs=[status_text_output, start_button, stop_button, prompt_input_ui])
        webcam_input.stream(fn=lambda frame, prompt: process_gradio_stream(frame, prompt, args_namespace),
                            inputs=[webcam_input, prompt_input_ui],
                            outputs=[webcam_input, status_text_output, latest_saved_file_ui])
    logger.info("Gradio 인터페이스 생성 완료 (SIGLIP - 단일 레이블 모드)")
    return demo


# --- 메인 실행 ---
def main():
    parser = argparse.ArgumentParser(description="SIGLIP 웹캠 단일 레이블 감지 및 녹화 (v17.2)")
    parser.add_argument("--siglip_model_name", type=str, default=DEFAULT_SIGLIP_MODEL_NAME, help="SIGLIP 모델 이름 또는 경로")
    parser.add_argument("--siglip_confidence_threshold", type=float, default=DEFAULT_SIGLIP_CONFIDENCE_THRESHOLD,
                        help="SIGLIP 감지 신뢰도 임계값")

    parser.add_argument("--gpu_idx", type=int, default=(0 if torch.cuda.is_available() else -1),
                        help="GPU 인덱스 (-1은 CPU)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR_BASE, help="로그 및 비디오 저장 디렉토리")
    parser.add_argument("--frame_queue_size", type=int, default=DEFAULT_FRAME_QUEUE_SIZE, help="모델 입력용 프레임 큐 크기")
    parser.add_argument("--recording_buffer_seconds", type=float, default=DEFAULT_RECORDING_BUFFER_SECONDS,
                        help="녹화 프레임 큐 버퍼링 시간(초)")
    parser.add_argument("--default_prompt", type=str, default=DEFAULT_PROMPT, help="기본 SIGLIP 단일 감지 레이블")  # 도움말 수정
    parser.add_argument("--stop_threshold", type=int, default=DEFAULT_STOP_THRESHOLD, help="연속 미감지 시 녹화 중지 카운트")
    parser.add_argument("--record_fps", type=int, default=DEFAULT_RECORD_FPS, help="녹화 비디오 FPS")
    args = parser.parse_args()

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir,
                                 f"log_v17.2_siglip_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")  # 로그 파일명 변경

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler()]
    )
    logger.info(f"애플리케이션 시작 (v17.2 - SIGLIP 단일 레이블, 쓰기 스레드 분리)")  # 버전 정보 업데이트
    logger.info(f"--- 실행 인자 ---");
    [logger.info(f"{n}: {v}") for n, v in vars(args).items()]
    logger.info(f"로그 파일: {os.path.abspath(log_file_path)}");
    logger.info(f"-----------------")

    videos_output_dir = args.output_dir#os.path.join(args.output_dir, "videos")
    os.makedirs(videos_output_dir, exist_ok=True)
    logger.info(f"녹화 비디오 저장 경로: {os.path.abspath(videos_output_dir)}")

    try:
        load_siglip_pipeline(args.siglip_model_name, args.gpu_idx)
    except Exception as e:
        logger.critical(f"SIGLIP 파이프라인 로드 실패로 프로그램 종료: {e}");
        return

    app_interface = create_gradio_app(args)
    try:
        logger.info("Gradio 애플리케이션 실행 (SIGLIP - 단일 레이블 모드)...")
        app_interface.launch(share=True, max_threads=32)
    except Exception as e:
        logger.critical(f"Gradio 앱 실행 중 치명적 오류: {e}", exc_info=True)
    finally:
        logger.info("애플리케이션 종료 절차 시작 (SIGLIP - 단일 레이블 모드)...")
        STOP_PROCESSING_EVENT.set()

        if RECORDING_FRAMES_Q:
            logger.info("애플리케이션 종료 시 녹화 스레드에 SHUTDOWN 명령 전달 시도")
            try:
                RECORDING_FRAMES_Q.put({'type': 'SHUTDOWN'}, timeout=0.5)
            except queue.Full:
                logger.warning("SHUTDOWN 명령 전달 실패: RECORDING_FRAMES_Q 가득 참")

        if RECORDING_ACTIVE_FLAG.is_set():
            logger.warning("애플리케이션 종료 시 RECORDING_ACTIVE_FLAG가 아직 설정되어 있습니다.")

        logger.info("애플리케이션 종료됨 (SIGLIP - 단일 레이블 모드).")


if __name__ == "__main__":
    main()