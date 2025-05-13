# -*- coding: utf-8 -*-
"""
Real-time SmolVLM2 image understanding + pre/post-event recorder.
Adapted from the previous SIGLIP version.

Fix history
-----------
* 2025-04-27: initial SmolVLM2 port.
* 2025-04-27 pm: dtype bug fixed (torch.bfloat16 attribute error).
* 2025-04-27 pm2: robust answer parsing.
* 2025-04-27 pm3: video speed patch (writer FPS).
* 2025-04-27 pm4: **SyntaxError patch** – fixed unterminated f-string.
"""

import os
import time
import cv2
import uuid
import re
from collections import deque
from PIL import Image
import argparse
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# -----------------------------------------------------------------------------
# Detection helper
# -----------------------------------------------------------------------------

def _extract_yes_no(raw: str) -> str:
    raw = raw.lower().strip()
    if "assistant:" in raw:
        raw = raw.split("assistant:")[-1].strip()
    raw = raw.lstrip("\n \t.:;-")
    return re.split(r"[\s\n\r\t\.:;!?]", raw, 1)[0]


def smolvlm2_detect(frame, labels, model, processor, dtype):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for lbl in labels:
        prompt = f"Does this image show {lbl}? Answer yes or no."
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,return_tensors="pt"
        ).to(model.device, dtype=dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=8)
        raw = processor.batch_decode(ids, skip_special_tokens=True)[0]
        if _extract_yes_no(raw) in ("yes", "true"):
            return True, lbl, raw
    return False, None, raw


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = f"cuda:{args.gpu_device}" if args.gpu_device >= 0 and torch.cuda.is_available() else "cpu"
    dtype_req = torch.bfloat16 if "cuda" in device else torch.float32

    print("Loading model…")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=dtype_req).to(device)
    param_dtype = next(model.parameters()).dtype
    labels = [s.strip() for s in args.labels.split(",")]

    # Camera
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {args.camera_index} 열기 실패")
    cam_fps = cap.get(cv2.CAP_PROP_FPS)

    # Writer FPS
    loop_fps = 1 / args.interval if args.interval > 0 else 30
    writer_fps = cam_fps if cam_fps and cam_fps > 0 and cam_fps < loop_fps + 1e-3 else loop_fps

    # Buffers
    buf_frames = int(args.pre_record / args.interval) if args.interval > 0 else int(args.pre_record * writer_fps)
    frame_buf = deque(maxlen=buf_frames)

    # ---- Console summary ----
    print("--- 설정 ---")
    cam_fps_str = f"{cam_fps:.2f}" if cam_fps and cam_fps > 0 else "0"
    print(f"Writer FPS           : {writer_fps:.2f} (cam={cam_fps_str})")
    print(f"처리 간격            : {args.interval:.2f}s (≈ {loop_fps:.2f} fps)")
    print(f"선행 버퍼            : {buf_frames} frames (~{args.pre_record}s)\n")

    recording = False
    false_cnt = 0
    writer = None
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            frame_buf.append(frame.copy())
            cv2.imshow("Live", frame)

            try:
                detected, label, raw_ans = smolvlm2_detect(frame, labels, model, processor, param_dtype)
                print(label, raw_ans)
            except Exception as e:
                print(f"검출 오류: {e}")
                detected = False

            # ---- Recording ----
            if not recording and detected:
                path = os.path.join(args.output_dir, f"rec_{uuid.uuid4().hex}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                h, w, _ = frame.shape
                writer = cv2.VideoWriter(path, fourcc, writer_fps, (w, h))
                if writer.isOpened():
                    for bf in frame_buf:
                        writer.write(bf)
                    recording = True
                    false_cnt = 0
                    print(f"🔴 REC start → {path}")
            elif recording:
                writer.write(frame)
                false_cnt = 0 if detected else false_cnt + 1
                if false_cnt >= args.stop_threshold_count:
                    writer.release()
                    recording = False
                    false_cnt = 0
                    print("⏹ REC stop")

            # Exit & pacing
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            dt = time.time() - t0
            if (sleep := args.interval - dt) > 0:
                time.sleep(sleep)
            t0 = time.time()

    finally:
        if writer and writer.isOpened():
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("SmolVLM2 real-time recorder")
    p.add_argument("--model-path", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    p.add_argument("--output-dir", default="./record_videos")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--labels", default="The image shows table service or a restaurant waiter.")
    p.add_argument("--interval", type=float, default=0.3)
    p.add_argument("--pre-record", type=float, default=5.0)
    p.add_argument("--stop-threshold-count", type=int, default=20)
    p.add_argument("--camera-index", type=int, default=0)
    main(p.parse_args())