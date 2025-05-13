import os
import time
import cv2
import uuid
from transformers import pipeline
from PIL import Image
from collections import deque
import argparse # argparse 라이브러리 추가

def main(args): # main 함수가 인자를 받도록 수정
    # 출력 디렉토리 생성 (인자값 사용)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # SIGLIP 제로샷 분류 파이프라인 초기화
    try:
        classifier = pipeline(
            task="zero-shot-image-classification",
            model="google/siglip2-base-patch16-384",
            device=args.gpu_device # 인자값 사용
        )
    except Exception as e:
        print(f"SIGLIP 파이프라인 초기화 중 오류 발생: {e}")
        print("CPU 또는 다른 GPU 장치 사용을 시도해보세요 (--gpu-device -1 또는 다른 ID).")
        return

    # 후보 레이블 처리 (인자값 사용, 쉼표로 구분된 문자열을 리스트로 변환)
    candidate_labels = [label.strip() for label in args.labels.split(',')]
    threshold = args.siglip_threshold # 인자값 사용

    print("--- 설정 값 ---")
    print(f"출력 디렉토리: {output_dir}")
    print(f"사용 장치 (GPU/CPU): {args.gpu_device}")
    print(f"탐지 레이블: {candidate_labels}")
    print(f"SIGLIP 임계값: {threshold}")
    print(f"카메라 인덱스: {args.camera_index}")

    # 카메라 열기 (인자값 사용)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"카메라 인덱스 {args.camera_index}를 열 수 없습니다.")
        return

    # 녹화 상태 관련 변수
    recording = False
    false_count = 0
    writer = None

    # --- 값 계산 (인자값 기반) ---
    processing_interval = args.interval
    # 초당 프레임 수 (FPS) 계산
    # 처리 간격이 0이 되는 것을 방지
    fps = 1 / processing_interval if processing_interval > 0 else 30
    # 이벤트 전 녹화 시간 (초)
    pre_record_time = args.pre_record
    # 필요한 버퍼 크기 계산 (녹화 시간 / 처리 간격)
    buffer_size = int(pre_record_time / processing_interval) if processing_interval > 0 else int(pre_record_time * 30)
    # 녹화 종료 조건 (연속 False 횟수)
    false_count_threshold = args.stop_threshold_count
    # --------------------------

    # 프레임 버퍼 설정 (계산된 크기 사용)
    frame_buffer = deque(maxlen=buffer_size)

    print(f"처리 간격: {processing_interval:.2f}초")
    print(f"계산된 FPS: {fps:.2f}")
    print(f"버퍼 크기: {buffer_size} (약 {pre_record_time}초 분량)")
    print(f"녹화 종료 임계 횟수: {false_count_threshold}")
    print("-----------------")
    print("실시간 SIGLIP 판별 및 이벤트 전후 녹화를 시작합니다. (종료: 'q')")


    try:
        loop_start_time = time.time() # 루프 시작 시간 기록

        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다. 잠시 후 다시 시도합니다.")
                time.sleep(0.5) # 카메라 문제 시 잠시 대기
                continue

            # 현재 프레임을 버퍼에 추가
            frame_buffer.append(frame.copy())

            # 현재 영상을 화면에 출력
            #cv2.imshow("Live Feed", frame)

            # SIGLIP 판별
            try:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                outputs = classifier(pil_image, candidate_labels=candidate_labels)
                # 가장 높은 점수를 가진 레이블과 점수 찾기 (여러 레이블 비교 시 유용)
                best_output = max(outputs, key=lambda x: x['score'])
                score = best_output['score']
                detected_label = best_output['label']
                result = score >= threshold
                if result:
                    print(f"SIGLIP 결과: True (Label: {detected_label}, Score: {score:.2f})")
                # False일 때는 간결하게 출력하거나 필요시 상세 정보 출력
                else:
                    print(f"SIGLIP 결과: False (Best Score: {score:.2f} for {detected_label})")

            except Exception as e:
                print(f"SIGLIP 판별 중 오류 발생: {e}")
                result = False

            # 녹화 로직
            if not recording:
                if result:
                    filename = os.path.join(output_dir, f"record_{uuid.uuid4().hex}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    height, width, _ = frame.shape
                    try:
                        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                        if not writer.isOpened():
                            raise IOError("VideoWriter를 열 수 없습니다.")

                        print(f"\n탐지! 녹화 시작: {filename}")
                        print(f"버퍼에 저장된 {len(frame_buffer)} 프레임부터 기록 시작...")

                        # 버퍼 프레임 기록
                        for pre_frame in list(frame_buffer):
                            writer.write(pre_frame)

                        recording = True
                        false_count = 0

                    except (IOError, cv2.error) as e:
                        print(f"VideoWriter 생성 또는 쓰기 오류: {e}")
                        if writer is not None and writer.isOpened():
                            writer.release()
                        writer = None
                        recording = False
                        frame_buffer.clear() # 오류 발생 시 버퍼 비우기

            else: # 녹화 중일 때
                try:
                    writer.write(frame)

                    if not result:
                        false_count += 1
                    else:
                        false_count = 0 # 다시 감지되면 카운트 리셋

                    if false_count >= false_count_threshold:
                        print(f"\n미감지 {false_count_threshold}회 연속: 녹화 종료")
                        writer.release()
                        writer = None
                        recording = False
                        false_count = 0

                except cv2.error as e:
                    print(f"녹화 중 프레임 쓰기 오류: {e}")
                    # 오류 발생 시 녹화 중단 및 자원 해제 시도
                    if writer is not None and writer.isOpened():
                        writer.release()
                    writer = None
                    recording = False
                    false_count = 0

            # 'q' 키 종료 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n사용자에 의해 종료됨.")
                break

            # 정확한 간격 제어
            processing_time = time.time() - loop_start_time
            wait_time = processing_interval - processing_time
            if wait_time > 0:
                time.sleep(wait_time)
            # else:
            #     # 처리 시간이 간격보다 길 경우 경고 (선택적)
            #     print(f"경고: 루프 처리 시간이 목표 간격({processing_interval:.2f}s)보다 깁니다 ({processing_time:.2f}s).")

            loop_start_time = time.time() # 다음 루프 시작 시간 기록

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨 (Ctrl+C).")

    finally:
        # 자원 정리
        print("자원 정리 중...")
        if writer is not None and writer.isOpened():
            print("녹화 중인 파일 저장 완료.")
            writer.release()
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("자원 해제 완료.")

if __name__ == '__main__':
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description="실시간 SIGLIP 판별 및 이벤트 전후 녹화 프로그램")

    parser.add_argument('--output-dir', type=str, default='./record_videos',
                        help='녹화된 비디오를 저장할 디렉토리 경로')
    parser.add_argument('--gpu-device', type=int, default=0,
                        help='사용할 GPU 장치 ID (CPU 사용 시 -1)')
    parser.add_argument('--labels', type=str, default='The image shows table service or a restaurant waiter.',
                        help='SIGLIP 분류 레이블 (쉼표로 구분하여 여러 개 지정 가능)')
    parser.add_argument('--siglip-threshold', type=float, default=0.2,
                        help='SIGLIP 분류 결과의 신뢰도 임계값')
    parser.add_argument('--interval', type=float, default=0.2,
                        help='프레임 처리 간격 (초)')
    parser.add_argument('--pre-record', type=float, default=5.0,
                        help='이벤트 탐지 전 녹화할 시간 (초)')
    parser.add_argument('--stop-threshold-count', type=int, default=30,
                        help='녹화를 중지하기 위한 연속 미감지(False) 프레임 횟수')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='사용할 카메라의 인덱스 번호')

    # 인자 파싱
    args = parser.parse_args()

    # 메인 함수 실행
    main(args)
