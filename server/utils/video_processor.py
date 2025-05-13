"""
비디오 처리 및 프레임 추출 기능을 제공하는 모듈
"""
import os
import shutil
import logging
import uuid
from typing import List, Tuple, Optional, Union

import cv2


class VideoProcessor:
    """
    비디오 처리 및 프레임 추출 기능을 제공하는 클래스
    """

    def extract_video_frames(self, video_path: str, output_folder: Optional[str] = None, fps: int = 1) -> Tuple[List[str], Optional[str]]:
        """
        비디오 파일에서 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_folder: 프레임 저장 폴더 경로 (None이면 자동 생성)
            fps: 초당 추출할 프레임 수
            
        Returns:
            Tuple[List[str], Optional[str]]: 추출된 프레임 파일 경로 리스트와 출력 폴더 경로
        """
        if not video_path:
            return [], None

        if output_folder is None:
            output_folder = f"frames_list/frames_{uuid.uuid4().hex}"

        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error(f"오류: 비디오 파일을 열 수 없습니다 - {video_path}")
            return [], None

        frame_paths = []
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        if frame_rate is None or frame_rate == 0:
            logging.warning("경고: FPS를 읽을 수 없습니다, 기본값 4으로 설정합니다.")
            frame_rate = 4.0

        frame_interval = int(frame_rate / fps) if fps > 0 else 1
        if frame_interval <= 0:
            frame_interval = 1

        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                logging.warning(f"경고: {frame_count}번째 프레임이 비어있습니다.")
                frame_count += 1
                continue

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
                try:
                    if cv2.imwrite(frame_path, frame):
                        frame_paths.append(frame_path)
                        saved_frame_count += 1
                    else:
                        logging.warning(f"경고: {frame_path} 저장 실패.")
                except Exception as e:
                    logging.warning(f"경고: 프레임 저장 오류 ({frame_path}): {e}")

            frame_count += 1

        cap.release()

        if not frame_paths:
            logging.warning("경고: 프레임 추출 실패.")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            return [], None

        return frame_paths, output_folder

    def cleanup_temp_files(self, video_path: Union[str, List[str], None], frame_folder: Optional[str]) -> None:
        """
        임시 비디오 파일 및 프레임 폴더 정리
        
        Args:
            video_path: 단일 비디오 경로 또는 비디오 경로 리스트
            frame_folder: 프레임 폴더 경로
        """
        if video_path:
            # video_path가 리스트면 각 항목을 순회하여 삭제
            if isinstance(video_path, list):
                for vp in video_path:
                    self._remove_temp_video(vp)
            else:
                self._remove_temp_video(video_path)

        if frame_folder and os.path.exists(frame_folder):
            try:
                shutil.rmtree(frame_folder)
                logging.info(f"프레임 폴더 삭제: {frame_folder}")
            except OSError as e:
                logging.error(f"프레임 폴더 삭제 오류: {e}")
    
    def _remove_temp_video(self, video_path: str) -> None:
        """
        임시 비디오 파일 삭제
        
        Args:
            video_path: 비디오 파일 경로
        """
        if video_path and "temp_video_" in video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logging.info(f"임시 비디오 파일 삭제: {video_path}")
            except OSError as e:
                logging.error(f"임시 비디오 파일 삭제 오류: {e}") 