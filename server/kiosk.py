"""
이 파일은 하위 모듈들을 임포트하고 애플리케이션을 실행시키는 메인 파일입니다.
각 컴포넌트는 별도 모듈로 분리되어 있으며, 여기서는 모든 모듈을 조합하여 전체 애플리케이션을 구동합니다.
"""

# pylint: disable=missing-docstring, line-too-long
import logging
import os
import sys
from pathlib import Path

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 내부 모듈 임포트
from models.google_reviews import GoogleReviewManager
from models.model_clients import ModelClients
from models.tip_calculator import TipCalculator
from utils.video_processor import VideoProcessor
from utils.config import Config
from ui.ui_handler import UIHandler


class App:
    """모든 컴포넌트를 연결하는 메인 애플리케이션 클래스"""

    def __init__(self):
        # 기본 디렉토리 설정 확인
        self._ensure_directories()
        
        # 설정 초기화
        self.config = Config()
        
        # 컴포넌트 초기화
        self.model_clients = ModelClients(self.config)
        self.video_processor = VideoProcessor()
        self.tip_calculator = TipCalculator(self.config, self.model_clients, self.video_processor)
        self.ui_handler = UIHandler(self.config, self.tip_calculator, self.video_processor)

    def _ensure_directories(self):
        """필요한 디렉토리 구조 생성"""
        # 서버 디렉토리 절대 경로
        server_dir = os.path.dirname(os.path.abspath(__file__))
        logging.info(f"서버 디렉토리: {server_dir}")
        logging.info(f"현재 작업 디렉토리: {os.getcwd()}")
        
        directories = [
            "images",
            "record_videos",
            "frames_list"
        ]
        for directory in directories:
            dir_path = os.path.join(server_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"디렉토리 생성/확인: {dir_path}")

    def run_gradio(self):
        """Gradio 인터페이스 실행"""
        interface = self.ui_handler.create_gradio_blocks()
        # video 폴더 경로 추가
        video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video")
        # 외부 비디오 경로 추가
        external_video_path = r"C:\Users\tm011\PycharmProjects\video"
        interface.launch(
            share=True, 
            allowed_paths=[video_path, external_video_path],
            show_api=False,  # API 문서 비활성화
            width=1200,      # 고정 너비 설정
            height=800,      # 고정 높이 설정
            #analytics_enabled=False,  # 분석 기능 비활성화
            inbrowser=True,   # 브라우저에서 자동 실행
            max_threads=10    # 최대 스레드 수 증가
        )


def get_recorded_videos():
    """녹화된 비디오 파일 목록 반환"""
    # 서버 디렉토리 절대 경로
    server_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(server_dir, "record_videos")
    
    if not os.path.exists(folder):
        return []
    
    # .avi, .mp4 확장자를 가진 파일만 포함
    video_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f))
                  and f.lower().endswith((".avi", ".mp4"))]
    
    return video_files


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("kiosk.log")
        ]
    )
    
    # 애플리케이션 실행
    app = App()
    app.run_gradio()
