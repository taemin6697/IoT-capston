"""
이 파일은 하위 모듈들을 임포트하고 애플리케이션을 실행시키는 메인 파일입니다.
각 컴포넌트는 별도 모듈로 분리되어 있으며, 여기서는 모든 모듈을 조합하여 전체 애플리케이션을 구동합니다.
"""

# pylint: disable=missing-docstring, line-too-long
import logging
import os
import sys
# from pathlib import Path # Path 객체는 현재 파일에서 사용되지 않음

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 내부 모듈 임포트
# from models.google_reviews import GoogleReviewManager # Config 모듈에서 직접 임포트하여 사용
# from models.model_clients import ModelClients # TipCalculator 모듈에서 직접 임포트하여 사용
from models.tip_calculator import TipCalculator
from utils.video_processor import VideoProcessor
from utils.config import Config
from ui.ui_handler import UIHandler


class App:
    """모든 컴포넌트를 연결하는 메인 애플리케이션 클래스"""

    def __init__(self):
        """
        App 클래스 초기화.
        필수 디렉토리를 확인/생성하고, 모든 주요 컴포넌트(설정, 모델 클라이언트,
        비디오 프로세서, 팁 계산기, UI 핸들러)를 초기화합니다.
        """
        # 기본 디렉토리 설정 확인
        self._ensure_directories()
        
        # 설정 초기화
        self.config = Config()
        
        # 컴포넌트 초기화
        self.model_clients = ModelClients(self.config)
        self.video_processor = VideoProcessor()
        self.tip_calculator = TipCalculator(self.config, self.model_clients, self.video_processor)

        # UI 핸들러에 전달할 녹화된 비디오 목록 가져오기
        self.recorded_videos_list = get_recorded_videos() # kiosk.py에 정의된 함수 사용
        logging.info(f"초기 녹화된 비디오 목록: {self.recorded_videos_list}")

        self.ui_handler = UIHandler(
            self.config,
            self.tip_calculator,
            self.video_processor,
            self.recorded_videos_list
        )

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

        # 프로젝트 루트의 'video' 폴더 경로 (예제 비디오 등)
        project_root_video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video")

        # 서버 디렉토리 내의 'record_videos' 폴더 경로 (녹화된 비디오 저장)
        # _ensure_directories에서 server_dir = os.path.dirname(os.path.abspath(__file__)) 로 정의됨
        server_dir = os.path.dirname(os.path.abspath(__file__))
        recorded_videos_path = os.path.join(server_dir, "record_videos")

        # 허용된 경로 목록: 프로젝트 루트의 video 폴더와 서버 내 record_videos 폴더
        # 사용자의 로컬 Windows 경로는 제거되었습니다.
        current_allowed_paths = [project_root_video_path, recorded_videos_path]
        logging.info(f"Gradio allowed_paths: {current_allowed_paths}")

        interface.launch(
            share=True, 
            allowed_paths=current_allowed_paths,
            show_api=False,  # API 문서 비활성화
            width=1200,      # 고정 너비 설정
            height=800,      # 고정 높이 설정
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
