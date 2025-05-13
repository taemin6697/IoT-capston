"""
다양한 AI 모델에 대한 API 클라이언트 관리 모듈
"""
import base64
import logging
from typing import Any

import openai
from ollama import Client
from google import genai

from utils.config import Config


class ModelClients:
    """
    다양한 AI 모델에 대한 API 클라이언트를 관리하는 클래스
    """

    def __init__(self, config: Config):
        """
        ModelClients 초기화
        
        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config

        # Ollama 클라이언트 (Mistral 모델용)
        self.local_client = Client(host='http://59.9.11.187:11434')

        # OpenAI 클라이언트 (GPT 모델용)
        self.gpt_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Google 클라이언트 (Gemini 모델용)
        self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

        # Qwen 클라이언트 (Alibaba 모델용)
        try:
            from openai import OpenAI as QwenOpenAI
            self.qwen_client = QwenOpenAI(
                api_key=config.QWEN_API_KEY,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
        except ImportError as e:
            logging.error(f"Qwen 클라이언트 초기화 실패: {e}")
            self.qwen_client = None

    def encode_video_qwen(self, video_path: str) -> str:
        """
        Qwen API용 비디오 인코딩
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            str: Base64로 인코딩된 비디오 데이터
        """
        try:
            with open(video_path, "rb") as video_file:
                return base64.b64encode(video_file.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"비디오 인코딩 오류 (Qwen): {e}")
            return "" 