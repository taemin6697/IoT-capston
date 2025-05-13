"""
구글 리뷰 API를 통해 리뷰 데이터를 가져와서 처리하는 모듈
"""
import logging
from typing import Dict, List, Optional

import requests
import pandas as pd


class GoogleReviewManager:
    """
    구글 리뷰 API를 통해 리뷰 데이터를 가져와 텍스트로 저장하고,
    DEFAULT_PROMPT_TEMPLATE에 적용할 리뷰 문자열을 생성하는 클래스
    """

    def __init__(self, url: str, target_review_count: int = 20):
        """
        GoogleReviewManager 초기화
        
        Args:
            url: 구글 리뷰를 가져올 장소 URL
            target_review_count: 가져올 리뷰 개수
        """
        self.url = url
        self.target_review_count = target_review_count
        self.reviews_text = self.fetch_reviews_text()

    def fetch_reviews_text(self) -> str:
        """
        API를 통해 구글 리뷰를 가져오는 함수
        
        Returns:
            str: 가져온 리뷰 텍스트 또는 오류 메시지
        """
        try:
            # API 엔드포인트 URL
            api_url = "http://59.9.11.187:8899/api/get_google_reviews"

            # 요청 파라미터
            params = {
                "url": self.url,
                "count": self.target_review_count
            }

            # API 요청
            response = requests.get(api_url, params=params)

            # 응답 확인
            if response.status_code == 200:
                data = response.json()
                return data.get("reviews_text", "(구글 리뷰를 불러오지 못했습니다.)")
            else:
                logging.error(f"API 요청 실패: {response.status_code}, {response.text}")
                return "(구글 리뷰를 불러오지 못했습니다.)"

        except Exception as e:
            logging.error(f"리뷰 가져오기 중 오류 발생: {e}")
            return "(구글 리뷰를 불러오지 못했습니다.)"

    @staticmethod
    def format_google_reviews(reviews_text: str) -> str:
        """
        리뷰 텍스트를 포맷팅하는 함수
        
        Args:
            reviews_text: 가공되지 않은 리뷰 텍스트
            
        Returns:
            str: 포맷팅된 리뷰 텍스트
        """
        # 각 줄로 분리하고, 이미 "####"가 포함된 줄은 제외하여 순수한 리뷰 내용만 남김
        reviews = [line for line in reviews_text.split("\n") if line.strip() and "####" not in line]
        formatted_reviews = []
        
        for i, review in enumerate(reviews, start=1):
            formatted_reviews.append(f"#### Google Review {i} ####\n{review}")
            
        return "\n\n".join(formatted_reviews)

    def get_reviews_dataframe(self) -> pd.DataFrame:
        """
        리뷰 텍스트를 DataFrame으로 변환하는 함수
        
        Returns:
            pd.DataFrame: 리뷰 데이터를 담은 DataFrame 객체
        """
        if self.reviews_text == "(구글 리뷰를 불러오지 못했습니다.)":
            return pd.DataFrame()

        reviews_data = []
        for line in self.reviews_text.split("\n"):
            if not line.strip():
                continue

            try:
                # 예: "[4.5 stars] Excellent service and food."
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    rating_part = parts[0].strip("[")
                    review_text = parts[1]

                    try:
                        rating = float(rating_part.split()[0])
                    except (ValueError, IndexError):
                        rating = None

                    reviews_data.append({
                        "Name": "Unknown",  # API는 이름을 반환하지 않을 수 있음
                        "Rating": rating,
                        "Date / Time Ago": "Unknown",  # API는 날짜를 반환하지 않을 수 있음
                        "Review Text": review_text
                    })
            except Exception as e:
                logging.error(f"리뷰 파싱 중 오류: {e}")

        return pd.DataFrame(reviews_data) 