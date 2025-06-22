"""
팁 계산 핵심 로직 모듈
"""
import os
import re
import json
import time
import logging
import base64
import requests
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config import Config
from models.model_clients import ModelClients
from utils.video_processor import VideoProcessor


class TipCalculator:
    """팁 계산 핵심 로직을 담당하는 클래스"""

    def __init__(self, config: Config, model_clients: ModelClients, video_processor: VideoProcessor):
        """
        TipCalculator 초기화
        
        Args:
            config: 애플리케이션 설정 객체
            model_clients: AI 모델 클라이언트 객체
            video_processor: 비디오 처리 객체
        """
        self.config = config
        self.model_clients = model_clients
        self.video_processor = video_processor

    def parse_llm_output(self, output_text: str) -> Tuple[str, float, float, str]:
        """
        LLM 출력을 파싱하여 팁 정보 추출
        
        Args:
            output_text: LLM이 생성한 텍스트 출력
            
        Returns:
            Tuple[str, float, float, str]: 분석 결과, 팁 퍼센트, 팁 금액, 원본 출력 텍스트
        """
        # 원본 전체 출력 텍스트 보존
        formatted_output = output_text
        
        # 이스케이프된 줄바꿈 문자 처리
        formatted_output = formatted_output.replace("\\n", "\n")
        
        # 팁 정보 초기화
        tip_percentage = 0.0
        tip_amount = 0.0
        total_bill = 0.0

        # 1) JSON 블록 탐색 (```json ... ``` 또는 { "final_tip_percentage": ... })
        json_block = None
        m = re.search(r"```json\s*(\{.*?\})\s*```", formatted_output, re.DOTALL | re.IGNORECASE)
        
        if not m:
            # 백틱이 없는 경우 탐색
            m = re.search(r"(\{[^{}]*\"final_tip_percentage\"[^{}]*\})", formatted_output, re.DOTALL | re.IGNORECASE)

        if m:
            json_block = m.group(0) if '```' in m.group(0) else m.group(0)  # 전체 매치 (백틱 포함 또는 미포함)
            json_content = m.group(1)  # JSON 내용만
            try:
                json_data = json.loads(json_content)
                tip_percentage = float(json_data.get("final_tip_percentage", 0.0))
                tip_amount = float(json_data.get("final_tip_amount", 0.0))
                total_bill = float(json_data.get("final_total_bill", 0.0))
                
                # JSON 포맷팅
                formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
                # 원본 JSON 블록을 포맷팅된 JSON으로 교체 (백틱 포함 여부 유지)
                if '```' in json_block:
                    formatted_json_block = f"```json\n{formatted_json}\n```"
                else:
                    formatted_json_block = formatted_json
                
                formatted_output = formatted_output.replace(json_block, formatted_json_block)
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON 파싱 실패: {e}")

        # 2) JSON이 없으면 구 포맷(**Final Tip …**)으로 파싱
        if json_block is None:
            # Tip %
            m = re.search(r"\*\*Final Tip Percentage\*\*:\s*([0-9]+(?:\.[0-9]+)?)%", formatted_output, re.IGNORECASE)
            if m:
                tip_percentage = float(m.group(1))

            # Tip Amount
            m = re.search(r"\*\*Final Tip Amount\*\*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", formatted_output, re.IGNORECASE)
            if m:
                tip_amount = float(m.group(1))

            # Total Bill
            m = re.search(r"\*\*Final Total Bill\*\*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", formatted_output, re.IGNORECASE)
            if m:
                total_bill = float(m.group(1))

        # 3) 전체 출력을 그대로 분석 결과로 사용
        return output_text, tip_percentage, tip_amount, output_text

    def process_video_gemini(self, video_file_paths: List[str], merged_video_info: str = "") -> Tuple[str, None, None]:
        """
        Gemini 모델을 사용하여 여러 비디오 파일 병렬 처리
        
        Args:
            video_file_paths: 비디오 파일 경로 리스트
            merged_video_info: 비디오 관련 추가 정보
            
        Returns:
            Tuple[str, None, None]: 생성된 캡션, None, None
        """
        number_to_word = {
            1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"
        }
        merged_caption = ""

        def worker(idx: int, single_video: str) -> Tuple[int, str]:
            """단일 비디오 처리 작업자 함수"""
            prefix = number_to_word.get(idx, f"{idx}th").capitalize()
            header = f"### {prefix} Video Caption ###\n\n"

            if not os.path.exists(single_video):
                return idx, header + f"[Video {idx}] Error: Invalid video file path.\n\n"

            try:
                # 비디오 업로드
                video_file = self.model_clients.gemini_client.files.upload(file=single_video)
                logging.info(f"[Video {idx}] Uploaded: {video_file.name}")

                # 처리 대기
                start_time = time.time()
                processing_timeout = 40  # 40초 타임아웃
                timeout_occurred = False

                while video_file.state.name == "PROCESSING":
                    if time.time() - start_time > processing_timeout:
                        logging.warning(f"[Video {idx}] Processing timed out after {processing_timeout} seconds.")
                        timeout_occurred = True
                        break

                    time.sleep(5)  # 5초 대기
                    try:
                        video_file = self.model_clients.gemini_client.files.get(name=video_file.name)
                    except Exception as e:
                        logging.error(f"[Video {idx}] Error getting file state: {e}")
                        return idx, header + f"[Video {idx}] Error checking file processing state: {e}\n\n"

                if timeout_occurred:
                    return idx, header + "Video uploaded but processing timed out.\n\n"

                if video_file.state.name == "FAILED":
                    return idx, header + f"[Video {idx}] File processing failed: {video_file.state.name}\n\n"

                # 프롬프트 구성
                prompt = self._get_gemini_video_prompt(merged_video_info)

                # 캡션 생성
                caption_summary = self.model_clients.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=[video_file, prompt]
                )
                return idx, header + caption_summary.text + "\n\n"
            except Exception as e:
                logging.error(f"[Video {idx}] Error during caption generation: {e}")
                return idx, header + f"[Video {idx}] Error during caption generation: {e}\n\n"

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, idx, path)
                       for idx, path in enumerate(video_file_paths, start=1)]

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # 순서대로 결과 병합
        for idx, caption in sorted(results, key=lambda x: x[0]):
            merged_caption += caption

        return merged_caption, None, None

    def _get_gemini_video_prompt(self, merged_video_info: str = "") -> str:
        """
        Gemini 비디오 분석용 프롬프트 생성
        
        Args:
            merged_video_info: 추가 비디오 정보
            
        Returns:
            str: 생성된 프롬프트
        """
        prompt = '''
        1. Video Caption Analysis
        1-1. After analyzing the video, present the analysis results. Then, provide an analysis for each evaluation criterion and assign a score.
        1-2. However, if the video caption analysis has already been completed, please use the existing scores for the video scoring process.
        1-3. You must either generate a score for all items or get something from an existing caption and display it again. This item must be present.
        1-4. You are required to justify every single item without exception.
        1-5. Video Evaluation Criteria
            a) Reliability
                1: Frequent order mistakes, multiple billing errors
                2: No major mistakes but minor omissions occur
                3: Mostly accurate orders, occasional lack of confirmation
                4: Custom orders correctly handled, allergy/cooking preferences considered
                5: Always 100% accurate, uses repeat confirmation procedures

            b) Responsiveness
                1: Ignores customer requests or extremely slow
                2: Responds but with significant delays
                3: Average response speed, acceptable performance
                4: Quick and polite responses
                5: Immediate response, friendly tone and facial expressions, proactive problem-solving

            c) Assurance
                1: Unable to answer menu questions, rude or unfriendly
                2: Insufficient explanations, basic courtesy only
                3: Can explain the menu but lacks confidence
                4: Professional explanations, confident handling
                5: Earns customer trust, can offer premium suggestions like wine pairings

            d) Empathy
                1: Shows no interest in customers, mechanical responses
                2: Provides only minimal consideration
                3: Basic friendliness but lacks personalization
                4: Remembers customer preferences, offers personalized services like birthday events
                5: Proactively understands customer emotions, provides impressive experiences

            e) Tangibles
                1: Dirty table, utensils, and uniform
                2: Only some aspects are clean (e.g., utensils only)
                3: Acceptable but lacks attention to detail
                4: Uniform and table settings are tidy, cleanliness maintained
                5: Meets FDA Food Code standards, ServSafe certification level cleanliness

        1-6. Video Evaluation Criteria entries must generate results unconditionally.
        1-7. For Video Score, you must print the evidence and score for Reliability, Responsiveness, Assurance, Empathy, and Tangibles before printing the final score.
        1-8. Each item is out of 20, and the final Video score is out of 100.
        1-9. Video Score Calculation
                Video Score = (Reliability + Responsiveness + Assurance + Empathy + Tangibles)
        '''
        # 추가 정보가 있으면 포함
        if merged_video_info:
            prompt += f"\n\nMerged Video info: {merged_video_info}"
            
        return prompt

    def process_tip_gemini(self, 
                          video_file_path: List[str], 
                          star_rating: float, 
                          user_review: str, 
                          calculated_subtotal: float, 
                          custom_prompt: Optional[str] = None, 
                          merged_video_info: str = "") -> Tuple[str, float, float, None, None, str]:
        """
        Gemini를 사용한 팁 계산
        
        Args:
            video_file_path: 비디오 파일 경로 리스트
            star_rating: 별점 (1-5)
            user_review: 사용자 리뷰 텍스트
            calculated_subtotal: 계산된 소계
            custom_prompt: 사용자 정의 프롬프트 (없으면 기본값 사용)
            merged_video_info: 비디오 관련 추가 정보
            
        Returns:
            Tuple[str, float, float, None, None, str]: 분석 결과, 팁 퍼센트, 팁 금액, None, None, 원본 출력 텍스트
        """
        # 비디오 처리
        caption_summary, _, _ = self.process_video_gemini(video_file_path, merged_video_info)
        user_review = user_review.strip() if user_review and user_review.strip() else "(No user review provided)"

        # 프롬프트 구성
        prompt = self._get_tip_prompt(custom_prompt, calculated_subtotal, star_rating, user_review)
        
        # 캡션 삽입
        final_prompt = prompt.replace("{caption_text}", caption_summary)

        try:
            # 팁 계산 요청
            response = self.model_clients.gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[final_prompt]
            )
            
            # 결과 파싱
            llm_output = response.text
            # 전체 응답 텍스트를 보존하여 반환
            full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(llm_output)
            return full_text, tip_percentage, tip_amount, None, None, llm_output
        except Exception as e:
            logging.error(f"Gemini 모델 팁 계산 오류: {e}")
            return f"Error processing tip calculation with Gemini model: {e}", 0.0, 0.0, None, None, ""

    def process_tip_local(self, 
                         video_file_path: List[str], 
                         star_rating: float, 
                         user_review: str, 
                         calculated_subtotal: float, 
                         custom_prompt: Optional[str] = None, 
                         merged_video_info: str = "") -> Tuple[str, float, float, List[str], Optional[str], str]:
        """
        로컬 서버를 통한 팁 계산
        
        Args:
            video_file_path: 비디오 파일 경로 리스트
            star_rating: 별점 (1-5)
            user_review: 사용자 리뷰 텍스트
            calculated_subtotal: 계산된 소계
            custom_prompt: 사용자 정의 프롬프트 (없으면 기본값 사용)
            merged_video_info: 비디오 관련 추가 정보
            
        Returns:
            Tuple[str, float, float, List[str], Optional[str], str]: 분석 결과, 팁 퍼센트, 팁 금액, 프레임 경로, 프레임 폴더, 원본 출력 텍스트
        """
        # 비디오 처리
        caption_summary, frame_paths, frame_folder = self.process_video_gemini(video_file_path, merged_video_info)
        user_review = user_review.strip() if user_review and user_review.strip() else "(No user review provided)"

        # 프롬프트 구성
        prompt = self._get_tip_prompt(custom_prompt, calculated_subtotal, star_rating, user_review)
        
        # 캡션 삽입
        final_prompt = prompt.replace("{caption_text}", caption_summary)

        try:
            # 로컬 API 호출
            url = "http://59.9.11.187:8899/process_video_api"
            data = {'text_input': final_prompt}
            
            # 요청 전송
            response = requests.post(url, data=data, timeout=60)

            if response.status_code == 200:
                logging.info("로컬 API 요청 성공")
                
                # 응답 파싱
                resp_json = response.json()
                llm_output = resp_json.get("output", "")
                
                # 결과 파싱
                # 전체 응답 텍스트를 보존하여 반환
                full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(llm_output)
                return full_text, tip_percentage, tip_amount, frame_paths, frame_folder, llm_output
            else:
                logging.error(f"로컬 API 요청 실패: {response.status_code}, {response.text}")
                return f"HTTP Error {response.status_code}", 0.0, 0.0, frame_paths, frame_folder, ""
        except Exception as e:
            logging.error(f"로컬 모델 팁 계산 오류: {e}")
            return f"Error processing tip calculation with local model: {e}", 0.0, 0.0, frame_paths, frame_folder, ""

    def process_tip_qwen(self, 
                        video_file_path: List[str], 
                        star_rating: float, 
                        user_review: str, 
                        calculated_subtotal: float, 
                        custom_prompt: Optional[str] = None, 
                        merged_video_info: str = "") -> Tuple[str, float, float, List[str], Optional[str], str]:
        """
        Qwen을 사용한 팁 계산
        
        Args:
            video_file_path: 비디오 파일 경로 리스트
            star_rating: 별점 (1-5)
            user_review: 사용자 리뷰 텍스트
            calculated_subtotal: 계산된 소계
            custom_prompt: 사용자 정의 프롬프트 (없으면 기본값 사용)
            merged_video_info: 비디오 관련 추가 정보
            
        Returns:
            Tuple[str, float, float, List[str], Optional[str], str]: 분석 결과, 팁 퍼센트, 팁 금액, 프레임 경로, 프레임 폴더, 원본 출력 텍스트
        """
        if not isinstance(video_file_path, list) or len(video_file_path) == 0:
            return "Error: Video file list is empty.", 0.0, 0.0, [], None, ""

        # 비디오 캡션 생성
        #combined_caption = self._generate_qwen_captions(video_file_path, merged_video_info)
        combined_caption, frame_paths, frame_folder = self.process_video_gemini(video_file_path, merged_video_info)

        # 리뷰 처리
        user_review = user_review.strip() if user_review else "(No user review)"
        
        # 프롬프트 구성
        prompt = self._get_tip_prompt(custom_prompt, calculated_subtotal, star_rating, user_review)
        
        # 캡션 삽입
        final_prompt = prompt.replace("{caption_text}", combined_caption)

        try:
            # Qwen VL 모델로 팁 계산
            qvq_result = self.model_clients.qwen_client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt}
                        ],
                    },
                ],
                modalities=["text"],
                stream=True,
            )
            
            # 스트림 결과 처리
            final_text = self._process_qwen_stream(qvq_result)
            
            # 결과 파싱
            # 전체 응답 텍스트를 보존하여 반환
            full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(final_text)
            return full_text, tip_percentage, tip_amount, [], None, final_text
        except Exception as e:
            logging.error(f"Qwen 모델 팁 계산 오류: {e}")
            return f"Error processing tip calculation with Qwen model: {e}", 0.0, 0.0, [], None, ""

    def _generate_qwen_captions(self, video_file_path: List[str], merged_video_info: str = "") -> str:
        """
        Qwen Omni 모델을 사용하여 비디오 캡션 생성
        
        Args:
            video_file_path: 비디오 파일 경로 리스트
            merged_video_info: 추가 비디오 정보
            
        Returns:
            str: 생성된 캡션
        """
        combined_caption = ""
        omni_caption_prompt = '''
Task 1: Describe the waiters' actions in these restaurant video frames. Please check for mistakes or negative behaviors.
Task 2: Provide a short chronological summary of the entire scene.
'''
        # 추가 정보 포함
        if merged_video_info:
            omni_caption_prompt += f"\n\nVideo info: {merged_video_info}"

        # 각 비디오 처리
        for idx, single_video in enumerate(video_file_path, start=1):
            if not os.path.exists(single_video):
                combined_caption += f"[Video {idx}] Error: Invalid video file path.\n\n"
                continue

            try:
                # 비디오 인코딩
                base64_video = self.model_clients.encode_video_qwen(single_video)
                
                # Omni 모델 호출
                omni_result = self.model_clients.qwen_client.chat.completions.create(
                    model="qwen2.5-omni-7b",
                    messages=[
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": omni_caption_prompt},
                                {
                                    "type": "video_url",
                                    "video_url": {"url": f"data:;base64,{base64_video}"}
                                }
                            ],
                        },
                    ],
                    modalities=["text"],
                    stream=True,
                    stream_options={"include_usage": True},
                )
                
                # 스트림 처리
                caption_text = ""
                all_omni_chunks = list(omni_result)
                
                for chunk in all_omni_chunks[:-1]:
                    if not chunk.choices:
                        continue
                    if chunk.choices[0].delta.content:
                        caption_text += chunk.choices[0].delta.content

                # 캡션이 없으면 기본 메시지 사용
                if not caption_text.strip():
                    caption_text = "(No caption from Omni)"

                # 결과 병합
                combined_caption += f"[Video {idx}]:\n{caption_text}\n\n"
            except Exception as e:
                logging.error(f"[Video {idx}] Qwen 캡션 생성 오류: {e}")
                combined_caption += f"[Video {idx}] Error: {e}\n\n"

        return combined_caption

    def _process_qwen_stream(self, stream_result: Any) -> str:
        """
        Qwen 스트림 결과 처리
        
        Args:
            stream_result: Qwen 스트림 응답 객체
            
        Returns:
            str: 처리된 응답 텍스트
        """
        all_chunks = list(stream_result)
        final_reasoning = ""
        final_answer = ""
        is_answering = False
        
        for chunk in all_chunks[:-1]:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                final_reasoning += delta.reasoning_content
                
            if delta.content:
                if not is_answering:
                    is_answering = True
                final_answer += delta.content
                
        return final_reasoning + "\n" + final_answer

    def process_tip_gpt(self, 
                       video_file_path: List[str], 
                       star_rating: float, 
                       user_review: str, 
                       calculated_subtotal: float, 
                       custom_prompt: Optional[str] = None, 
                       merged_video_info: str = "") -> Tuple[str, float, float, None, None, str]:
        """
        GPT를 사용한 팁 계산
        
        Args:
            video_file_path: 비디오 파일 경로 리스트
            star_rating: 별점 (1-5)
            user_review: 사용자 리뷰 텍스트
            calculated_subtotal: 계산된 소계
            custom_prompt: 사용자 정의 프롬프트 (없으면 기본값 사용)
            merged_video_info: 비디오 관련 추가 정보
            
        Returns:
            Tuple[str, float, float, None, None, str]: 분석 결과, 팁 퍼센트, 팁 금액, None, None, 원본 출력 텍스트
        """
        # 비디오 처리
        caption_summary, _, _ = self.process_video_gemini(video_file_path, merged_video_info)
        user_review = user_review.strip() if user_review and user_review.strip() else "(No user review provided)"

        # 프롬프트 구성
        prompt = self._get_tip_prompt(custom_prompt, calculated_subtotal, star_rating, user_review)
        
        # 캡션 삽입
        final_prompt = prompt.replace("{caption_text}", caption_summary)

        try:
            # GPT 모델 호출
            response = self.model_clients.gpt_client.chat.completions.create(
                model=self.config.GPT_MODEL, 
                messages=[{"role": "user", "content": final_prompt}], 
                temperature=0.0, 
                max_tokens=2048
            )
            
            # 결과 파싱
            llm_output = response.choices[0].message.content
            # 전체 응답 텍스트를 보존하여 반환
            full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(llm_output)
            return full_text, tip_percentage, tip_amount, None, None, llm_output
        except Exception as e:
            logging.error(f"GPT 모델 팁 계산 오류: {e}")
            return f"Error processing tip calculation with GPT model: {e}", 0.0, 0.0, None, None, ""

    def _get_tip_prompt(self, custom_prompt: Optional[str], calculated_subtotal: float, 
                       star_rating: float, user_review: str) -> str:
        """
        팁 계산을 위한 프롬프트 생성
        
        Args:
            custom_prompt: 사용자 정의 프롬프트 (없으면 기본값 사용)
            calculated_subtotal: 계산된 소계
            star_rating: 별점 (1-5)
            user_review: 사용자 리뷰 텍스트
            
        Returns:
            str: 생성된 프롬프트
        """
        if custom_prompt is None:
            prompt = self.config.DEFAULT_PROMPT_TEMPLATE.format(
                calculated_subtotal=calculated_subtotal, 
                star_rating=star_rating, 
                user_review=user_review,
                google_reviews=self.config.GOOGLE_REVIEWS
            )
        else:
            try:
                prompt = custom_prompt.format(
                    calculated_subtotal=calculated_subtotal, 
                    star_rating=star_rating, 
                    user_review=user_review,
                    google_reviews=self.config.GOOGLE_REVIEWS
                )
            except KeyError as e:
                logging.warning(f"커스텀 프롬프트에 필요한 키가 없습니다: {e}. 기본 템플릿을 사용합니다.")
                prompt = self.config.DEFAULT_PROMPT_TEMPLATE.format(
                    calculated_subtotal=calculated_subtotal, 
                    star_rating=star_rating, 
                    user_review=user_review,
                    google_reviews=self.config.GOOGLE_REVIEWS
                )
        
        return prompt

    def calculate_manual_tip(self, tip_percent: float, subtotal: float) -> Tuple[str, str, str]:
        """
        백분율에 따른 수동 팁 계산
        
        Args:
            tip_percent: 팁 퍼센트 (%)
            subtotal: 소계
            
        Returns:
            Tuple[str, str, str]: 분석 출력, 팁 출력, 총액 출력
        """
        tip_amount = subtotal * (tip_percent / 100)
        total_bill = subtotal + tip_amount
        
        analysis_output = f"Manual calculation using fixed tip percentage of {tip_percent}%."
        tip_output = f"${tip_amount:.2f} ({tip_percent:.1f}%)"
        total_bill_output = f"${total_bill:.2f}"
        
        return analysis_output, tip_output, total_bill_output 