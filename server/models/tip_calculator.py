"""
팁 계산 핵심 로직 모듈
"""
import os
import re
import json
import time
import logging
# import base64 # 사용되지 않음
import requests
# import cv2 # 사용되지 않음
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
        LLM 출력을 파싱하여 팁 관련 정보를 추출합니다.

        이 함수는 먼저 LLM 출력에서 JSON 블록을 찾으려고 시도합니다.
        JSON 블록은 ```json ... ``` 형식으로 감싸져 있거나, 단순히 { ... } 형태일 수 있습니다.
        JSON이 발견되면 "final_tip_percentage", "final_tip_amount", "final_total_bill" 키를 사용하여 값을 추출합니다.
        추출된 JSON은 가독성을 위해 포맷팅되어 원본 출력 내의 해당 블록을 대체합니다.

        JSON 블록을 찾지 못하거나 파싱에 실패하면, 이전의 텍스트 기반 형식 (예: "**Final Tip Percentage**: ...")으로
        팁 정보를 파싱하려고 시도합니다.

        Args:
            output_text: LLM이 생성한 원본 텍스트 출력.

        Returns:
            Tuple[str, float, float, str]:
                - formatted_output (str): JSON이 포맷팅된 버전의 원본 텍스트 (또는 변경 없음).
                - tip_percentage (float): 추출된 팁 비율 (실패 시 0.0).
                - tip_amount (float): 추출된 팁 금액 (실패 시 0.0).
                - output_text (str): LLM의 원본, 수정되지 않은 출력 텍스트.
                                     (주: 이전 버전에서는 formatted_output을 반환했으나,
                                      호출자들이 원본 LLM 출력을 기대하는 경우가 있어 output_text로 변경.
                                      formatted_output은 이제 로깅 또는 디버깅 목적으로 사용될 수 있음.)
        """
        # 원본 전체 출력 텍스트 보존 (분석 및 로깅용)
        # formatted_output은 JSON 포맷팅 등 내부 변경을 추적하는 데 사용
        formatted_output = output_text
        
        # 이스케이프된 줄바꿈 문자 처리 (LLM 출력에서 흔히 발견됨)
        formatted_output = formatted_output.replace("\\n", "\n")
        
        # 팁 정보 초기화
        tip_percentage = 0.0
        tip_amount = 0.0
        total_bill = 0.0 # 참고: 이 값은 현재 반환 튜플에 포함되지 않음

        # 1) JSON 블록 탐색
        # 먼저 ```json ... ``` 형식으로 감싸진 JSON 블록을 찾습니다.
        # re.DOTALL 플래그는 '.'이 줄바꿈 문자도 포함하도록 하며, re.IGNORECASE는 대소문자를 무시합니다.
        json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", formatted_output, re.DOTALL | re.IGNORECASE)
        
        if not json_block_match:
            # ```json ... ``` 형식이 없으면, 일반적인 JSON 객체 패턴 ({...})을 찾습니다.
            # "final_tip_percentage" 키가 포함된 JSON 객체를 특정하여 대상 범위를 좁힙니다.
            # [^{}]* 는 중괄호가 아닌 모든 문자를 의미하여, 중첩된 JSON 객체를 잘못 파싱하는 것을 방지합니다.
            json_block_match = re.search(r"(\{[^{}]*\"final_tip_percentage\"[^{}]*\})", formatted_output, re.DOTALL | re.IGNORECASE)

        if json_block_match:
            # json_block_text는 ```json ... ``` 전체 또는 { ... } 전체를 포함합니다.
            json_block_text = json_block_match.group(0)
            # json_content_text는 실제 JSON 내용 ({...})만 추출합니다.
            # group(1)은 정규식에서 첫 번째 괄호로 캡처된 부분을 의미합니다.
            json_content_text = json_block_match.group(1)
            try:
                json_data = json.loads(json_content_text)
                tip_percentage = float(json_data.get("final_tip_percentage", 0.0))
                tip_amount = float(json_data.get("final_tip_amount", 0.0))
                total_bill = float(json_data.get("final_total_bill", 0.0)) # 이 값은 사용될 수 있음

                # JSON 포맷팅 (가독성 향상)
                formatted_json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                
                # 원본 JSON 블록을 포맷팅된 JSON으로 교체 (백틱 포함 여부 유지)
                # 이는 formatted_output을 업데이트하여 로깅이나 디버깅 시 더 읽기 쉬운 형태로 만듭니다.
                if '```' in json_block_text:
                    formatted_json_block_replacement = f"```json\n{formatted_json_str}\n```"
                else:
                    formatted_json_block_replacement = formatted_json_str
                
                formatted_output = formatted_output.replace(json_block_text, formatted_json_block_replacement)
                logging.info("JSON 블록 파싱 성공.")
            except (ValueError, json.JSONDecodeError) as e:
                # JSON 파싱 중 오류 발생 시 (예: 잘못된 형식의 JSON)
                logging.warning(f"JSON 파싱 실패: {e}. 텍스트 기반 파싱으로 대체합니다.")
                json_block_match = None # 파싱 실패 시 json_block_match를 None으로 설정하여 텍스트 기반 파싱 실행

        # 2) JSON 블록이 없거나 파싱에 실패한 경우, 이전의 텍스트 기반 형식으로 파싱
        if json_block_match is None:
            logging.info("JSON 블록을 찾지 못했거나 파싱에 실패하여 텍스트 기반 파싱을 시도합니다.")
            # Tip Percentage: "**Final Tip Percentage**: <value>%" 형식으로 찾습니다.
            # (?:...)는 비캡처 그룹을 의미하며, \.?는 소수점이 있을 수도 있고 없을 수도 있음을 나타냅니다.
            m_percent = re.search(r"\*\*Final Tip Percentage\*\*:\s*([0-9]+(?:\.[0-9]+)?)%", formatted_output, re.IGNORECASE)
            if m_percent:
                tip_percentage = float(m_percent.group(1))

            # Tip Amount: "**Final Tip Amount**: $<value>" 또는 "**Final Tip Amount**: <value>" 형식으로 찾습니다.
            # \$?는 달러 기호가 있을 수도 있고 없을 수도 있음을 나타냅니다.
            m_amount = re.search(r"\*\*Final Tip Amount\*\*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", formatted_output, re.IGNORECASE)
            if m_amount:
                tip_amount = float(m_amount.group(1))

            # Total Bill: "**Final Total Bill**: $<value>" 또는 "**Final Total Bill**: <value>" 형식으로 찾습니다.
            m_total = re.search(r"\*\*Final Total Bill\*\*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", formatted_output, re.IGNORECASE)
            if m_total:
                total_bill = float(m_total.group(1)) # 이 값은 사용될 수 있음

        # 3) 최종 반환:
        # output_text는 LLM의 원본 출력을 그대로 반환합니다.
        # formatted_output (JSON 포맷팅된 버전)은 필요한 경우 로깅 등에 사용될 수 있습니다.
        # tip_percentage와 tip_amount는 추출된 값을 반환합니다.
        return formatted_output, tip_percentage, tip_amount, output_text

    def process_video_gemini(self, video_file_paths: List[str], merged_video_info: str = "") -> Tuple[str, None, None]:
        """
        Gemini 모델을 사용하여 여러 비디오 파일을 병렬로 처리하고 캡션을 생성합니다.

        각 비디오는 Gemini API에 업로드되고 처리됩니다. 타임아웃 및 오류 상태가 처리됩니다.
        성공적으로 처리된 각 비디오에 대해 캡션이 생성되고, 순서대로 병합되어 단일 문자열로 반환됩니다.
        `merged_video_info`는 모든 비디오에 공통적으로 적용될 수 있는 추가 컨텍스트 정보를 제공하는 데 사용됩니다.

        Args:
            video_file_paths: 처리할 비디오 파일 경로의 리스트.
            merged_video_info: (선택 사항) 모든 비디오 프롬프트에 추가될 공통 정보 문자열.
                               예: 레스토랑 이름, 날짜 등 컨텍셔닝에 사용될 수 있습니다.

        Returns:
            Tuple[str, None, None]:
                - merged_caption (str): 모든 비디오에서 생성된 캡션을 순서대로 병합한 문자열.
                                       오류 발생 시 오류 메시지가 포함될 수 있습니다.
                - None: (프레임 경로용 플레이스홀더, 이 함수에서는 사용되지 않음).
                - None: (프레임 폴더용 플레이스홀더, 이 함수에서는 사용되지 않음).
        """
        # 숫자를 서수 단어로 변환하기 위한 딕셔너리 (예: 1 -> "first")
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
        Gemini 비디오 분석을 위한 표준 프롬프트를 생성합니다.

        이 프롬프트는 Gemini 모델에게 직원 행동 분석, 장면 요약,
        그리고 정의된 기준(신뢰성, 응답성, 확신성, 공감성, 유형성)에 따른
        서비스 품질 평가 및 점수 매기기를 지시합니다.
        최종 비디오 점수는 이러한 기준 점수들의 합계에 4를 곱하여 계산됩니다.

        Args:
            merged_video_info: (선택 사항) 프롬프트 끝에 추가될 수 있는 추가적인 컨텍스트 정보.
                               이는 특정 비디오나 분석 작업에 대한 추가 지침이나 데이터를 제공할 수 있습니다.

        Returns:
            str: Gemini 비디오 분석을 위해 구성된 전체 프롬프트 문자열.
        """
        prompt = '''
Task 1: Staff Action Analysis
Describe in detail the actions of any waiters or staff visible in this restaurant video.
* Note any specific interactions, mistakes, or positive actions.

Task 2: Scene Summary
Provide a concise overall summary of the scene depicted in the frames, in chronological order if possible.

Task 3: Service Quality Evaluation and Scoring
Based on the observations from Task 1 and the overall scene from Task 2, analyze the staff's performance according to the following criteria. For each criterion, provide a brief justification for your score and then assign a score from 1 to 5. Finally, calculate the total Video Score.

3-1. Video Evaluation Criteria and Scoring:
   *You are required to justify every single item without exception.* (Justification is required)
   a) Reliability: (Score: __/5)
      * 1: Frequent order mistakes, multiple billing errors
      * 2: No major mistakes but minor omissions occur
      * 3: Mostly accurate orders, occasional lack of confirmation
      * 4: Custom orders correctly handled, allergy/cooking preferences considered
      * 5: Always 100% accurate, uses repeat confirmation procedures
      * Justification:

   b) Responsiveness: (Score: __/5)
      * 1: Ignores customer requests or extremely slow
      * 2: Responds but with significant delays
      * 3: Average response speed, acceptable performance
      * 4: Quick and polite responses
      * 5: Immediate response, friendly tone and facial expressions, proactive problem-solving
      * Justification:

   c) Assurance: (Score: __/5)
      * 1: Unable to answer menu questions, rude or unfriendly
      * 2: Insufficient explanations, basic courtesy only
      * 3: Can explain the menu but lacks confidence
      * 4: Professional explanations, confident handling
      * 5: Earns customer trust, can offer premium suggestions like wine pairings
      * Justification:

   d) Empathy: (Score: __/5)
      * 1: Shows no interest in customers, mechanical responses
      * 2: Provides only minimal consideration
      * 3: Basic friendliness but lacks personalization
      * 4: Remembers customer preferences, offers personalized services like birthday events
      * 5: Proactively understands customer emotions, provides impressive experiences
      * Justification:

   e) Tangibles: (Score: __/5)
      * 1: Dirty table, utensils, and uniform
      * 2: Only some aspects are clean (e.g., utensils only)
      * 3: Acceptable but lacks attention to detail
      * 4: Uniform and table settings are tidy, cleanliness maintained
      * 5: Meets FDA Food Code standards, ServSafe certification level cleanliness
      * Justification:

3-2. Video Score Calculation:
After assigning scores for each of the five criteria, calculate and present the final 'Video Score' using the following formula:
* Video Score = (Reliability Score + Responsiveness Score + Assurance Score + Empathy Score + Tangibles Score) * 4
* Final Video Score:
* Final Video Caption:
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
        Gemini 모델을 사용하여 팁을 계산하고 관련 분석을 생성합니다.

        이 프로세스에는 다음 단계가 포함됩니다:
        1. `process_video_gemini`를 호출하여 제공된 비디오 파일에서 캡션 요약을 생성합니다.
           `merged_video_info`는 비디오 분석 컨텍스트를 제공하기 위해 이 단계로 전달됩니다.
        2. 사용자 리뷰가 제공되지 않은 경우 기본 문자열로 처리합니다.
        3. `_get_tip_prompt`를 사용하여 LLM에 대한 프롬프트를 구성합니다. 이 프롬프트는
           비디오 캡션, 별점, 사용자 리뷰, 계산된 소계 및 Google 리뷰를 통합합니다.
           사용자 정의 프롬프트가 제공되면 기본 프롬프트 대신 사용됩니다.
        4. 생성된 비디오 캡션을 프롬프트의 `{caption_text}` 플레이스홀더에 삽입합니다.
        5. Gemini 모델을 호출하여 팁 계산 및 분석을 수행합니다.
        6. `parse_llm_output`을 사용하여 모델의 응답에서 구조화된 팁 정보와 전체 텍스트를 추출합니다.

        Args:
            video_file_path: 처리할 비디오 파일 경로의 리스트.
            star_rating: 사용자가 제공한 별점 (1-5).
            user_review: 사용자의 텍스트 리뷰.
            calculated_subtotal: 주문의 계산된 소계.
            custom_prompt: (선택 사항) 기본 프롬프트 템플릿 대신 사용할 사용자 정의 프롬프트.
            merged_video_info: (선택 사항) `process_video_gemini`에 전달될 비디오 관련 추가 정보.

        Returns:
            Tuple[str, float, float, None, None, str]:
                - full_text (str): 모델의 포맷팅된 응답 텍스트 (JSON 포함 가능).
                - tip_percentage (float): 계산된 팁 비율.
                - tip_amount (float): 계산된 팁 금액.
                - None: (프레임 경로용 플레이스홀더, 이 모델에서는 사용되지 않음).
                - None: (프레임 폴더용 플레이스홀더, 이 모델에서는 사용되지 않음).
                - llm_output (str): 모델의 원본, 수정되지 않은 응답 텍스트.
        """
        # 1. 비디오 처리 (캡션 생성)
        # `merged_video_info`는 비디오 분석 시 추가 컨텍스트를 제공할 수 있습니다.
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
            
            # 6. 결과 파싱
            llm_output = response.text
            # `parse_llm_output`은 포맷팅된 텍스트와 원본 LLM 출력을 모두 반환합니다.
            # 여기서 full_text는 포맷팅된 버전이고, llm_output은 원본입니다.
            full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(llm_output) # _ 는 원본 output_text를 무시
            return full_text, tip_percentage, tip_amount, None, None, llm_output
        except Exception as e:
            logging.error(f"Gemini 모델 팁 계산 오류: {e}")
            # 오류 발생 시 기본값과 오류 메시지 반환
            return f"Error processing tip calculation with Gemini model: {e}", 0.0, 0.0, None, None, ""

    def process_tip_local(self, 
                         video_file_path: List[str], 
                         star_rating: float, 
                         user_review: str, 
                         calculated_subtotal: float, 
                         custom_prompt: Optional[str] = None, 
                         merged_video_info: str = "") -> Tuple[str, float, float, List[str], Optional[str], str]:
        """
        로컬에서 호스팅되는 모델을 사용하여 팁을 계산하고 관련 분석을 생성합니다.

        이 프로세스는 `process_tip_gemini`와 유사한 단계를 따르지만,
        Gemini API 대신 로컬 서버 엔드포인트(`http://59.9.11.187:8899/process_video_api`)로 요청을 보냅니다.
        비디오 처리는 여전히 `process_video_gemini`를 통해 수행됩니다 (캡션 생성 목적).

        Args:
            video_file_path: 처리할 비디오 파일 경로의 리스트.
            star_rating: 사용자가 제공한 별점 (1-5).
            user_review: 사용자의 텍스트 리뷰.
            calculated_subtotal: 주문의 계산된 소계.
            custom_prompt: (선택 사항) 기본 프롬프트 템플릿 대신 사용할 사용자 정의 프롬프트.
            merged_video_info: (선택 사항) `process_video_gemini`에 전달될 비디오 관련 추가 정보.

        Returns:
            Tuple[str, float, float, List[str], Optional[str], str]:
                - full_text (str): 모델의 포맷팅된 응답 텍스트.
                - tip_percentage (float): 계산된 팁 비율.
                - tip_amount (float): 계산된 팁 금액.
                - frame_paths (List[str]): `process_video_gemini`에서 반환된 프레임 경로 (이 컨텍스트에서는 주로 Gemini가 생성).
                - frame_folder (Optional[str]): `process_video_gemini`에서 반환된 프레임 폴더 (이 컨텍스트에서는 주로 Gemini가 생성).
                - llm_output (str): 모델의 원본, 수정되지 않은 응답 텍스트.
        """
        # 1. 비디오 처리 (캡션 생성, Gemini 사용)
        # `merged_video_info`는 비디오 분석 시 추가 컨텍스트를 제공합니다.
        # 로컬 모델은 자체 비디오 처리 기능이 없을 수 있으므로, Gemini를 캡션 생성에 활용합니다.
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
                
                # 6. 결과 파싱
                # `parse_llm_output`은 포맷팅된 텍스트와 원본 LLM 출력을 모두 반환합니다.
                full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(llm_output) # _ 는 원본 output_text를 무시
                return full_text, tip_percentage, tip_amount, frame_paths, frame_folder, llm_output
            else:
                # HTTP 오류 처리
                logging.error(f"로컬 API 요청 실패: {response.status_code}, {response.text}")
                return f"HTTP Error {response.status_code} from local API: {response.text}", 0.0, 0.0, frame_paths, frame_folder, ""
        except requests.exceptions.RequestException as e: # requests 관련 예외 처리
            logging.error(f"로컬 모델 API 요청 중 네트워크 오류: {e}")
            return f"Network error during local model API request: {e}", 0.0, 0.0, frame_paths, frame_folder, ""
        except Exception as e: # 기타 예외 처리
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
        Qwen 모델 (qwen2.5-72b-instruct)을 사용하여 팁을 계산하고 관련 분석을 생성합니다.

        이 프로세스는 다음 단계를 포함합니다:
        1. 비디오 파일 목록이 비어 있는지 확인합니다.
        2. `process_video_gemini`를 호출하여 비디오 캡션을 생성합니다. 현재 Qwen 모델은
           비디오 입력을 직접 받지 않으므로, Gemini를 통해 생성된 텍스트 캡션을 사용합니다.
           `merged_video_info`는 비디오 분석 컨텍스트를 제공하기 위해 이 단계로 전달됩니다.
           (주: `_generate_qwen_captions`는 현재 사용되지 않습니다.)
        3. 사용자 리뷰를 처리합니다 (비어 있는 경우 기본값 사용).
        4. `_get_tip_prompt`를 사용하여 LLM 프롬프트를 구성합니다.
        5. 생성된 비디오 캡션을 프롬프트에 삽입합니다.
        6. Qwen 모델 (qwen2.5-72b-instruct, 텍스트 전용)을 호출하여 팁 계산을 수행합니다.
           스트리밍 응답을 사용하고 `_process_qwen_stream`으로 처리합니다.
        7. `parse_llm_output`을 사용하여 모델 응답에서 팁 정보를 추출합니다.

        Args:
            video_file_path: 처리할 비디오 파일 경로의 리스트.
            star_rating: 사용자가 제공한 별점 (1-5).
            user_review: 사용자의 텍스트 리뷰.
            calculated_subtotal: 주문의 계산된 소계.
            custom_prompt: (선택 사항) 기본 프롬프트 템플릿 대신 사용할 사용자 정의 프롬프트.
            merged_video_info: (선택 사항) `process_video_gemini`에 전달될 비디오 관련 추가 정보.

        Returns:
            Tuple[str, float, float, List[str], Optional[str], str]:
                - full_text (str): 모델의 포맷팅된 응답 텍스트.
                - tip_percentage (float): 계산된 팁 비율.
                - tip_amount (float): 계산된 팁 금액.
                - [] (List[str]): 프레임 경로 (Qwen은 현재 Gemini 캡션을 사용하므로 비어 있음).
                - None (Optional[str]): 프레임 폴더 (Qwen은 현재 Gemini 캡션을 사용하므로 None).
                - llm_output (str): 모델의 원본, 수정되지 않은 응답 텍스트.
        """
        # 1. 비디오 파일 목록 유효성 검사
        if not isinstance(video_file_path, list) or len(video_file_path) == 0:
            logging.error("process_tip_qwen: 비디오 파일 목록이 비어 있습니다.")
            return "Error: Video file list is empty.", 0.0, 0.0, [], None, ""

        # 2. 비디오 캡션 생성 (Gemini 사용)
        # `merged_video_info`는 비디오 분석 시 추가 컨텍스트를 제공합니다.
        # Qwen 모델은 현재 텍스트 입력을 사용하므로, Gemini로 캡션을 생성합니다.
        # frame_paths와 frame_folder는 Gemini 호출에서 오지만, Qwen 컨텍스트에서는 직접 사용되지 않습니다.
        combined_caption, frame_paths_from_gemini, frame_folder_from_gemini = self.process_video_gemini(video_file_path, merged_video_info)

        # 3. 리뷰 처리
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
            
            # 7. 결과 파싱
            # `parse_llm_output`은 포맷팅된 텍스트와 원본 LLM 출력을 모두 반환합니다.
            full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(final_text) # _ 는 원본 output_text를 무시
            # Qwen은 Gemini를 통해 캡션을 얻으므로, 프레임 관련 정보는 Gemini의 것을 반환할 수도 있으나,
            # 혼동을 피하기 위해 여기서는 Qwen이 직접 생성한 프레임 정보가 없는 것으로 명시합니다.
            return full_text, tip_percentage, tip_amount, [], None, final_text
        except Exception as e: # Qwen API 호출 또는 기타 예외 처리
            logging.error(f"Qwen 모델 팁 계산 오류: {e}")
            return f"Error processing tip calculation with Qwen model: {e}", 0.0, 0.0, [], None, ""

    def _generate_qwen_captions(self, video_file_path: List[str], merged_video_info: str = "") -> str:
        """
        (현재 사용되지 않음) Qwen Omni 모델을 사용하여 비디오 캡션을 생성합니다.

        이 함수는 각 비디오 파일을 Base64로 인코딩하고 Qwen Omni 모델 (qwen2.5-omni-7b)을 호출하여
        캡션을 생성하려고 시도합니다. 생성된 캡션은 비디오별로 병합됩니다.
        `merged_video_info`는 모든 비디오 프롬프트에 공통 컨텍스트를 추가하는 데 사용됩니다.

        참고: 이 함수는 현재 `process_tip_qwen`에서 직접 호출되지 않으며,
        대신 `process_video_gemini`를 사용하여 캡션을 생성합니다.

        Args:
            video_file_path: 처리할 비디오 파일 경로의 리스트.
            merged_video_info: (선택 사항) 모든 비디오 프롬프트에 추가될 공통 정보.

        Returns:
            str: 모든 비디오에서 생성된 캡션을 병합한 문자열. 오류 발생 시 오류 메시지가 포함될 수 있습니다.
        """
        combined_caption = ""
        # Qwen Omni 모델에 전달할 기본 프롬프트
        omni_caption_prompt = '''
Task 1: Describe the waiters' actions in these restaurant video frames. Please check for mistakes or negative behaviors.
Task 2: Provide a short chronological summary of the entire scene.
'''
        # `merged_video_info`가 제공되면 프롬프트에 추가
        if merged_video_info:
            omni_caption_prompt += f"\n\nVideo info: {merged_video_info}"

        # 각 비디오 파일에 대해 반복 처리
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
        Qwen 모델의 스트리밍 응답을 처리합니다.

        이 함수는 Qwen 모델에서 반환된 청크(chunk) 스트림을 반복하여,
        `reasoning_content` (있는 경우)와 `content`를 수집하여
        하나의 완전한 응답 문자열로 결합합니다.

        Args:
            stream_result: Qwen 모델의 `chat.completions.create` 호출에서 반환된 스트림 객체.

        Returns:
            str: 스트림에서 모든 텍스트 조각을 결합하여 생성된 전체 응답 문자열.
        """
        all_chunks = list(stream_result) # 스트림을 리스트로 변환하여 모든 청크를 한 번에 처리
        final_reasoning = "" # 모델의 추론 과정을 저장 (일부 모델에서 제공)
        final_answer = ""
        
        for chunk in all_chunks[:-1]:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # 모델이 추론 과정을 제공하는 경우 수집
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                final_reasoning += delta.reasoning_content

            # 모델이 실제 답변 내용을 제공하는 경우 수집
            if delta.content:
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
        OpenAI GPT 모델을 사용하여 팁을 계산하고 관련 분석을 생성합니다.

        이 프로세스는 `process_tip_gemini`와 매우 유사한 단계를 따릅니다:
        1. `process_video_gemini`를 호출하여 비디오 캡션 요약을 생성합니다.
           `merged_video_info`는 비디오 분석 컨텍스트를 제공합니다.
        2. 사용자 리뷰를 처리합니다 (비어 있는 경우 기본값 사용).
        3. `_get_tip_prompt`를 사용하여 LLM 프롬프트를 구성합니다.
        4. 생성된 비디오 캡션을 프롬프트에 삽입합니다.
        5. OpenAI GPT 모델 (설정에서 지정된 모델)을 호출하여 팁 계산을 수행합니다.
        6. `parse_llm_output`을 사용하여 모델 응답에서 팁 정보를 추출합니다.

        Args:
            video_file_path: 처리할 비디오 파일 경로의 리스트.
            star_rating: 사용자가 제공한 별점 (1-5).
            user_review: 사용자의 텍스트 리뷰.
            calculated_subtotal: 주문의 계산된 소계.
            custom_prompt: (선택 사항) 기본 프롬프트 템플릿 대신 사용할 사용자 정의 프롬프트.
            merged_video_info: (선택 사항) `process_video_gemini`에 전달될 비디오 관련 추가 정보.

        Returns:
            Tuple[str, float, float, None, None, str]:
                - full_text (str): 모델의 포맷팅된 응답 텍스트.
                - tip_percentage (float): 계산된 팁 비율.
                - tip_amount (float): 계산된 팁 금액.
                - None: (프레임 경로용 플레이스홀더, 이 모델에서는 사용되지 않음).
                - None: (프레임 폴더용 플레이스홀더, 이 모델에서는 사용되지 않음).
                - llm_output (str): 모델의 원본, 수정되지 않은 응답 텍스트.
        """
        # 1. 비디오 처리 (캡션 생성, Gemini 사용)
        # GPT 모델은 현재 텍스트 입력을 사용하므로, Gemini로 캡션을 생성합니다.
        # `merged_video_info`는 비디오 분석 시 추가 컨텍스트를 제공합니다.
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
            
            # 6. 결과 파싱
            llm_output = response.choices[0].message.content
            # `parse_llm_output`은 포맷팅된 텍스트와 원본 LLM 출력을 모두 반환합니다.
            full_text, tip_percentage, tip_amount, _ = self.parse_llm_output(llm_output) # _ 는 원본 output_text를 무시
            return full_text, tip_percentage, tip_amount, None, None, llm_output
        except Exception as e: # OpenAI API 호출 또는 기타 예외 처리
            logging.error(f"GPT 모델 팁 계산 오류: {e}")
            return f"Error processing tip calculation with GPT model: {e}", 0.0, 0.0, None, None, ""

    def _get_tip_prompt(self, custom_prompt: Optional[str], calculated_subtotal: float, 
                       star_rating: float, user_review: str) -> str:
        """
        팁 계산을 위한 LLM 프롬프트를 생성하고 포맷합니다.

        사용자 정의 프롬프트(`custom_prompt`)가 제공되면 해당 프롬프트를 사용합니다.
        그렇지 않으면 `self.config.DEFAULT_PROMPT_TEMPLATE`에 정의된 기본 템플릿을 사용합니다.

        프롬프트는 다음 플레이스홀더를 동적 값으로 대체하여 포맷됩니다:
        - `{calculated_subtotal}`: 계산된 주문 소계.
        - `{star_rating}`: 사용자가 제공한 별점.
        - `{user_review}`: 사용자의 텍스트 리뷰.
        - `{google_reviews}`: 설정에서 로드된 Google 리뷰 텍스트.
        - `{caption_text}`: 이 값은 이 함수 호출 이후에 별도로 삽입됩니다 (예: `final_prompt = prompt.replace("{caption_text}", caption_summary)`).

        Args:
            custom_prompt: (선택 사항) 사용할 사용자 정의 프롬프트 문자열.
            calculated_subtotal: 계산된 주문 소계.
            star_rating: 사용자가 제공한 별점.
            user_review: 사용자의 텍스트 리뷰.

        Returns:
            str: 포맷팅된 프롬프트 문자열. `{caption_text}` 플레이스홀더는 아직 대체되지 않았습니다.

        Raises:
            KeyError: 사용자 정의 프롬프트에 필요한 포맷 키가 누락된 경우 경고가 기록되고 기본 템플릿이 사용됩니다.
        """
        # 사용할 프롬프트 템플릿 결정 (사용자 정의 또는 기본값)
        prompt_to_use = None
        if custom_prompt is not None:
            try:
                # 사용자 정의 프롬프트 포맷 시도
                # {caption_text}는 이 함수 반환 후 별도로 처리됩니다.
                prompt_to_use = custom_prompt.format(
                    calculated_subtotal=calculated_subtotal, 
                    star_rating=star_rating, 
                    user_review=user_review,
                    google_reviews=self.config.GOOGLE_REVIEWS
                )
            except KeyError as e:
                logging.warning(
                    f"커스텀 프롬프트에 필요한 키가 없습니다: {e}. "
                    f"입력된 custom_prompt (앞 200자): '{custom_prompt[:200]}...' 기본 템플릿을 사용합니다."
                )
                # 이 경우 prompt_to_use는 None으로 유지되어 아래에서 기본 템플릿이 사용됩니다.
        
        # 사용자 정의 프롬프트가 제공되지 않았거나 포맷팅에 실패한 경우 기본 템플릿 사용
        if prompt_to_use is None:
            prompt_to_use = self.config.DEFAULT_PROMPT_TEMPLATE.format(
                calculated_subtotal=calculated_subtotal,
                star_rating=star_rating,
                user_review=user_review,
                google_reviews=self.config.GOOGLE_REVIEWS
            )

        return prompt_to_use

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