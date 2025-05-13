"""
Gradio UI 이벤트 및 콜백 처리 모듈
"""
import os
import uuid
import shutil
import logging
import json
import re
from typing import List, Dict, Tuple, Any, Optional, Union

import gradio as gr

from utils.config import Config
from utils.video_processor import VideoProcessor
from models.tip_calculator import TipCalculator
from models.google_reviews import GoogleReviewManager


def get_recorded_videos():
    """녹화된 비디오 파일 목록 반환"""
    folder = "./record_videos"
    if not os.path.exists(folder):
        return []
    
    # .avi, .mp4 확장자를 가진 파일만 포함
    video_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f))
                  and f.lower().endswith((".avi", ".mp4"))]
    
    return video_files


def format_json_output(text: str) -> str:
    """
    JSON 출력을 포맷팅하고 이스케이프 문자를 처리합니다.
    
    Args:
        text: 포맷팅할 텍스트
        
    Returns:
        str: 포맷팅된 텍스트
    """
    if not text:
        return ""
        
    # 이스케이프된 줄바꿈 문자 처리
    formatted_text = text.replace("\\n", "\n")
    
    # JSON 블록 찾기 및 포맷팅
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = list(re.finditer(json_pattern, formatted_text, re.DOTALL))
    
    # 뒤에서부터 처리하여 문자열 위치가 변경되는 문제 방지
    for match in reversed(matches):
        # 전체 JSON 블록 (백틱 포함)
        full_match = match.group(0)  
        # JSON 내용만
        json_str = match.group(1)
        
        try:
            # JSON 파싱 및 예쁘게 출력
            parsed_json = json.loads(json_str)
            pretty_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            # 원본 JSON 블록을 포맷팅된 JSON으로 교체
            formatted_text = formatted_text.replace(
                full_match, 
                f"```json\n{pretty_json}\n```"
            )
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 원본 유지
            pass
    
    # 백틱이 없는 독립적인 JSON 블록 찾기
    bare_json_pattern = r'(?<![`\w])\s*(\{[^{}]*\"final_tip_percentage\"[^{}]*\})\s*(?![`\w])'
    bare_matches = list(re.finditer(bare_json_pattern, formatted_text, re.DOTALL))
    
    # 뒤에서부터 처리
    for match in reversed(bare_matches):
        json_str = match.group(1)
        try:
            parsed_json = json.loads(json_str)
            pretty_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            formatted_text = formatted_text.replace(json_str, pretty_json)
        except json.JSONDecodeError:
            pass
    
    # 마크다운 헤더 강조
    markdown_headers = re.finditer(r'^(#+)\s+(.+)$', formatted_text, re.MULTILINE)
    for match in markdown_headers:
        header = match.group(0)
        level = len(match.group(1))
        text = match.group(2)
        formatted_text = formatted_text.replace(header, f"{'#' * level} **{text}**")
    
    return formatted_text


class UIHandler:
    """
    Gradio UI 이벤트 및 콜백 처리 클래스
    """

    def __init__(self, config: Config, tip_calculator: TipCalculator, video_processor: VideoProcessor):
        """
        UIHandler 초기화
        
        Args:
            config: 애플리케이션 설정 객체
            tip_calculator: 팁 계산 객체
            video_processor: 비디오 처리 객체
        """
        self.config = config
        self.tip_calculator = tip_calculator
        self.video_processor = video_processor

    def update_subtotal_and_prompt(self, *args) -> Tuple[float, str]:
        """
        사용자 입력에 따라 소계 및 프롬프트 업데이트
        
        Args:
            *args: 사용자 입력값 (수량, 별점, 리뷰)
            
        Returns:
            Tuple[float, str]: 계산된 소계와 업데이트된 프롬프트
        """
        num_food_items = len(self.config.FOOD_ITEMS)
        quantities = args[:num_food_items]
        star_rating = args[num_food_items]
        user_review = args[num_food_items + 1]

        # 소계 계산
        calculated_subtotal = 0.0
        for i in range(num_food_items):
            calculated_subtotal += self.config.FOOD_ITEMS[i]['price'] * quantities[i]

        # 리뷰 텍스트 준비
        user_review_text = user_review.strip() if user_review and user_review.strip() else "(No user review provided)"

        # 구글 리뷰 포맷팅
        formatted_google_reviews = self.config.GOOGLE_REVIEWS
        
        # 프롬프트 업데이트
        updated_prompt = self.config.DEFAULT_PROMPT_TEMPLATE.format(
            calculated_subtotal=calculated_subtotal,
            star_rating=star_rating,
            user_review=user_review_text,
            google_reviews=formatted_google_reviews
        )
        
        # 캡션 플레이스홀더 유지
        updated_prompt = updated_prompt.replace("{caption_text}", "{{caption_text}}")

        return calculated_subtotal, updated_prompt

    def compute_tip(self, model_type: str, video_file_obj: Any, subtotal: float, 
                   star_rating: float, user_review: str, custom_prompt_text: str) -> Tuple[str, str, str, str, Any]:
        """
        비디오, 리뷰 및 선택한 모델에 따라 팁 계산
        
        Args:
            model_type: 사용할 모델 유형 ('local', 'gpt', 'qwen', 'gemini')
            video_file_obj: 비디오 파일 객체 또는 경로
            subtotal: 계산된 소계
            star_rating: 별점 (1-5)
            user_review: 사용자 리뷰 텍스트
            custom_prompt_text: 커스텀 프롬프트 텍스트
            
        Returns:
            Tuple[str, str, str, str, Any]: 분석 결과, 팁 출력, 총액 출력, 프롬프트, 비디오 UI 업데이트
        """
        analysis_output = "계산을 시작합니다..."
        tip_output = "$0.00"
        total_bill_output = f"${subtotal:.2f}"

        # 비디오 파일 확인
        if video_file_obj is None:
            return "오류: 비디오 파일을 업로드해주세요.", tip_output, total_bill_output, custom_prompt_text, gr.update(value=None)

        # 임시 비디오 파일 생성
        temp_video_path = self._create_temp_videos(video_file_obj)
        if isinstance(temp_video_path, str) and temp_video_path.startswith("오류:"):
            return temp_video_path, tip_output, total_bill_output, custom_prompt_text, None

        frame_folder = None
        try:
            # 팁 계산 프로세스 실행
            logging.info('팁계산 프로세스 시작')
            logging.info(f'동영상 path: {temp_video_path}')
            
            # 모델 타입에 따른 처리
            if model_type == 'local':
                full_analysis, tip_percentage, tip_amount, _, frame_folder, raw_output = self.tip_calculator.process_tip_local(
                    temp_video_path, star_rating, user_review, subtotal, custom_prompt_text, merged_video_info=""
                )
            elif model_type == 'gpt':
                full_analysis, tip_percentage, tip_amount, _, _, raw_output = self.tip_calculator.process_tip_gpt(
                    temp_video_path, star_rating, user_review, subtotal, custom_prompt_text, merged_video_info=""
                )
            elif model_type == 'qwen':
                full_analysis, tip_percentage, tip_amount, _, _, raw_output = self.tip_calculator.process_tip_qwen(
                    temp_video_path, star_rating, user_review, subtotal, custom_prompt_text, merged_video_info=""
                )
            elif model_type == 'gemini':
                full_analysis, tip_percentage, tip_amount, _, _, raw_output = self.tip_calculator.process_tip_gemini(
                    temp_video_path, star_rating, user_review, subtotal, custom_prompt_text, merged_video_info=""
                )
            else:
                raise ValueError(f"알 수 없는 모델 타입: {model_type}")

            # 결과 포맷팅
            if "Error" in full_analysis:
                analysis_output = full_analysis
                tip_amount = 0.0
            else:
                # LLM의 전체 출력을 그대로 표시
                analysis_output = raw_output
                tip_output = f"${tip_amount:.2f} ({tip_percentage:.1f}%)"
                total_bill = subtotal + tip_amount
                total_bill_output = f"${total_bill:.2f}"
                
        except Exception as e:
            logging.error(f"팁 계산 중 오류 발생 ({model_type}): {e}")
            analysis_output = f"오류 발생: {e}"
            tip_output = "$0.00"
            total_bill_output = f"${subtotal:.2f}"
        finally:
            # 임시 파일 정리
            self.video_processor.cleanup_temp_files(temp_video_path, frame_folder)

        return analysis_output, tip_output, total_bill_output, custom_prompt_text, gr.update(value=None)

    def _create_temp_videos(self, video_file_obj: Any) -> Union[List[str], str]:
        """
        임시 비디오 파일 생성
        
        Args:
            video_file_obj: 비디오 파일 객체 또는 경로
            
        Returns:
            Union[List[str], str]: 임시 비디오 파일 경로 리스트 또는 오류 메시지
        """
        # 서버 디렉토리 절대 경로
        server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 리스트로 처리
        if isinstance(video_file_obj, list):
            temp_video_paths = []
            for video in video_file_obj:
                try:
                    temp_path = os.path.join(server_dir, f"temp_video_{uuid.uuid4().hex}.mp4")
                    original_path = video.name if hasattr(video, "name") else video
                    shutil.copyfile(original_path, temp_path)
                    logging.info(f"임시 비디오 파일 생성: {temp_path}")
                    temp_video_paths.append(temp_path)
                except Exception as e:
                    logging.error(f"임시 비디오 파일 생성 오류: {e}")
                    return f"오류: 비디오 파일을 처리할 수 없습니다: {e}"
            return temp_video_paths
        else:
            # 단일 파일 처리
            try:
                temp_video_path = os.path.join(server_dir, f"temp_video_{uuid.uuid4().hex}.mp4")
                original_path = video_file_obj.name if hasattr(video_file_obj, "name") else video_file_obj
                shutil.copyfile(original_path, temp_video_path)
                logging.info(f"임시 비디오 파일 생성: {temp_video_path}")
                return [temp_video_path]  # 단일 비디오도 리스트로 반환
            except Exception as e:
                logging.error(f"임시 비디오 파일 생성 오류: {e}")
                return f"오류: 비디오 파일을 처리할 수 없습니다: {e}"

    def process_payment(self, total_bill: str) -> str:
        """
        결제 처리 및 녹화된 비디오 파일 정리
        
        Args:
            total_bill: 총액 문자열
            
        Returns:
            str: 결제 결과 메시지
        """
        # 서버 디렉토리 절대 경로
        # server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # video_dir = os.path.join(server_dir, "record_videos")
        #
        # # 디렉토리가 존재하는지 확인
        # if os.path.exists(video_dir):
        #     logging.info(f"'{video_dir}' 디렉토리에서 파일 삭제를 시작합니다.")
        #
        #     # 디렉토리 내 파일 삭제
        #     for filename in os.listdir(video_dir):
        #         file_path = os.path.join(video_dir, filename)
        #         try:
        #             if os.path.isfile(file_path) or os.path.islink(file_path):
        #                 os.unlink(file_path)  # 파일 삭제
        #                 logging.info(f"파일 삭제됨: {file_path}")
        #         except Exception as e:
        #             logging.error(f"'{file_path}' 삭제 중 오류 발생: {e}")
        # else:
        #     logging.warning(f"디렉토리를 찾을 수 없습니다: '{video_dir}'. 파일 삭제를 건너뜁니다.")

        return f"{total_bill} 결제되었습니다."

    def update_invoice_summary(self, *args) -> str:
        """
        수량 및 팁에 따라 청구서 요약 업데이트
        
        Args:
            *args: 수량, 팁, 총액 등 정보
            
        Returns:
            str: 업데이트된 청구서 요약
        """
        num_items = len(self.config.FOOD_ITEMS)
        quantities = args[:num_items]

        # 팁과 총액 정보 추출
        if len(args) >= num_items + 2:
            tip_str = args[num_items]
            total_bill_str = args[num_items + 1]
        else:
            tip_str = "$0.00"
            total_bill_str = "$0.00"

        # 청구서 생성
        summary = ""
        for i, q in enumerate(quantities):
            try:
                q_val = float(q)
            except (ValueError, TypeError):
                q_val = 0

            if q_val > 0:
                item = self.config.FOOD_ITEMS[i]
                total_price = item['price'] * q_val
                summary += f"{item['name']} x{int(q_val)} : ${total_price:.2f}\n"

        # 메뉴가 없는 경우
        if summary == "":
            summary = "주문한 메뉴가 없습니다."

        # 팁과 총액 정보 추가
        summary += f"\nTip: {tip_str}\nTotal Bill: {total_bill_str}"

        return summary

    def manual_tip_and_invoice(self, tip_percent: float, subtotal: float, *quantities) -> Tuple[str, str, str, str]:
        """
        수동 팁 계산 및 청구서 업데이트
        
        Args:
            tip_percent: 팁 퍼센트 (%)
            subtotal: 소계
            *quantities: 각 메뉴 수량
            
        Returns:
            Tuple[str, str, str, str]: 분석 결과, 팁 출력, 총액 출력, 청구서 요약
        """
        # 수동 팁 계산
        analysis, tip_disp, total_bill_disp = self.tip_calculator.calculate_manual_tip(tip_percent, subtotal)
        
        # 청구서 업데이트
        invoice = self.update_invoice_summary(*quantities, tip_disp, total_bill_disp)
        
        return analysis, tip_disp, total_bill_disp, invoice

    def auto_tip_and_invoice(self, model_type: str, video_file_obj: Any, subtotal: float, 
                            star_rating: float, review: str, prompt: str, *quantities) -> Tuple[str, str, str, str, Any, str]:
        """
        AI 모델을 사용한 자동 팁 계산 및 청구서 업데이트
        
        Args:
            model_type: 사용할 모델 유형
            video_file_obj: 비디오 파일 객체 또는 경로
            subtotal: 계산된 소계
            star_rating: 별점 (1-5)
            review: 사용자 리뷰 텍스트
            prompt: 커스텀 프롬프트 텍스트
            *quantities: 각 메뉴 수량
            
        Returns:
            Tuple[str, str, str, str, Any, str]: 분석 결과, 팁 출력, 총액 출력, 프롬프트, 비디오 UI 업데이트, 청구서 요약
        """
        # 팁 계산
        analysis, tip_disp, total_bill_disp, prompt_out, vid_out = self.compute_tip(
            model_type, video_file_obj, subtotal, star_rating, review, prompt
        )
        # 청구서 업데이트
        invoice = self.update_invoice_summary(*quantities, tip_disp, total_bill_disp)
        
        return analysis, tip_disp, total_bill_disp, prompt_out, vid_out, invoice

    def create_gradio_blocks(self) -> gr.Blocks:
        """
        Gradio Blocks 인터페이스 구성 - 가로형 키오스크 레이아웃용
        
        Returns:
            gr.Blocks: Gradio 인터페이스 객체
        """
        with gr.Blocks(title="Peter Luger Steak House\nAI-based Tip Calculation Kiosk", theme=gr.themes.Soft(),
                      css=self.config.CUSTOM_CSS,
                      head="""
                        <meta name="viewport" content="width=1200, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                        <meta name="HandheldFriendly" content="false">
                        <meta name="MobileOptimized" content="1200">
                        <meta http-equiv="ScreenOrientation" content="autoRotate:disabled">
                        <style>
                          html, body {
                            min-width: 1200px !important;
                            width: 1200px !important;
                            overflow-x: auto !important;
                            touch-action: pan-x pan-y !important;
                            -ms-touch-action: pan-x pan-y !important;
                            -webkit-touch-callout: none !important;
                            -webkit-text-size-adjust: none !important;
                            -webkit-tap-highlight-color: rgba(0,0,0,0) !important;
                          }
                          
                          /* Safari/Chrome용 확대/축소 방지 */
                          * { 
                            -webkit-user-select: none;
                            -webkit-touch-callout: none;
                          }
                          
                          /* 모든 입력 필드는 선택 가능하게 유지 */
                          input, textarea { 
                            -webkit-user-select: text;
                            -webkit-touch-callout: default;
                          }
                          
                          /* 탭 버튼 네비게이션 스타일 */
                          .tab-nav {
                            display: flex;
                            justify-content: center;
                            margin-bottom: 20px;
                            background-color: #f5f5f5;
                            padding: 10px;
                            border-radius: 10px;
                          }
                          
                          .tab-button {
                            margin: 0 10px;
                            padding: 10px 20px;
                            background-color: #ffffff;
                            border: 1px solid #dddddd;
                            border-radius: 5px;
                            cursor: pointer;
                            transition: all 0.3s;
                          }
                          
                          .tab-button:hover, .tab-button.active {
                            background-color: #4CAF50;
                            color: white;
                            transform: translateY(-2px);
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                          }
                        </style>
                      """) as interface:
            gr.Markdown("# Peter Luger Steak House\n # AI-based Tip Calculation Kiosk", elem_id="main-title")

            # 컴포넌트 변수 선언
            quantity_inputs = []
            subtotal_display = gr.Number(label="Subtotal ($)", value=0.0, interactive=False, visible=False)
            subtotal_visible_display = None
            review_input, rating_input = None, None
            btn_5, btn_10, btn_15, btn_20, btn_25 = None, None, None, None, None
            local_btn, gpt_btn, qwen_btn, gemini_btn = None, None, None, None
            tip_display, total_bill_display, payment_btn, payment_result = None, None, None, None
            video_input = None
            analysis_display, order_summary_display = None, None
            prompt_editor = None

            # 커스텀 탭 네비게이션 버튼
            with gr.Row(elem_id="tab-navigation"):
                main_tab_btn = gr.Button("메인 화면", elem_id="main-tab-btn", elem_classes=["tab-button", "active"])
                example_tab_btn = gr.Button("예제", elem_id="example-tab-btn", elem_classes=["tab-button"])
                analysis_nav_btn = gr.Button("AI 결과 분석", elem_id="analysis-nav-btn", elem_classes=["tab-button"])
                prompt_nav_btn = gr.Button("프롬프트 편집", elem_id="prompt-nav-btn", elem_classes=["tab-button"])

            # 메인 화면 (메뉴 선택 등)
            with gr.Group(elem_id="main-content-group", visible=True) as main_content_group:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, elem_id="menu-column"):
                        gr.Markdown("## 메뉴 선택", elem_id="menu-title")
                        with gr.Column(elem_id="food-container"):
                            items_per_row = 4
                            total_items = min(12, len(self.config.FOOD_ITEMS))
                            for row_idx in range(0, (total_items + items_per_row - 1) // items_per_row):
                                with gr.Row(elem_id=f"menu-row-{row_idx}"):
                                    for col_idx in range(items_per_row):
                                        item_idx = row_idx * items_per_row + col_idx
                                        if item_idx < total_items:
                                            item = self.config.FOOD_ITEMS[item_idx]
                                            with gr.Column(elem_id=f"food-item-{item_idx}", variant="compact", elem_classes=["food-item"]):
                                                img = gr.Image(value=item["image"], label=None, show_label=False, interactive=False, elem_id=f"food-image-{item_idx}", width=300, height=300, type="filepath")
                                                name_display = gr.Markdown(f"**{item['name']}** ${item['price']:.2f}", elem_id=f"food-name-{item_idx}")
                                                q_input = gr.Number(label="수량", value=0, minimum=0, step=1, elem_id=f"qty_{item['name'].replace(' ', '_')}")
                                                quantity_inputs.append(q_input)
                                                def update_quantity(qty, evt: gr.SelectData):
                                                    return qty + 1
                                                img.select(update_quantity, q_input, q_input, api_name=f"click_menu_{item_idx}")
                        with gr.Row(elem_id="subtotal-row"):
                            with gr.Column(scale=2):
                                gr.Markdown("### 소계")
                                subtotal_visible_display = gr.Textbox(value="$0.00", label="Subtotal", interactive=False)
                    with gr.Column(scale=3, elem_id="right-sidebar"):
                        gr.Markdown("## 결제", elem_id="rating-title")
                        with gr.Group(elem_id="rating-group"):
                            gr.Markdown("### 별점")
                            with gr.Row(equal_height=True):
                                rating_input = gr.Radio(choices=[1, 2, 3, 4, 5], value=3, label="", type="value", elem_id="rating-input", container=False)
                        with gr.Group(elem_id="review-group"):
                            review_input = gr.Textbox(label="리뷰", placeholder="서비스 리뷰 작성", lines=2, elem_id="review-input")
                        with gr.Group(elem_id="tip-calculation-group"):
                            gr.Markdown("### 팁", elem_id="tip-title")
                            with gr.Row(equal_height=True, elem_id="manual-tip-buttons"):
                                btn_5 = gr.Button("5%", elem_id="tip-5", size="sm")
                                btn_10 = gr.Button("10%", elem_id="tip-10", size="sm")
                                btn_15 = gr.Button("15%", elem_id="tip-15", size="sm")
                                btn_20 = gr.Button("20%", elem_id="tip-20", size="sm")
                                btn_25 = gr.Button("25%", elem_id="tip-25", size="sm")
                            with gr.Row(equal_height=True, elem_id="ai-tip-buttons"):
                                with gr.Column(scale=1):
                                    local_btn = gr.Button("Mistral", elem_id="mistral-button", size="sm")
                                    gemini_btn = gr.Button('Google Gemini', elem_id="google-button", size="sm")
                                with gr.Column(scale=1):
                                    gpt_btn = gr.Button("OpenAI GPT", variant="primary", elem_id="gpt-button", size="sm")
                                    qwen_btn = gr.Button("Alibaba Qwen", elem_id="qwen-button", size="sm")
                        with gr.Group(elem_id="results-group"):
                            with gr.Row():
                                with gr.Column():
                                    tip_display = gr.Textbox(label="팁 %", value="$0.00", interactive=False, elem_id="tip-display")
                                    total_bill_display = gr.Textbox(label="결제 비용", value="$0.00", interactive=False, elem_id="total-bill-display")
                        with gr.Group(elem_id="bill-group"):
                            gr.Markdown("### 청구서")
                            order_summary_display = gr.Textbox(label="청구서 내역", value="주문 메뉴 없음", lines=3, interactive=False, elem_id="order-summary")
                        with gr.Group(elem_id="payment-group"):
                            payment_btn = gr.Button("결제하기", size="lg", elem_id="payment-button")
                            payment_result = gr.Textbox(label="결제 결과", value="", interactive=False, visible=False, elem_id="payment-result")

            # 예제 탭
            with gr.Group(elem_id="example-content-group", visible=False) as example_content_group:
                gr.Markdown("## 예제 서비스 (클릭하여 적용)", elem_id="examples-title")
                def apply_example_1():
                    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "sample.mp4"), 1, "He drop the tray..so bad"
                def apply_example_2():
                    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "odette_singapore_10.mp4"), 5, "Good service!"
                def apply_example_3():
                    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video","back2.mp4"), 4, "Nice service!"

                with gr.Row(elem_id="examples-container"):
                    with gr.Column(scale=1, elem_id="example-bad"):
                        gr.Markdown("### 예제 1: 나쁜 서비스")
                        with gr.Row():
                            example1_btn = gr.Button("예제 1 적용", elem_id="apply-example1-btn")
                        example1_video = gr.Video(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "sample.mp4"), label="나쁜 서비스 예시", elem_id="example-video-bad", interactive=False)
                        gr.Markdown("리뷰: He drop the tray..so bad")
                        gr.Markdown("별점: ⭐")

                    with gr.Column(scale=1, elem_id="example-good"):
                        gr.Markdown("### 예제 2: 좋은 서비스")
                        with gr.Row():
                            example2_btn = gr.Button("예제 2 적용", elem_id="apply-example2-btn")
                        example2_video = gr.Video(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "odette_singapore_10.mp4"), label="좋은 서비스 예시", elem_id="example-video-good", interactive=False)
                        gr.Markdown("리뷰: Good service!")
                        gr.Markdown("별점: ⭐⭐⭐⭐⭐")

                    with gr.Column(scale=1, elem_id="example-good"):
                        gr.Markdown("### 예제 3: 괜찮은 서비스")
                        with gr.Row():
                            example3_btn = gr.Button("예제 3 적용", elem_id="apply-example2-btn")
                        example2_video = gr.Video(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "back2.mp4"), label="괜찮은 서비스 예시", elem_id="example-nice-good", interactive=False)
                        gr.Markdown("리뷰: Nice service!")
                        gr.Markdown("별점: ⭐⭐⭐⭐")

                gr.Markdown("## 서비스 비디오 업로드/촬영")
                video_input = gr.Video(label="서비스 비디오 업로드 또는 촬영", elem_id="video-input-standalone")
                gr.Markdown("""
                ### 비디오 촬영 안내
                1. 위 컴포넌트를 클릭하여 비디오를 업로드하거나 웹캠으로 촬영하세요.
                2. 서비스 장면을 선명하게 담아주세요.
                3. 10-30초 정도의 서비스 장면이 적합합니다.
                4. 촬영 후 메인화면으로 돌아가셔서 평가를 진행해주세요.
                """)

            # AI 결과 분석 탭
            with gr.Group(elem_id="analysis-content-group", visible=False) as analysis_content_group:
                gr.Markdown("### AI 결과 분석")
                analysis_display = gr.Markdown(label="AI Analysis")#, lines=100, max_lines=150, interactive=True, elem_id="analysis_display_component")
            
            # 프롬프트 편집 탭
            with gr.Group(elem_id="prompt-content-group", visible=False) as prompt_content_group:
                gr.Markdown("### 프롬프트 편집기")
                gr.Markdown(
                    "자동 생성된 프롬프트를 여기서 확인하고 **직접 수정**할 수 있습니다. "
                    "AI 분석 시 여기에 있는 최종 내용이 사용됩니다."
                )
                prompt_editor = gr.Code(
                    label="Tip Calculation Prompt (Editable)",
                    language="python",
                    value="Loading prompt...",
                    lines=35,
                    elem_id="prompt_editor_component"
                )
            
            # 모든 콘텐츠 그룹 리스트
            content_groups = [main_content_group, example_content_group, analysis_content_group, prompt_content_group]
            
            def show_content(tab_to_show_idx): # 0:main, 1:example, 2:AI Analysis, 3:Prompt Edit
                updates = [gr.update(visible=False) for _ in content_groups]
                if 0 <= tab_to_show_idx < len(content_groups):
                    updates[tab_to_show_idx] = gr.update(visible=True)
                return tuple(updates)

            # 네비게이션 버튼 클릭 이벤트
            main_tab_btn.click(lambda: show_content(0), [], content_groups)
            example_tab_btn.click(lambda: show_content(1), [], content_groups)
            analysis_nav_btn.click(lambda: show_content(2), [], content_groups)
            prompt_nav_btn.click(lambda: show_content(3), [], content_groups)

            # Helper function to define components for checking if all are defined
            def all_components_defined(*components):
                return all(comp is not None for comp in components)

            if all_components_defined(
                subtotal_display, subtotal_visible_display, review_input, rating_input, 
                prompt_editor, order_summary_display, video_input, analysis_display, 
                tip_display, total_bill_display, payment_result, 
                local_btn, gpt_btn, qwen_btn, gemini_btn,
                btn_5, btn_10, btn_15, btn_20, btn_25,
                *content_groups, # 모든 그룹이 생성되었는지 확인
                *quantity_inputs
            ):
                subtotal_display.change(fn=lambda x: f"${x:.2f}", inputs=subtotal_display, outputs=subtotal_visible_display)
                inputs_for_prompt_update = quantity_inputs + [rating_input, review_input]
                outputs_for_prompt_update = [subtotal_display, prompt_editor]
                interface.load(fn=self.update_subtotal_and_prompt, inputs=inputs_for_prompt_update, outputs=outputs_for_prompt_update)
                for comp in inputs_for_prompt_update:
                    comp.change(fn=self.update_subtotal_and_prompt, inputs=inputs_for_prompt_update, outputs=outputs_for_prompt_update)
                for comp in quantity_inputs:
                    comp.change(fn=self.update_invoice_summary, inputs=quantity_inputs, outputs=order_summary_display)

                ai_compute_inputs = [video_input, subtotal_display, rating_input, review_input, prompt_editor] + quantity_inputs
                ai_core_outputs_list = [analysis_display, tip_display, total_bill_display, prompt_editor, video_input, order_summary_display]
                
                # AI 계산 및 AI 분석 그룹으로 전환하는 함수
                def run_ai_and_switch_to_analysis(model_type_str, vid_up, sub, rat, rev, prom, *qty):
                    analysis, tip_disp, total_bill_disp, prompt_out, vid_out, invoice = self.auto_tip_and_invoice(
                        model_type_str, compute_video_source(vid_up, get_recorded_videos()), sub, rat, rev, prom, *qty
                    )
                    group_visibility_updates = show_content(2) # 2 for AI Analysis Group
                    return tuple([analysis, tip_disp, total_bill_disp, prompt_out, vid_out, invoice] + list(group_visibility_updates))

                ai_button_outputs = ai_core_outputs_list + content_groups

                local_btn.click(fn=lambda vid_up, sub, rat, rev, prom, *qty: run_ai_and_switch_to_analysis('local', vid_up, sub, rat, rev, prom, *qty), inputs=ai_compute_inputs, outputs=ai_button_outputs)
                gpt_btn.click(fn=lambda vid_up, sub, rat, rev, prom, *qty: run_ai_and_switch_to_analysis('gpt', vid_up, sub, rat, rev, prom, *qty), inputs=ai_compute_inputs, outputs=ai_button_outputs)
                qwen_btn.click(fn=lambda vid_up, sub, rat, rev, prom, *qty: run_ai_and_switch_to_analysis('qwen', vid_up, sub, rat, rev, prom, *qty), inputs=ai_compute_inputs, outputs=ai_button_outputs)
                gemini_btn.click(fn=lambda vid_up, sub, rat, rev, prom, *qty: run_ai_and_switch_to_analysis('gemini', vid_up, sub, rat, rev, prom, *qty), inputs=ai_compute_inputs, outputs=ai_button_outputs)

                manual_tip_outputs = [analysis_display, tip_display, total_bill_display, order_summary_display]
                btn_5.click(fn=lambda sub, *qty: self.manual_tip_and_invoice(5, sub, *qty), inputs=[subtotal_display] + quantity_inputs, outputs=manual_tip_outputs)
                btn_10.click(fn=lambda sub, *qty: self.manual_tip_and_invoice(10, sub, *qty), inputs=[subtotal_display] + quantity_inputs, outputs=manual_tip_outputs)
                btn_15.click(fn=lambda sub, *qty: self.manual_tip_and_invoice(15, sub, *qty), inputs=[subtotal_display] + quantity_inputs, outputs=manual_tip_outputs)
                btn_20.click(fn=lambda sub, *qty: self.manual_tip_and_invoice(20, sub, *qty), inputs=[subtotal_display] + quantity_inputs, outputs=manual_tip_outputs)
                btn_25.click(fn=lambda sub, *qty: self.manual_tip_and_invoice(25, sub, *qty), inputs=[subtotal_display] + quantity_inputs, outputs=manual_tip_outputs)
                
                payment_btn.click(fn=self.process_payment, inputs=[total_bill_display], outputs=[payment_result])

                # 예제 적용 및 메인 화면으로 전환하는 함수
                def apply_example_and_switch_to_main(example_func):
                    video_val, rating_val, review_val = example_func()
                    group_visibility_updates = show_content(0) # 0 for Main Content Group
                    return tuple([video_val, rating_val, review_val] + list(group_visibility_updates))
                
                example_core_outputs_list = [video_input, rating_input, review_input]
                example_button_outputs = example_core_outputs_list + content_groups

                example1_btn.click(fn=lambda: apply_example_and_switch_to_main(apply_example_1), inputs=[], outputs=example_button_outputs)
                example2_btn.click(fn=lambda: apply_example_and_switch_to_main(apply_example_2), inputs=[], outputs=example_button_outputs)
                example3_btn.click(fn=lambda: apply_example_and_switch_to_main(apply_example_3), inputs=[], outputs=example_button_outputs)

            else:
                logging.warning("컴포넌트 초기화 확인 실패. 이벤트 핸들러가 제대로 연결되지 않을 수 있습니다.")
            return interface

# compute_video_source는 클래스 메서드가 아니므로 외부 또는 static으로 정의 필요
# UIHandler 클래스 외부 또는 static 메서드로 이동하거나, self 없이 호출되도록 수정
def compute_video_source(video_upload, video_list):
    if video_upload is not None and video_upload != "":
        return [video_upload]
    else:
        return video_list 