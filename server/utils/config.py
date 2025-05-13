"""
애플리케이션 설정 및 상수 관리 모듈
"""
import os
import logging
from typing import Dict, List, Any
from pathlib import Path

from models.google_reviews import GoogleReviewManager


class Config:
    """
    애플리케이션 설정 및 상수를 관리하는 클래스
    """

    # 서버 디렉토리의 절대 경로 계산
    SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 음식 메뉴 데이터 - 절대 경로 사용
    FOOD_ITEMS = [
        {"name": "Single Steak", "image": os.path.join(SERVER_DIR, "images", "single_steak.JPG"), "price": 73.95}, #
        {"name": "Rib Steak", "image": os.path.join(SERVER_DIR, "images", "rib_steak.jpg"), "price": 92.95}, #
        {"name": "Steak For 2", "image": os.path.join(SERVER_DIR, "images", "steak_for_2.JPG"), "price": 147.90},
        {"name": "Pecan Pie", "image": os.path.join(SERVER_DIR, "images", "pecan_pie.JPG"), "price": 14.95},
        {"name": "Chocolate Mousse Cake", "image": os.path.join(SERVER_DIR, "images", "chocolate_mousse_cake.JPG"), "price": 14.95},
        {"name": "Cheese Cake", "image": os.path.join(SERVER_DIR, "images", "cheese_cake.JPG"), "price": 14.95},
        {"name": "Lamb Chops", "image": os.path.join(SERVER_DIR, "images", "lamb_chops.jpg"), "price": 72.95},
        {"name": "Grilled Atlantic Salmon", "image": os.path.join(SERVER_DIR, "images", "grilled_atlantic_salmon.jpg"), "price": 39.95},
        {"name": "French Fried Potatoes (For 2)", "image": os.path.join(SERVER_DIR, "images", "french_fried_potatoes_for_2.jpg"), "price": 18.95},
        {"name": "Sliced Tomatoes & Onions (For 2)", "image": os.path.join(SERVER_DIR, "images", "sliced_tomatoes_onions_for_2.jpg"), "price": 17.95},
        {"name": "Iceberg Wedge Salad", "image": os.path.join(SERVER_DIR, "images", "iceberg_wedge_salad.jpg"), "price": 18.95},
        {"name": "Caesar Salad", "image": os.path.join(SERVER_DIR, "images", "caesar_salad.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
        # {"name": "사이다", "image": os.path.join(SERVER_DIR, "images", "food6.jpg"), "price": 12.00},
    ]

    # API 키 및 설정
    OPENAI_API_KEY = ""
    QWEN_API_KEY = ""
    GPT_MODEL = ""
    GEMINI_API_KEY = ''

    # Gradio UI용 CSS
    CUSTOM_CSS = """
    /* 레이아웃 고정 설정 - 모바일 포함 모든 디바이스에서 동일하게 표시 */
    :root {
        --fixed-width: 1200px;
        --fixed-height: 800px;
    }
    
    /* 미디어 쿼리 무시하고 항상 고정 크기 적용 */
    @media (max-width: 1200px) {
        .gradio-container {
            min-width: var(--fixed-width) !important;
            width: var(--fixed-width) !important;
            max-width: var(--fixed-width) !important;
            overflow-x: auto !important;
        }
    }
    
    /* 컨테이너 고정 크기 설정 */
    .gradio-container {
        width: var(--fixed-width) !important;
        min-width: var(--fixed-width) !important;
        max-width: var(--fixed-width) !important;
        margin: 0 auto !important;
        overflow-x: auto !important;
    }
    
    /* 제한된 크기의 디바이스에서 수평 스크롤 허용 */
    body {
        min-width: var(--fixed-width) !important;
        overflow-x: auto !important;
    }
    
    /* 키오스크 가로형 레이아웃을 위한 스타일 */
    .gradio-container {
        padding: 0;
    }
    
    /* 메인 타이틀 스타일 */
    #main-title {
        text-align: center;
        font-size: 32px !important;
        margin-bottom: 20px !important;
        color: #333;
        background: linear-gradient(to right, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    
    /* 메인 화면 레이아웃 균형 맞추기 */
    #main-tab .gradio-row {
        align-items: flex-start !important;
        gap: 10px !important; /* 메뉴와 결제 영역 간격 축소 */
    }
    
    /* 메뉴 그리드 확장 */
    #menu-column {
        width: 100%;
        display: flex;
        flex-direction: column;
    }
    
    /* 메뉴 행 스타일 */
    [id^="menu-row-"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: space-between !important;
        gap: 8px !important;
        margin-bottom: 8px !important;
    }
    
    /* 메뉴 아이템 스타일 - 정사각형 */
    [id^="food-item-"] {
        flex: 1 1 0 !important;
        width: 25% !important; /* 4개 아이템, 각 25% */
        max-width: 25% !important;
        min-width: 0 !important;
        padding: 10px !important;
        border: 1px solid #eaeaea !important;
        border-radius: 12px !important;
        background-color: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        overflow: hidden !important;
        margin: 0 !important;
        /* 정사각형 형태로 설정 */
        aspect-ratio: 1/1 !important;
    }
    
    [id^="food-item-"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* 이미지 컨테이너 크기 조정 */
    [id^="food-image-"] {
        width: 100% !important;
        flex: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    
    /* 이미지 크기 최대화 */
    [id^="food-image-"] img {
        max-width: 100% !important;
        max-height: 100% !important;
        object-fit: contain !important;
    }
    
    /* 텍스트 스타일 */
    [id^="food-name-"] {
        font-size: 14px !important;
        margin: 8px 0 4px 0 !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    /* 수량 입력 필드 축소 */
    [id^="food-item-"] .gradio-number {
        transform: scale(0.9) !important;
        margin-top: 5px !important;
    }
    
    /* 소계 행 스타일 */
    #subtotal-row {
        margin-top: 10px !important;
        padding: 8px !important;
        background-color: #f8f8f8 !important;
        border-radius: 8px !important;
    }
    
    /* 오른쪽 사이드바 스타일 */
    #right-sidebar {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 12px;
        box-shadow: -2px 0 10px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        min-width: 250px;
    }
    
    /* 오른쪽 사이드바 내부 그룹 스타일 */
    #right-sidebar .gradio-group {
        background-color: white;
        border-radius: 8px;
        padding: 8px; /* 패딩 축소 */
        margin-bottom: 8px; /* 마진 축소 */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 컴팩트한 버튼 디자인 */
    #manual-tip-buttons button {
        padding: 4px 8px !important;
        min-width: 32px !important;
        font-size: 12px !important;
    }
    
    #ai-tip-buttons button {
        padding: 4px 8px !important;
        font-size: 11px !important;
        min-width: 80px !important;
    }
    
    /* 리뷰 입력 필드 최적화 */
    #review-input textarea {
        font-size: 12px !important;
        padding: 6px !important;
    }
    
    /* 결과 및 총액 디스플레이 최적화 */
    #tip-display, #total-bill-display, #order-summary {
        font-size: 12px !important;
        padding: 6px !important;
    }
    
    /* 그룹 제목 최적화 */
    #right-sidebar h3 {
        font-size: 14px !important;
        margin: 5px 0 !important;
    }
    
    /* 결제 버튼 최적화 */
    #payment-button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 14px !important;
        padding: 8px 15px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s !important;
        width: 100% !important;
    }
    
    #payment-button:hover {
        background-color: #45a049 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    #payment-button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* 평가 및 결제 제목 */
    #rating-title {
        margin-bottom: 10px;
        text-align: center;
        background: linear-gradient(to right, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 별점 그룹 */
    #rating-group {
        margin-bottom: 10px;
    }
    
    /* 리뷰 그룹 */
    #review-group {
        margin-bottom: 10px;
    }
    
    /* 팁 계산 그룹 */
    #tip-calculation-group {
        margin-bottom: 10px;
    }
    
    /* 결과 그룹 */
    #results-group {
        margin-bottom: 10px;
    }
    
    /* 청구서 그룹 */
    #bill-group {
        margin-top: auto;
        margin-bottom: 10px;
    }
    
    #bill-group .gradio-markdown {
        margin-bottom: 5px;
    }
    
    /* 결제 그룹 */
    #payment-group {
        margin-bottom: 0;
    }
    
    /* 예제 적용 버튼 스타일 */
    #apply-example1-btn, #apply-example2-btn {
        background-color: #4CAF50;
        color: white;
        width: 100%;
        margin-bottom: 10px;
    }
    
    #apply-example1-btn:hover, #apply-example2-btn:hover {
        background-color: #3e8e41;
    }
    
    #food-container .gradio-image {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        margin: 0 auto;
        display: block;
    }
    
    #food-container .gradio-image:hover {
        transform: scale(1.05);
    }
    
    /* 예제 섹션 스타일 */
    .gradio-examples {
        margin-top: 20px;
        border-top: 1px solid #ddd;
        padding-top: 20px;
    }
    
    .gradio-examples .gradio-table {
        font-size: 14px;
    }
    
    .gradio-examples img {
        max-width: 120px;
        max-height: 120px;
        object-fit: cover;
    }
    
    /* 별점 스타일 */
    #rating-input .gradio-radio {
        display: flex;
        gap: 5px;
    }
    
    #rating-input .gradio-radio label {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        font-size: 18px !important;
        background-color: #FFD700;
        color: white;
        border-radius: 50%;
        margin: 0 3px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    #rating-input .gradio-radio input:checked + label {
        transform: scale(1.2);
        background-color: #FFA500;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.7);
    }
    
    #rating-input .gradio-radio label:hover {
        transform: scale(1.1);
    }
    
    /* 작은 버튼 스타일 */
    .gradio-button[size="sm"] {
        font-size: 14px !important;
        padding: 8px 12px !important;
        margin: 4px !important;
        min-height: 36px !important;
    }
    
    /* 팁 버튼 스타일 */
    #tip-5, #tip-10, #tip-15, #tip-20, #tip-25 {
        width: 45px !important;
        height: 36px !important;
        padding: 0 !important;
        font-size: 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* AI 모델 버튼 스타일 - 작은 공간에 맞게 조정 */
    #mistral-button, #google-button, #gpt-button, #qwen-button {
        font-size: 12px !important;
        padding: 8px 6px !important;
        min-width: 90px !important;
    }
    
    /* 비디오 탭 스타일 */
    #video-input-standalone {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* 탭 스타일 */
    .gradio-tab {
        font-size: 18px !important;
        padding: 12px 16px !important;
    }

    /* Mistral 버튼 색상 */
    #mistral-button {
        background-color: #FF8C00 !important;
        color: white !important;
        border-color: #FF8C00 !important;
    }

    #mistral-button:hover {
        background-color: #FFA500 !important;
    }
    
    /* Google 버튼 색상 */
    #google-button {
        background-color: #50C878 !important;
        color: white !important;
        border-color: #50C878 !important;
    }
    
    #google-button:hover {
        background-color: #3CB371 !important;
    }

    /* Qwen 버튼 색상 */
    #qwen-button {
        background-color: #8A2BE2 !important;
        color: white !important;
        border-color: #8A2BE2 !important;
    }

    #qwen-button:hover {
        background-color: #7722CC !important;
    }
    
    /* 키오스크 텍스트 출력 스타일 */
    .gradio-container textarea.scroll-hide {
        font-family: monospace !important;
        white-space: pre-wrap !important;
        line-height: 1.5 !important;
    }

    /* JSON 출력 스타일 */
    .gradio-container textarea.scroll-hide pre {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 3px;
        padding: 10px;
        margin: 10px 0;
        overflow-x: auto;
    }

    /* 예제 비디오 섹션 스타일 */
    #examples-container {
        margin-bottom: 20px;
    }
    
    #examples-title {
        margin-bottom: 15px;
        color: #2196F3;
    }
    
    #example-bad, #example-good {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 0 10px;
    }
    
    #example-video-bad, #example-video-good {
        margin-bottom: 10px;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* 좌우 패널 간격 조정 */
    #menu-column, #right-sidebar {
        margin: 0 5px;
    }
    
    /* 메인 탭과 평가 탭 동시 표시 */
    #main-tab, #rating-tab {
        display: block !important;
    }
    
    /* 고정 레이아웃 추가 보강 */
    .gradio-app {
        min-width: var(--fixed-width) !important;
        width: var(--fixed-width) !important;
        max-width: var(--fixed-width) !important;
    }
    
    /* 모바일 디바이스 레이아웃 고정 */
    @media screen and (max-width: 600px), 
           screen and (max-device-width: 600px),
           screen and (orientation: portrait) {
        body {
            min-width: var(--fixed-width) !important;
            width: var(--fixed-width) !important;
            max-width: var(--fixed-width) !important;
            overflow-x: scroll !important;
        }
        
        .gradio-container {
            transform: none !important;
            width: var(--fixed-width) !important;
            max-width: var(--fixed-width) !important;
            margin: 0 !important;
            overflow-x: scroll !important;
        }
        
        /* 가로 스크롤 UI 최적화 */
        ::-webkit-scrollbar {
            height: 8px !important;
            background-color: #f5f5f5 !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background-color: #888 !important;
            border-radius: 4px !important;
        }
    }
    
    /* 확대/축소 방지 및 스크롤 UI 최적화 */
    * {
        -ms-content-zooming: none !important;
        -ms-user-select: none !important;
        touch-action: pan-x pan-y !important;
    }
    """

    # 팁 계산을 위한 기본 프롬프트 템플릿
    DEFAULT_PROMPT_TEMPLATE = '''
###Persona###
You are a tip calculation assistant. Based on the country, waiter's behavior, Google reviews, user reviews, and star rating, you must calculate an appropriate tip for the waiter. Since tipping percentages vary by country, follow the instructions below.
    a. Base Tip Percentages by Country
       1. USA: Casual dining 15–20%, Fine dining 20%, Buffet 10%
       2. UK: Casual dining 10–15%, Fine dining 10–15%
       3. Germany: Casual dining 5–10%, Fine dining 10%
       4. Argentina: Casual dining 10%, Fine dining 10%
            
###Task###
   1. Video Caption Analysis
    After analyzing the video, present the analysis results. Then, provide an analysis for each evaluation criterion and assign a score.
    *However, if the video caption analysis has already been completed, please use the existing scores for the video scoring process.*
    *You must either generate a score for all items or get something from an existing Caption and display it again. This item must be present.*
    You are required to justify every single item without exception.
      1-1. Video Evaluation Criteria
            a) Reliability:
                1: Frequent order mistakes, multiple billing errors
                2: No major mistakes but minor omissions occur
                3: Mostly accurate orders, occasional lack of confirmation
                4: Custom orders correctly handled, allergy/cooking preferences considered
                5: Always 100% accurate, uses repeat confirmation procedures

            b) Responsiveness:
                1: Ignores customer requests or extremely slow
                2: Responds but with significant delays
                3: Average response speed, acceptable performance
                4: Quick and polite responses
                5: Immediate response, friendly tone and facial expressions, proactive problem-solving

            c) Assurance:
                1: Unable to answer menu questions, rude or unfriendly
                2: Insufficient explanations, basic courtesy only
                3: Can explain the menu but lacks confidence
                4: Professional explanations, confident handling
                5: Earns customer trust, can offer premium suggestions like wine pairings

            d) Empathy:
                1: Shows no interest in customers, mechanical responses
                2: Provides only minimal consideration
                3: Basic friendliness but lacks personalization
                4: Remembers customer preferences, offers personalized services like birthday events
                5: Proactively understands customer emotions, provides impressive experiences

            e) Tangibles:
                1: Dirty table, utensils, and uniform
                2: Only some aspects are clean (e.g., utensils only)
                3: Acceptable but lacks attention to detail
                4: Uniform and table settings are tidy, cleanliness maintained
                5: Meets FDA Food Code standards, ServSafe certification level cleanliness
        
        **Video Evaluation Criteria entries must generate results unconditionally. **

        1-2. Video Score Calculation
            Video Score = (Reliability + Responsiveness + Assurance + Empathy + Tangibles)*4

    2. Google Review Analysis
        2-1. Analyze Google reviews and provide the results first.
        2-2. Scoring Method:
            a) A higher score for more positive mentions, lower for negative mentions. Assign 0 for ethical violations.
            b) Google review score is calculated out of 100.

    3. User Review Analysis
        3-1. Analyze user reviews and provide the results first.
        3-2. Scoring Method:
            a) A higher score for more positive mentions, lower for negative mentions. Assign 0 for ethical violations.
            b) User review score is calculated out of 100.

    4. Star rating
        4-1. Star rating is based on a 5-star scale.
        4-2. Star rating Score Calculation
            Star rating Score = (Star rating / 5) * 100
                              
    5. Total Score
        5-1. Total Score Calculation
           Total Score = Video Score + Google Review Score + User Review Score + Star rating Score
                              
    6. Tip Calculation
        6-1. Calculate the tip based on the total score and analysis.
        
            Tip Calculation Guide
               a) Categorize the service level as Poor, Normal, or Good based on the total score and review content.
                b) Determine the tipping percentage within the culturally appropriate range according to the selected country and restaurant type.
                    i) Tipping Ranges by Country and Restaurant Type
                        1) USA
                             Casual dining: Poor 3%, Normal 12~15%, Good 20%
                             Fine dining: Poor 4%, Normal 15~18%, Good 20%
                             Buffet: Poor 2%, Normal 7%, Good 10%

                        2) UK
                             Casual dining: Poor 2%, Normal 7~10%, Good 15%
                             Fine dining: Poor 2%, Normal 8~12%, Good 15%
                                 
                        3) Germany
                             Casual dining: Poor 1%, Normal 4~7%, Good 10%
                             Fine dining: Poor 2%, Normal 6~8%, Good 10%
                                 
                        4) Argentina
                             Casual dining: Poor 2%, Normal 5~7%, Good 10%
                             Fine dining: Poor 2%, Normal 5~7%, Good 10%
                                 
                    ii) Within the same level (Poor/Normal/Good), choose the lower or upper end of the range based on the positivity or negativity of the reviews and video.
                    iii) If there are any ethical issues, the tip must be set to 0%.
        6-2. Format
            Following the ###Output indicator### format.
                              
    7. Format
        7-1. Analysis Output Format
            The analysis must be presented in Markdown format.
        7-2. Tip Calculation Output Format
            Output indicator format as shown below.

###Guide###
    1. If there are ethical issues like racism or sexism mentioned in Google reviews, the tip percentage should be 0%.
    2. Even if the video score is high, if the user review score is low, user reviews should take priority and the weighting must be adjusted accordingly.
    3. Even if the waiter made a serious mistake, user reviews should take precedence.
    4. If there are issues in Google reviews but they have been resolved according to the user reviews, the Google review evaluation should be adjusted accordingly.
    5. After analyzing the video, clearly state the results of the video analysis, the scores for each criterion, and the reasons for those scores.
    6. Clearly state the reasons for each analysis.
    7. Clearly explain the reason for determining the final tip amount.
    8. You must complete all the tasks in order and then finally do the Json output. Never do the Json output alone.

###Output indicator###
    ```json
    {{{{
      "### Video Caption i(th) ###": i(th) "Full Video Scene Caption",
      "Final Reason": "Final Reason Summary",
      "final_tip_percentage": <calculated_percentage_int>,
      "final_tip_amount": <calculated_tip_float>,
      "final_total_bill": <calculated_total_bill_float>
    }}}}
    ```

###Video Caption###
{{caption_text}}

###Google Reviews###
{google_reviews}

###User Input###
    1. Country: USA
    2. Restaurant name: Peter Luger Steak House
    3. Calculated subtotal: ${calculated_subtotal:.2f}
    4. User reviews: {user_review}
    5. Star rating: {star_rating}
'''

    def __init__(self):
        """
        설정 클래스 초기화 및 리뷰 관리자 설정
        """
        # 구글 리뷰 URL 설정
        review_url = "https://www.google.com/maps/place/Peter+Luger+Steak+House/@40.7098814,-73.9650801,17z/data=!3m1!4b1!4m6!3m5!1s0x89c25bdedbcaf647:0x469a1d01423a03f2!8m2!3d40.7098774!4d-73.9625052!16zL20vMGI2MWt4?hl=en&entry=ttu&g_ep=EgoyMDI1MDUwNy4wIKXMDSoASAFQAw%3D%3D"
        self.google_review_manager = GoogleReviewManager(review_url, target_review_count=3)
        
        # 리뷰를 미리 포맷하여 저장
        self.GOOGLE_REVIEWS = GoogleReviewManager.format_google_reviews(
            self.google_review_manager.reviews_text
        )
        
        # 이미지 디렉토리 및 파일 확인
        self._validate_resources()
    
    def _validate_resources(self) -> None:
        """이미지 디렉토리 및 파일 확인"""
        # 서버 디렉토리 및 현재 작업 디렉토리 로깅
        logging.info(f"서버 디렉토리 경로: {self.SERVER_DIR}")
        logging.info(f"현재 작업 디렉토리: {os.getcwd()}")
        
        # 이미지 디렉토리 경로 확인
        images_dir = os.path.join(self.SERVER_DIR, "images")
        logging.info(f"이미지 디렉토리 경로: {images_dir}")
        
        if not os.path.exists(images_dir):
            logging.warning(f"경고: '{images_dir}' 폴더를 찾을 수 없습니다. 음식 이미지가 표시되지 않을 수 있습니다.")
            os.makedirs(images_dir, exist_ok=True)
            
        # 각 이미지 파일 존재 확인
        for item in self.FOOD_ITEMS:
            if not os.path.exists(item["image"]):
                logging.warning(f"경고: 이미지 파일을 찾을 수 없습니다 - {item['image']}")
            else:
                logging.info(f"이미지 파일 확인됨: {item['image']}") 