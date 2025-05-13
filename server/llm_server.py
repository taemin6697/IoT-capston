# llm_server.py

from flask import Flask, request, jsonify
import os
import shutil
import uuid
import cv2  # server.py의 VideoProcessor에서 사용하던 것으로, 여기서는 직접 사용 안함
import base64  # server.py의 VideoProcessor에서 사용하던 것으로, 여기서는 직접 사용 안함
from ollama import Client

# --- GoogleReviewManager 및 Selenium 관련 import ---
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# --- GoogleReviewManager 관련 import 끝 ---

# Flask 애플리케이션 생성
app = Flask(__name__)

# Ollama Mistral 모델을 로컬에서 호출하기 위한 클라이언트
# host='http://localhost:11434'는 Ollama 기본 주소 예시
try:
    local_client = Client(host='http://localhost:11434')
    local_client.list()  # Ollama 서버 연결 테스트
    print("Ollama client initialized and connected successfully.")
except Exception as e:
    print(f"Failed to initialize or connect to Ollama client: {e}")
    local_client = None


# --- GoogleReviewManager 클래스 정의 (server.py에서 가져옴) ---
class GoogleReviewManager:
    """
    구글 리뷰 크롤링을 통해 리뷰 데이터를 한 번만 가져와 텍스트로 저장하고,
    리뷰 문자열을 생성하는 클래스.
    """

    def __init__(self, url, target_review_count=20):
        self.url = url
        self.target_review_count = target_review_count
        self.reviews_text = self._fetch_reviews_selenium()

    def _fetch_reviews_selenium(self):
        df_reviews = self.google_review_crawling(self.target_review_count, self.url)
        if df_reviews.empty:
            return "(구글 리뷰를 불러오지 못했습니다.)"
        reviews = []
        for index, row in df_reviews.iterrows():
            # 예: [4.5 stars] Excellent service and food.
            reviews.append(f"[{row['Rating']} stars] {row['Review Text']}")
        # 각 리뷰를 개행 문자로 구분하여 하나의 문자열로 생성
        return "\n".join(reviews)

    def google_review_crawling(self, TARGET_REVIEW_COUNT, url):
        try:
            service = Service(ChromeDriverManager().install())
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")  # 헤드리스 모드
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")  # Docker 또는 특정 환경에서 필요할 수 있음
            options.add_argument("--disable-dev-shm-usage")  # Docker 또는 특정 환경에서 필요할 수 있음
            options.add_argument("--window-size=1200,900")  # 적절한 창 크기 설정
            options.add_argument("--lang=en-US,en;q=0.9")  # 언어 설정 (영어 우선)
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
            )  # 일반적인 User-Agent
            driver = webdriver.Chrome(service=service, options=options)
            app.logger.info("웹 드라이버 설정 완료 (헤드리스 모드).")
        except Exception as e:
            app.logger.error(f"웹 드라이버 설정 중 오류 발생: {e}")
            return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환

        reviews_data = []
        processed_keys = set()

        try:
            driver.get(url)
            app.logger.info(f"Google Maps 접속 시도: {url}")
            time.sleep(5)  # 페이지 로드 대기 시간 증가

            # 쿠키 동의창 등이 있을 경우 처리 (예시, 실제 버튼 선택자는 다를 수 있음)
            try:
                consent_button_selectors = [
                    (By.XPATH, '//button[.//span[contains(text(), "Accept all")]]'),
                    (By.XPATH, '//button[.//span[contains(text(), "Reject all")]]'),  # 또는 "모두 거부"
                    (By.CSS_SELECTOR, 'form[action*="consent"] button'),
                ]
                for selector in consent_button_selectors:
                    try:
                        consent_button = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable(selector)
                        )
                        consent_button.click()
                        app.logger.info("쿠키 동의 버튼 클릭됨.")
                        time.sleep(2)
                        break  # 하나라도 클릭되면 루프 종료
                    except TimeoutException:
                        continue  # 다음 선택자 시도
            except Exception as e:
                app.logger.warning(f"쿠키 동의창 처리 중 오류 또는 창 없음: {e}")

            # 리뷰 탭으로 이동
            review_tab_button = None
            possible_review_selectors = [
                (By.XPATH,
                 "//button[@role='tab'][@aria-label][contains(@aria-label, 'Reviews') or contains(@aria-label, '리뷰')]"),
                # Google의 aria-label이 더 구체적일 수 있음
                (By.CSS_SELECTOR, "button[jsaction*='pane.tabPane. συγκεκριмена καρτέλα'][data-tab-index='1']"),
                # data-tab-index가 리뷰탭인 경우
                (By.XPATH, "//button[contains(., '리뷰') or contains(., 'Reviews')]"),
            ]
            wait_for_review_tab = WebDriverWait(driver, 15)  # 대기 시간 증가
            for selector_type, selector_value in possible_review_selectors:
                try:
                    review_tab_button = wait_for_review_tab.until(
                        EC.element_to_be_clickable((selector_type, selector_value))
                    )
                    app.logger.info(f"리뷰 탭 버튼 찾음 (방식: {selector_type}, 값: {selector_value})")
                    break
                except TimeoutException:
                    app.logger.warning(f"리뷰 탭 버튼 못찾음 (방식: {selector_type}, 값: {selector_value})")
                    continue

            if not review_tab_button:
                app.logger.error("리뷰 탭 버튼을 찾을 수 없습니다.")
                # 현재 페이지 스크린샷 저장 (디버깅용)
                # driver.save_screenshot("debug_no_review_tab.png")
                raise NoSuchElementException("리뷰 탭 버튼 없음")

            driver.execute_script("arguments[0].click();", review_tab_button)  # JavaScript 클릭 시도
            app.logger.info("리뷰 탭 클릭 시도 완료.")
            time.sleep(3)

            # 정렬 버튼 클릭 후 '최신순' 선택
            try:
                sort_button_selector_str = "//button[contains(@aria-label, 'Sort reviews') or contains(@aria-label, '리뷰 정렬') or .//span[contains(text(), 'Sort') or contains(text(), '정렬')]]"
                sort_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, sort_button_selector_str))
                )
                app.logger.info("정렬 기준 버튼 찾음. 클릭 시도...")
                driver.execute_script("arguments[0].click();", sort_button)
                time.sleep(1)

                newest_option_selector_str = "//div[@role='menuitemradio'][contains(., 'Newest') or contains(., '최신순')]"
                newest_option = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, newest_option_selector_str))
                )
                app.logger.info("최신순 옵션 찾음. 클릭 시도...")
                driver.execute_script("arguments[0].click();", newest_option)
                app.logger.info("최신순으로 정렬 적용됨. 잠시 대기...")
                time.sleep(3)
            except (TimeoutException, NoSuchElementException) as e:
                app.logger.warning(f"정렬 적용 중 오류 발생 또는 관련 요소 없음: {e}. 기본 정렬 상태로 진행합니다.")

            # 스크롤 가능한 영역 찾기 (중요: 이 선택자는 Google Maps 업데이트에 따라 자주 변경됨)
            scrollable_div_selector = (By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde")  # server.py와 동일한 선택자
            try:
                scrollable_div = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(scrollable_div_selector)
                )
                app.logger.info("리뷰 스크롤 영역 찾음.")
            except TimeoutException:
                app.logger.error("리뷰 스크롤 영역을 찾을 수 없습니다. 전체 페이지 스크롤로 대체합니다.")
                # driver.save_screenshot("debug_no_scroll_div.png") # 디버깅용
                scrollable_div = driver.find_element(By.TAG_NAME, "body")  # body로 대체하거나 다른 방법 강구

            time.sleep(2)  # 스크롤 영역 로드 대기

            review_elements_selector = (By.CSS_SELECTOR, "div.jftiEf.fontBodyMedium")  # 개별 리뷰 컨테이너

            loop_count = 0
            max_loop = 30  # 스크롤 루프 최대 횟수
            no_new_reviews_count = 0  # 새로운 리뷰가 추가되지 않은 연속 횟수

            while len(reviews_data) < TARGET_REVIEW_COUNT and loop_count < max_loop and no_new_reviews_count < 3:
                loop_count += 1
                previous_review_count = len(reviews_data)

                all_review_elements = driver.find_elements(*review_elements_selector)
                app.logger.info(
                    f"Loop {loop_count}: 총 {len(all_review_elements)}개의 리뷰 요소 발견. 현재 수집된 리뷰 수: {len(reviews_data)}")

                for review_el in all_review_elements:
                    try:
                        # "더보기" 버튼 처리
                        try:
                            more_button = review_el.find_element(By.CSS_SELECTOR, "button.w8nwRe.kyuRq")
                            # JavaScript로 버튼 클릭 시도 (요소가 다른 것에 가려져 있을 수 있음)
                            driver.execute_script("arguments[0].scrollIntoView(true);", more_button)
                            time.sleep(0.2)
                            driver.execute_script("arguments[0].click();", more_button)
                            time.sleep(0.3)  # 내용 로드 대기
                        except NoSuchElementException:
                            pass  # "더보기" 버튼이 없을 수 있음
                        except Exception as e_more:
                            app.logger.warning(f"더보기 버튼 처리 중 오류: {e_more}")

                        reviewer_name_el = review_el.find_element(By.CSS_SELECTOR, "div.d4r55")
                        reviewer_name = reviewer_name_el.text.strip() if reviewer_name_el else "N/A"

                        review_date_el = review_el.find_element(By.CSS_SELECTOR, "span.rsqaWe")
                        review_date = review_date_el.text.strip() if review_date_el else "N/A"

                        unique_key = reviewer_name + "##" + review_date  # 중복 방지 키

                        # 리뷰 텍스트
                        review_text_el = review_el.find_element(By.CSS_SELECTOR, "span.wiI7pd")
                        review_text = review_text_el.text.strip() if review_text_el else ""

                        # 고유 키 생성 시 리뷰 텍스트 일부도 포함하여 동일인이 동일 날짜에 여러 리뷰 남기는 경우 구분
                        unique_key += "##" + review_text[:50]

                        if unique_key in processed_keys:
                            continue  # 이미 처리된 리뷰 건너뛰기
                        processed_keys.add(unique_key)

                        if review_text:  # 리뷰 텍스트가 있는 경우에만 추가
                            rating_str = "N/A"
                            try:
                                rating_span = review_el.find_element(By.CSS_SELECTOR, "span.kvMYJc")
                                rating_str = rating_span.get_attribute("aria-label") or "N/A"
                            except NoSuchElementException:
                                app.logger.warning("평점 요소를 찾지 못했습니다.")

                            rating_num = None
                            if "star" in rating_str.lower():  # "평점: 별표 5개 중 X개" 형식 등
                                try:
                                    rating_num = float(rating_str.lower().split("star")[0].split()[-1])
                                except (ValueError, IndexError):
                                    try:  # 다른 형식 시도 "X out of 5 stars"
                                        rating_num = float(rating_str.lower().split(" out of")[0])
                                    except:
                                        app.logger.warning(f"평점 파싱 실패: {rating_str}")

                            review_info = {
                                "Name": reviewer_name,
                                "Rating": rating_num,
                                "Date / Time Ago": review_date,
                                "Review Text": review_text.replace('\n', ' ')
                            }
                            reviews_data.append(review_info)
                            app.logger.info(
                                f"리뷰 추가: {reviewer_name}, 평점: {rating_num}, 날짜: {review_date} (총 {len(reviews_data)}개)")
                            if len(reviews_data) >= TARGET_REVIEW_COUNT:
                                break
                    except Exception as e_review:
                        app.logger.error(f"개별 리뷰 처리 중 오류: {e_review}")
                        # driver.save_screenshot(f"debug_review_processing_error_{len(reviews_data)}.png")
                        continue  # 다음 리뷰로 넘어감

                if len(reviews_data) >= TARGET_REVIEW_COUNT:
                    app.logger.info(f"목표 리뷰 수 ({TARGET_REVIEW_COUNT}개) 도달. 크롤링 중단.")
                    break

                if len(reviews_data) == previous_review_count:
                    no_new_reviews_count += 1
                    app.logger.info(f"새로운 리뷰가 추가되지 않음 (연속 {no_new_reviews_count}회).")
                else:
                    no_new_reviews_count = 0  # 새로운 리뷰가 추가되면 카운터 리셋

                if no_new_reviews_count >= 3:
                    app.logger.info("3회 연속 새로운 리뷰가 없어 스크롤을 중단합니다.")
                    break

                # 스크롤 실행
                if scrollable_div:
                    driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
                    app.logger.info("스크롤 실행됨.")
                else:  # 스크롤 div 못찾았으면 body 스크롤
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    app.logger.info("Body 스크롤 실행됨.")
                time.sleep(3)  # 스크롤 후 새 컨텐츠 로드 대기

            df_reviews = pd.DataFrame(reviews_data[:TARGET_REVIEW_COUNT])

        except Exception as e:
            app.logger.error(f"스크립트 실행 중 예기치 않은 오류 발생: {e}")
            # driver.save_screenshot("debug_unexpected_error.png")
            df_reviews = pd.DataFrame()  # 오류 시 빈 DataFrame
        finally:
            if 'driver' in locals() and driver:
                driver.quit()
                app.logger.info("웹 드라이버 종료됨.")

        return df_reviews


# --- GoogleReviewManager 클래스 정의 끝 ---


def process_text_with_ollama(text_input):
    """Ollama Mistral 모델로 텍스트를 처리하는 함수 (기존 process_video 함수 역할)"""
    if not local_client:
        return "Ollama client is not available."

    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': text_input,  # server.py에서 만들어진 전체 프롬프트가 여기로 전달됨
        }
    ]
    try:
        response = local_client.chat(
            model='mistral-small3.1:latest',  # 사용할 모델명
            messages=messages,
            options={
                "temperature": 0.0,
                "num_gpu": 99,  # 사용 가능한 GPU 코어 수 (Ollama 설정에 따라 다름)
                "top_p": 0.95,
            }
        )
        output_text = response['message']['content']
    except Exception as e:
        app.logger.error(f"Ollama 추론 중 오류: {e}")
        output_text = f"Error during Ollama inference: {e}"
    return output_text


@app.route('/process_video_api', methods=['POST'])
def api_process_video():
    """
    POST /process_video_api
    Form-Data:
      - text_input: 사용자 텍스트 입력 (Tip 계산, 추가 요청 등. 실제로는 전체 프롬프트)
    """
    text_input = request.form.get('text_input', '')
    if not text_input:
        return jsonify({'error': 'text_input is required'}), 400

    try:
        result_text = process_text_with_ollama(text_input)
    except Exception as e:
        app.logger.error(f"Error in /process_video_api: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({'output': result_text})


@app.route('/api/get_google_reviews', methods=['GET'])
def api_get_google_reviews():
    """
    GET /api/get_google_reviews
    Query Parameters:
      - url: Google Maps 장소 URL (필수)
      - count: 가져올 리뷰 수 (선택, 기본값 5)
    """
    place_url = request.args.get('url')
    target_count = request.args.get('count', default=5, type=int)

    if not place_url:
        return jsonify({"error": "URL parameter is required"}), 400

    app.logger.info(f"Google 리뷰 요청 수신: URL='{place_url}', Count={target_count}")

    try:
        review_manager = GoogleReviewManager(url=place_url, target_review_count=target_count)
        reviews_text_result = review_manager.reviews_text  # __init__에서 생성된 리뷰 텍스트

        if "(구글 리뷰를 불러오지 못했습니다.)" in reviews_text_result:
            app.logger.warning(f"리뷰 가져오기 실패: {place_url}")
            return jsonify({"reviews_text": reviews_text_result, "message": "Failed to fetch reviews from Google"}), 500

        app.logger.info(f"리뷰 성공적으로 가져옴: {place_url}, 첫 50자: {reviews_text_result[:50]}")
        return jsonify({"reviews_text": reviews_text_result})
    except Exception as e:
        app.logger.error(f"Google 리뷰 API 처리 중 심각한 오류 발생 for {place_url}: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # 로깅 설정 (선택 사항, 하지만 디버깅에 유용)
    if not app.debug:  # 운영 모드일 때 파일 로깅 등 설정 가능
        import logging
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler('llm_server.log', maxBytes=1024 * 1024 * 10, backupCount=5)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)

    app.logger.info("Flask 서버 시작 중...")
    # threaded=True 옵션은 개발 서버에서 여러 요청을 동시에 처리하는 데 도움을 줄 수 있으나,
    # Selenium과 같은 블로킹 I/O 작업이 많은 경우 Gunicorn, uWSGI 같은 WSGI 서버 사용을 권장합니다.
    app.run(host='0.0.0.0', port=8899, debug=True, threaded=True)