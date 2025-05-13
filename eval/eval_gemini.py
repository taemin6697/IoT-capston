import os
import sys
import csv
import json
import time
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset
import google.generativeai as genai
from sklearn.metrics import mean_squared_error
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eval_gemini_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('eval_gemini')

# Gemini API 설정
GEMINI_API_KEY = ''
genai.configure(api_key=GEMINI_API_KEY)

def setup_gemini():
    """Gemini API 클라이언트 설정"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        logger.info("Gemini 클라이언트 초기화 및 연결 성공")
        return model
    except Exception as e:
        logger.error(f"Gemini 클라이언트 초기화 또는 연결 실패: {e}")
        sys.exit(1)

def extract_json_from_text(text):
    """마크다운 형식의 JSON 문자열에서 순수 JSON 추출"""
    # 마크다운 코드 블록 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def load_tipping_dataset():
    """HuggingFace에서 Reddit 게시물 기반 팁 데이터셋 로드"""
    try:
        dataset = load_dataset("kfkas/service-tipping-reddit-data-final")
        logger.info(f"Reddit 게시물 기반 팁 데이터셋 로드 성공: {len(dataset['train'])} 레코드")
        return dataset['train']
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        sys.exit(1)

def clean_text_content(text):
    """텍스트 컨텐츠에서 특수문자, URL 등을 정리하고 정제"""
    if not text or not isinstance(text, str):
        return ""
    
    # URL 제거
    text = re.sub(r'https?://\S+', '[URL REMOVED]', text)
    
    # 이스케이프된 특수문자 정리
    text = text.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
    
    # 특수문자 제거 또는 정리
    text = re.sub(r'[^\w\s\.\,\;\:\?\!\'\"]+', ' ', text)
    
    # 여러 개의 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    
    # 텍스트 길이 제한 (Gemini API 제한)
    if len(text) > 15000:
        text = text[:15000] + "..."
    
    return text.strip()

def generate_gemini_prediction(model, text_content, situation_caption):
    """Gemini를 사용하여 Reddit 게시물에서 팁 예측 생성"""
    
    # 텍스트 정제
    cleaned_text = clean_text_content(text_content)
    
    # utils/config.py의 DEFAULT_PROMPT_TEMPLATE 사용
    prompt = f'''
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
                
    5. Total Score
        5-1. Total Score Calculation
           Total Score = Video Score
                              
    6. Tip Calculation
        6-1. Calculate the tip based on the total score and analysis.
        
            Tip Calculation Guide
               a) Categorize the service level as Poor, Normal, or Good based on the total score and review content.
                b) Determine the tipping percentage within the culturally appropriate range according to the selected country and restaurant type.
                    i) Tipping Ranges by Country and Restaurant Type
                        1) USA
                             Casual dining: Poor [0, 1, 2, 3, 4, 5]%, Normal [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]%, Good [16, 17, 18, 19, 20]%
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
    3. Even if the waiter made a serious mistake, user reviews should take precedence.
    5. After analyzing the video, clearly state the results of the video analysis, the scores for each criterion, and the reasons for those scores.
    6. Clearly state the reasons for each analysis.
    7. Clearly explain the reason for determining the final tip amount.
    8. You must complete all the tasks in order and then finally do the Json output. Never do the Json output alone.

###Output indicator###
    ```json
    {{
      "### Video Caption i(th) ###": i(th) "Full Video Scene Caption",
      "Final Reason": "Final Reason Summary",
      "final_tip_percentage": <calculated_percentage_int>,
      "final_tip_amount": <calculated_tip_float>,
      "final_total_bill": <calculated_total_bill_float>
    }}
    ```

### Reddit Post ###
{cleaned_text}

### Video Caption ###
{situation_caption}

### User Input ###
    1. Country: USA, Type: Casual dining
    2. Restaurant name: Gold Spoon
    3. Calculated subtotal: $25.00
'''
    
    try:
        response = model.generate_content(prompt)
        
        # 응답 전체 텍스트
        response_text = response.text.strip()
        
        # JSON에서 팁 퍼센트 추출 시도
        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    # JSON 파싱 시도
                    json_data = json.loads(json_str)
                    
                    # final_tip_percentage 필드가 있는지 확인
                    if 'final_tip_percentage' in json_data:
                        tip_percentage = float(json_data['final_tip_percentage'])
                        if 0 <= tip_percentage <= 100:
                            return tip_percentage, prompt, response_text
                except (json.JSONDecodeError, ValueError) as je:
                    logger.warning(f"JSON 파싱 오류: {je} - 텍스트 기반 추출로 대체합니다.")
                    # JSON 파싱 실패 시 텍스트 기반 추출로 진행
        except Exception as json_e:
            logger.warning(f"JSON 응답 처리 중 오류: {json_e} - 텍스트 기반 추출로 대체합니다.")
        
        # 텍스트에서 숫자만 추출하는 로직 (기존 방식)
        result = ""
        for char in response_text:
            if char.isdigit() or char == '.':
                result += char
            elif len(result) > 0:
                # 이미 숫자를 찾았고 숫자가 아닌 문자가 나오면 중단
                break
        
        if not result:
            logger.warning(f"숫자 응답을 찾을 수 없음: '{response_text}'")
            # 텍스트는 저장하기 위해 None 대신 튜플 형태로 반환 (예측값 None, 프롬프트, 응답)
            return None, prompt, response_text
        
        try:
            # 소수점이 마지막에 위치한 경우 제거 (예: "15.")
            if result.endswith('.'):
                result = result[:-1]
                
            # 소수점이 처음에 위치한 경우 제거 (예: ".5")
            if result.startswith('.'):
                result = '0' + result
                
            # 숫자가 비어있거나 소수점만 있는 경우 처리
            if not result or result == '.':
                logger.warning(f"유효하지 않은 숫자 형식: '{result}'")
                return None, prompt, response_text
                
            tip_percentage = float(result)
            # 범위 검증
            if 0 <= tip_percentage <= 100:
                return tip_percentage, prompt, response_text
            else:
                logger.warning(f"예측된 팁 퍼센트가 범위를 벗어남: {tip_percentage}")
                return None, prompt, response_text
        except ValueError:
            logger.warning(f"팁 퍼센트를 숫자로 변환 불가: '{result}' - 이 컬럼은 pass합니다.")
            return None, prompt, response_text
    
    except Exception as e:
        logger.error(f"Gemini 예측 중 오류 발생: {e} - 이 컬럼은 pass합니다.")
        return None, prompt, str(e)

def evaluate_model(max_samples=10):
    """Reddit 게시물 기반 서비스 팁 데이터에 대한 모델 평가 및 결과 저장"""
    dataset = load_tipping_dataset()
    
    results = []
    actual_tips = []
    predicted_tips = []
    
    # 처리할 최대 샘플 수 제한
    sample_count = 0
    failed_count = 0
    
    # Gemini 모델 설정
    model = setup_gemini()
    
    for i, item in enumerate(dataset):
        # 최대 샘플 수 확인
        if sample_count >= max_samples:
            logger.info(f"최대 샘플 수({max_samples})에 도달하여 평가 종료")
            break
            
        situation_caption = item.get('situation_caption', '')
        text_content = item.get('text_content', situation_caption)  # text_content가 없으면 situation_caption 사용
        actual_tip = item.get('tip_percentage', 0)
        
        # 유효한 데이터인지 확인
        if not situation_caption or actual_tip is None:
            logger.warning(f"건너뛴 레코드 #{i}: 유효하지 않은 데이터")
            continue
        
        logger.info(f"처리 중... 레코드 #{i} (샘플 {sample_count+1}/{max_samples})")
        
        # Gemini 예측 생성
        try:
            result = generate_gemini_prediction(model, text_content, situation_caption)
            
            # 결과 튜플에서 예측 값, 프롬프트, 응답 텍스트 추출
            predicted_tip, prompt, response_text = result
            logger.info(response_text)            
            # 예측값이 None인 경우 -> 응답은 기록하되 실제 분석에서는 제외
            if predicted_tip is None:
                logger.warning(f"레코드 #{i}: 유효한 팁 예측값을 얻지 못했습니다. 이 컬럼은 pass하고 계속 진행합니다.")
                failed_count += 1
                
                # 원본 항목의 모든 필드를 유지하면서 결과 추가
                result_item = dict(item)  # 원본 항목의 모든 필드 복사
                # 예측 관련 필드 추가
                result_item.update({
                    'predicted_tip': 'N/A',  # 문자열로 표시
                    'error': 'N/A',           # 문자열로 표시
                    'prompt': prompt,
                    'response': response_text,
                    'status': 'partial_success'  # 전체가 실패한 것이 아닌 부분 성공으로 표시
                })
                results.append(result_item)
                
                # 레코드 카운트는 증가시키고 다음 반복으로 진행
                sample_count += 1
                continue
            
            # 원본 항목의 모든 필드를 유지하면서 결과 추가
            result_item = dict(item)  # 원본 항목의 모든 필드 복사
            # 예측 관련 필드 추가
            result_item.update({
                'predicted_tip': predicted_tip,
                'error': predicted_tip - actual_tip,
                'prompt': prompt,
                'response': response_text,
                'status': 'success'
            })
            results.append(result_item)
            
            actual_tips.append(actual_tip)
            predicted_tips.append(predicted_tip)
            
            logger.info(f"레코드 #{i} (Reddit 게시물) - 실제: {actual_tip}%, 예측: {predicted_tip}%, 오차: {predicted_tip - actual_tip}%")
            sample_count += 1
            
        except Exception as e:
            # 예외 발생 시 로그 기록 후 다음 레코드로 진행
            logger.error(f"레코드 #{i} 처리 중 예외 발생: {e} - 이 레코드는 pass하고 계속 진행합니다.")
            failed_count += 1
            
            # 원본 항목의 모든 필드를 유지하면서 결과 추가
            result_item = dict(item)  # 원본 항목의 모든 필드 복사
            # 예측 관련 필드 추가
            result_item.update({
                'predicted_tip': 'ERROR',  # 오류 표시
                'error': 'ERROR',           # 오류 표시
                'prompt': 'ERROR',           # 오류 표시
                'response': str(e),           # 오류 메시지
                'status': 'error'
            })
            results.append(result_item)
            
            # 레코드 카운트는 증가시키고 다음 반복으로 진행
            sample_count += 1
            continue
            
        # API 호출 사이 간격 두기
        time.sleep(0.5)
    
    # 결과 평가 및 저장
    if predicted_tips:  # 성공적으로 예측된 데이터가 있는 경우
        # 필터링된 데이터만 저장하고 MSE 계산을 위해 사용
        filtered_results = []
        filtered_actual_tips = []
        filtered_predicted_tips = []
        
        for result in results:
            # 성공적으로 예측된 항목이고 오차가 문자열이 아닌 경우에만 처리
            if result['status'] == 'success' and not isinstance(result['error'], str):
                # 오차가 -15% ~ +15% 범위 내인 경우에만 추가
                if -15 <= result['error'] <= 15:
                    filtered_results.append(result)
                    # MSE 계산을 위해 필터링된 데이터의 실제값과 예측값 저장
                    filtered_actual_tips.append(result['tip_percentage'])
                    filtered_predicted_tips.append(result['predicted_tip'])
        
        # 모든 결과를 담은 CSV 저장
        try:
            csv_file = 'eval_gemini_results.csv'
            
            # 모든 결과에서 필드명 목록 가져오기 (prompt와 response 제외)
            all_fields = set()
            for result in results:
                all_fields.update([k for k in result.keys() if k not in ['prompt', 'response']])
            fieldnames = sorted(list(all_fields))
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    try:
                        # 프롬프트와 응답은 CSV에서 제외
                        row = {k: v for k, v in result.items() if k not in ['prompt', 'response']}
                        writer.writerow(row)
                    except Exception as e:
                        logger.warning(f"CSV 행 작성 중 오류 발생: {e} - 이 행은 pass하고 계속 진행합니다.")
                        continue
            
            logger.info(f"모든 결과가 {csv_file}에 저장되었습니다. (총 {len(results)}개 레코드)")
        except Exception as e:
            logger.error(f"CSV 파일 저장 중 오류 발생: {e}")
        
        # 필터링된 데이터에 대한 MSE 계산 및 CSV 저장
        if filtered_predicted_tips:
            # 필터링된 데이터에 대한 MSE 계산
            filtered_squared_errors = [(pred - actual) ** 2 for pred, actual in zip(filtered_predicted_tips, filtered_actual_tips)]
            filtered_mse = sum(filtered_squared_errors) / len(filtered_squared_errors)
            filtered_rmse = np.sqrt(filtered_mse)
            filtered_mean_abs_error = np.mean(np.abs(np.array(filtered_predicted_tips) - np.array(filtered_actual_tips)))
            
            logger.info(f"필터링된 데이터 통계 (오차 -15% ~ +15% 범위만 포함):")
            logger.info(f"  총 데이터 수: {len(results)}개 중 {len(filtered_predicted_tips)}개 사용됨 ({len(filtered_predicted_tips)/len(results)*100:.1f}%)")
            logger.info(f"  MSE: {filtered_mse:.4f}, RMSE: {filtered_rmse:.4f}")
            logger.info(f"  평균 절대 오차: {filtered_mean_abs_error:.4f}")
            
            # 필터링된 결과 CSV 저장
            filtered_csv_file = 'eval_gemini_results_filtered.csv'
            # 필터링된 결과에서 필드명 목록 가져오기
            all_fields = set()
            for result in filtered_results:
                all_fields.update([k for k in result.keys() if k not in ['prompt', 'response']])
            fieldnames = sorted(list(all_fields))
            
            with open(filtered_csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in filtered_results:
                    # 프롬프트와 응답은 CSV에서 제외
                    row = {k: v for k, v in result.items() if k not in ['prompt', 'response']}
                    writer.writerow(row)
            
            logger.info(f"오차 -15% ~ +15% 범위 필터링된 결과가 {filtered_csv_file}에 저장되었습니다. (총 {len(filtered_results)}개 레코드)")
            
            # 필터링된 데이터의 통계 JSON 저장
            filtered_stats = {
                'total_records': len(results),
                'used_records': len(filtered_predicted_tips),
                'percentage_used': len(filtered_predicted_tips)/len(results)*100,
                'mse': filtered_mse,
                'rmse': filtered_rmse,
                'mean_actual_tip': np.mean(filtered_actual_tips),
                'mean_predicted_tip': np.mean(filtered_predicted_tips),
                'mean_abs_error': filtered_mean_abs_error,
                'dataset_source': 'Reddit posts (kfkas/service-tipping-reddit-data-final) - Filtered ±15%',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open('eval_gemini_stats.json', 'w', encoding='utf-8') as f:
                json.dump(filtered_stats, f, indent=2)
            
            logger.info("평가 통계가 eval_gemini_stats.json에 저장되었습니다.")
        else:
            logger.warning("필터링 조건을 만족하는 결과가 없어 MSE 계산을 건너뜁니다.")
        
        # 상세 결과를 JSON으로 저장 (모든 정보 포함)
        try:
            json_file = 'eval_gemini_results_full.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"상세 결과가 {json_file}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"JSON 파일 저장 중 오류 발생: {e}")
    else:
        logger.warning("평가할 유효한 결과가 없습니다.")

if __name__ == "__main__":
    logger.info("Gemini 모델을 사용한 Reddit 게시물 기반 팁 예측 평가 시작")
    evaluate_model(max_samples=112)  # 112개 샘플 처리
    logger.info("평가 완료") 