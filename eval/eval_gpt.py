import os
import sys
import csv
import json
import time
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import mean_squared_error

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eval_gpt_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('eval_gpt')

# OpenAI API 설정값 (직접 설정)
OPENAI_API_KEY = ""
GPT_MODEL = "gpt-4.1-nano"

# OpenAI 클라이언트 설정
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info(f"OpenAI 클라이언트 초기화 성공 (모델: {GPT_MODEL})")
except Exception as e:
    logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
    sys.exit(1)

def load_tipping_dataset():
    """HuggingFace에서 Reddit 게시물 기반 팁 데이터셋 로드"""
    try:
        dataset = load_dataset("kfkas/service-tipping-reddit-data-filtered_v2")
        logger.info(f"Reddit 게시물 기반 팁 데이터셋 로드 성공: {len(dataset['train'])} 레코드")
        return dataset['train']
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        sys.exit(1)

def generate_gpt_prediction(text_content, situation_caption, model_name=None):
    """GPT를 사용하여 Reddit 게시물에서 팁 예측 생성"""
    
    # 모델이 지정되지 않은 경우 기본값 사용
    if model_name is None:
        model_name = GPT_MODEL
    
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
    {{{{
      "### Video Caption i(th) ###": i(th) "Full Video Scene Caption",
      "Final Reason": "Final Reason Summary",
      "final_tip_percentage": <calculated_percentage_int>,
      "final_tip_amount": <calculated_tip_float>,
      "final_total_bill": <calculated_total_bill_float>
    }}}}
    ```



### Video Caption ###
{situation_caption}

### User Input ###
    1. Country: USA, Type: Casual dining
    2. Restaurant name: Gold Spoon
    3. Calculated subtotal: $25.00
'''
    
    try:
        # GPT 모델 호출
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )
        
        # 응답 전체 텍스트
        response_text = response.choices[0].message.content.strip()
        
        # JSON에서 팁 퍼센트 추출 시도
        try:
            import re
            import json
            
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
        logger.error(f"GPT 예측 중 오류 발생: {e} - 이 컬럼은 pass합니다.")
        return None, prompt, str(e)

def evaluate_model(max_samples=10, gpt_model=None):
    """Reddit 게시물 기반 서비스 팁 데이터에 대한 모델 평가 및 결과 저장"""
    # 모델이 지정되지 않은 경우 기본값 사용
    if gpt_model is None:
        gpt_model = GPT_MODEL
        
    dataset = load_tipping_dataset()
    
    results = []
    actual_tips = []
    predicted_tips = []
    
    # 처리할 최대 샘플 수 제한
    sample_count = 0
    failed_count = 0
    
    # 결과를 위한 디렉토리 생성
    os.makedirs('eval_results', exist_ok=True)
    
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
        
        # GPT 예측 생성
        try:
            result = generate_gpt_prediction(text_content, situation_caption, model_name=gpt_model)
            
            # 결과 튜플에서 예측 값, 프롬프트, 응답 텍스트 추출
            predicted_tip, prompt, response_text = result
            logger.info(response_text)            
            # 예측값이 None인 경우 -> 응답은 기록하되 실제 분석에서는 제외
            if predicted_tip is None:
                logger.warning(f"레코드 #{i}: 유효한 팁 예측값을 얻지 못했습니다. 이 컬럼은 pass하고 계속 진행합니다.")
                failed_count += 1
                
                # 결과를 JSON에 저장하되, 분석에서는 제외
                results.append({
                    **item,  # 모든 원본 필드 포함
                    'predicted_tip': 'N/A',  # 문자열로 표시
                    'error': 'N/A',           # 문자열로 표시
                    'prompt': prompt,
                    'response': response_text,
                    'status': 'partial_success'  # 전체가 실패한 것이 아닌 부분 성공으로 표시
                })
                
                # 레코드 카운트는 증가시키고 다음 반복으로 진행
                sample_count += 1
                continue
            
            # 유효한 예측값인 경우
            error = predicted_tip - actual_tip
            
            result_item = {
                **item,  # 모든 원본 필드 포함
                'predicted_tip': predicted_tip,
                'error': error,
                'prompt': prompt,
                'response': response_text,
                'status': 'success'
            }
            
            results.append(result_item)
            actual_tips.append(actual_tip)
            predicted_tips.append(predicted_tip)
            
            logger.info(f"레코드 #{i} (Reddit 게시물) - 실제: {actual_tip}%, 예측: {predicted_tip}%, 오차: {error}%")
            sample_count += 1
            
        except Exception as e:
            # 예외 발생 시 로그 기록 후 다음 레코드로 진행
            logger.error(f"레코드 #{i} 처리 중 예외 발생: {e} - 이 레코드는 pass하고 계속 진행합니다.")
            failed_count += 1
            
            # 예외가 발생했지만 기본 정보는 저장
            results.append({
                **item,  # 모든 원본 필드 포함
                'predicted_tip': 'ERROR',  # 오류 표시
                'error': 'ERROR',           # 오류 표시
                'prompt': 'ERROR',           # 오류 표시
                'response': str(e),           # 오류 메시지
                'status': 'error'
            })
            
            # 레코드 카운트는 증가시키고 다음 반복으로 진행
            sample_count += 1
            continue
            
        # API 호출 사이 간격 두기
        time.sleep(0.5)
    
    # 결과 평가 및 저장
    if predicted_tips:  # 성공적으로 예측된 데이터가 있는 경우
        # MSE 계산 - 각 샘플의 오차 제곱을 평균
        squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predicted_tips, actual_tips)]
        mse = sum(squared_errors) / len(squared_errors)  # 평균 제곱 오차
        rmse = np.sqrt(mse)  # 평균 제곱근 오차
        
        logger.info(f"평가 완료. 총 {len(predicted_tips)} Reddit 게시물 레코드 처리됨 (실패: {failed_count}개)")
        logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        # 결과를 CSV 파일로 저장 (프롬프트와 응답 제외)
        try:
            csv_file = os.path.join('eval_results', 'eval_gpt_results.csv')
            
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
            
            logger.info(f"결과가 {csv_file}에 저장되었습니다. (총 {len(results)}개 레코드)")
        except Exception as e:
            logger.error(f"CSV 파일 저장 중 오류 발생: {e}")
        
        # 상세 결과를 JSON으로 저장 (모든 정보 포함)
        try:
            json_file = os.path.join('eval_results', 'eval_gpt_results_full.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"상세 결과가 {json_file}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"JSON 파일 저장 중 오류 발생: {e}")
        
        # 간단한 통계 저장
        try:
            stats = {
                'total_records': len(results),
                'successful_predictions': len(predicted_tips),
                'failed_predictions': failed_count,
                'mse': mse,
                'rmse': rmse,
                'mean_actual_tip': np.mean(actual_tips),
                'mean_predicted_tip': np.mean(predicted_tips),
                'mean_abs_error': np.mean(np.abs(np.array(predicted_tips) - np.array(actual_tips))),
                'dataset_source': 'Reddit posts (kfkas/service-tipping-reddit-data-filtered_v2)',
                'model': gpt_model,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            stats_file = os.path.join('eval_results', 'eval_gpt_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"평가 통계가 {stats_file}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"통계 JSON 파일 저장 중 오류 발생: {e}")
    else:
        logger.warning("평가할 유효한 결과가 없습니다.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT 모델을 사용한 팁 계산 평가')
    parser.add_argument('--model', type=str, default=GPT_MODEL, 
                        help=f'사용할 GPT 모델 (기본값: {GPT_MODEL})')
    parser.add_argument('--samples', type=int, default=105, 
                        help='평가할 최대 샘플 수 (기본값: 10)')
    
    args = parser.parse_args()
    
    logger.info(f"GPT 모델({args.model})을 사용한 Reddit 게시물 기반 팁 예측 평가 시작")
    evaluate_model(max_samples=args.samples, gpt_model=args.model)
    logger.info("평가 완료") 