import OpenDartReader
import pandas as pd
import time
import numpy as np
import os
from datetime import datetime

# ---------------------------------------------------------
# [필수] API 키 입력
# ---------------------------------------------------------
API_KEY = '6f6f89581f4ebd606b31de2907e50fe9a8c3628e'  # 여기에 본인의 API 키를 입력하세요
dart = OpenDartReader(API_KEY)

# ---------------------------------------------------------
# [설정] 경로 및 파일명
# ---------------------------------------------------------
BASE_DIR = 'C:/Users/123/Documents/개인논문 - 금융 제약과 투자 파급효과/데이터'
SAVE_DIR = BASE_DIR
DEBUG_DIR = os.path.join(BASE_DIR, 'debug_logs') # 디버깅용 폴더
INPUT_CSV_FILENAME = 'KRX TMI.csv' 

# ---------------------------------------------------------
# 1. 기업 목록 로드
# ---------------------------------------------------------
def load_target_stocks(csv_path):
    try:
        try:
            df = pd.read_csv(csv_path, dtype={'종목코드': str}, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, dtype={'종목코드': str}, encoding='utf-8-sig')
            
        if '종목코드' in df.columns and '종목명' in df.columns:
            return df[['종목코드', '종목명']].drop_duplicates()
        else:
            print("[Error] CSV에 '종목코드' 또는 '종목명' 컬럼이 없습니다.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[Error] 파일 로드 실패: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# 2. 재무 데이터 추출 함수 (이자비용 추가 및 로직 보완)
# ---------------------------------------------------------
def get_financial_panel_data(corp_code, year, corp_name_debug='Unknown', reprt_code='11011'):
    """
    한 기업의 특정 연도 사업보고서에서 연구에 필요한 모든 변수를 추출
    """
    try:
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)

        # 1. 연결재무제표(CFS) 우선 조회
        fs = dart.finstate_all(corp_code, year, reprt_code=reprt_code, fs_div='CFS')
        
        # 2. 연결 데이터가 없으면 개별재무제표(OFS) 조회
        if fs is None or fs.empty:
            print(f"   [Debug] {year}년 연결(CFS) 없음. 개별(OFS) 시도...")
            fs = dart.finstate_all(corp_code, year, reprt_code=reprt_code, fs_div='OFS')
        
        # -------------------------------------------------------
        # [DEBUG] 원본 데이터 저장
        # -------------------------------------------------------
        debug_filename = f"raw_{corp_name_debug}_{year}.csv"
        debug_path = os.path.join(DEBUG_DIR, debug_filename)
        
        if fs is not None and not fs.empty:
            fs.to_csv(debug_path, index=False, encoding='utf-8-sig')
        else:
            print(f"   [Critical] API가 데이터를 반환하지 않음 (None or Empty).")
            return None
        # -------------------------------------------------------

        # [전처리] 컬럼 유효성 검사 및 결측치 방어
        if 'frmtrm_amount' not in fs.columns:
            fs['frmtrm_amount'] = '0'
        
        # 쉼표(,) 제거 후 숫자 변환
        fs['thstrm_amount'] = fs['thstrm_amount'].astype(str).str.replace(',', '')
        fs['frmtrm_amount'] = fs['frmtrm_amount'].astype(str).str.replace(',', '')

        fs['thstrm_amount'] = pd.to_numeric(fs['thstrm_amount'], errors='coerce').fillna(0)
        fs['frmtrm_amount'] = pd.to_numeric(fs['frmtrm_amount'], errors='coerce').fillna(0)
        
        # 계정명 정규화 (공백 제거)
        fs['account_nm_clean'] = fs['account_nm'].astype(str).str.replace(" ", "").str.strip()
        
        # -------------------------------------------------------
        # A. 현금흐름표 (CFS/OFS) - OCF, 이자지급(Cash), CAPEX
        # -------------------------------------------------------
        
        # 1. OCF (영업활동현금흐름)
        ocf_val = fs.loc[fs['account_nm_clean'].isin(['영업활동으로인한현금흐름', '영업활동현금흐름']), 'thstrm_amount'].max()
        
        # 2. 이자지급 (Interest Paid - Cash Basis)
        # "이자의지급", "이자지급", "이자비용지급" 등 다양할 수 있음
        int_paid_accs = ['이자의지급', '이자지급', '금융이자의지급']
        int_paid_raw = fs.loc[fs['account_nm_clean'].isin(int_paid_accs), 'thstrm_amount'].max()
        interest_paid = abs(int_paid_raw) if int_paid_raw != 0 else 0
        
        # 3. CAPEX (유형자산 취득)
        capex_raw = fs.loc[fs['account_nm_clean'].isin(['유형자산의취득', '유형자산취득', '설비자산의취득']), 'thstrm_amount'].max()
        capex = abs(capex_raw)

        # -------------------------------------------------------
        # B. 재무상태표 (BS) - 자산총계 (Total Assets)
        # -------------------------------------------------------
        assets_curr = fs.loc[fs['account_nm_clean'].isin(['자산총계']), 'thstrm_amount'].max()
        assets_prev = fs.loc[fs['account_nm_clean'].isin(['자산총계']), 'frmtrm_amount'].max()

        # -------------------------------------------------------
        # C. 손익계산서 (IS/CIS) - 매출액, 당기순이익, [추가] 이자비용
        # -------------------------------------------------------
        # 1. 매출액
        sales_accs = ['매출액', '수익(매출액)', '영업수익']
        sales = fs.loc[fs['account_nm_clean'].isin(sales_accs), 'thstrm_amount'].max()
        
        # 2. 당기순이익
        ni_accs = ['당기순이익', '당기순이익(손실)']
        net_income = fs.loc[fs['account_nm_clean'].isin(ni_accs), 'thstrm_amount'].max()

        # 3. [추가] 이자비용 (Interest Expense - Accrual Basis)
        # 손익계산서상의 이자비용. 이자지급(CF)이 0일 때 대용치로 사용 가능
        int_exp_accs = ['이자비용', '금융원가', '금융비용'] 
        # 금융원가/금융비용은 이자비용보다 넓은 개념이지만, 이자비용 단독 계정이 없으면 사용
        # 우선순위: 이자비용 > 금융원가 > 금융비용 (max값은 위험하므로 우선순위 로직 적용 필요하나, 여기선 이자비용 우선 탐색)
        
        interest_expense = 0
        for acc in ['이자비용', '금융원가', '금융비용']:
            val = fs.loc[fs['account_nm_clean'] == acc, 'thstrm_amount'].max()
            if val > 0:
                interest_expense = val
                break
        
        # -------------------------------------------------------
        # D. 파생 변수 계산
        # -------------------------------------------------------
        
        # CFCR 계산 (우선순위: 현금흐름표상 이자지급 -> 없으면 손익계산서상 이자비용)
        # 분모로 사용할 이자 항목 결정
        denominator_interest = interest_paid if interest_paid > 0 else interest_expense

        if denominator_interest > 0:
            cfcr = (ocf_val + denominator_interest) / denominator_interest
        else:
            cfcr = 100.0 # 무차입으로 간주

        return {
            'ocf': ocf_val,
            'interest_paid': interest_paid,       # 현금주의 이자
            'interest_expense': interest_expense, # 발생주의 이자 (NEW)
            'capex': capex,
            'assets_curr': assets_curr,
            'assets_prev': assets_prev,
            'sales': sales,
            'net_income': net_income,
            'cfcr': cfcr
        }

    except Exception as e:
        print(f"   [Error] 처리 중 예외 발생: {e}")
        return None

# ---------------------------------------------------------
# 3. 메인 로직
# ---------------------------------------------------------
def main():
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
        except Exception:
            pass

    input_csv_path = os.path.join(BASE_DIR, INPUT_CSV_FILENAME)
    target_df = load_target_stocks(input_csv_path)
    
    if target_df.empty:
        return

    # [테스트 설정]
    start_year = 2024
    end_year = 2024
    target_years = [str(y) for y in range(start_year, end_year + 1)]
    
    test_limit = None 
    
    if test_limit:
        print(f"DEBUG MODE: 시총 상위 {test_limit}개 기업만 수행합니다.")
        target_df = target_df.head(test_limit)

    print(f"\n[Info] 총 {len(target_df)}개 기업, {len(target_years)}년치 데이터 수집 시작")
    
    try:
        corp_list = dart.corp_codes.copy()
        corp_list['stock_code'] = corp_list['stock_code'].astype(str).str.strip()
        target_df = target_df.merge(corp_list[['corp_code', 'stock_code']], left_on='종목코드', right_on='stock_code', how='left')
        target_df = target_df.dropna(subset=['corp_code'])
    except Exception:
        print("[Error] 기업 코드 매핑 실패 (API Key 확인 필요)")
        return

    all_results = []
    total_steps = len(target_df)
    
    for idx, row in target_df.iterrows():
        stock_code = row['종목코드']
        corp_name = row['종목명']
        corp_code = row['corp_code']
        
        print(f"[{idx+1}/{total_steps}] {corp_name} ({stock_code}) 데이터 수집 중...")
        
        for year in target_years:
            data = get_financial_panel_data(corp_code, year, corp_name_debug=corp_name)
            time.sleep(0.3) 
            
            if data:
                record = {
                    'stock_code': stock_code,
                    'company_name': corp_name,
                    'year': year,
                    'total_assets': data['assets_curr'],
                    'total_assets_prev': data['assets_prev'],
                    'capex': data['capex'],
                    'ocf': data['ocf'],
                    'interest_paid': data['interest_paid'],       # 현금흐름표 이자
                    'interest_expense': data['interest_expense'], # 손익계산서 이자
                    'sales': data['sales'],
                    'net_income': data['net_income'],
                    'cfcr': data['cfcr']
                }
                all_results.append(record)
            else:
                pass
        
        if (idx + 1) % 100 == 0:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(os.path.join(SAVE_DIR, f'panel_data_intermediate_{idx+1}.csv'), index=False, encoding='utf-8-sig')

    if all_results:
        final_df = pd.DataFrame(all_results)
        
        final_df['inv_rate'] = final_df['capex'] / final_df['total_assets_prev']
        final_df['roa'] = final_df['net_income'] / final_df['total_assets']
        final_df['size'] = np.log(final_df['total_assets'].replace(0, np.nan))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'research_panel_data_{start_year}_{end_year}_{timestamp}.csv'
        final_path = os.path.join(SAVE_DIR, filename)
        
        final_df.to_csv(final_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[Success] 최종 패널 데이터 저장 완료: {final_path}")
        print(f"총 관측치(N*T): {len(final_df)}개")
        # 이자비용 컬럼도 출력하여 확인
        print(final_df[['company_name', 'year', 'interest_paid', 'interest_expense', 'cfcr']].head())
    else:
        print("\n[Fail] 수집된 데이터가 없습니다.")
        print(f" -> '{DEBUG_DIR}' 폴더 내의 raw_*.csv 파일들을 확인해보세요.")

if __name__ == "__main__":
    main()