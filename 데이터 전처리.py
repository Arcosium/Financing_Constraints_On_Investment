import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# [설정] 경로 및 파일명
# ---------------------------------------------------------
# 요청하신 저장 경로
BASE_DIR = r"C:/Users/123/Documents/개인논문 - 금융 제약과 투자 파급효과/데이터"

# 수집된 파일 리스트 (파일명이 정확해야 합니다)
file_names = [
    'research_panel_data_2015_2018_20251201_1146.csv',
    'research_panel_data_2019_2023_20251201_0628.csv',
    'research_panel_data_2024_2024_20251201_1417.csv'
]

# 경로가 없으면 생성
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f"[Info] 폴더 생성됨: {BASE_DIR}")

# ---------------------------------------------------------
# 1. 데이터 로드 및 병합
# ---------------------------------------------------------
df_list = []

print("[Step 1] 파일 로드 및 병합 시작...")
for fname in file_names:
    # 스크립트 실행 위치에 파일이 있다고 가정하거나, 절대 경로를 지정해야 함
    # 여기서는 편의상 BASE_DIR에 원본 파일도 옮겨져 있다고 가정하거나,
    # 현재 작업 디렉토리에서 찾습니다.
    
    # 1. 현재 작업 폴더에서 찾기
    if os.path.exists(fname):
        path = fname
    # 2. 혹은 BASE_DIR에 넣어두셨다면
    elif os.path.exists(os.path.join(BASE_DIR, fname)):
        path = os.path.join(BASE_DIR, fname)
    else:
        print(f"[Warning] 파일을 찾을 수 없습니다: {fname} (경로 확인 필요)")
        continue
        
    # stock_code는 문자로 읽어야 '005930' 등이 유지됨
    temp_df = pd.read_csv(path, dtype={'stock_code': str})
    df_list.append(temp_df)
    print(f"  -> 로드 완료: {fname} ({len(temp_df)}건)")

if not df_list:
    raise ValueError("병합할 데이터가 없습니다. 파일 경로를 확인해주세요.")

# 전체 병합
df = pd.concat(df_list, axis=0, ignore_index=True)

# 중복 제거 (기업-연도 기준)
df = df.drop_duplicates(subset=['stock_code', 'year'], keep='last')
df = df.sort_values(by=['stock_code', 'year']).reset_index(drop=True)

print(f"[Merge] 총 {len(df)}건 데이터 병합 완료.\n")

# ---------------------------------------------------------
# 2. 결측치 보정 (Imputation Strategy)
# ---------------------------------------------------------
print("[Step 2] 결측치 보정 및 변수 재계산...")

# (1) 기초자산(Total Assets t-1) 복원
# API에서 total_assets_prev가 누락된 경우, 패널 구조를 이용해 전년도 자산을 가져옴
df['calc_assets_prev'] = df.groupby('stock_code')['total_assets'].shift(1)

# API 값이 0이거나 결측이면, 우리가 찾은 값으로 대체
df['total_assets_prev'] = df['total_assets_prev'].fillna(0)
mask_prev_missing = (df['total_assets_prev'] == 0)
df.loc[mask_prev_missing, 'total_assets_prev'] = df.loc[mask_prev_missing, 'calc_assets_prev']

# (2) 이자 비용(Interest) 확정
# 현금흐름표(interest_paid) 우선, 없으면 손익계산서(interest_expense)
df['interest_final'] = df['interest_paid'].fillna(0)
mask_no_cash_int = (df['interest_final'] == 0)
df.loc[mask_no_cash_int, 'interest_final'] = df.loc[mask_no_cash_int, 'interest_expense'].fillna(0)

# ---------------------------------------------------------
# 3. 파생변수 재계산 (Recalculation)
# ---------------------------------------------------------

# 1. 설비투자율 (Inv_Rate = CAPEX / Assets_prev)
# 분모가 0이면 계산 불가 -> NaN 처리
df['total_assets_prev'] = df['total_assets_prev'].replace(0, np.nan)
df['inv_rate'] = df['capex'] / df['total_assets_prev']

# 2. CFCR (Cash Flow Coverage Ratio)
# 분모(이자)가 0인 경우 -> 무차입 기업 -> CFCR 100(High) 부여
# 분모가 0이 아니면 정상 계산
df['cfcr_final'] = np.where(
    df['interest_final'] > 0,
    (df['ocf'] + df['interest_final']) / df['interest_final'],
    100.0  # 무차입 간주
)

# 3. ROA & Size
df['roa'] = df['net_income'] / df['total_assets']
df['size'] = np.log(df['total_assets'].replace(0, np.nan))

# 4. Sales Growth (매출액 증가율) - 패널 데이터라 계산 가능
df['sales_prev'] = df.groupby('stock_code')['sales'].shift(1)
df['sales_growth'] = (df['sales'] - df['sales_prev']) / df['sales_prev']

# ---------------------------------------------------------
# 4. 이상치 제거 (Winsorization) - 상하위 1%
# ---------------------------------------------------------
print("[Step 3] 이상치 제어 (Winsorization 1%)...")

def winsorize_series(series, limits=(0.01, 0.01)):
    # NaN 무시하고 quantile 계산
    return series.clip(lower=series.quantile(limits[0]), upper=series.quantile(1 - limits[1]))

# 윈저라이징 대상 변수
win_vars = ['inv_rate', 'cfcr_final', 'roa', 'sales_growth'] # Size는 로그변환 했으므로 보통 놔둠

for col in win_vars:
    # 무한대 제거
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    # 윈저라이징 적용
    df[f'{col}_win'] = winsorize_series(df[col])

# ---------------------------------------------------------
# 5. 최종 저장
# ---------------------------------------------------------
# 분석에 필요한 컬럼만 선택
final_cols = [
    'stock_code', 'company_name', 'year',
    'inv_rate_win', 'cfcr_final_win', 'roa_win', 'size', 'sales_growth_win', # 분석용(Y, X, Controls)
    'inv_rate', 'cfcr_final', 'roa', 'sales_growth', # 원본(참고용)
    'capex', 'total_assets', 'total_assets_prev', 'ocf', 'interest_final' # 기초 데이터
]

# 투자율(Y)이나 CFCR(X)이 없는 행은 분석 불가하므로 제거 (선택 사항)
df_final = df[final_cols].dropna(subset=['inv_rate_win', 'cfcr_final_win'])

save_name = 'korean_firms_panel_2015_2024_final.csv'
save_path = os.path.join(BASE_DIR, save_name)

df_final.to_csv(save_path, index=False, encoding='utf-8-sig')

print("-" * 60)
print(f"[Success] 전처리 완료 및 저장됨")
print(f"경로: {save_path}")
print(f"관측치 수: {len(df_final)}")
print("-" * 60)
print(df_final[['stock_code', 'year', 'inv_rate_win', 'cfcr_final_win']].head())