import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ---------------------------------------------------------
# [설정] 경로 및 파일명
# ---------------------------------------------------------
BASE_DIR = r"C:/Users/123/Documents/개인논문 - 금융 제약과 투자 파급효과/데이터"
PANEL_FILE = "korean_firms_panel_2015_2024_final.csv"
MACRO_FILE = "거시경제지표.csv" # 사용자가 직접 올린 파일

# ---------------------------------------------------------
# 1. 사용자 거시 데이터 로드 및 전처리
# ---------------------------------------------------------
macro_path = os.path.join(BASE_DIR, MACRO_FILE)

if not os.path.exists(macro_path):
    raise FileNotFoundError(f"거시경제지표 파일을 찾을 수 없습니다: {macro_path}")

# ECOS csv는 구조가 복잡하므로, 데이터가 있는 부분만 깔끔하게 파싱합니다.
# csv 읽기 (헤더가 첫 줄에 있다고 가정)
df_raw = pd.read_csv(macro_path, encoding='utf-8-sig') # 혹은 cp949

print("[Raw Data Loaded]")
# 데이터 구조 확인을 위한 출력
# print(df_raw.head())

# 데이터 정제 (Wide -> Long Format 변환)
# 1. 필요한 행만 추출 (키워드 기반)
# 파일의 항목명을 정확히 매핑해야 합니다.
# snippet 기준: "무담보콜금리 전체", "국내총생산(실질성장률)", "소비자물가지수" (총지수)

# 키워드 매핑
indicators = {
    '무담보콜금리 전체': 'call_rate',
    '국내총생산(실질성장률)': 'gdp_growth',
    '총지수': 'cpi_index' # 소비자물가지수 행의 항목명이 '총지수'로 되어 있을 가능성 높음
}

# 2. 연도 컬럼만 식별 (2014 ~ 2024)
year_cols = [str(y) for y in range(2014, 2025)]
available_cols = [c for c in df_raw.columns if c in year_cols]

# 3. 데이터 추출 및 전치
records = []

# 콜금리
row_call = df_raw[df_raw.iloc[:, 1].str.contains('무담보콜금리', na=False)]
if not row_call.empty:
    vals = row_call[available_cols].iloc[0].values.astype(float)
    records.append(pd.DataFrame({'year': available_cols, 'value': vals, 'type': 'call_rate'}))

# GDP (실질성장률)
row_gdp = df_raw[df_raw.iloc[:, 1].str.contains('국내총생산|실질성장률', na=False)]
if not row_gdp.empty:
    vals = row_gdp[available_cols].iloc[0].values.astype(float)
    records.append(pd.DataFrame({'year': available_cols, 'value': vals, 'type': 'gdp_growth'}))

# CPI (지수) - "총지수" 혹은 "소비자물가지수"
row_cpi = df_raw[df_raw.iloc[:, 1].str.contains('소비자물가지수|총지수', na=False)]
# 주의: 여러 개 잡힐 수 있으므로 첫번째 것(보통 총지수) 사용
if not row_cpi.empty:
    vals = row_cpi[available_cols].iloc[0].values.astype(float)
    records.append(pd.DataFrame({'year': available_cols, 'value': vals, 'type': 'cpi_index'}))

# 병합
df_macro_long = pd.concat(records)
df_macro = df_macro_long.pivot(index='year', columns='type', values='value').reset_index()
df_macro['year'] = df_macro['year'].astype(int)

# ---------------------------------------------------------
# [중요] CPI 지수를 인플레이션율(%)로 변환
# ---------------------------------------------------------
# 현재 cpi_index는 94.196, 94.861 등 지수 형태임.
# 성장률(%) = (현재 - 과거)/과거 * 100
df_macro = df_macro.sort_values('year')
df_macro['cpi_growth'] = df_macro['cpi_index'].pct_change() * 100

# 주의: 2014년은 전년도(2013) 데이터가 없어서 NaN이 됨.
# 사용자의 엄밀함을 위해 2014년은 MP Shock 계산에서 제외되거나, 
# 2013년 CPI(92.9, ECOS 기준)를 안다면 보정 가능.
# 여기서는 데이터의 순수성을 위해 NaN인 2014년을 제외하고 2015년부터 MP Shock 산출
df_macro_analysis = df_macro.dropna(subset=['cpi_growth', 'gdp_growth', 'call_rate'])

print("\n[Calculated Macro Data]")
print(df_macro_analysis[['year', 'call_rate', 'gdp_growth', 'cpi_growth']].head())

# ---------------------------------------------------------
# 2. MP Shock 추출 (Taylor Rule Residuals)
# ---------------------------------------------------------
X = df_macro_analysis[['cpi_growth', 'gdp_growth']]
X = sm.add_constant(X)
y = df_macro_analysis['call_rate']

model = sm.OLS(y, X).fit()
df_macro_analysis['mp_shock'] = model.resid

print("\n[Taylor Rule Regression Result]")
print(f"R-squared: {model.rsquared:.3f}")
print(df_macro_analysis[['year', 'mp_shock']])

# ---------------------------------------------------------
# 3. 기업 패널 데이터와 병합
# ---------------------------------------------------------
panel_path = os.path.join(BASE_DIR, PANEL_FILE)

if os.path.exists(panel_path):
    df_panel = pd.read_csv(panel_path)
    print(f"\n[Load] 기업 패널 데이터: {len(df_panel)}건")
    
    # 거시 변수 병합
    df_merged = pd.merge(df_panel, df_macro_analysis[['year', 'mp_shock', 'gdp_growth', 'cpi_growth']], on='year', how='left')
    
    # 교호작용항 생성
    df_merged = df_merged.sort_values(by=['stock_code', 'year'])
    
    # 윈저라이징된 CFCR 사용 권장
    if 'cfcr_final_win' in df_merged.columns:
        cfcr_col = 'cfcr_final_win'
    else:
        cfcr_col = 'cfcr_final'

    # 시차 변수 생성 (t-1)
    df_merged['cfcr_lag'] = df_merged.groupby('stock_code')[cfcr_col].shift(1)
    
    # Interaction Term = CFCR(t-1) * MP_Shock(t)
    df_merged['interaction_term'] = df_merged['cfcr_lag'] * df_merged['mp_shock']
    
    # 분석 불가한 행 제거 (MP Shock이 없는 2014년 등)
    df_final = df_merged.dropna(subset=['inv_rate_win', 'interaction_term'])
    
    # 저장
    save_path = os.path.join(BASE_DIR, "korean_firms_panel_rigorous_final.csv")
    df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[Success] 엄밀한 데이터셋 생성 완료")
    print(f"파일 경로: {save_path}")
    print(f"최종 관측치: {len(df_final)}개")

else:
    print(f"[Error] 패널 데이터 파일을 찾을 수 없습니다: {PANEL_FILE}")