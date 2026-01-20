import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# [설정] 한글 폰트 및 저장 경로 설정
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준 (Mac은 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False    # 마이너스 기호 깨짐 방지

BASE_DIR = r"C:/Users/123/Documents/개인논문 - 금융 제약과 투자 파급효과/데이터"
RESULT_DIR = os.path.join(BASE_DIR, "논문_분석_결과")
DATA_FILE = "최종데이터.csv" # 사용자가 업로드한 파일명

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"[Info] 결과 저장 폴더 생성: {RESULT_DIR}")

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
file_path = os.path.join(BASE_DIR, DATA_FILE)
df = pd.read_csv(file_path)

# 분석에 필요한 변수 정의
# 종속변수: inv_rate_win
# 주요 설명변수: cfcr_lag (전기 CFCR), mp_shock, interaction_term
# 통제변수: roa_win, size, sales_growth_win

# 데이터 타입 변환 (문자열 -> 숫자) 및 결측치 제거
numeric_cols = ['inv_rate_win', 'cfcr_lag', 'mp_shock', 'interaction_term', 'roa_win', 'size', 'sales_growth_win']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 결측치가 있는 행 제거 (엄밀한 분석을 위해 완전한 케이스만 사용)
df_clean = df.dropna(subset=numeric_cols + ['stock_code', 'year']).copy()

# 기업별, 연도별 인덱스 설정 (패널 구조 확인용)
df_clean = df_clean.sort_values(['stock_code', 'year'])
print(f"[Data Loaded] 분석 대상 관측치: {len(df_clean)}건")

# ---------------------------------------------------------
# 2. 기술통계량 (Descriptive Statistics) - Table 1
# ---------------------------------------------------------
desc_stats = df_clean[numeric_cols].describe().T
desc_save_path = os.path.join(RESULT_DIR, "Table1_기술통계량.csv")
desc_stats.to_csv(desc_save_path)
print(" -> Table 1 저장 완료")

# 상관관계 행렬 - Table 2
corr_matrix = df_clean[numeric_cols].corr()
corr_save_path = os.path.join(RESULT_DIR, "Table2_상관관계표.csv")
corr_matrix.to_csv(corr_save_path)
print(" -> Table 2 저장 완료")

# ---------------------------------------------------------
# 3. 회귀분석 (Regression Analysis)
# ---------------------------------------------------------
# 모델링 전략: OLS with Firm Fixed Effects & Clustered Standard Errors
# 기업 고정효과(Entity FE)를 포함하여 기업 고유의 관찰되지 않은 특성을 통제
# 표준오차는 기업(stock_code) 기준으로 군집화(Cluster)하여 자기상관 문제 해결

# [Model 1] 가설 1 검증: CFCR(t-1)이 투자에 미치는 영향
# 식: Inv = a + b1*CFCR(t-1) + Controls + Year_Dummy + Firm_FE
print("\n[Analysis] Model 1 (가설 1) 분석 중...")

# C(year)를 넣어 시간 고정효과(Time FE) 통제
formula_m1 = "inv_rate_win ~ cfcr_lag + roa_win + size + sales_growth_win + C(year)"
model_m1 = smf.ols(formula=formula_m1, data=df_clean).fit(
    cov_type='cluster', cov_kwds={'groups': df_clean['stock_code']}
)

# [Model 2] 가설 2 검증: 통화정책 충격과 CFCR의 교호작용
# 식: Inv = a + b1*CFCR + b2*MP + b3*(CFCR*MP) + Controls + Firm_FE
# 주의: MP Shock은 연도별로 동일하므로 Time FE(C(year))와 공선성 문제 발생 가능.
# 따라서 Model 2에서는 Time FE 대신 거시 변수(MP Shock)를 직접 투입합니다.
print("[Analysis] Model 2 (가설 2) 분석 중...")

formula_m2 = "inv_rate_win ~ cfcr_lag + mp_shock + interaction_term + roa_win + size + sales_growth_win"
model_m2 = smf.ols(formula=formula_m2, data=df_clean).fit(
    cov_type='cluster', cov_kwds={'groups': df_clean['stock_code']}
)

# ---------------------------------------------------------
# 4. 결과 정리 및 저장 - Table 3
# ---------------------------------------------------------
# 결과를 보기 좋게 정리
def extract_results(model, model_name):
    results = pd.DataFrame({
        'Coefficient': model.params,
        'Std. Error': model.bse,
        't-Statistic': model.tvalues,
        'P-value': model.pvalues
    })
    results['Significance'] = results['P-value'].apply(lambda x: '***' if x<0.01 else ('**' if x<0.05 else ('*' if x<0.1 else '')))
    results.columns = pd.MultiIndex.from_product([[model_name], results.columns])
    return results

res_m1 = extract_results(model_m1, "Model 1 (Baseline)")
res_m2 = extract_results(model_m2, "Model 2 (Interaction)")

final_table = pd.concat([res_m1, res_m2], axis=1)
reg_save_path = os.path.join(RESULT_DIR, "Table3_회귀분석결과.csv")
final_table.to_csv(reg_save_path, encoding='utf-8-sig')

# 요약본 텍스트 저장
with open(os.path.join(RESULT_DIR, "Regression_Summary.txt"), "w", encoding='utf-8') as f:
    f.write(model_m2.summary().as_text())

print(" -> Table 3 (회귀분석 결과) 저장 완료")

# ---------------------------------------------------------
# 5. 논문용 그래프 그리기 (Visualization)
# ---------------------------------------------------------

# [Figure 1] 금융 제약 그룹별 연도별 평균 투자율 추이
# CFCR 중앙값 기준으로 그룹 나누기 (매년)
df_clean['cfcr_group'] = df_clean.groupby('year')['cfcr_lag'].transform(
    lambda x: np.where(x > x.median(), 'Low Constraint (High CFCR)', 'High Constraint (Low CFCR)')
)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clean, x='year', y='inv_rate_win', hue='cfcr_group', marker='o', errorbar=None)
plt.title('금융 제약 여부에 따른 연도별 설비투자율 추이 (2015-2024)', fontsize=14)
plt.ylabel('평균 설비투자율 (Investment Rate)', fontsize=12)
plt.xlabel('연도 (Year)', fontsize=12)
plt.legend(title='Financial Constraint Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "Figure1_Investment_Trend.png"), dpi=300)
print(" -> Figure 1 저장 완료")

# [Figure 2] 통화정책 충격에 대한 반응 민감도 (Interaction Effect)
# 가설 2 시각화: MP Shock이 증가할 때(긴축), CFCR 수준에 따라 투자가 어떻게 변하는가?
# 회귀계수를 바탕으로 예측값(Fitted Value) 시각화

# 시뮬레이션 데이터 생성
mp_range = np.linspace(df_clean['mp_shock'].min(), df_clean['mp_shock'].max(), 100)
# CFCR 10분위수(하위 10% - 제약 심함) vs 90분위수(상위 10% - 제약 없음)
cfcr_low = df_clean['cfcr_lag'].quantile(0.1)
cfcr_high = df_clean['cfcr_lag'].quantile(0.9)

# 통제변수는 평균값으로 고정
means = df_clean.mean(numeric_only=True)

# 예측값 계산 (b1*CFCR + b2*MP + b3*(CFCR*MP) + Controls)
params = model_m2.params

# 제약 기업 (Low CFCR) 예측
y_constrained = (params['Intercept'] + 
                 params['cfcr_lag'] * cfcr_low + 
                 params['mp_shock'] * mp_range + 
                 params['interaction_term'] * (cfcr_low * mp_range) +
                 params['roa_win'] * means['roa_win'] + 
                 params['size'] * means['size'] + 
                 params['sales_growth_win'] * means['sales_growth_win'])

# 비제약 기업 (High CFCR) 예측
y_unconstrained = (params['Intercept'] + 
                   params['cfcr_lag'] * cfcr_high + 
                   params['mp_shock'] * mp_range + 
                   params['interaction_term'] * (cfcr_high * mp_range) +
                   params['roa_win'] * means['roa_win'] + 
                   params['size'] * means['size'] + 
                   params['sales_growth_win'] * means['sales_growth_win'])

plt.figure(figsize=(10, 6))
plt.plot(mp_range, y_constrained, label='High Constraint (Low CFCR)', color='red', linestyle='-')
plt.plot(mp_range, y_unconstrained, label='Low Constraint (High CFCR)', color='blue', linestyle='--')

plt.title('통화정책 충격(MP Shock)이 설비투자에 미치는 차별적 영향', fontsize=14)
plt.xlabel('통화정책 충격 (MP Shock, +:긴축 / -:완화)', fontsize=12)
plt.ylabel('예측 설비투자율 (Predicted Investment Rate)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 0점 기준선 (MP Shock = 0)
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "Figure2_Interaction_Effect.png"), dpi=300)
print(" -> Figure 2 저장 완료")

print(f"\n[Done] 모든 분석이 완료되었습니다. 결과 폴더를 확인하세요: {RESULT_DIR}")