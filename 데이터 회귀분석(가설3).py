import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# [설정] 한글 폰트 및 저장 경로
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"C:/Users/123/Documents/개인논문 - 금융 제약과 투자 파급효과/데이터"
RESULT_DIR = os.path.join(BASE_DIR, "논문_분석_결과")
DATA_FILE = "최종데이터.csv"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
file_path = os.path.join(BASE_DIR, DATA_FILE)
df = pd.read_csv(file_path)

# 숫자형 변환 및 결측치 제거
numeric_cols = ['inv_rate_win', 'cfcr_lag', 'mp_shock', 'roa_win', 'size', 'sales_growth_win']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=numeric_cols + ['stock_code', 'year']).copy()

# 기본 모델용 더미 (30% 기준)
threshold_30 = df_clean['cfcr_lag'].quantile(0.30)
df_clean['high_constraint_dummy'] = np.where(df_clean['cfcr_lag'] <= threshold_30, 1, 0)

# [핵심 수정] 비대칭 충격 분리 (Asymmetric Shocks)
# MP Shock > 0 (긴축, 금리인상 충격) / MP Shock < 0 (완화, 금리인하 충격)
df_clean['mp_tight'] = df_clean['mp_shock'].apply(lambda x: x if x > 0 else 0)
df_clean['mp_loose'] = df_clean['mp_shock'].apply(lambda x: x if x < 0 else 0)

# 교호작용항도 분리
# 가설: "긴축(Tight) 충격 시, 제약 기업(Dummy=1)이 투자를 더 많이 줄인다(-)"
df_clean['inter_tight'] = df_clean['high_constraint_dummy'] * df_clean['mp_tight']
df_clean['inter_loose'] = df_clean['high_constraint_dummy'] * df_clean['mp_loose']

print(f"[Data Prepared] 총 관측치: {len(df_clean)}건")

# ---------------------------------------------------------
# 2. 회귀분석 (Model 4: Asymmetric Interaction)
# ---------------------------------------------------------
print("\n[Analysis] Model 4 (비대칭 효과: 긴축 vs 완화 분리 분석)...")

# 식: Inv ~ Dummy + MP_Tight + MP_Loose + (Dummy*Tight) + (Dummy*Loose) + Controls
formula_m4 = """
inv_rate_win ~ high_constraint_dummy + mp_tight + mp_loose + 
               inter_tight + inter_loose + 
               roa_win + size + sales_growth_win
"""

model_m4 = smf.ols(formula=formula_m4, data=df_clean).fit(
    cov_type='cluster', cov_kwds={'groups': df_clean['stock_code']}
)

# 결과 저장
results = pd.DataFrame({
    'Coefficient': model_m4.params,
    'Std. Error': model_m4.bse,
    't-Statistic': model_m4.tvalues,
    'P-value': model_m4.pvalues
})
results['Significance'] = results['P-value'].apply(lambda x: '***' if x<0.01 else ('**' if x<0.05 else ('*' if x<0.1 else '')))

# Model 4 결과만 따로 저장 (이게 메인이 될 가능성 높음)
save_csv_path = os.path.join(RESULT_DIR, "Table6_Model4_Asymmetric.csv")
results.to_csv(save_csv_path, encoding='utf-8-sig')

# 요약본 저장
with open(os.path.join(RESULT_DIR, "Model4_Summary.txt"), "w", encoding='utf-8') as f:
    f.write(model_m4.summary().as_text())

print(f" -> 결과 저장 완료: {save_csv_path}")

# ---------------------------------------------------------
# 3. 시각화 (비대칭 효과)
# ---------------------------------------------------------
# MP Shock 전체 범위에 대해, 양수 구간과 음수 구간의 기울기가 다르게 나타나는지 확인

mp_range = np.linspace(df_clean['mp_shock'].min(), df_clean['mp_shock'].max(), 100)
means = df_clean.mean(numeric_only=True)
params = model_m4.params

# 예측값 계산 함수
def predict_investment(dummy_val, shock_val):
    shock_tight = max(0, shock_val)
    shock_loose = min(0, shock_val)
    
    pred = (params['Intercept'] + 
            params['high_constraint_dummy'] * dummy_val + 
            params['mp_tight'] * shock_tight + 
            params['mp_loose'] * shock_loose + 
            params['inter_tight'] * (dummy_val * shock_tight) + 
            params['inter_loose'] * (dummy_val * shock_loose) + 
            params['roa_win'] * means['roa_win'] + 
            params['size'] * means['size'] + 
            params['sales_growth_win'] * means['sales_growth_win'])
    return pred

y_con = [predict_investment(1, x) for x in mp_range]
y_unc = [predict_investment(0, x) for x in mp_range]

plt.figure(figsize=(10, 6))
plt.plot(mp_range, y_con, color='#E63946', linewidth=2.5, label='금융 제약 기업 (Bottom 30%)')
plt.plot(mp_range, y_unc, color='#1D3557', linewidth=2.5, linestyle='--', label='비제약 기업 (Top 70%)')

plt.title('비대칭적 통화정책 충격과 설비투자 (Model 4)', fontsize=14, fontweight='bold')
plt.xlabel('통화정책 충격 (MP Shock)\n(-) 완화  <--  0  -->  (+) 긴축', fontsize=12)
plt.ylabel('예측 설비투자율', fontsize=12)
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.6)

# 긴축/완화 구간 표시
plt.text(0.5, plt.ylim()[0]*1.05, "긴축 구간 (Tightening)", fontsize=10, color='black', ha='center')
plt.text(-0.5, plt.ylim()[0]*1.05, "완화 구간 (Easing)", fontsize=10, color='black', ha='center')

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_img_path = os.path.join(RESULT_DIR, "Figure4_Asymmetric_Effect.png")
plt.savefig(save_img_path, dpi=300)
print(f" -> 그래프 저장 완료: {save_img_path}")

print("\n[Done] 비대칭 모형(Model 4) 분석 완료.")