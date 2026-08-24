# %% [markdown]
# 천안 교차로 안전진단: 1차 SPF(3A 패널기반 베이지안) + 2차 A(Track A 교차표) + 2차 B(시간분할 검증)
#
# ============================================================
# 최종 확정 구조 (2026-08-24 정정 — 3A를 패널 파일로 실행 가능함을 재확인)
# ============================================================
# 1차: 3A - 패널 파일(연도별패널_432행.csv)의 y/기대_연/추정_연을 이용한
#      Poisson-Gamma 베이지안 사후분포(위험배수 EB) 모형   ★최종
#      → 처음엔 "완전히 새로운 원사고 파일(교차로_연도별_실사고.csv)이 있어야만
#        3A를 돌릴 수 있다"고 판단했으나 이는 과도하게 엄격한 기준이었음.
#        패널 파일에는 이미 링크기반 위험지역 데이터로 만든 실제 카운트(y, 관측=True인 연도)와
#        조건부 기댓값(추정_연, 관측=False인 연도)이 들어있고, 이걸로 v7의 초과위험 계산을
#        100% 재현 가능함을 검증함(추정_6년총합/기대_6년 완전 일치).
#        → 새 파일 없이 지금 바로 실행 가능.
#      → 남은 한계(원래부터 알던 것, 새로 생긴 문제 아님): 46곳(한 번도 지정 안 된 교차로)의
#        연도별 값은 여전히 "조건부 기댓값"이지 진짜 관측값이 아님. 팀이 이미
#        "절단관측 46곳은 순위 없이 확인목록으로만 제시"하기로 결정해둔 바로 그 한계.
#        완전히 풀려면 100m 버퍼 내 진짜 원사고 원자료가 필요한데, 이건 2026-08-24
#        확인 결과 공개 경로로 확보 불가(TAAS 개별원자료는 위치가 구 단위로만 익명화).
# 2차 A: Track A - 고집중 26곳, 초과위험 상위10 vs 비교16, 교차표+Fisher/Mann-Whitney   ★최종
#      → 원래 계획한 L1 회귀(구조·운영변수 필요)는 데이터 확보 불가로 포기.
#        해당 코드는 맨 아래 "보류 코드"에 남겨둠(향후 데이터 확보 시 재사용).
# 2차 B: 시간분할 검증 - 2019~2023 학습 → 2024 검증, 임계위험지역 발생 예측   ★최종
#      → 아래 3B(GEE, 전체기간 확률)는 1차 모델 후보에서 제외하고 참고용 함수로만 남김.
#        2차 B는 별도의 시간분할 로지스틱(temporal_validation_threshold, 6번 셀)을 사용한다.
# ============================================================
#
# 실행 전 준비 (지금 시점 기준 — 최종)
# Colab 첫 셀: !pip install -q pandas numpy scipy statsmodels scikit-learn
# DATA_DIR에 다음 파일만 있으면 전체 파이프라인이 끝까지 실행된다 (전부 이미 확보됨):
#    - 연도별패널_432행.csv   ← 1차 모델(3A)의 유일한 입력, 새 파일 불필요
#    - 교차로_통합데이터_v7.csv
#    - 교차로_1차모델결과_v7.csv  (Track A용 태그·EPDO 포함)
# 아래 두 파일은 최종적으로 확보하지 않기로 확정됨 (경로만 남겨둠, 없어도 정상 실행됨):
#    - 교차로_연도별_실사고.csv (완전 비마스킹 원자료 — 확보 불가 최종 확인, 3A는 이제 이 파일 불필요)
#    - 교차로_구조운영변수.csv
#
# 이 파일은 VS Code/Jupyter의 # %% 셀 단위로 실행할 수 있다.

# %% 0. 패키지와 경로
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import gamma, spearmanr, fisher_exact, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# 수정: 팀 폴더 또는 복사한 데이터 폴더로 지정
DATA_DIR = Path("/Users/j/Downloads/천안 공모전 1차 모델 수정본")
OUT_DIR = Path("./spf_outputs")
OUT_DIR.mkdir(exist_ok=True)

PANEL_FILE = DATA_DIR / "연도별패널_432행.csv"
BASE_FILE = DATA_DIR / "교차로_통합데이터_v7.csv"
RESULT_FILE = DATA_DIR / "교차로_1차모델결과_v7.csv"           # Track A용 (태그·EPDO 포함)
RAW_CRASH_FILE = DATA_DIR / "교차로_연도별_실사고.csv"           # 미확보. 확보되면 자동으로 3A 실행됨
SECOND_STAGE_FILE = DATA_DIR / "교차로_구조운영변수.csv"         # 미확보. 팀 결정으로 사용 안 함 (보류 코드 참고)

SEED = 20260824
np.random.seed(SEED)


# %% 1. 현재 자료 감사(audit): 절단 가정이 무엇을 뜻하는지 확인
panel = pd.read_csv(PANEL_FILE)
base = pd.read_csv(BASE_FILE)

required_panel = {"교차로명", "연도", "관측", "y"}
required_base = {"교차로명", "사고건수", "지정연도수", "일평균_교통량", "야간_비율", "교차로형태", "인구_200m"}
assert required_panel <= set(panel.columns), "패널 필수 열을 확인하세요."
assert required_base <= set(base.columns), "기초자료 필수 열을 확인하세요."

audit = (
    panel.groupby("교차로명", as_index=False)
    .agg(패널_관측연도수=("관측", "sum"), 패널_y합=("y", "sum"))
    .merge(base[["교차로명", "사고건수", "지정연도수"]], on="교차로명", validate="one_to_one")
)
audit["관측연도수_일치"] = audit["패널_관측연도수"].eq(audit["지정연도수"])
audit["사고합계_일치"] = audit["패널_y합"].eq(audit["사고건수"])
audit.to_csv(OUT_DIR / "01_current_data_audit.csv", index=False, encoding="utf-8-sig")

print("패널 행 수:", len(panel), "| 관측 행:", int(panel["관측"].sum()))
print("관측연도수 불일치:", int((~audit["관측연도수_일치"]).sum()))
print("사고합계 불일치:", int((~audit["사고합계_일치"]).sum()))
print("\n중요: 현재 y=0은 실제 0이 아니라 '12건 이상 위험지역 기록 없음'이다.")
print("따라서 원사고 집계 전에는 이를 교차로 100m 총사고 0~11건으로 해석하지 않는다.")


# %% 2. 공통 전처리: 설명변수와 품질점검
# 팀 결정: 교차로형태 연속형 유지 / 인구변수 포함
def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """SPF 기본 설명변수의 유효 범위를 검사하고 변환한다."""
    out = df.copy()
    if (out["일평균_교통량"] <= 0).any():
        raise ValueError("일평균_교통량은 모두 양수여야 log 변환이 가능합니다.")
    if ~out["야간_비율"].between(0, 1).all():
        raise ValueError("야간_비율은 0~1 비율이어야 합니다.")
    out["log_aadt"] = np.log(out["일평균_교통량"])
    out["인구_천명"] = out["인구_200m"] / 1000
    return out

base_x = prepare_covariates(base)
print(base_x[["일평균_교통량", "야간_비율", "교차로형태", "인구_천명"]].describe().round(3))


# %% 3A. [최종 확정, 1차 모델] 패널 파일 기반 Poisson-Gamma 베이지안 EB 위험배수
# 새로운 원사고 파일 불필요. 연도별패널_432행.csv의 y(관측연도 실제값)와
# 추정_연(미관측연도 조건부 기댓값)만으로 실행된다.
# alpha_hat은 기존 v7 결과에서 역산 검증된 값(1.9037)을 기본으로 쓰되,
# 필요시 재추정 로직으로 교체 가능하도록 인자로 노출해둔다.
ALPHA_HAT_DEFAULT = 1.9037  # v7 결과 역산 검증 완료 (교차로_1차모델결과_v7.csv와 완전 일치)

def fit_panel_bayesian_spf(panel_df: pd.DataFrame, alpha_hat: float = ALPHA_HAT_DEFAULT):
    """패널(교차로x연도)의 실제값+조건부기댓값으로 교차로별 위험배수 사후분포를 계산한다.

    한계(원래부터 알던 것): 관측=False인 연도의 값은 진짜 관측치가 아니라
    NB 분포 하의 조건부 기댓값이다. 완전한 비마스킹 원자료는 여전히 없음.
    """
    required = {"교차로명", "연도", "관측", "y", "기대_연", "추정_연"}
    if not required <= set(panel_df.columns):
        raise ValueError(f"패널 파일 필수 열 누락: {required - set(panel_df.columns)}")

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "관측사고": g.loc[g["관측"], "y"].sum() + g.loc[~g["관측"], "추정_연"].sum(),
            "SPF_기대사고": g["기대_연"].sum(),
            "관측연도수": int(g["관측"].sum()),
        })

    site = panel_df.groupby("교차로명", as_index=False).apply(_agg, include_groups=False).reset_index()

    k = 1 / alpha_hat
    site["theta_사후_shape"] = k + site["관측사고"]
    site["theta_사후_rate"] = k + site["SPF_기대사고"]
    site["EB_위험배수"] = site["theta_사후_shape"] / site["theta_사후_rate"]
    site["EB_위험배수_90CI_하한"] = gamma.ppf(0.05, a=site["theta_사후_shape"], scale=1 / site["theta_사후_rate"])
    site["EB_위험배수_90CI_상한"] = gamma.ppf(0.95, a=site["theta_사후_shape"], scale=1 / site["theta_사후_rate"])
    site["P_초과위험"] = gamma.sf(1.0, a=site["theta_사후_shape"], scale=1 / site["theta_사후_rate"])
    site["P_실질초과위험_1_25배"] = gamma.sf(1.25, a=site["theta_사후_shape"], scale=1 / site["theta_사후_rate"])
    site["EB_사고"] = site["EB_위험배수"] * site["SPF_기대사고"]
    site["EB_초과사고"] = site["EB_사고"] - site["SPF_기대사고"]
    site = site.sort_values("P_실질초과위험_1_25배", ascending=False)
    site["초과위험_순위"] = np.arange(1, len(site) + 1)

    site.to_csv(OUT_DIR / "03A_panel_bayesian_site_ranking.csv", index=False, encoding="utf-8-sig")
    return site


# %% 3B. [참고용, 1차 모델 아님] 임계 위험지역 발생 GEE 확률 — 3A로 대체됨
# 2차 B(시간분할 검증)는 이 함수가 아니라 별도의 temporal_validation_threshold(6번 셀)를 쓴다.
# 이 함수는 "전체기간 평균 발생확률"이 필요할 때만 참고용으로 호출한다.
def fit_threshold_surveillance(panel_df: pd.DataFrame, base_covariates: pd.DataFrame):
    d = panel_df.merge(
        base_covariates[["교차로명", "log_aadt", "야간_비율", "교차로형태", "인구_천명"]],
        on="교차로명", how="inner", validate="many_to_one"
    ).copy()
    d["임계위험지역_발생"] = d["관측"].astype(int)

    gee = smf.gee(
        "임계위험지역_발생 ~ log_aadt + 야간_비율 + 교차로형태 + 인구_천명 + C(연도)",
        groups="교차로명", data=d,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    d["P_임계위험지역"] = gee.predict(d)
    site = d.groupby("교차로명", as_index=False).agg(**{
        "6년_발생연도수": ("임계위험지역_발생", "sum"),
        "연평균_발생확률": ("P_임계위험지역", "mean"),
    })
    site["6년내_최소1회_발생확률"] = 1 - (1 - site["연평균_발생확률"]).pow(6)
    site = site.sort_values("연평균_발생확률", ascending=False)
    site.to_csv(OUT_DIR / "03B_threshold_surveillance_ranking.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "03B_threshold_surveillance_summary.txt").write_text(gee.summary().as_text(), encoding="utf-8")
    return gee, d, site


# %% 4. 1차 모델 실행 (최종: 3A 패널 기반 베이지안 EB)
excess_ranking = fit_panel_bayesian_spf(panel, alpha_hat=ALPHA_HAT_DEFAULT)
print("1차 모델(3A, 패널 기반 베이지안 EB 위험배수) 실행 완료 — 이번 대회 최종 1차 모델.")
print(excess_ranking[["교차로명", "관측사고", "SPF_기대사고", "EB_위험배수",
                       "EB_위험배수_90CI_하한", "EB_위험배수_90CI_상한",
                       "P_실질초과위험_1_25배", "초과위험_순위"]].head(10))


# %% 5. 2차 A [현재 사용]: Track A - 고집중 26곳 교차표 진단
# 원래 계획한 L1 회귀(구조·운영변수 필요)는 데이터 확보 불가로 대체됨.
# 회귀 아님: 초과위험 상위10(우선개선군) vs 비교16곳의 태그·사상자 지표를 개별 검정.
TOP_N = 10  # 우선개선군 크기. 8/12로 바꿔 민감도 분석 가능.

def explode_tags(series: pd.Series) -> pd.DataFrame:
    """콤마로 구분된 다발유형 문자열을 태그별 0/1 컬럼으로 변환한다."""
    filled = series.fillna("")
    all_tags = sorted({t.strip() for row in filled for t in row.split(",") if t.strip()})
    out = pd.DataFrame(0, index=series.index, columns=all_tags)
    for idx, row in filled.items():
        for t in row.split(","):
            t = t.strip()
            if t:
                out.loc[idx, t] = 1
    return out

def run_track_a_diagnosis(result_file: Path, top_n: int = TOP_N):
    df = pd.read_csv(result_file)
    hi = df[df["위험등급"] == "고집중"].sort_values("초과위험_순위").copy()
    assert len(hi) >= top_n + 5, "고집중 표본이 너무 작아 비교군 확보가 어렵습니다."
    hi["집단"] = np.where(hi["초과위험_순위"] <= top_n, "우선개선군", "비교군")

    tag_matrix = explode_tags(hi["다발유형"])
    for c in ["노인", "어린이", "보행자", "이륜차", "음주"]:
        if c in hi.columns:
            tag_matrix[c] = hi[c].notna().astype(int)
    hi_tags = pd.concat([hi[["교차로명", "집단"]].reset_index(drop=True),
                          tag_matrix.reset_index(drop=True)], axis=1)

    # 태그별 Fisher 정확검정
    rows = []
    for tag in tag_matrix.columns:
        a = hi_tags.loc[hi_tags["집단"] == "우선개선군", tag].sum()
        b = (hi_tags["집단"] == "우선개선군").sum() - a
        c = hi_tags.loc[hi_tags["집단"] == "비교군", tag].sum()
        d = (hi_tags["집단"] == "비교군").sum() - c
        if a + c == 0:
            continue
        odds, p = fisher_exact([[a, b], [c, d]])
        rows.append({
            "태그": tag,
            "우선개선군_보유": int(a), "우선개선군_비율": round(a / max(a + b, 1), 2),
            "비교군_보유": int(c), "비교군_비율": round(c / max(c + d, 1), 2),
            "오즈비": round(odds, 2) if np.isfinite(odds) else np.inf,
            "p_value": round(p, 4),
        })
    tag_test = pd.DataFrame(rows).sort_values("p_value")

    # 연속변수(EPDO, 사망/중상/경상) Mann-Whitney + 효과크기(순위이연상관)
    def rank_biserial(g1, g2):
        u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
        return round(1 - (2 * u) / (len(g1) * len(g2)), 3)

    cont_rows = []
    for col in ["EPDO", "사망", "중상", "경상"]:
        if col not in hi.columns:
            continue
        g1 = hi.loc[hi["집단"] == "우선개선군", col].fillna(0)
        g2 = hi.loc[hi["집단"] == "비교군", col].fillna(0)
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        cont_rows.append({
            "변수": col, "우선개선군_중앙값": g1.median(), "비교군_중앙값": g2.median(),
            "순위이연상관": rank_biserial(g1, g2), "p_value": round(p, 4),
        })
    cont_test = pd.DataFrame(cont_rows)

    label_cols = [c for c in ["교차로명", "초과위험_순위", "집단", "다발유형", "노인", "어린이",
                               "보행자", "이륜차", "음주", "EPDO", "사망", "중상", "경상"] if c in hi.columns]
    site_labels = hi[label_cols]

    tag_test.to_csv(OUT_DIR / "05_tag_fisher_test.csv", index=False, encoding="utf-8-sig")
    cont_test.to_csv(OUT_DIR / "05_continuous_mannwhitney_test.csv", index=False, encoding="utf-8-sig")
    site_labels.to_csv(OUT_DIR / "05_site_prescription_labels.csv", index=False, encoding="utf-8-sig")
    return tag_test, cont_test, site_labels


# %% 6. 2차 B [현재 사용]: 잠재위험 - '다음 연도 임계위험지역 발생' 시간외 검증
# 2019~2023으로 학습하고 2024를 평가한다. 2025 자료가 있으면 같은 방식으로 외부검증한다.
def temporal_validation_threshold(panel_df: pd.DataFrame, base_covariates: pd.DataFrame):
    d = panel_df.merge(
        base_covariates[["교차로명", "log_aadt", "야간_비율", "교차로형태", "인구_천명"]],
        on="교차로명", how="inner", validate="many_to_one"
    ).copy()
    d["event"] = d["관측"].astype(int)
    xcols = ["log_aadt", "야간_비율", "교차로형태", "인구_천명"]
    train, test = d[d["연도"] <= 2023].copy(), d[d["연도"] == 2024].copy()

    train_x = sm.add_constant(train[xcols], has_constant="add")
    test_x = sm.add_constant(test[xcols], has_constant="add")
    logit = sm.Logit(train["event"], train_x).fit(disp=False)
    test["P_임계위험지역"] = logit.predict(test_x)

    metrics = {
        "test_year": 2024, "n_test": int(len(test)), "event_test": int(test["event"].sum()),
        "pr_auc": float(average_precision_score(test["event"], test["P_임계위험지역"])),
        "roc_auc": float(roc_auc_score(test["event"], test["P_임계위험지역"])),
        "brier": float(brier_score_loss(test["event"], test["P_임계위험지역"])),
    }
    test.sort_values("P_임계위험지역", ascending=False).to_csv(
        OUT_DIR / "06_2024_threshold_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (OUT_DIR / "06_2024_threshold_holdout_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return logit, test, metrics


# %% 7. 실행 예시 (지금 있는 파일로 바로 실행 가능)
#
# tag_test, cont_test, site_labels = run_track_a_diagnosis(RESULT_FILE)
# validation_model, holdout_2024, validation_metrics = temporal_validation_threshold(panel, base_x)
#
# 1차(3A) 결과에 처방 라벨(주요원인 등) 붙이기:
# labels_with_cause = excess_ranking.merge(
#     base[["교차로명", "주요원인"]], on="교차로명", how="left"
# )
# labels_with_cause.to_csv(OUT_DIR / "07_prescription_labels_final.csv", index=False, encoding="utf-8-sig")


# %% ============================================================
# %% 보류 코드: 구조·운영변수 확보 시 재사용 (지금은 호출하지 않음)
# %% ============================================================
#
# 아래 SECOND_STAGE_FEATURES / fit_second_stage_diagnosis / stability_select_second_stage는
# 팀이 원래 설계했던 L1 회귀 기반 2차 A안이다. 차로수·신호주기·제한속도 등
# `교차로_구조운영변수.csv`를 확보하지 못해 2026-08-24 팀 결정으로 실행하지 않기로 함.
# 향후 데이터가 생기면 이 블록을 되살려서 Track A(교차표)와 결과를 비교·보완하는 데 쓸 수 있다.

SECOND_STAGE_FEATURES = [
    "접근차로수", "좌회전전용차로_유무", "보호좌회전_유무", "제한속도",
    "신호주기", "횡단보도수", "보행신호_유무", "조명_등급",
    "버스정류장_100m", "접근부_시거", "중앙분리대_유무",
]

def load_second_stage_features(file: Path) -> pd.DataFrame:
    if not file.exists():
        raise FileNotFoundError(
            "2차 구조·운영변수 파일이 없습니다. (2026-08-24 팀 결정: 확보 포기, Track A로 대체)"
        )
    x = pd.read_csv(file)
    if "교차로명" not in x.columns:
        raise ValueError("2차 변수 파일에는 교차로명 열이 필요합니다.")
    available = [c for c in SECOND_STAGE_FEATURES if c in x.columns]
    if len(available) < 3:
        raise ValueError("2차 설명변수는 최소 3개 이상 필요합니다.")
    return x[["교차로명"] + available].copy()

def fit_second_stage_diagnosis(excess_site: pd.DataFrame, feature_file: Path):
    features = load_second_stage_features(feature_file)
    d = excess_site.merge(features, on="교차로명", how="inner", validate="one_to_one").copy()
    available = [c for c in SECOND_STAGE_FEATURES if c in d.columns]

    d["고초과위험"] = (d["P_실질초과위험_1_25배"] >= 0.80).astype(int)
    if d["고초과위험"].sum() < 8:
        raise ValueError("사후확률 0.80 이상 사례가 8개 미만입니다.")

    numeric = [c for c in available if pd.api.types.is_numeric_dtype(d[c])]
    categorical = [c for c in available if c not in numeric]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegressionCV(
            Cs=12, cv=inner_cv, penalty="l1", solver="saga", class_weight="balanced",
            scoring="average_precision", max_iter=20000, random_state=SEED,
        )),
    ])
    clf.fit(d[available], d["고초과위험"])
    p = clf.predict_proba(d[available])[:, 1]
    d["2차_고초과위험_점검확률"] = p
    stability = stability_select_second_stage(clf, d, available)
    return clf, d, stability

def stability_select_second_stage(fitted_pipeline, d, available, n_resamples=300,
                                   train_fraction=0.75, selection_threshold=0.70):
    y = d["고초과위험"].to_numpy()
    selected_c = float(fitted_pipeline.named_steps["model"].C_[0])
    stable_model = LogisticRegression(
        C=selected_c, penalty="l1", solver="saga", class_weight="balanced",
        max_iter=20000, random_state=SEED,
    )
    template = Pipeline([
        ("prep", clone(fitted_pipeline.named_steps["prep"])), ("model", stable_model),
    ])
    feature_names = fitted_pipeline.named_steps["prep"].get_feature_names_out().tolist()
    coefficients = pd.DataFrame(0.0, index=range(n_resamples), columns=feature_names)
    sampler = StratifiedShuffleSplit(n_splits=n_resamples, train_size=train_fraction, random_state=SEED)
    X = d[available]
    for iteration, (train_idx, _) in enumerate(sampler.split(X, y)):
        fitted = clone(template).fit(X.iloc[train_idx], y[train_idx])
        names = fitted.named_steps["prep"].get_feature_names_out()
        coefficients.loc[iteration, names] = fitted.named_steps["model"].coef_.ravel()
    selected = coefficients.ne(0)
    selection_frequency = selected.mean(axis=0)
    sign_consistency = pd.Series(0.0, index=coefficients.columns)
    for col in coefficients.columns:
        chosen = coefficients.loc[selected[col], col]
        if len(chosen):
            sign_consistency[col] = max((chosen > 0).mean(), (chosen < 0).mean())
    output = pd.DataFrame({
        "변수": coefficients.columns, "선택빈도": selection_frequency.values,
        "선택시_계수중앙값": coefficients.replace(0, np.nan).median(axis=0).fillna(0).values,
        "부호일관성": sign_consistency.values,
    })
    output["안정적_후보"] = (output["선택빈도"] >= selection_threshold) & (output["부호일관성"] >= 0.90)
    return output.sort_values(["안정적_후보", "선택빈도"], ascending=[False, False])
