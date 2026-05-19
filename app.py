"""
토압 계산 프로그램 (Earth Pressure Calculator)
- 랭킨(Rankine) 토압 이론
- 쿨롱(Coulomb) 토압 이론
- 옹벽 설계용 주동/수동 토압 계산 및 시각화
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# ----------------------------
# 페이지 설정
# ----------------------------
st.set_page_config(
    page_title="Earth Pressure Calculator (Rankine & Coulomb)",
    page_icon="🧱",
    layout="wide"
)

# Matplotlib 기본 폰트 (영문 전용)
plt.rcParams['axes.unicode_minus'] = False


# ============================
# 계산 함수들
# ============================
def rankine_coefficients(phi_deg, i_deg=0.0):
    """
    랭킨 토압 계수
    - 수평지표면(i=0): Ka = tan²(45° - φ/2), Kp = tan²(45° + φ/2)
    - 경사지표면: Ka = cos(i) * [cos(i) - √(cos²i - cos²φ)] / [cos(i) + √(cos²i - cos²φ)]
    """
    phi = math.radians(phi_deg)
    i = math.radians(i_deg)

    if abs(i_deg) < 1e-6:
        Ka = math.tan(math.pi/4 - phi/2) ** 2
        Kp = math.tan(math.pi/4 + phi/2) ** 2
    else:
        if abs(i_deg) >= abs(phi_deg):
            return None, None
        cos_i = math.cos(i)
        cos_phi = math.cos(phi)
        root = math.sqrt(cos_i**2 - cos_phi**2)
        Ka = cos_i * (cos_i - root) / (cos_i + root)
        Kp = cos_i * (cos_i + root) / (cos_i - root)
    return Ka, Kp


def coulomb_coefficients(phi_deg, delta_deg, alpha_deg=90.0, beta_deg=0.0):
    """
    쿨롱 토압 계수
    phi   : 흙의 내부마찰각
    delta : 벽면 마찰각
    alpha : 옹벽 배면이 수평면과 이루는 각도 (수직벽이면 90°)
    beta  : 뒤채움 지표면 경사각 (수평이면 0°)
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    # --- Ka (주동) ---
    try:
        num_a = math.sin(alpha + phi) ** 2
        denom_a_inner = math.sin(phi + delta) * math.sin(phi - beta) / \
                        (math.sin(alpha - delta) * math.sin(alpha + beta))
        if denom_a_inner < 0:
            Ka = None
        else:
            denom_a = (math.sin(alpha)**2) * math.sin(alpha - delta) * \
                      (1 + math.sqrt(denom_a_inner)) ** 2
            Ka = num_a / denom_a
    except Exception:
        Ka = None

    # --- Kp (수동) ---
    try:
        num_p = math.sin(alpha - phi) ** 2
        denom_p_inner = math.sin(phi + delta) * math.sin(phi + beta) / \
                        (math.sin(alpha + delta) * math.sin(alpha + beta))
        if denom_p_inner < 0:
            Kp = None
        else:
            denom_p = (math.sin(alpha)**2) * math.sin(alpha + delta) * \
                      (1 - math.sqrt(denom_p_inner)) ** 2
            Kp = num_p / denom_p
    except Exception:
        Kp = None

    return Ka, Kp


def calc_earth_pressure(gamma, H, K, c=0.0, q=0.0, gw_depth=None, gamma_w=1.0):
    """
    토압 분포 및 합력 계산
    """
    n = 200
    depths = np.linspace(0, H, n)

    sigma_soil = np.zeros_like(depths)
    sigma_water = np.zeros_like(depths)

    if gw_depth is not None and gw_depth < H:
        gamma_sub = gamma - gamma_w
    else:
        gamma_sub = gamma

    for idx, z in enumerate(depths):
        if gw_depth is None or z <= (gw_depth if gw_depth is not None else H):
            sigma_v = gamma * z
        else:
            sigma_v = gamma * gw_depth + gamma_sub * (z - gw_depth)
            sigma_water[idx] = gamma_w * (z - gw_depth)
        sigma_soil[idx] = K * (sigma_v + q) - 2 * c * math.sqrt(K)
        if sigma_soil[idx] < 0:
            sigma_soil[idx] = 0.0

    sigma_total = sigma_soil + sigma_water

    # NumPy 2.x 호환 (trapezoid). 옛 버전 fallback 처리
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    P_total = _trapz(sigma_total, depths)

    if P_total > 1e-9:
        moment = _trapz(sigma_total * (H - depths), depths)
        y_bar = moment / P_total
    else:
        y_bar = 0.0

    return depths, sigma_total, sigma_soil, sigma_water, P_total, y_bar


# ============================
# UI - 사이드바 입력
# ============================
st.title("🧱 토압 계산 프로그램")
st.markdown("**Rankine & Coulomb 토압 이론** 기반 옹벽 설계용 토압 계산기")

with st.sidebar:
    st.header("📥 입력 데이터")

    st.subheader("기본 흙 물성")
    gamma = st.number_input("흙의 단위중량 γ (t/m³)", 1.0, 3.0, 1.8, 0.05)
    phi   = st.number_input("내부마찰각 φ (°)", 0.0, 50.0, 30.0, 0.5)
    c     = st.number_input("점착력 c (t/m²)", 0.0, 20.0, 0.0, 0.1)

    st.subheader("옹벽 형상")
    H     = st.number_input("옹벽 높이 H (m)", 0.5, 30.0, 5.0, 0.1)
    i     = st.number_input("지표면 경사각 i (°)", 0.0, 45.0, 0.0, 0.5)
    alpha = st.number_input("옹벽 배면 각도 α (°)", 60.0, 120.0, 90.0, 1.0)
    delta = st.number_input("벽면마찰각 δ (°)", 0.0, 40.0, 15.0, 0.5)

    st.subheader("외부 조건 (선택)")
    use_q  = st.checkbox("등분포하중 적용", False)
    q      = st.number_input("등분포하중 q (t/m²)", 0.0, 50.0, 1.0, 0.1) if use_q else 0.0

    use_gw = st.checkbox("지하수위 고려", False)
    gw_depth = st.number_input("지하수위 깊이 (지표에서 m)", 0.0, H, H/2, 0.1) if use_gw else None
    gamma_w = 1.0

    st.markdown("---")
    st.caption("※ 단위: t/m³, t/m², t/m (단위길이당 합력)")


# ============================
# 계산 수행
# ============================
Ka_r, Kp_r = rankine_coefficients(phi, i)
Ka_c, Kp_c = coulomb_coefficients(phi, delta, alpha, i)

warnings = []
if Ka_r is None or Kp_r is None:
    warnings.append("⚠️ 랭킨식: 경사각 i 가 내부마찰각 φ 보다 크거나 같아 계산 불가합니다.")
if Ka_c is None:
    warnings.append("⚠️ 쿨롱식 주동: 입력 조건으로 계산이 불가합니다 (제곱근 내부 음수).")
if Kp_c is None:
    warnings.append("⚠️ 쿨롱식 수동: 입력 조건으로 계산이 불가합니다.")
for w in warnings:
    st.warning(w)


# ============================
# 결과 - 토압 계수 표
# ============================
st.header("📊 1. 토압 계수 비교")

data = {
    "구분": ["주동토압계수 Ka", "수동토압계수 Kp"],
    "Rankine": [f"{Ka_r:.4f}" if Ka_r else "—",
                f"{Kp_r:.4f}" if Kp_r else "—"],
    "Coulomb": [f"{Ka_c:.4f}" if Ka_c else "—",
                f"{Kp_c:.4f}" if Kp_c else "—"],
}
st.table(pd.DataFrame(data))


# ============================
# 결과 - 토압 크기/작용점
# ============================
st.header("📐 2. 토압 크기 및 작용점")

results = []
press_curves = {}

for theory, Ka, Kp in [("Rankine", Ka_r, Kp_r), ("Coulomb", Ka_c, Kp_c)]:
    for kind, K in [("Active(Pa)", Ka), ("Passive(Pp)", Kp)]:
        if K is None:
            results.append({"Theory": theory, "Type": kind,
                            "K": "—", "P (t/m)": "—",
                            "y_bar (m, from base)": "—"})
            continue

        c_use = c if "Active" in kind else 0.0
        q_use = q if "Active" in kind else 0.0

        depths, sigma_total, sigma_soil, sigma_water, P, ybar = calc_earth_pressure(
            gamma, H, K, c=c_use, q=q_use,
            gw_depth=gw_depth, gamma_w=gamma_w
        )
        results.append({
            "Theory": theory,
            "Type": kind,
            "K": f"{K:.4f}",
            "P (t/m)": f"{P:.3f}",
            "y_bar (m, from base)": f"{ybar:.3f}",
        })
        press_curves[f"{theory}-{kind}"] = (depths, sigma_total, sigma_soil, sigma_water, P, ybar)

st.dataframe(pd.DataFrame(results), hide_index=True)


# ============================
# 결과 - 그래프 (토압 분포도) - 영어 라벨
# ============================
st.header("📈 3. 토압 분포도")

tab1, tab2 = st.tabs(["Active Pressure", "Passive Pressure"])

def plot_pressure(curves_keys, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Rankine": "#1f77b4", "Coulomb": "#d62728"}
    has_curve = False

    for key in curves_keys:
        if key not in press_curves:
            continue
        depths, sigma_total, sigma_soil, sigma_water, P, ybar = press_curves[key]
        theory = key.split("-")[0]
        has_curve = True

        ax.plot(sigma_total, depths,
                label=f"{theory} (P={P:.2f} t/m)",
                color=colors[theory], linewidth=2)
        ax.fill_betweenx(depths, 0, sigma_total,
                         color=colors[theory], alpha=0.15)
        # Resultant action point line
        ax.axhline(H - ybar, color=colors[theory], linestyle="--", alpha=0.5)
        ax.text(0.02, H - ybar - 0.08,
                f"{theory}: y_bar = {ybar:.2f} m",
                color=colors[theory], fontsize=9)

    ax.set_xlabel("Earth Pressure (t/m^2)")
    ax.set_ylabel("Depth z (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    if has_curve:
        ax.legend(loc="lower right")
    ax.axvline(0, color="gray", linewidth=3)
    return fig

with tab1:
    fig1 = plot_pressure(["Rankine-Active(Pa)", "Coulomb-Active(Pa)"],
                         "Active Earth Pressure")
    st.pyplot(fig1)

with tab2:
    fig2 = plot_pressure(["Rankine-Passive(Pp)", "Coulomb-Passive(Pp)"],
                         "Passive Earth Pressure")
    st.pyplot(fig2)


# ============================
# 옹벽 단면 개념도 - 영어 라벨
# ============================
st.header("🏗️ 4. 옹벽 단면도")

fig3, ax3 = plt.subplots(figsize=(8, 6))

# 옹벽
wall_x = [0, 0, 0.4, 0.4]
wall_y = [0, H, H, 0]
ax3.fill(wall_x, wall_y, color="#808080", alpha=0.7, label="Wall")

# 뒤채움 흙
i_rad = math.radians(i)
backfill_x = [0.4, 0.4 + H*1.5, 0.4 + H*1.5]
top_y = H + (H*1.5) * math.tan(i_rad)
backfill_y = [0, top_y, 0]
ax3.fill(backfill_x, backfill_y, color="#c2a878", alpha=0.5, label="Backfill")

# 지하수위
if gw_depth is not None:
    ax3.axhline(H - gw_depth, xmin=0.05, xmax=0.95,
                color="blue", linestyle="--", linewidth=2)
    ax3.text(0.4 + H*0.8, H - gw_depth + 0.1,
             f"GWL (-{gw_depth:.1f} m)",
             color="blue", fontsize=10)

# 등분포하중
if use_q and q > 0:
    arrow_y = top_y + 0.3
    for x_arrow in np.linspace(0.5, 0.4 + H*1.3, 8):
        ax3.annotate("", xy=(x_arrow, top_y),
                     xytext=(x_arrow, arrow_y),
                     arrowprops=dict(arrowstyle="->", color="red"))
    ax3.text(0.4 + H*0.5, arrow_y + 0.1,
             f"q = {q} t/m^2",
             color="red", fontsize=11, ha="center", fontweight="bold")

# 토압 작용 화살표 (랭킨 주동 기준)
if Ka_r is not None and "Rankine-Active(Pa)" in press_curves:
    _, _, _, _, P_show, ybar_show = press_curves["Rankine-Active(Pa)"]
    ax3.annotate("", xy=(0.4, ybar_show), xytext=(1.5, ybar_show),
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5))
    ax3.text(1.6, ybar_show,
             f"Pa = {P_show:.2f} t/m\n(y_bar = {ybar_show:.2f} m)",
             color="darkred", fontsize=10, va="center")

ax3.set_xlim(-0.5, 0.4 + H*1.7)
ax3.set_ylim(-0.5, max(top_y, H) + 1.5)
ax3.set_aspect("equal")
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Height (m)")
ax3.set_title(f"Wall Section (H={H} m, i={i} deg, alpha={alpha} deg)")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right")

st.pyplot(fig3)


# ============================
# 푸터
# ============================
st.markdown("---")
with st.expander("ℹ️ 사용 안내 및 가정사항"):
    st.markdown("""
- **부호 규약**: 깊이 z 는 지표면에서 아래로(+), 작용점 y_bar 는 옹벽 바닥에서 위로(+).
- **점착력 c** 는 주동토압에 한해 `-2c√Ka` 로 반영하며, 인장영역(σ<0)은 0으로 처리합니다.
- **지하수위** 적용 시 수위 아래 흙은 수중단위중량 `γ' = γ - γw` 를 사용하고, 정수압을 별도로 더합니다.
- **수동토압**의 점착력/등분포하중 효과는 보수적으로 미반영(일반 설계 관행).
- **쿨롱식**은 입력 조건에 따라 제곱근 내부가 음수가 되면 계산 불가 처리합니다.
- 본 프로그램은 **개략 설계/학습용** 입니다. 실무 설계 시 KDS/KSCE 기준 및 안전율을 별도 적용하세요.
""")

st.caption("© Earth Pressure Calculator — Built with Streamlit")
