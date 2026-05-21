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


def calc_effective_vertical_stress(depth, gamma, q=0.0, gw_depth=None, gamma_w=1.0):
    """특정 깊이에서의 유효 연직응력 계산 (등분포하중 포함)"""
    if gw_depth is None or depth <= gw_depth:
        return gamma * depth + q
    gamma_sub = gamma - gamma_w
    return gamma * gw_depth + gamma_sub * (depth - gw_depth) + q


def calc_mohr_stresses(depth, gamma, K, state="active", c=0.0, q=0.0, gw_depth=None, gamma_w=1.0):
    """선택 깊이에서의 간략 Mohr circle용 주응력 계산"""
    sigma_v = calc_effective_vertical_stress(depth, gamma, q=q, gw_depth=gw_depth, gamma_w=gamma_w)

    if state == "active":
        sigma_h = K * sigma_v - 2 * c * math.sqrt(K)
        sigma_h = max(0.0, sigma_h)
    else:
        sigma_h = K * sigma_v + 2 * c * math.sqrt(K)

    sigma_1 = max(sigma_v, sigma_h)
    sigma_3 = min(sigma_v, sigma_h)
    return sigma_v, sigma_h, sigma_1, sigma_3


def plot_mohr_circle(theory_name, Ka, Kp, depth, gamma, phi, c=0.0, q=0.0, gw_depth=None, gamma_w=1.0):
    fig, ax = plt.subplots(figsize=(7, 6))
    max_sigma = 1.0

    if Ka is not None:
        _, _, s1a, s3a = calc_mohr_stresses(depth, gamma, Ka, state="active", c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w)
        center_a = (s1a + s3a) / 2
        radius_a = (s1a - s3a) / 2
        theta = np.linspace(0, 2 * np.pi, 400)
        x_a = center_a + radius_a * np.cos(theta)
        y_a = radius_a * np.sin(theta)
        ax.plot(x_a, y_a, color="#1f77b4", linewidth=2, label=f"Active ({theory_name})")
        ax.plot([s3a, s1a], [0, 0], 'o', color="#1f77b4", markersize=4)
        max_sigma = max(max_sigma, s1a)

    if Kp is not None:
        _, _, s1p, s3p = calc_mohr_stresses(depth, gamma, Kp, state="passive", c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w)
        center_p = (s1p + s3p) / 2
        radius_p = (s1p - s3p) / 2
        theta = np.linspace(0, 2 * np.pi, 400)
        x_p = center_p + radius_p * np.cos(theta)
        y_p = radius_p * np.sin(theta)
        ax.plot(x_p, y_p, color="#d62728", linewidth=2, label=f"Passive ({theory_name})")
        ax.plot([s3p, s1p], [0, 0], 'o', color="#d62728", markersize=4)
        max_sigma = max(max_sigma, s1p)

    x_env = np.linspace(0, max_sigma * 1.15, 300)
    tau_env = c + x_env * math.tan(math.radians(phi))
    ax.plot(x_env, tau_env, color="black", linestyle="--", linewidth=1.5, label="Failure Envelope")
    if c > 0:
        ax.plot(x_env, -tau_env, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    else:
        ax.plot(x_env, -x_env * math.tan(math.radians(phi)), color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Normal Stress σ (t/m^2)")
    ax.set_ylabel("Shear Stress τ (t/m^2)")
    ax.set_title(f"Mohr Circle at z = {depth:.2f} m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return fig


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

with st.expander("📐 계산식 보기"):
    st.markdown("**Rankine (수평 지표면)**")
    st.latex(r"K_a = \tan^2\left(45^\circ - \frac{\phi}{2}\right), \quad K_p = \tan^2\left(45^\circ + \frac{\phi}{2}\right)")
    st.markdown("**Rankine (경사 지표면)**")
    st.latex(r"K_a = \cos i \cdot \frac{\cos i - \sqrt{\cos^2 i - \cos^2 \phi}}{\cos i + \sqrt{\cos^2 i - \cos^2 \phi}}")
    st.markdown("**Coulomb**")
    st.latex(r"K_a = \frac{\sin^2(\alpha+\phi)}{\sin^2\alpha\,\sin(\alpha-\delta)\left[1+\sqrt{\frac{\sin(\phi+\delta)\sin(\phi-\beta)}{\sin(\alpha-\delta)\sin(\alpha+\beta)}}\right]^2}")
    st.latex(r"K_p = \frac{\sin^2(\alpha-\phi)}{\sin^2\alpha\,\sin(\alpha+\delta)\left[1-\sqrt{\frac{\sin(\phi+\delta)\sin(\phi+\beta)}{\sin(\alpha+\delta)\sin(\alpha+\beta)}}\right]^2}")
    st.markdown("**유효 연직응력 / 토압 / 합력**")
    st.latex(r"\sigma_v' = \gamma z + q \quad (\text{수위 아래에서는 } \gamma' = \gamma-\gamma_w)")
    st.latex(r"\sigma_h = K\sigma_v' \; (\text{active}), \qquad P = \int_0^H \sigma_h(z)\,dz")
    st.latex(r"\bar{y} = \frac{\int_0^H \sigma_h(z)(H-z)\,dz}{\int_0^H \sigma_h(z)\,dz}")
    st.caption("※ 본 버전은 주동/수동토압 중심입니다. 벽체 수평변위가 거의 허용되지 않으면 정지토압(K₀) 검토가 필요합니다.")


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
# 결과 - Mohr Circle
# ============================
st.header("📈 3. 모어원 그래프")

mohr_col1, mohr_col2 = st.columns([1, 2])
with mohr_col1:
    mohr_depth = st.slider("깊이 z (m)", min_value=0.0, max_value=float(H), value=float(min(1.0, H)), step=0.1)
    st.caption("모어원은 Rankine 이론 기준으로만 표시합니다.")
    st.info("Coulomb 토압은 벽면마찰을 포함한 쐐기 평형해석이므로, 점 응력 기반 모어원과 직접 대응시키지 않습니다.")

with mohr_col2:
    mohr_fig = plot_mohr_circle("Rankine", Ka_r, Kp_r, mohr_depth, gamma, phi, c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w)
    st.pyplot(mohr_fig)

# 선택 깊이 응력값 표 (Rankine only)
mohr_rows = []
if Ka_r is not None:
    sigma_v, sigma_h, s1, s3 = calc_mohr_stresses(mohr_depth, gamma, Ka_r, state="active", c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w)
    mohr_rows.append({
        "Theory": "Rankine",
        "State": "Active",
        "σv' (t/m²)": f"{sigma_v:.3f}",
        "σh' (t/m²)": f"{sigma_h:.3f}",
        "σ1 (t/m²)": f"{s1:.3f}",
        "σ3 (t/m²)": f"{s3:.3f}",
    })
if Kp_r is not None:
    sigma_v, sigma_h, s1, s3 = calc_mohr_stresses(mohr_depth, gamma, Kp_r, state="passive", c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w)
    mohr_rows.append({
        "Theory": "Rankine",
        "State": "Passive",
        "σv' (t/m²)": f"{sigma_v:.3f}",
        "σh' (t/m²)": f"{sigma_h:.3f}",
        "σ1 (t/m²)": f"{s1:.3f}",
        "σ3 (t/m²)": f"{s3:.3f}",
    })

if mohr_rows:
    st.dataframe(pd.DataFrame(mohr_rows), hide_index=True)


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
x_far = 0.4 + H * 1.5
top_y = H + (x_far - 0.4) * math.tan(i_rad)
backfill_x = [0.4, 0.4, x_far, x_far]
backfill_y = [0, H, top_y, 0]
ax3.fill(backfill_x, backfill_y, color="#c2a878", alpha=0.5, label="Backfill")
ax3.plot([0.4, x_far], [H, top_y], color="#8c6d3f", linewidth=2)

# 지하수위
if gw_depth is not None:
    ax3.axhline(H - gw_depth, xmin=0.05, xmax=0.95,
                color="blue", linestyle="--", linewidth=2)
    ax3.text(0.4 + H*0.8, H - gw_depth + 0.1,
             f"GWL (-{gw_depth:.1f} m)",
             color="blue", fontsize=10)

# 등분포하중
if use_q and q > 0:
    x_start = 0.8
    x_end = max(1.2, x_far - 0.5)
    x_mid = (x_start + x_end) / 2
    y_mid = H + (x_mid - 0.4) * math.tan(i_rad)
    for x_arrow in np.linspace(x_start, x_end, 8):
        y_surface = H + (x_arrow - 0.4) * math.tan(i_rad)
        ax3.annotate("", xy=(x_arrow, y_surface),
                     xytext=(x_arrow, y_surface + 0.45),
                     arrowprops=dict(arrowstyle="->", color="red"))
    ax3.text(x_mid, y_mid + 0.55,
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
- **정지토압 / 주동토압 / 수동토압** 은 벽체의 수평변위 조건에 따라 구분됩니다. 벽체 변위가 거의 허용되지 않으면 **정지토압(K₀)** 검토가 필요합니다.
- **점착력 c** 는 주동토압에 한해 `-2c√Ka` 로 반영하며, 인장영역(σ<0)은 0으로 처리합니다.
- **지하수위** 적용 시 수위 아래 흙은 수중단위중량 `γ' = γ - γw` 를 사용하고, 정수압을 별도로 더합니다.
- **수동토압**의 점착력/등분포하중 효과는 보수적으로 미반영(일반 설계 관행)하며, 전도 검토에서 수동토압은 보통 제외합니다.
- **Rankine** 은 벽면마찰을 무시한 이론이며, **Coulomb** 은 벽면마찰과 배면조건을 고려한 쐐기 평형해석입니다.
- 본 프로그램은 **개략 설계/학습용** 입니다. 실무 설계 시 KDS/KSCE 기준, 활동·전도·지지력·전체안정 검토를 별도로 수행하세요.
""")

st.caption("© Earth Pressure Calculator — Built with Streamlit")
