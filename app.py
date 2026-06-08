"""
토압 계산 프로그램 (Earth Pressure Calculator)
- 랭킨(Rankine) 토압 이론
- 쿨롱(Coulomb) 토압 이론
- 정지토압(K0), 주동토압(Ka), 수동토압(Kp)
- 옹벽 설계용 토압 계산 및 시각화
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

# Matplotlib 기본 설정
plt.rcParams["axes.unicode_minus"] = False


# ============================
# 계산 함수들
# ============================
def at_rest_coefficient_jaky(phi_deg):
    """
    Jaky 식 (정규압밀토 가정)
    K0 = 1 - sin(phi)
    """
    phi = math.radians(phi_deg)
    return 1.0 - math.sin(phi)


def rankine_coefficients(phi_deg, i_deg=0.0):
    """
    랭킨 토압 계수
    - 수평지표면(i=0): Ka = tan²(45° - φ/2), Kp = tan²(45° + φ/2)
    - 경사지표면: Ka = cos(i) * [cos(i) - √(cos²i - cos²φ)] / [cos(i) + √(cos²i - cos²φ)]
    """
    phi = math.radians(phi_deg)
    i = math.radians(i_deg)

    if abs(i_deg) < 1e-6:
        Ka = math.tan(math.pi / 4 - phi / 2) ** 2
        Kp = math.tan(math.pi / 4 + phi / 2) ** 2
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
        denom_a_inner = (
            math.sin(phi + delta) * math.sin(phi - beta)
            / (math.sin(alpha - delta) * math.sin(alpha + beta))
        )
        if denom_a_inner < 0:
            Ka = None
        else:
            denom_a = (
                (math.sin(alpha) ** 2)
                * math.sin(alpha - delta)
                * (1 + math.sqrt(denom_a_inner)) ** 2
            )
            Ka = num_a / denom_a
    except Exception:
        Ka = None

    # --- Kp (수동) ---
    try:
        num_p = math.sin(alpha - phi) ** 2
        denom_p_inner = (
            math.sin(phi + delta) * math.sin(phi + beta)
            / (math.sin(alpha + delta) * math.sin(alpha + beta))
        )
        if denom_p_inner < 0:
            Kp = None
        else:
            denom_p = (
                (math.sin(alpha) ** 2)
                * math.sin(alpha + delta)
                * (1 - math.sqrt(denom_p_inner)) ** 2
            )
            Kp = num_p / denom_p
    except Exception:
        Kp = None

    return Ka, Kp


def calc_lateral_pressure_profile(
    gamma, H, K, state="active", c=0.0, q=0.0,
    gw_depth=None, gamma_w=1.0, gamma_sat=None
):
    """
    state:
    - "at_rest" : sigma_h = K0 * sigma_v'
    - "active"  : sigma_h = Ka * sigma_v' - 2c√Ka
    - "passive" : sigma_h = Kp * sigma_v' + 2c√Kp

    gamma     : 지하수면 위 단위중량
    gamma_sat : 지하수면 아래 포화단위중량
    gw_depth  : 지표면으로부터 지하수위 깊이
    """
    n = 300
    depths = np.linspace(0, H, n)

    sigma_soil = np.zeros_like(depths)
    sigma_water = np.zeros_like(depths)

    for idx, z in enumerate(depths):
        # 유효 연직응력 계산
        if gw_depth is None or z <= gw_depth:
            sigma_v_eff = gamma * z
        else:
            if gamma_sat is None:
                gamma_sub = gamma - gamma_w
            else:
                gamma_sub = gamma_sat - gamma_w

            sigma_v_eff = gamma * gw_depth + gamma_sub * (z - gw_depth)
            sigma_water[idx] = gamma_w * (z - gw_depth)

        # 횡토압 계산
        if state == "at_rest":
            sigma_h = K * (sigma_v_eff + q)

        elif state == "active":
            sigma_h = K * (sigma_v_eff + q) - 2 * c * math.sqrt(K)
            sigma_h = max(0.0, sigma_h)

        elif state == "passive":
            sigma_h = K * (sigma_v_eff + q) + 2 * c * math.sqrt(K)
            sigma_h = max(0.0, sigma_h)

        else:
            sigma_h = K * (sigma_v_eff + q)

        sigma_soil[idx] = sigma_h

    sigma_total = sigma_soil + sigma_water

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    P_total = _trapz(sigma_total, depths)

    if P_total > 1e-9:
        moment = _trapz(sigma_total * (H - depths), depths)
        y_bar = moment / P_total
    else:
        y_bar = 0.0

    return depths, sigma_total, sigma_soil, sigma_water, P_total, y_bar


def calc_effective_vertical_stress(depth, gamma, q=0.0, gw_depth=None, gamma_w=1.0, gamma_sat=None):
    """특정 깊이에서의 유효 연직응력 계산"""
    if gw_depth is None or depth <= gw_depth:
        return gamma * depth + q

    if gamma_sat is None:
        gamma_sub = gamma - gamma_w
    else:
        gamma_sub = gamma_sat - gamma_w

    return gamma * gw_depth + gamma_sub * (depth - gw_depth) + q


def calc_mohr_stresses(
    depth, gamma, K, state="active", c=0.0, q=0.0,
    gw_depth=None, gamma_w=1.0, gamma_sat=None
):
    """선택 깊이에서의 간략 Mohr circle용 주응력 계산"""
    sigma_v = calc_effective_vertical_stress(
        depth, gamma, q=q, gw_depth=gw_depth,
        gamma_w=gamma_w, gamma_sat=gamma_sat
    )

    if state == "active":
        sigma_h = K * sigma_v - 2 * c * math.sqrt(K)
        sigma_h = max(0.0, sigma_h)
    elif state == "passive":
        sigma_h = K * sigma_v + 2 * c * math.sqrt(K)
    else:
        sigma_h = K * sigma_v

    sigma_1 = max(sigma_v, sigma_h)
    sigma_3 = min(sigma_v, sigma_h)

    return sigma_v, sigma_h, sigma_1, sigma_3


def plot_mohr_circle(
    theory_name, Ka, Kp, depth, gamma, phi, c=0.0, q=0.0,
    gw_depth=None, gamma_w=1.0, gamma_sat=None
):
    fig, ax = plt.subplots(figsize=(7, 6))
    max_sigma = 1.0

    if Ka is not None:
        _, _, s1a, s3a = calc_mohr_stresses(
            depth, gamma, Ka, state="active", c=c, q=q,
            gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
        )
        center_a = (s1a + s3a) / 2
        radius_a = (s1a - s3a) / 2
        theta = np.linspace(0, 2 * np.pi, 400)
        x_a = center_a + radius_a * np.cos(theta)
        y_a = radius_a * np.sin(theta)
        ax.plot(x_a, y_a, color="#1f77b4", linewidth=2, label=f"Active ({theory_name})")
        ax.plot([s3a, s1a], [0, 0], "o", color="#1f77b4", markersize=4)
        max_sigma = max(max_sigma, s1a)

    if Kp is not None:
        _, _, s1p, s3p = calc_mohr_stresses(
            depth, gamma, Kp, state="passive", c=c, q=q,
            gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
        )
        center_p = (s1p + s3p) / 2
        radius_p = (s1p - s3p) / 2
        theta = np.linspace(0, 2 * np.pi, 400)
        x_p = center_p + radius_p * np.cos(theta)
        y_p = radius_p * np.sin(theta)
        ax.plot(x_p, y_p, color="#d62728", linewidth=2, label=f"Passive ({theory_name})")
        ax.plot([s3p, s1p], [0, 0], "o", color="#d62728", markersize=4)
        max_sigma = max(max_sigma, s1p)

    x_env = np.linspace(0, max_sigma * 1.15, 300)
    tau_env = c + x_env * math.tan(math.radians(phi))
    ax.plot(x_env, tau_env, color="black", linestyle="--", linewidth=1.5, label="Failure Envelope")

    if c > 0:
        ax.plot(x_env, -tau_env, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    else:
        ax.plot(
            x_env,
            -x_env * math.tan(math.radians(phi)),
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Normal Stress σ (t/m²)")
    ax.set_ylabel("Shear Stress τ (t/m²)")
    ax.set_title(f"Mohr Circle at z = {depth:.2f} m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return fig


def plot_wall_section(H, i, alpha, q=0.0, use_q=False, gw_depth=None, P_show=None, ybar_show=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 옹벽
    wall_x = [0, 0, 0.4, 0.4]
    wall_y = [0, H, H, 0]
    ax.fill(wall_x, wall_y, color="#808080", alpha=0.75, label="Wall")

    # 뒤채움
    i_rad = math.radians(i)
    x_far = 0.4 + H * 1.5
    top_y = H + (x_far - 0.4) * math.tan(i_rad)
    backfill_x = [0.4, 0.4, x_far, x_far]
    backfill_y = [0, H, top_y, 0]
    ax.fill(backfill_x, backfill_y, color="#c2a878", alpha=0.55, label="Backfill")
    ax.plot([0.4, x_far], [H, top_y], color="#8c6d3f", linewidth=2)

    # 지하수위
    if gw_depth is not None:
        y_gw = H - gw_depth
        ax.axhline(y_gw, xmin=0.05, xmax=0.95, color="blue", linestyle="--", linewidth=2)
        ax.text(0.4 + H * 0.8, y_gw + 0.12, f"GWL (-{gw_depth:.1f} m from top)", color="blue", fontsize=10)

    # 등분포하중
    if use_q and q > 0:
        x_start = 0.8
        x_end = max(1.2, x_far - 0.5)
        x_mid = (x_start + x_end) / 2
        y_mid = H + (x_mid - 0.4) * math.tan(i_rad)

        for x_arrow in np.linspace(x_start, x_end, 8):
            y_surface = H + (x_arrow - 0.4) * math.tan(i_rad)
            ax.annotate(
                "",
                xy=(x_arrow, y_surface),
                xytext=(x_arrow, y_surface + 0.45),
                arrowprops=dict(arrowstyle="->", color="red"),
            )

        ax.text(
            x_mid,
            y_mid + 0.55,
            f"q = {q} t/m²",
            color="red",
            fontsize=11,
            ha="center",
            fontweight="bold",
        )

    # 대표 토압 화살표
    if P_show is not None and ybar_show is not None:
        ax.annotate(
            "",
            xy=(0.4, ybar_show),
            xytext=(1.6, ybar_show),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5),
        )

    ax.text(0.2, H + 0.2, f"alpha = {alpha:.1f}°", ha="center", fontsize=10)

    ax.set_xlim(-0.5, 0.4 + H * 1.7)
    ax.set_ylim(-0.5, max(top_y, H) + 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title(f"Wall Section (H={H:.1f} m, i={i:.1f}°)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    return fig


def plot_single_distribution(profile, H, title="Distribution", side="right",
                             fill_color="#f4a261", line_color="#e76f51",
                             soil_color="#8d5524", water_color="#1d4ed8"):
    """
    그래프 내부 글자 겹침 방지를 위해
    그래프 내부 텍스트는 최소화하고
    합력 작용점은 점(marker)만 표시
    """
    depths, sigma_total, sigma_soil, sigma_water, P, ybar = profile

    fig, ax = plt.subplots(figsize=(7.5, 6))

    if side == "left":
        x_total = -sigma_total
        x_soil = -sigma_soil
        x_water = -sigma_water
    else:
        x_total = sigma_total
        x_soil = sigma_soil
        x_water = sigma_water

    ax.fill_betweenx(depths, 0, x_total, color=fill_color, alpha=0.45, label="Total")
    ax.plot(x_total, depths, color=line_color, linewidth=2.5)

    if np.max(sigma_water) > 0:
        ax.plot(x_soil, depths, color=soil_color, linestyle="--", linewidth=1.5, label="Soil")
        ax.plot(x_water, depths, color=water_color, linestyle=":", linewidth=2.0, label="Water")

    # 합력 작용점
    depth_resultant = H - ybar
    sigma_resultant = np.interp(depth_resultant, depths, sigma_total)
    sigma_resultant_plot = -sigma_resultant if side == "left" else sigma_resultant
    ax.plot([sigma_resultant_plot], [depth_resultant], "o", color="black", markersize=6)

    max_sigma = max(1.0, float(np.max(sigma_total)))
    if side == "left":
        ax.set_xlim(-max_sigma * 1.2, 0.15 * max_sigma)
    else:
        ax.set_xlim(-0.15 * max_sigma, max_sigma * 1.2)

    ax.set_ylim(H, 0)
    ax.set_xlabel("Pressure sigma_h (t/m²)")
    ax.set_ylabel("Depth z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    return fig


def plot_compare_distribution(profile_k0, profile_ka, profile_kp, H, compare_title="Comparison"):
    """
    비교 그래프에서는 곡선/범례만 표시
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))

    d0, s0, _, _, _, _ = profile_k0
    da, sa, _, _, _, _ = profile_ka
    dp, sp, _, _, _, _ = profile_kp

    ax.plot(s0, d0, color="#6a4c93", linewidth=2.5, label="K0")
    ax.plot(sa, da, color="#e76f51", linewidth=2.5, label="Ka")
    ax.plot(-sp, dp, color="#2d6a4f", linewidth=2.5, label="Kp")

    ax.fill_betweenx(d0, 0, s0, color="#6a4c93", alpha=0.15)
    ax.fill_betweenx(da, 0, sa, color="#e76f51", alpha=0.15)
    ax.fill_betweenx(dp, 0, -sp, color="#2d6a4f", alpha=0.12)

    max_sigma = max(
        1.0,
        float(np.max(s0)),
        float(np.max(sa)),
        float(np.max(sp))
    )

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlim(-max_sigma * 1.2, max_sigma * 1.2)
    ax.set_ylim(H, 0)
    ax.set_xlabel("Pressure sigma_h (t/m²)")
    ax.set_ylabel("Depth z (m)")
    ax.set_title(compare_title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    return fig


# ============================
# UI - 사이드바 입력
# ============================
st.title("🧱 토압 계산 프로그램")
st.markdown("**Rankine & Coulomb 토압 이론** 기반 옹벽 설계용 토압 계산기")

with st.sidebar:
    st.header("📥 입력 데이터")

    st.subheader("기본 흙 물성")
    gamma = st.number_input("지하수면 위 단위중량 γ (t/m³)", 1.0, 3.0, 1.8, 0.05)
    phi = st.number_input("내부마찰각 φ (°)", 0.0, 50.0, 30.0, 0.5)
    c = st.number_input("점착력 c (t/m²)", 0.0, 20.0, 0.0, 0.1)

    st.subheader("옹벽 형상")
    H = st.number_input("옹벽 높이 H (m)", 0.5, 30.0, 5.0, 0.1)
    i = st.number_input("지표면 경사각 i (°)", 0.0, 45.0, 0.0, 0.5)
    alpha = st.number_input("옹벽 배면 각도 α (°)", 60.0, 120.0, 90.0, 1.0)
    delta = st.number_input("벽면마찰각 δ (°)", 0.0, 40.0, 15.0, 0.5)

    st.subheader("외부 조건 (선택)")
    use_q = st.checkbox("등분포하중 적용", False)
    q = st.number_input("등분포하중 q (t/m²)", 0.0, 50.0, 1.0, 0.1) if use_q else 0.0

    use_gw = st.checkbox("지하수위 고려", False)

    if use_gw:
        gw_ref = st.radio(
            "지하수위 입력 기준",
            ["지표면 기준", "옹벽 저면 기준"],
            horizontal=True
        )

        gw_value = st.number_input("지하수위 위치 (m)", 0.0, H, H / 2, 0.1)

        if gw_ref == "지표면 기준":
            gw_depth = gw_value
        else:
            gw_depth = H - gw_value

        gamma_sat = st.number_input("지하수면 아래 포화단위중량 γsat (t/m³)", 1.0, 3.0, 1.90, 0.05)
    else:
        gw_depth = None
        gamma_sat = None

    gamma_w = 1.0

    st.markdown("---")
    st.caption("※ 지하수위가 있으면 위쪽은 γ, 아래쪽은 γsat를 사용합니다.")
    st.caption("※ 단위: t/m³, t/m², t/m (단위길이당 합력)")


# ============================
# 계산 수행
# ============================
K0 = at_rest_coefficient_jaky(phi)
Ka_r, Kp_r = rankine_coefficients(phi, i)
Ka_c, Kp_c = coulomb_coefficients(phi, delta, alpha, i)

warnings = []

if Ka_r is None or Kp_r is None:
    warnings.append("⚠️ Rankine: 경사각 i 가 내부마찰각 φ 보다 크거나 같아 계산 불가합니다.")
if Ka_c is None:
    warnings.append("⚠️ Coulomb 주동: 입력 조건으로 계산이 불가합니다.")
if Kp_c is None:
    warnings.append("⚠️ Coulomb 수동: 입력 조건으로 계산이 불가합니다.")

for w in warnings:
    st.warning(w)

# 프로파일 계산
profile_k0 = calc_lateral_pressure_profile(
    gamma, H, K0, state="at_rest", c=0.0, q=q,
    gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
)

profile_rankine_active = None
profile_rankine_passive = None
profile_coulomb_active = None
profile_coulomb_passive = None

if Ka_r is not None:
    profile_rankine_active = calc_lateral_pressure_profile(
        gamma, H, Ka_r, state="active", c=c, q=q,
        gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )

if Kp_r is not None:
    profile_rankine_passive = calc_lateral_pressure_profile(
        gamma, H, Kp_r, state="passive", c=c, q=q,
        gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )

if Ka_c is not None:
    profile_coulomb_active = calc_lateral_pressure_profile(
        gamma, H, Ka_c, state="active", c=c, q=q,
        gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )

if Kp_c is not None:
    profile_coulomb_passive = calc_lateral_pressure_profile(
        gamma, H, Kp_c, state="passive", c=c, q=q,
        gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )


# ============================
# 결과 - 토압 계수 표
# ============================
st.header("📊 1. 토압 계수 비교")

data = {
    "구분": ["정지토압계수 K0", "주동토압계수 Ka", "수동토압계수 Kp"],
    "Jaky / At-rest": [f"{K0:.4f}", "—", "—"],
    "Rankine": ["—",
                f"{Ka_r:.4f}" if Ka_r is not None else "—",
                f"{Kp_r:.4f}" if Kp_r is not None else "—"],
    "Coulomb": ["—",
                f"{Ka_c:.4f}" if Ka_c is not None else "—",
                f"{Kp_c:.4f}" if Kp_c is not None else "—"],
}
st.table(pd.DataFrame(data))

with st.expander("📐 계산식 보기"):
    st.markdown("**At-rest (Jaky)**")
    st.latex(r"K_0 = 1 - \sin\phi")

    st.markdown("**Rankine (수평 지표면)**")
    st.latex(r"K_a = \tan^2\left(45^\circ - \frac{\phi}{2}\right), \quad K_p = \tan^2\left(45^\circ + \frac{\phi}{2}\right)")

    st.markdown("**Rankine (경사 지표면)**")
    st.latex(r"K_a = \cos i \cdot \frac{\cos i - \sqrt{\cos^2 i - \cos^2 \phi}}{\cos i + \sqrt{\cos^2 i - \cos^2 \phi}}")

    st.markdown("**Coulomb**")
    st.latex(r"K_a = \frac{\sin^2(\alpha+\phi)}{\sin^2\alpha\,\sin(\alpha-\delta)\left[1+\sqrt{\frac{\sin(\phi+\delta)\sin(\phi-\beta)}{\sin(\alpha-\delta)\sin(\alpha+\beta)}}\right]^2}")
    st.latex(r"K_p = \frac{\sin^2(\alpha-\phi)}{\sin^2\alpha\,\sin(\alpha+\delta)\left[1-\sqrt{\frac{\sin(\phi+\delta)\sin(\phi+\beta)}{\sin(\alpha+\delta)\sin(\alpha+\beta)}}\right]^2}")

    st.markdown("**유효 연직응력 / 토압 / 합력**")
    st.latex(r"\sigma_v' = \gamma z + q \quad (\text{수위 아래에서는 } \gamma' = \gamma_{sat}-\gamma_w)")
    st.latex(r"\sigma_h = K_0 \sigma_v' \; (\text{at-rest})")
    st.latex(r"\sigma_h = K_a \sigma_v' - 2c\sqrt{K_a} \; (\text{active})")
    st.latex(r"\sigma_h = K_p \sigma_v' + 2c\sqrt{K_p} \; (\text{passive})")
    st.latex(r"P = \int_0^H \sigma_h(z)\,dz, \qquad \bar{y} = \frac{\int_0^H \sigma_h(z)(H-z)\,dz}{\int_0^H \sigma_h(z)\,dz}")


# ============================
# 결과 - 토압 크기 및 작용점
# ============================
st.header("📐 2. 토압 크기 및 작용점")

results = []

# K0
_, _, _, _, P0, y0 = profile_k0
results.append({
    "Theory": "At-rest",
    "Type": "K0",
    "K": f"{K0:.4f}",
    "P (t/m)": f"{P0:.3f}",
    "y_bar (m, from base)": f"{y0:.3f}",
})

# Rankine
if profile_rankine_active is not None:
    _, _, _, _, P, ybar = profile_rankine_active
    results.append({
        "Theory": "Rankine",
        "Type": "Active (Ka)",
        "K": f"{Ka_r:.4f}",
        "P (t/m)": f"{P:.3f}",
        "y_bar (m, from base)": f"{ybar:.3f}",
    })

if profile_rankine_passive is not None:
    _, _, _, _, P, ybar = profile_rankine_passive
    results.append({
        "Theory": "Rankine",
        "Type": "Passive (Kp)",
        "K": f"{Kp_r:.4f}",
        "P (t/m)": f"{P:.3f}",
        "y_bar (m, from base)": f"{ybar:.3f}",
    })

# Coulomb
if profile_coulomb_active is not None:
    _, _, _, _, P, ybar = profile_coulomb_active
    results.append({
        "Theory": "Coulomb",
        "Type": "Active (Ka)",
        "K": f"{Ka_c:.4f}",
        "P (t/m)": f"{P:.3f}",
        "y_bar (m, from base)": f"{ybar:.3f}",
    })

if profile_coulomb_passive is not None:
    _, _, _, _, P, ybar = profile_coulomb_passive
    results.append({
        "Theory": "Coulomb",
        "Type": "Passive (Kp)",
        "K": f"{Kp_c:.4f}",
        "P (t/m)": f"{P:.3f}",
        "y_bar (m, from base)": f"{ybar:.3f}",
    })

st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)


# ============================
# 결과 - Mohr Circle
# ============================
st.header("📈 3. 모어원 그래프")

mohr_col1, mohr_col2 = st.columns([1, 2])

with mohr_col1:
    mohr_depth = st.slider(
        "깊이 z (m)",
        min_value=0.0,
        max_value=float(H),
        value=float(min(1.0, H)),
        step=0.1,
    )
    st.caption("모어원은 Rankine 이론 기준으로만 표시합니다.")
    st.info("Coulomb 토압은 벽면마찰을 포함한 쐐기 평형해석이므로 점 응력 기반 모어원과 직접 대응시키지 않습니다.")

with mohr_col2:
    mohr_fig = plot_mohr_circle(
        "Rankine", Ka_r, Kp_r, mohr_depth, gamma, phi,
        c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )
    st.pyplot(mohr_fig)

mohr_rows = []

if Ka_r is not None:
    sigma_v, sigma_h, s1, s3 = calc_mohr_stresses(
        mohr_depth, gamma, Ka_r, state="active",
        c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )
    mohr_rows.append({
        "Theory": "Rankine",
        "State": "Active",
        "σv' (t/m²)": f"{sigma_v:.3f}",
        "σh' (t/m²)": f"{sigma_h:.3f}",
        "σ1 (t/m²)": f"{s1:.3f}",
        "σ3 (t/m²)": f"{s3:.3f}",
    })

if Kp_r is not None:
    sigma_v, sigma_h, s1, s3 = calc_mohr_stresses(
        mohr_depth, gamma, Kp_r, state="passive",
        c=c, q=q, gw_depth=gw_depth, gamma_w=gamma_w, gamma_sat=gamma_sat
    )
    mohr_rows.append({
        "Theory": "Rankine",
        "State": "Passive",
        "σv' (t/m²)": f"{sigma_v:.3f}",
        "σh' (t/m²)": f"{sigma_h:.3f}",
        "σ1 (t/m²)": f"{s1:.3f}",
        "σ3 (t/m²)": f"{s3:.3f}",
    })

if mohr_rows:
    st.dataframe(pd.DataFrame(mohr_rows), hide_index=True, use_container_width=True)


# ============================
# 4. 옹벽 단면도
# ============================
st.header("🏗️ 4. 옹벽 단면도")

wall_P = None
wall_y = None
if profile_rankine_active is not None:
    _, _, _, _, wall_P, wall_y = profile_rankine_active

wall_fig = plot_wall_section(
    H, i, alpha, q=q, use_q=use_q,
    gw_depth=gw_depth, P_show=wall_P, ybar_show=wall_y
)
st.pyplot(wall_fig)


# ============================
# 5. 토압 분포도
# ============================
st.header("📉 5. 토압 분포도")

tab_k0, tab_rankine, tab_coulomb, tab_compare = st.tabs([
    "K0",
    "Rankine",
    "Coulomb",
    "Comparison"
])

with tab_k0:
    st.subheader("정지토압 K0")
    fig_k0 = plot_single_distribution(
        profile_k0,
        H,
        title="At-rest Earth Pressure (K0)",
        side="right",
        fill_color="#b8a1d9",
        line_color="#6a4c93",
        soil_color="#6a4c93",
        water_color="#1d4ed8",
    )
    st.pyplot(fig_k0)

    _, _, _, _, P0, y0 = profile_k0
    st.caption(f"K0 = {K0:.4f} | P0 = {P0:.3f} t/m | y_bar = {y0:.3f} m from base")

with tab_rankine:
    st.subheader("Rankine 분포도")
    col_ra, col_rp = st.columns(2)

    with col_ra:
        if profile_rankine_active is not None:
            fig_ra = plot_single_distribution(
                profile_rankine_active,
                H,
                title="Rankine Active (Ka)",
                side="right",
                fill_color="#f4a261",
                line_color="#e76f51",
                soil_color="#8d5524",
                water_color="#1d4ed8",
            )
            st.pyplot(fig_ra)
            _, _, _, _, P, ybar = profile_rankine_active
            st.caption(f"Ka = {Ka_r:.4f} | Pa = {P:.3f} t/m | y_bar = {ybar:.3f} m from base")
        else:
            st.warning("Rankine 주동토압을 계산할 수 없습니다.")

    with col_rp:
        if profile_rankine_passive is not None:
            fig_rp = plot_single_distribution(
                profile_rankine_passive,
                H,
                title="Rankine Passive (Kp)",
                side="left",
                fill_color="#95d5b2",
                line_color="#2d6a4f",
                soil_color="#1b4332",
                water_color="#1d4ed8",
            )
            st.pyplot(fig_rp)
            _, _, _, _, P, ybar = profile_rankine_passive
            st.caption(f"Kp = {Kp_r:.4f} | Pp = {P:.3f} t/m | y_bar = {ybar:.3f} m from base")
        else:
            st.warning("Rankine 수동토압을 계산할 수 없습니다.")

with tab_coulomb:
    st.subheader("Coulomb 분포도")
    col_ca, col_cp = st.columns(2)

    with col_ca:
        if profile_coulomb_active is not None:
            fig_ca = plot_single_distribution(
                profile_coulomb_active,
                H,
                title="Coulomb Active (Ka)",
                side="right",
                fill_color="#ffd6a5",
                line_color="#e85d04",
                soil_color="#9c6644",
                water_color="#1d4ed8",
            )
            st.pyplot(fig_ca)
            _, _, _, _, P, ybar = profile_coulomb_active
            st.caption(f"Ka = {Ka_c:.4f} | Pa = {P:.3f} t/m | y_bar = {ybar:.3f} m from base")
        else:
            st.warning("Coulomb 주동토압을 계산할 수 없습니다.")

    with col_cp:
        if profile_coulomb_passive is not None:
            fig_cp = plot_single_distribution(
                profile_coulomb_passive,
                H,
                title="Coulomb Passive (Kp)",
                side="left",
                fill_color="#cdeac0",
                line_color="#40916c",
                soil_color="#1b4332",
                water_color="#1d4ed8",
            )
            st.pyplot(fig_cp)
            _, _, _, _, P, ybar = profile_coulomb_passive
            st.caption(f"Kp = {Kp_c:.4f} | Pp = {P:.3f} t/m | y_bar = {ybar:.3f} m from base")
        else:
            st.warning("Coulomb 수동토압을 계산할 수 없습니다.")

with tab_compare:
    st.subheader("K0 - Ka - Kp 비교")
    compare_theory = st.radio("비교 기준", ["Rankine", "Coulomb"], horizontal=True)

    if compare_theory == "Rankine":
        if profile_rankine_active is not None and profile_rankine_passive is not None:
            fig_cmp = plot_compare_distribution(
                profile_k0,
                profile_rankine_active,
                profile_rankine_passive,
                H,
                compare_title="K0 / Ka / Kp Comparison (Rankine)"
            )
            st.pyplot(fig_cmp)
        else:
            st.warning("Rankine 비교 그래프를 그릴 수 없습니다.")
    else:
        if profile_coulomb_active is not None and profile_coulomb_passive is not None:
            fig_cmp = plot_compare_distribution(
                profile_k0,
                profile_coulomb_active,
                profile_coulomb_passive,
                H,
                compare_title="K0 / Ka / Kp Comparison (Coulomb)"
            )
            st.pyplot(fig_cmp)
        else:
            st.warning("Coulomb 비교 그래프를 그릴 수 없습니다.")


# ============================
# 문제 입력 도움말
# ============================
with st.expander("📝 문제 입력 팁"):
    st.markdown("""
- **연직배면(β = 90°)** 문제는 이 앱에서 보통 **α = 90°** 로 입력합니다.
- **뒤채움 경사각**은 `i` 에 입력합니다.
- **지하수위가 저면 기준**으로 주어지면, `지하수위 입력 기준`에서 **옹벽 저면 기준**을 선택하면 됩니다.
- **지하수면 위/아래 단위중량이 다르면**, 위는 `γ`, 아래는 `γsat` 에 각각 입력하세요.
- **Coulomb 이론 문제에서 벽면마찰각 δ가 주어지지 않으면**, 보통 **δ = 0°** 로 두고 계산합니다.
""")


# ============================
# 푸터
# ============================
st.markdown("---")
with st.expander("ℹ️ 사용 안내 및 가정사항"):
    st.markdown("""
- **정지토압 K0** 는 Jaky 식 `K0 = 1 - sin(phi)` 를 사용합니다. (정규압밀토 가정)
- **부호 규약**: 깊이 z 는 지표면에서 아래로(+), 작용점 y_bar 는 옹벽 바닥에서 위로(+).
- **점착력 c** 는 주동/수동토압 식에 반영하고, 주동토압에서 인장영역(σ<0)은 0으로 처리합니다.
- **지하수위** 적용 시 수위 위는 `γ`, 수위 아래는 `γsat`를 사용하며, 수중단위중량은 `gamma' = gamma_sat - gamma_w` 로 계산하고 정수압을 별도로 더합니다.
- **Rankine** 은 벽면마찰을 무시한 이론이며, **Coulomb** 은 벽면마찰과 배면조건을 고려한 쐐기 평형해석입니다.
- 본 프로그램은 **학습용 / 개략 검토용** 입니다. 실무 설계 시 KDS/KSCE 기준, 활동·전도·지지력·전체안정 검토를 별도로 수행하세요.
""")

st.caption("© Earth Pressure Calculator — Built with Streamlit")
