#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:42:20 2025

@author: junsaito
"""
# fluxes.py
import numpy as np
import pandas as pd
from scipy.ndimage import sobel, laplace, gaussian_filter


def compute_specific_humidity(press, dew):
    C1, C2, C3 = 611.21, 17.67, 243.5
    tmp_diff = max(dew - C3, 1e-6)  # 負の値防止
    e_s = max(C1 * np.exp((C2 * (dew - 273.15)) / tmp_diff), 1e-9)  # ゼロ以下防止
    return 0.622 * e_s / max(press - 0.378 * e_s, 1e-9)  # ゼロ割り防止


def downscale_pressure(press, t_orig, t_corr, dz):
    """ 標高補正を考慮して気圧をダウンスケール """
    R = 287.0  # 乾燥空気の気体定数 [J/(kg*K)]
    g = 9.81   # 重力加速度 [m/s^2]
    
    T_mean = (t_orig + t_corr) / 2.0
    T_mean = max(T_mean, 200)  # 低温の過小評価防止
    
    return press * np.exp(-g * dz / (R * T_mean))

def compute_radiative_equilibrium_temp(LW_down, emissivity=0.98):
    """
    下向き長波放射 (LW_down [W/m^2]) から放射平衡温度 (K) を計算する。
    パラメータ:
      LW_down: float - 下向き長波放射 [W/m^2]
      emissivity: float - 地表面の放射率（例: 雪面の場合は約0.98）
    戻り値:
      T_rad (K)
    """
    sigma = 5.670374419e-8  # Stefan-Boltzmann定数 [W/(m^2 K^4)]
    LW_eff = max(LW_down, 1e-9)
    emissivity =0.98
    T_rad = (LW_eff / (emissivity * sigma)) ** 0.25
    return T_rad


def compute_sublimation(Ts, Ta, U, pressure, RH, dt_hours=1.0, CE=0.001, Ls=2.83e6):
    """
    風速などの情報を用いた昇華量（sublimation）を計算する関数（バルク法）
    
    Parameters:
      Ts        : 表面温度 (K)
      Ta        : 大気温度 (K)
      U         : 風速 (m/s)
      pressure  : 気圧 (Pa)
      RH        : 相対湿度 (%)（大気側の）
      dt_hours  : 時間刻み (hour)
      CE        : 潜熱乱流交換係数（例: 0.001）
      Ls        : 昇華潜熱 (J/kg)（例: 2.83e6）
      
    Returns:
      sublimation_mm : 昇華量 (mm 水当量 / dt_hours)
    """
    # 表面での飽和比湿
    e_sat_surface = calc_esat(Ts)
    q_sat_surface = 0.622 * e_sat_surface / (pressure - 0.378 * e_sat_surface)
    
    # 大気側の比湿
    e_sat_air = calc_esat(Ta)
    q_sat_air = 0.622 * e_sat_air / (pressure - 0.378 * e_sat_air)
    q_a = (RH / 100.0) * q_sat_air
    
    # 空気密度（理想気体の法則）
    R_d = 287.05
    rho_air = pressure / (R_d * Ta)
    
    # 昇華フラックス (kg/m²/s)
    sublimation_flux = rho_air * CE * U * max(q_sat_surface - q_a, 0)
    
    dt_seconds = dt_hours * 3600.0
    sublimation_mm = sublimation_flux * dt_seconds
    return sublimation_mm

# def drift_accumulation_factor(wind_speed, slope, curvature, wind_direction, aspect, 
#                                 threshold=7.0, erosion_coef=0.01):
#     wind_slope_factor = np.cos(np.radians(wind_direction - aspect))
#     base_factor = 1 + 0.1 * slope * wind_slope_factor + 0.3 * curvature
#     deposition_term = 0.22 * np.minimum(wind_speed, threshold)
#     excess = np.maximum(wind_speed - threshold, 0)
#     erosion_term = erosion_coef * (excess ** 3)
#     factor = base_factor + deposition_term - erosion_term
#     return np.clip(factor, 0.1, 3.0)

def drift_accumulation_factor(wind_speed, slope, curvature, wind_direction, aspect, 
                                threshold=7, erosion_coef=0.01):
    wind_slope_factor = np.cos(np.radians(wind_direction - aspect))
    base_factor = 1 + 0.2 * slope * wind_slope_factor + 0.3 * curvature
    deposition_term = 0.2 * np.minimum(wind_speed, threshold)
    excess = np.maximum(wind_speed - threshold, 0)
    erosion_term = erosion_coef * (excess ** 3)
    factor = base_factor + deposition_term - erosion_term
    return np.clip(factor, 0.1, 3.0)


def compute_emissivity_iziomon(qv, p, T, z):
    """ Iziomon et al. (2003) の方法に基づく大気放射率の計算 """
    e_pa = (qv * p) / (0.622 + qv)  # 水蒸気圧 [Pa]
    def compute_coeff(C1, C2, z1=200, z2=3000):
        if z <= z1:
            return C1
        elif z1 < z < z2:
            return C1 + (z - z1) / (z2 - z1) * (C2 - C1)
        else:
            return C2
    Xs = compute_coeff(0.85, 0.75) # 標高200m:0.85, 3000m:0.75
    Ys = compute_coeff(0.0007, 0.0005)
    Zs = compute_coeff(1.2, 1.0)
    emissivity = 1 - Zs * np.exp(-Ys * e_pa / T)
    return emissivity


def downscale_longwave(lw, t_orig, t_corr, d2m_orig, d2m_corr, p_orig, p_corr, z_orig, z_corr):
    """ 長波放射のダウンスケール（ERA5の比湿・放射率を利用） """
    delta_z = z_corr - z_orig
    p_corr = downscale_pressure(p_orig, t_orig, t_corr, delta_z)
    qv_orig = compute_specific_humidity(p_orig, d2m_orig)
    qv_corr = compute_specific_humidity(p_corr, d2m_corr)
    emi_orig = compute_emissivity_iziomon(qv_orig, p_orig, t_orig, z_orig)
    emi_corr = compute_emissivity_iziomon(qv_corr, p_corr, t_corr, z_corr)
    t_o = max(t_orig, 200)
    t_c = max(t_corr, 200)
    return lw * (emi_corr / emi_orig) * ((t_c / t_o) ** 4)


def calc_snow_ratio(pressure2d, qv2d, temp_corr2d):
    pressure2d_hpa = pressure2d / 100.0
    vap = (pressure2d_hpa * qv2d) / (0.622 + 0.378 * qv2d)
    Tw = 0.584 * (temp_corr2d - 273.15) + 0.875 * vap - 5.32
    Tw_low = 1 - 0.5 * np.exp(-2.2 * np.abs(1.1 - Tw)**1.3)
    Tw_high = 0.5 * np.exp(-2.2 * np.abs(Tw - 1.1)**1.3)
    Tw_low = np.where(Tw >= 1.1, 0, Tw_low)
    Tw_high = np.where(Tw < 1.1, 0, Tw_high)
    s_rate = Tw_low + Tw_high
    return np.clip(s_rate, 0, 1)

# def calc_esat(T):
#     return 611.2 * np.exp(17.62*(T - 273.15)/(T - 30.03))

def calc_esat(T):
    """
    飽和水蒸気圧の計算
    オーバーフローを防ぐため、指数部の値をクリップしています。
    """
    exponent = 17.62 * (T - 273.15) / np.clip(T - 30.03, 1e-6, None)
    exponent = np.clip(exponent, -700, 700)  # np.exp(700) 程度でオーバーフロー防止
    return 611.2 * np.exp(exponent)


import numpy as np

def compute_solar_incidence_angle(zen_deg, azi_deg, slope_deg, aspect_deg):
    """
    太陽の入射角の余弦を計算する関数
    """
    z_rad = np.radians(zen_deg)
    a_rad = np.radians(azi_deg)
    sp_rad = np.radians(slope_deg)
    as_rad = np.radians(aspect_deg)
    return np.cos(z_rad) * np.cos(sp_rad) + np.sin(z_rad) * np.sin(sp_rad) * np.cos(a_rad - as_rad)

def apply_topo_correction(sw_wm2, zen_deg, azi_deg, slope_deg, aspect_deg,
                          dem2d, dx, i0, j0, max_steps=50, min_cos=0.2, max_ratio=2.0):
    """
    地形補正を適用した短波放射値を計算する関数（改変版）
    
    Parameters:
      sw_wm2    : 短波放射 (W/m²)
      zen_deg   : 太陽天頂角 (度)
      azi_deg   : 太陽方位角 (度)
      slope_deg : 斜面角 (度)
      aspect_deg: 斜面方位 (度)
      dem2d     : 標高データの2次元配列
      dx        : DEM のセル間隔（距離）
      i0, j0    : 補正対象セルのインデックス
      max_steps : 影判定に使う最大ステップ数（デフォルト50）
      min_cos   : 太陽天頂角のcos値の下限（ここでは0.2に設定）
      max_ratio : 補正比の上限（ここでは2.0に設定）
      
    Returns:
      補正後の短波放射値 (W/m²)
      
    ※ 観測データとの比較で補正値が高く出すぎる場合、max_ratio と min_cos の値を調整することで対応を試みます。
    """
    # 太陽が水平線下なら放射はゼロ
    if zen_deg > 90:
        return 0.0

    # 斜面への太陽光入射角の余弦を計算
    cos_incidence = compute_solar_incidence_angle(zen_deg, azi_deg, slope_deg, aspect_deg)
    sun_rad = np.radians(azi_deg)
    
    # DEMのラップアラウンドを防ぐため、パディングを追加
    pad = max_steps
    dem_padded = np.pad(dem2d, pad, mode='edge')
    target_i = i0 + pad
    target_j = j0 + pad
    
    # 対象セルの標高
    elev_target = dem_padded[target_i, target_j]
    
    # 影判定: 太陽方向に沿って、対象セルより高い障害がある場合は影として扱う
    for k in range(1, max_steps):
        offset_x = int(np.round(np.cos(sun_rad) * k))
        offset_y = int(np.round(np.sin(sun_rad) * k))
        elev_offset = dem_padded[target_i - offset_y, target_j - offset_x]
        distance = dx * k  # 距離（m）
        # 対象セルとオフセットセル間の角度を計算（単位：度）
        shadow_angle = np.degrees(np.arctan((elev_offset - elev_target) / (distance + 1e-6)))
        # もし障害物の角度が太陽天頂角により遮蔽される場合、対象セルは影にあると判定
        if shadow_angle > (90 - zen_deg):
            return 0.0

    # 水平面での太陽光の入射角のcos値に下限を設定して極端な値を防ぐ
    horizontal_cos = max(np.cos(np.radians(zen_deg)), min_cos)
    # 補正比は、斜面に対する入射角と水平面での入射角の比率
    ratio = np.clip(cos_incidence / horizontal_cos, 0, max_ratio)
    
    # 補正後の短波放射値を返す
    return sw_wm2 * ratio


def compute_rel_humidity(t2d, dew2d):
    C1, C2, C3 = 611.21, 17.67, 243.5
    e_s = C1 * np.exp((C2 * (t2d - 273.15)) / (t2d - 273.15 + C3))
    e = C1 * np.exp((C2 * (dew2d - 273.15)) / (dew2d - 273.15 + C3))
    return np.clip(e / e_s, 0, 1)

def downscale_dew_point(dew2d, lapse2d, dz2d):
    return dew2d + (lapse2d * dz2d)

def to_year_fraction(dt):
    dt = pd.to_datetime(dt)
    year = dt.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year+1, month=1, day=1)
    return year + (dt - start).total_seconds() / ((end - start).total_seconds())

def calc_lapse_rate_block(block_indices, temp2d, elev2d):
    results = []
    for (i, j) in block_indices:
        if i < 2 or j < 2 or i >= temp2d.shape[0]-2 or j >= temp2d.shape[1]-2:
            results.append((i, j, -6.5e-3))
            continue
        t_win = temp2d[i-2:i+3, j-2:j+3].flatten()
        e_win = elev2d[i-2:i+3, j-2:j+3].flatten()
        if np.std(e_win) == 0:
            slope = -6.5e-3
        else:
            slope_, _, _, _, _ = linregress(e_win, t_win)
            slope = np.clip(slope_, -9e-3, -4e-3)
        results.append((i, j, slope))
    return results

# ------------------------------------------------------------
def calc_lapse_rate_parallel(temp2d, elev2d):
    H, W = temp2d.shape
    indices = [(i, j) for i in range(H) for j in range(W)]
    block_size = 1000
    lapse = np.full((H, W), -6.5e-3, dtype=np.float32)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    blocks = [indices[k:k+block_size] for k in range(0, len(indices), block_size)]
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(calc_lapse_rate_block, np.array(blk), temp2d, elev2d) for blk in blocks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Calculating Lapse Rate"):
            for i, j, slope in fut.result():
                lapse[i, j] = slope
    return lapse



def psi_m_stable(xi):
    a, b, c, d = 6.1, 2.5, 5.3, 1.1
    return a*xi + b*(xi - c/d)*np.exp(-d*xi) + (b*c/d)

def psi_h_stable(xi):
    a, b, c, d = 6.1, 2.5, 5.3, 1.1
    return a*xi + b*(xi - c/d)*np.exp(-d*xi) + (b*c/d)

def psi_m_unstable(xi):
    return 2.0 * np.log((1.0 + (1.0 - 16.0*xi)**0.25)/2.0) + np.log((1.0 + (1.0 - 16.0*xi)**0.5)/2.0) - 2.0*np.arctan((1.0 - 16.0*xi)**0.25) + np.pi/2.0

def psi_h_unstable(xi):
    return 2.0 * np.log((1.0 + (1.0 - 16.0*xi)**0.25)/2.0)


def wind_derection(v10,u10):
    wind_direction = float((np.degrees(np.arctan2(v10, u10) + 360)) % 360)
    return wind_direction
    
# --- 風速補正用関数 ---
def convert_wind_speed(U_meas, z_meas=10.0, z_target=2.0, z0w=1e-3):
    """
    対数則に基づき、測定高度 (z_meas) の風速 U_meas を目標高度 (z_target) の風速に変換する。
    """
    return U_meas * np.log(z_target / z0w) / np.log(z_meas / z0w)


def calc_flux_bulk_iter(Tz, ez, Uz, T0, e0, pz=101325.0, z=2.0, 
                        z_meas=10.0,  # measurement height (ERA5 is typically 10 m)
                        z0w=1e-3, z0T=1e-4, z0E=1e-4,
                        rho=1.3, cp=1005.0, Lv=2.5e6,
                        g=9.81, k=0.4, max_iter=20,
                        tol=1.0,  # convergence tolerance for L_current
                        QH_max=200, QE_max=200, damp_factor=0.5,
                        RH=None):
    """
    Calculate surface sensible (QH) and latent (QE) heat fluxes using the bulk method.
    
    If an observed relative humidity (RH, in %) is provided (not None), then compute the
    atmospheric water vapor pressure as:
         e_atm = (RH / 100) * calc_esat(Tz)
    and use delta_e = e_atm - e0 in the latent heat flux calculation.
    
    Parameters:
      Tz : float
          Air temperature at height z (K)
      ez : float
          Water vapor pressure at height z (Pa)
      Uz : float
          Wind speed at height z_meas (m/s)
      T0 : float
          Surface temperature (K)
      e0 : float
          Surface water vapor pressure (Pa)
      pz : float, optional
          Air pressure at height z (Pa) (default 101325 Pa)
      z : float, optional
          Target height for flux calculations (m) (default 2 m)
      z_meas : float, optional
          Measurement height for wind speed (m) (default 10 m)
      z0w, z0T, z0E : float, optional
          Roughness lengths for momentum, heat, and moisture (m)
      rho, cp, Lv, g, k : float, optional
          Air density, specific heat, latent heat, gravitational acceleration, von Karman constant
      max_iter : int, optional
          Maximum number of iterations
      tol : float, optional
          Convergence tolerance for Obukhov length
      QH_max, QE_max : float, optional
          Maximum absolute flux values for clipping (W/m²)
      damp_factor : float, optional
          Damping factor for iterative update of Obukhov length
      RH : float or None, optional
          Observed relative humidity (%) at height z (if provided, used to compute e_atm)
          
    Returns:
      (QH, QE) : tuple of floats
          Sensible and latent heat fluxes (W/m²)
    """
    # Return zero fluxes if wind speed is too small
    if Uz < 0.1:
        return 0.0, 0.0

    # Convert measured wind speed from z_meas to target height z using logarithmic profile
    if z_meas != z:
        Uz = convert_wind_speed(Uz, z_meas, z, z0w)

    # Clip wind speed and temperatures to physically plausible ranges
    Uz = np.clip(Uz, 0.5, 30.0)
    Tz = np.clip(Tz, 200, 320)
    T0 = np.clip(T0, 200, 320)
    delta_T = np.clip(Tz - T0, -10, 10)

    # Use observed relative humidity if provided; otherwise, use the input water vapor pressure ez
    if RH is not None:
        # Compute the atmospheric water vapor pressure from RH and air temperature
        e_atm = (RH / 100.0) * calc_esat(Tz)
        delta_e = np.clip(e_atm - e0, -50, 50)
    else:
        delta_e = np.clip(ez - e0, -50, 50)

    # Compute air density using the ideal gas law
    T_mean = (Tz + T0) / 2.0
    R_d = 287.05
    rho_air = np.clip(pz / (R_d * T_mean), 0.8, 1.5)

    # Iteratively compute the Obukhov length L_current
    L_current = 1e6
    for _ in range(max_iter):
        xi = z / L_current
        if L_current > 0:
            psi_m_val = psi_m_stable(xi)
            psi_h_val = psi_h_stable(xi)
        else:
            psi_m_val = psi_m_unstable(xi)
            psi_h_val = psi_h_unstable(xi)
        denom_m = max(np.log(z / z0w) - psi_m_val, 1e-6)
        denom_h = max(np.log(z / z0T) - psi_h_val, 1e-6)
        u_star = k * Uz / denom_m
        CH = (k**2) / (denom_m * denom_h)
        QH = rho_air * cp * CH * Uz * delta_T
        QE = rho_air * Lv * CH * Uz * (delta_e / pz)
        if abs(QH) > 1e-9:
            L_new = -rho_air * cp * (u_star**3) * Tz / (k * g * QH)
        else:
            L_new = 1e6
        L_new = L_current + damp_factor * (L_new - L_current)
        L_current_new = max(min(L_new, 1e6), -1e6)
        if abs(L_current_new - L_current) < tol:
            L_current = L_current_new
            break
        L_current = L_current_new
        if np.isnan(L_current) or np.isinf(L_current):
            L_current = 1e6
    QH_final = np.clip(QH, -QH_max, QH_max)
    QE_final = np.clip(QE, -QE_max, QE_max)
    return QH_final, QE_final


def adjust_wind_speed(W_coarse, u10, v10, slope, curvature, delta_z, roughness, d=0.1, measurement_height=10.0):
    kappa = 0.41
    z0 = np.maximum(roughness, 1e-3)
    u_star = kappa * W_coarse / np.log((measurement_height - d) / z0)
    U_log = u_star / kappa * np.log((measurement_height - d) / z0)
    wind_dir_rad = np.arctan2(v10, u10)
    slope_rad = np.radians(slope)
    Vs = slope_rad * np.cos(wind_dir_rad)
    terrain_factor = 1 + 0.5 * Vs + 0.5 * curvature
    altitude_factor = np.exp(0.0015 * np.clip(delta_z, -500, 500))
    U_fine = U_log * terrain_factor * altitude_factor
    adjusted_speed = gaussian_filter(U_fine, sigma=2)
    return np.clip(adjusted_speed, 0.1, 50)

def compute_slope_and_aspect(dem, dx):
    dzdx = sobel(dem, axis=1) / (8.0 * dx)
    dzdy = sobel(dem, axis=0) / (8.0 * dx)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)
    aspect = np.arctan2(-dzdy, dzdx) * (180 / np.pi)
    aspect[aspect < 0] += 360
    return slope, aspect

def adjust_precipitation(P0, z, z0, kappa):
    dz_km = (z - z0) / 1000.0
    correction = 1 + dz_km / (1 + dz_km / kappa)
    return P0 * correction


def calc_new_snow_density(T2m, Ts, RH, U10m):
    T2m_C = T2m - 273.15
    Ts_C = Ts - 273.15
    rho_new = (70 + 6.5 * T2m_C + 7.5 * Ts_C + 0.26 * RH + 13 * U10m
               - 4.5 * T2m_C * Ts_C - 0.65 * T2m_C * U10m - 0.17 * RH * U10m
               + 0.06 * T2m_C * Ts_C * RH)
    return np.clip(rho_new, 30, 150)



def calc_redeposit_density(U10m):
    if U10m > 1:
        return 361 * np.log10(U10m) + 33
    else:
        return 33

def calc_threshold_friction_velocity(SP, rg, rb, N3, rho_air=1.1, rho_ice=917, g=9.8, sigma=300):
    A, B = 0.02, 0.0015
    return np.sqrt((A * rho_ice * rg * (SP + 1) + B * sigma * N3 * (rb**2 / rg**2)) / rho_air)



import numpy as np
from scipy.optimize import bisect


heat_diff_cache = {}

# def albedo_from_temperature(T, alpha_cold=0.89, alpha_melt=0.4, T0=274, k_albedo=0.5):
#     """
#     温度に基づいてアルベドを推定する関数
#     参考: Sturm et al. (1997), Bair (2003) など
#     パラメータ:
#       T          : 表面温度 (K)
#       alpha_cold : 融解前のアルベド（例: 0.89）
#       alpha_melt : 融解状態のアルベド（例: 0.4）
#       T0         : 遷移温度 (K)
#       k_albedo   : 遷移の急峻さ
#     """
#     return alpha_cold - (alpha_cold - alpha_melt) / (1 + np.exp(-k_albedo * (T - T0)))


import numpy as np
import matplotlib.pyplot as plt
# Pirazzini, 2004
def albedo_from_temperature(T, alpha_cold=0.80, alpha_melt=0.65, T0=270, k_albedo=1.0):
    return alpha_cold - (alpha_cold - alpha_melt) / (1 + np.exp(-k_albedo * (T - T0)))


def solve_heat_diffusion(T_surface, T_deep, L, kappa, z_ref, dt, dz, total_time, max_iter_total=10000, tol=1e-3):
    """
    1次元非定常熱伝導方程式（熱拡散方程式）をクランク・ニコルソン法で解く関数
    Parameters:
      T_surface : 表面温度 (K)
      T_deep    : 深部温度 (K)
      L         : モデルの全深さ (m)
      kappa     : 熱拡散係数 (m²/s)
      z_ref     : 推定したい深さ (m)
      dt        : 時間刻み (s)
      dz        : 空間刻み (m)
      total_time: シミュレーション総時間 (s)
    Returns:
      T_sub     : z=z_ref における温度 (K)
      T         : 最終時刻の温度プロファイル (K)
      z         : 深さの配列 (m)
    """
    N = int(L / dz) + 1
    z = np.linspace(0, L, N)
    n_steps = int(total_time / dt)
    T = np.linspace(T_surface, T_deep, N)
    r = kappa * dt / (dz**2)
    
    A = np.zeros((N-2, N-2))
    B = np.zeros((N-2, N-2))
    for i in range(N-2):
        if i > 0:
            A[i, i-1] = -r/2
            B[i, i-1] = r/2
        A[i, i] = 1 + r
        B[i, i] = 1 - r
        if i < N-3:
            A[i, i+1] = -r/2
            B[i, i+1] = r/2

    iter_count = 0
    for _ in range(n_steps):
        iter_count += 1
        if iter_count > max_iter_total:
            print("最大反復回数に達しました。途中で終了します。")
            break
        b = B @ T[1:-1]
        b[0] += (r/2) * T_surface
        b[-1] += (r/2) * T_deep
        T[1:-1] = np.linalg.solve(A, b)
        T[0] = T_surface
        T[-1] = T_deep
    T_sub = np.interp(z_ref, z, T)
    return T_sub, T, z


def compute_surface_temperature(Sin, albedo, Lin, Ta, U, pressure, q_a, 
                                rho_new, rho_redeposit,
                                Tsub=None, Delta_T=0.5, method_Tsub='conduction', 
                                q_g=0.05, T_deep=270.0, z_ref=1.0, L=2.0,
                                T_lower=200, T_upper=273,
                                albedo_method='temp_based',   
                                # alpha_cold=0.85, alpha_melt=0.6, k_albedo=0.2,
                                alpha_cold=0.80, alpha_melt=0.65, k_albedo=1.0,
                                previous_snow_albedo=0.85,  # 前時刻の雪面アルベド（初期値など）
                                dt_hours=1.0,               # 時間刻み (hour)
                                BDOT=0.0,                   # 降雪量 (mm水当量)
                                RH=None                   # 大気側の相対湿度 (%)
                               ):
    """
    表面エネルギーバランス式
      Sin - (albedo * Sin) + Lin - Lout + H + LE + QG = 0
    を解いて、表面温度 Ts と各種フラックス、推定アルベドなどを計算する関数。
    
    Returns:
      Ts_solution     : 表面温度 (K)
      QM              : 融解に利用される余剰エネルギー (W/m²)
      melt_rate       : 融解量 (kg/m²/h)
      G               : 地中熱フラックス (W/m²)
      estimated_albedo: 更新された表面アルベド
      Tsub_est        : 補完または推定した Tsub (K)
      Lnet            : ネット長波放射 (W/m²)
      Snet            : ネット短波放射 (W/m²)
      Sublimation     : 昇華量 (mm水当量/時間)
    """
    # --- 定数 ---
    sigma = 5.670374419e-8
    epsilon = 0.988
    rho_air = 1.3
    cp = 1005.0
    CH = 0.001
    CE = 0.001
    Lv = 2.5e6
    d = 0.5
    L_f = 3.34e5

    # --- Tsub (サブ表面温度) の補完 ---
    if Tsub is None:
        if method_Tsub == 'Delta_T':
            Tsub = Ta - Delta_T
            Tsub_est = Tsub
        elif method_Tsub == 'geothermal':
            Tsub_est = np.nan
        elif method_Tsub == 'conduction':
            key = (round(Ta, 1), round(T_deep, 1), L, round(z_ref, 2))
            if key in heat_diff_cache:
                Tsub_est, _, _ = heat_diff_cache[key]
            else:
                Tsub_est, T_profile, z_arr = solve_heat_diffusion(T_surface=Ta, T_deep=T_deep, L=L, 
                                                                   kappa=1e-6, z_ref=z_ref, 
                                                                   dt=3600, dz=0.5, total_time=3600)
                heat_diff_cache[key] = (Tsub_est, T_profile, z_arr)
            Tsub = Tsub_est
        else:
            raise ValueError("Unknown method_Tsub. Use 'Delta_T', 'geothermal' or 'conduction'.")
    else:
        Tsub_est = Tsub

    # --- K_eff の計算 (Calonne et al. 2019) ---
    K_ice = 2.1
    RHO_I = 917.0
    rho_eff = rho_new + rho_redeposit
    rho_transition = 450.0
    a_param = 0.02
    theta = 1.0 / (1.0 + np.exp(-2 * a_param * (rho_eff - rho_transition)))
    kref_firn = 2.107 + 0.003618 * (rho_eff - RHO_I)
    kref_snow = 0.024 - 1.23e-4 * rho_eff + 2.5e-6 * rho_eff**2
    kref_i = 2.107
    kref_a = 0.024
    K_air = kref_a
    K_eff = (1 - theta) * (K_ice * K_air / (kref_i * kref_a)) * kref_snow + theta * (K_ice / kref_i) * kref_firn

    def calc_esat(T):
        exponent = 17.62 * (T - 273.15) / np.clip(T - 30.03, 1e-6, None)
        exponent = np.clip(exponent, -700, 700)
        return 611.2 * np.exp(exponent)

    def q_sat(Ts_local):
        es = calc_esat(Ts_local)
        return 0.622 * es / max(pressure - 0.378 * es, 1e-9)

    # --- エネルギーバランス式の定義 ---
    def energy_balance_equation(Ts):
        Lout = epsilon * sigma * Ts**4
        H_term = rho_air * cp * CH * U * (Ta - Ts)
        LE_term = rho_air * Lv * CE * U * (q_a - q_sat(Ts))
        if Tsub is None:
            if method_Tsub == 'geothermal':
                QG_term = - (q_g * z_ref) / d
            elif method_Tsub == 'conduction':
                key_local = round(Ts, 1)
                if key_local in heat_diff_cache:
                    Tsub_est_local, _, _ = heat_diff_cache[key_local]
                else:
                    Tsub_est_local, _, _ = solve_heat_diffusion(T_surface=Ts, T_deep=T_deep, L=L, 
                                                                 kappa=1e-6, z_ref=z_ref, 
                                                                 dt=3600, dz=0.01, total_time=3600)
                    heat_diff_cache[key_local] = (Tsub_est_local, None, None)
                QG_term = -K_eff * (Ts - Tsub_est_local) / d
            else:  # Delta_T
                QG_term = -K_eff * (Ts - (Ta - Delta_T)) / d
        else:
            QG_term = -K_eff * (Ts - Tsub) / d
        return Sin - (albedo * Sin) + Lin - (epsilon * sigma * Ts**4) + H_term + LE_term + QG_term

    # --- 表面温度 Ts の求解 ---
    f_lower = energy_balance_equation(T_lower)
    f_upper = energy_balance_equation(T_upper)
    if f_lower * f_upper > 0:
        Ts_solution = T_lower if abs(f_lower) < abs(f_upper) else T_upper
        QM = energy_balance_equation(Ts_solution)
    else:
        Ts_solution = bisect(energy_balance_equation, T_lower, T_upper)
        if Ts_solution > 273:
            QM = energy_balance_equation(273)
            Ts_solution = 273
        else:
            QM = 0.0

    melt_rate = QM / L_f * 3600  # 融解量 [kg/(m²·h)]

    # --- 地中熱フラックス G の計算 ---
    if Tsub is None:
        if method_Tsub == 'geothermal':
            G = - (q_g * z_ref) / d
        elif method_Tsub == 'conduction':
            if round(Ts_solution, 1) in heat_diff_cache:
                Tsub_est = heat_diff_cache[round(Ts_solution, 1)][0]
            else:
                Tsub_est, _, _ = solve_heat_diffusion(T_surface=Ts_solution, T_deep=T_deep, L=L, 
                                                       kappa=1e-6, z_ref=z_ref, 
                                                       dt=3600, dz=0.01, total_time=3600)
                heat_diff_cache[round(Ts_solution, 1)] = (Tsub_est, None, None)
            G = -K_eff * (Ts_solution - Tsub_est) / d
        else:
            Tsub_calc = Ta - Delta_T
            G = -K_eff * (Ts_solution - Tsub_calc) / d
    else:
        G = -K_eff * (Ts_solution - Tsub) / d

    # --- アルベドの計算 ---
    if albedo_method.lower() == "temp_based":
        # 従来の温度依存アルベド
        used_albedo = albedo_from_temperature(Ts_solution, alpha_cold=alpha_cold, alpha_melt=alpha_melt, T0=273.15, k_albedo=k_albedo)
    elif albedo_method.lower() == "snow_albedo":
        # 雪面アルベド更新: Kondo and Xu (1997) に基づく
        alpha_f = 0.4  # 最低の雪面アルベド（下限値）
        Ta_C = Ta - 273.15  # 摂氏変換
        # 空気温度により減衰定数を設定
        if Ta_C < 0.5:
            k_decay = 5.5 - 3.0 * Ta_C
        else:
            k_decay = 4.0
        # 降雪があった場合（BDOT >= 5 mm なら新雪とみなす）
        if BDOT >= 5:
            if Ta_C < -1.0:
                alpha0 = 0.88
            elif -1.0 <= Ta_C <= 3.0:
                alpha0 = (alpha_f - 0.88) * (Ta_C + 1.0) / 4.0 + 0.88
            else:
                alpha0 = alpha_f
            used_albedo = alpha0
        else:
            dt_days = dt_hours / 24.0
            used_albedo = (previous_snow_albedo - alpha_f) * np.exp(-dt_days / k_decay) + alpha_f
    elif albedo_method.lower() == "modis":
        used_albedo = albedo
    elif albedo_method.lower() == "seb":
        Lout_sol = epsilon * sigma * Ts_solution**4
        H_sol = rho_air * cp * CH * U * (Ta - Ts_solution)
        LE_sol = rho_air * Lv * CE * U * (q_a - q_sat(Ts_solution))
        if Tsub is None:
            if method_Tsub == 'geothermal':
                QG_sol = - (q_g * z_ref) / d
            elif method_Tsub == 'conduction':
                key_local = round(Ts_solution, 1)
                if key_local in heat_diff_cache:
                    Tsub_est_local, _, _ = heat_diff_cache[key_local]
                else:
                    Tsub_est_local, _, _ = solve_heat_diffusion(T_surface=Ts_solution, T_deep=T_deep, L=L, 
                                                                 kappa=1e-6, z_ref=z_ref, 
                                                                 dt=3600, dz=0.01, total_time=3600)
                    heat_diff_cache[key_local] = (Tsub_est_local, None, None)
                QG_sol = -K_eff * (Ts_solution - Tsub_est_local) / d
            else:
                QG_sol = -K_eff * (Ts_solution - (Ta - Delta_T)) / d
        else:
            QG_sol = -K_eff * (Ts_solution - Tsub) / d
        Net_sw = Lout_sol - H_sol - LE_sol - QG_sol - Lin
        used_albedo = 1 - (Net_sw / Sin) if Sin > 1e-6 else albedo
        used_albedo = np.clip(used_albedo, 0, 0.9)
    else:
        raise ValueError("Unknown albedo_method. Use 'temp_based', 'modis', 'seb', or 'snow_albedo'.")
    estimated_albedo = used_albedo

    # --- ネット長波放射 (Lnet) とネット短波放射 (Snet) の計算 ---
    Lout_sol = epsilon * sigma * Ts_solution**4
    Lnet = Lin - Lout_sol
    Snet = Sin - (used_albedo * Sin)

    # --- 昇華量 (Sublimation) の計算 ---
    # ここでは、compute_sublimation を呼び出して昇華フラックスを計算する
    # 引数として、Ts_solution, Ta, U, pressure, および RH を用いる
    Sublimation = compute_sublimation(Ts_solution, Ta, U, pressure, RH, dt_hours=dt_hours)

    return Ts_solution, QM, melt_rate, G, estimated_albedo, Tsub_est, Lnet, Snet, Sublimation



