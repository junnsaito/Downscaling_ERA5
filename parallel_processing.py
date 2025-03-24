#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:44:09 2025

@author: junsaito
"""

# parallel_processing.py

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from tqdm import tqdm
import numpy as np
import pandas as pd
# ERA5再投影＆補正関連
from era5 import reproject_to_rema, adjust_temperature_with_station, adjust_dewpoint_with_station
# fluxes関連
from fluxes import (
    compute_specific_humidity, calc_esat, downscale_pressure,compute_emissivity_iziomon,
    downscale_longwave, calc_snow_ratio, compute_rel_humidity,
    calc_flux_bulk_iter,apply_topo_correction,calc_new_snow_density,calc_redeposit_density,adjust_wind_speed,
    convert_wind_speed, compute_surface_temperature,drift_accumulation_factor,wind_derection,adjust_precipitation)


def process_time_step(
    i, 
    ds_year,
    src_transform_era5,
    rema_transform,
    W_dem, H_dem,
    i0, j0,
    lapse_value,        # t2mラプスレート
    lapse_value2,       # d2mラプスレート
    lapse_value3,       # sktラプスレート
    delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array,
    modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev,
    z_orig,
    wind_slope,  # 追加: 風速スロープ
    wind_intercept,  # 追加: 風速intercept
    P_slope,  # 追加: 風速スロープ
    P_intercept,  # 追加: 風速intercept
    RH_slope,  # 追加: 風速スロープ
    RH_intercept,  # 追加: 風速intercept
    sw_bias,      # 追加: SW の平均バイアス
    lw_bias,       # 追加: LW の平均バイアス
    kappa_field,apply_altitude_corr=True
):
    """
    1タイムステップ (i番目) のERA5データを再投影し、温度や降水などを補正計算して返す。
    
    戻り値: (sw_corr, lw_corr, temp_corr, tskin_corr, QH_val, QE_val,
             RAIN_val, BDOT_val, time_val, ALBEDO_val, rho_new_val, rho_redeposit_val)
    """

    # ---------------------------
    # 1) ds_year から i番目の2次元フィールドを取り出す
    # ---------------------------

    t2m_2d  = ds_year["t2m"].isel(valid_time=i).values
    skt_2d  = ds_year["skt"].isel(valid_time=i).values
    tp_2d   = ds_year["tp"].isel(valid_time=i).values
    ssrd_2d = ds_year["ssrd"].isel(valid_time=i).values
    strd_2d = ds_year["strd"].isel(valid_time=i).values
    sp_2d   = ds_year["sp"].isel(valid_time=i).values
    u10_2d  = ds_year["u10"].isel(valid_time=i).values
    v10_2d  = ds_year["v10"].isel(valid_time=i).values
    d2m_2d  = ds_year["d2m"].isel(valid_time=i).values
    # fsr_2d  = ds_year["fsr"].isel(valid_time=i).values  # 例: fsr (optional変数)

    # ---------------------------
    # 2) ERA5をDEM座標系に再投影 (reproject_to_rema)
    # ---------------------------
    src_crs = "EPSG:4326"
    dst_crs = "EPSG:4326"  # この例では同じEPSG:4326にしているが、実際にはDEMに合わせる想定

    t2m_re  = reproject_to_rema(t2m_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    skt_re  = reproject_to_rema(skt_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    tp_re   = reproject_to_rema(tp_2d,   src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    ssrd_re = reproject_to_rema(ssrd_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    strd_re = reproject_to_rema(strd_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    sp_re   = reproject_to_rema(sp_2d,   src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    u10_re  = reproject_to_rema(u10_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    v10_re  = reproject_to_rema(v10_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    d2m_re  = reproject_to_rema(d2m_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    # fsr_re  = reproject_to_rema(fsr_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)

    # ---------------------------
    # 3) DEMインデックス (i0, j0) の値を取り出す & 観測所補正など
    # ---------------------------
    current_date = pd.Timestamp(ds_year.valid_time.values[i])
    grid_elev = rema_dem_data[i0, j0]

    # 例: adjust_temperature_with_station(...) で実測補正＋ラプスレート補正
    t2m_val  = t2m_re[i0, j0]
    # 例: 露点 (d2m)
    d2m_val = d2m_re[i0, j0]

    
    
    # skt (Tskin from ERA5 does not fit!)
    skt_val = skt_re[i0, j0]
    tskin_corr = skt_val + lapse_value3 * delta_z_point
    tskin_degC = tskin_corr - 273.15
    
    # ERA5の標高（Geopotential）を取得
    z_era5_m = z_orig[i0, j0]  # ERA5の標高 (m)
    z_rema_m = rema_dem_data[i0, j0]  # REMAの標高 (m)
    dz = z_rema_m - z_era5_m  # 標高差 (m)
    
    
    
 
    
    T_adjusted = adjust_temperature_with_station(
    era5_temp=t2m_val,
    z_coarse=z_era5_m,
    grid_elev=grid_elev,
    station_elev=station_elev,
    date=current_date,
    station_data_1H=station_data,   # 1H補間された観測データ
    lapse_rate=lapse_value,
    method="lapse_rate"             # ← ここを明示
)
    
    temp_corr = T_adjusted
    
    
    
    
    T_adjusted2 = adjust_dewpoint_with_station(
    era5_dew=d2m_val,
    z_coarse=z_era5_m,
    grid_elev=grid_elev,
    station_elev=station_elev,
    date=current_date,
    station_data_1H=station_data2,
    dew_lapse_rate=lapse_value2,
    method="lapse_rate"             # ← 同様にラプス補正
)
    dt_corr = T_adjusted2

    # ---------------------------
    # 4) 風速, 気圧, 湿度, etc.
    # ---------------------------
    sp_val = sp_re[i0, j0]  
    # ERA5の気圧補正
    pressure_corr = downscale_pressure(sp_val, t2m_val, temp_corr, dz)
    
    # さらに観測との平均差 press_bias を加える
    pressure_final = pressure_corr*P_slope + P_intercept
    qv_orig = compute_specific_humidity(sp_re[i0, j0], d2m_re[i0, j0])
    qv_corr = compute_specific_humidity(pressure_final, dt_corr)

    u10_val = u10_re[i0, j0]
    v10_val = v10_re[i0, j0]
    wind_speed_raw = np.sqrt(u10_val**2 + v10_val**2)
    wind_speed_intermediate = adjust_wind_speed(
        W_coarse = wind_speed_raw,
        u10 = u10_val,
        v10 = v10_val,
        slope = slope2d[i0, j0],
        curvature = curvature2d[i0, j0],
        delta_z = (z_rema_m - z_era5_m),
        roughness= 1e-3,  # placeholder, 適宜 z0w など?
        d=0.1,
        measurement_height=10.0
    )
    # 2) 昭和基地との回帰 wind_slope, wind_intercept を適用
    wind_speed_final = wind_slope * wind_speed_intermediate + wind_intercept
    wind_speed_final = np.clip(wind_speed_final, 0.1, 50)
    
    
    # 相対湿度など
    RH_val = compute_rel_humidity(temp_corr, dt_corr) * 100
    RH_final = RH_slope * RH_val + RH_intercept


    # -------------------1---------
    # 6) 降水, RAIN, BDOT (例: 雨 + 雪)
    # ---------------------------
    tp_val = tp_re[i0, j0] * 1000.0  # ERA5 の tp は m なので mm に変換
    s_rate = calc_snow_ratio(pressure_final, qv_corr, temp_corr)
    
    # 標高依存補正の適用フラグに応じて降水量補正を行う
    if apply_altitude_corr:
        z_local = rema_dem_data[i0, j0]
        z0 = np.median(rema_dem_data)  # DEM全体の中央値を基準にする（例）
        kappa_val = kappa_field[i0, j0]
        P0_elev = adjust_precipitation(tp_val, z_local, z0, kappa_val)
    else:
        P0_elev = tp_val
    
    # 補正後の降水量 P0_elev を用いて、雨と雪に分割
    rain_mm = P0_elev * (1 - s_rate)
    snow_mm = P0_elev * s_rate
    RAIN_val = rain_mm
    BDOT_val = rain_mm + snow_mm
    BDOT_val_s = snow_mm

    # ---------------------------
    # 7) 短波・長波 (ssrd, strd)
    # ---------------------------
    ssrd_val = ssrd_re[i0, j0]
    strd_val = strd_re[i0, j0]

    sw_val = ssrd_val / time_interval_s
    lw_val = strd_val / time_interval_s
    
    
    zen_val = zenith_array[i]
    azi_val = azimuth_array[i]

    # 短波放射の補正
    if sw_val < 1e-6:
        sw_corr = 0.0
    else:
        sw_corr = sw_val + sw_bias  # バイアスを加える
        sw_corr = apply_topo_correction(sw_corr, zen_val, azi_val, slope2d[i0, j0],
                                        aspect2d[i0, j0], rema_dem_data, dx=500, i0=i0, j0=j0)
    sw_corr = np.maximum(sw_corr, 0)  # 負の値は 0 に
    
    # 長波放射の補正
    if lw_val < 1e-6:
        lw_corr = 0.0
    else:
        lw_corr = lw_val + lw_bias  # バイアスを加える
        lw_corr = downscale_longwave(lw_corr, t2m_val, temp_corr, d2m_val, dt_corr,
                                     sp_val, pressure_final, z_era5_m, z_rema_m)
        lw_corr = np.clip(lw_corr, 0, 500)


# 既存の変数（sw_corr, lw_corr, temp_corr, wind_speed_final, pressure_final, qv_corr 等）はそのまま
# また、BDOT_value, previous_snow_albedo_value, dt_hours_value はモデルの出力または設定値から取得する
# 例：前時刻の雪面アルベド、時間刻み、降雪量の設定
    from fluxes import compute_surface_temperature
    previous_snow_albedo_value = 0.85   # 前時刻の雪面アルベド（初期値など）
    dt_hours_value = 1.0                # 1時間刻み   
    # BDOT_value = BDOT_val_s             # BDOT_val_s は既に計算または与えられている降雪量の値(mm水当量)
    WD = wind_derection(v10_val,u10_val)
    
    slope_cell      = slope2d[i0, j0]
    curvature_cell  = curvature2d[i0, j0]
    aspect_cell     = aspect2d[i0, j0]

    drift_factor = drift_accumulation_factor(wind_speed_final, slope_cell, curvature_cell, WD, aspect_cell, threshold=7.0, erosion_coef=0.01)
    BDOT_value = drift_factor * BDOT_val
    # ここでは、snow_albedo として前時刻の値を初期値として使用する
    snow_albedo = previous_snow_albedo_value
    
    # rho_new と rho_redeposit の計算（calc_new_snow_density, calc_redeposit_density は fluxes.py 内の関数）
    rho_new_val = calc_new_snow_density(temp_corr, skt_val, RH_final, wind_speed_final)
    rho_redeposit_val = calc_redeposit_density(wind_speed_final)
    
    # ALBEDO_val は参考値として 0.8 を指定（ここでは使われず、albedo_method が "snow_albedo" の場合は内部で更新される）
    ALBEDO_val = 0.8
    Ts_surface, QM, melt_rate, G, estimated_albedo, Tsub_est, Lnet, Snet, Sublimation = compute_surface_temperature(
        Sin=sw_corr,
        albedo=ALBEDO_val,  
        Lin=lw_corr,
        Ta=temp_corr,
        U=wind_speed_final,
        pressure=pressure_final,
        q_a=qv_corr,
        rho_new=rho_new_val,
        rho_redeposit=rho_redeposit_val,
        Tsub=None,
        Delta_T=0.5,
        method_Tsub='conduction',
        q_g=0.05,
        T_deep=270.0,
        z_ref=1.0,
        L=2.0,
        T_lower=200,
        T_upper=273,
        albedo_method='temp_based',  # または 'snow_albedo' に変更可能
        alpha_cold=0.80,
        alpha_melt=0.65,
        k_albedo=1.0,
        previous_snow_albedo=previous_snow_albedo_value,
        dt_hours=dt_hours_value,
        BDOT=BDOT_value,
        RH=RH_final  # ここで大気側の相対湿度を渡す
    )


    # 再計算： compute_surface_temperature() で得た Ts_surface を用いて密度計算
    rho_new_val = calc_new_snow_density(temp_corr, Ts_surface, RH_final, wind_speed_final)
    rho_redeposit_val = calc_redeposit_density(wind_speed_final)
    ez_val = calc_esat(dt_corr)  # 大気水蒸気圧
    e0_val = calc_esat(Ts_surface)
    
    
    # 5) フラックス計算に補正後の風速/気圧を使用
    QH_val, QE_val = calc_flux_bulk_iter(
        Tz = temp_corr,
        ez = ez_val,
        Uz = wind_speed_final,  # 補正済み風速
        T0 =  Ts_surface,
        e0 = e0_val,
        pz = pressure_final     # 補正済み気圧
    )    
    
    
    # 9) その他： 雪密度補正、放射平衡温度の計算
    from fluxes import compute_radiative_equilibrium_temp
    Tskin_rad = compute_radiative_equilibrium_temp(lw_corr, emissivity=0.98)
    time_val = current_date


    return (
        sw_corr,
        lw_corr,
        temp_corr,
        tskin_corr,
        QH_val,
        QE_val,
        RAIN_val,
        BDOT_val,
        time_val,
        ALBEDO_val,
        rho_new_val,
        rho_redeposit_val,
        wind_speed_final,
        pressure_final,
        RH_final,
        Ts_surface,
        melt_rate,
        G,estimated_albedo, Tsub_est, Lnet, Snet,QM ,Sublimation,u10_val,v10_val,BDOT_value
    )


def process_time_steps_chunk(
    time_indices, ds_year,
    src_transform_era5, rema_transform,
    W_dem, H_dem, i0, j0,
    lapse_value, lapse_value2, lapse_value3, delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array, modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev, z_orig,
    wind_slope,
    wind_intercept,
    P_slope,
    P_intercept,
    RH_slope,
    RH_intercept,
    sw_bias,      # 追加: SW の平均バイアス
    lw_bias,
kappa_field,apply_altitude_corr=False       # 追加: LW の平均バイアス
):
    results = []
    for i in time_indices:
        row_result = process_time_step(
            i, ds_year,
            src_transform_era5, rema_transform,
            W_dem, H_dem, i0, j0,
            lapse_value, lapse_value2, lapse_value3, delta_z_point,
            slope2d, aspect2d, curvature2d,
            zenith_array, azimuth_array, modis_albedo_value,
            rema_dem_data,
            time_interval_s,
            station_data,
            station_data2,
            station_elev, z_orig,
            wind_slope,
            wind_intercept,
            P_slope,
            P_intercept,
            RH_slope,
            RH_intercept,
            sw_bias,      # 追加: SW の平均バイアス
            lw_bias,kappa_field,apply_altitude_corr=False      # 追加: LW の平均バイアス
        )
        results.append(row_result)
    return results

def task(
    i, ds_year,
    src_transform_era5, rema_transform,
    W_dem, H_dem, i0, j0,
    lapse_value, lapse_value2, lapse_value3, delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array, modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev, z_orig,
    wind_slope,
    wind_intercept,
    P_slope,
    P_intercept,
    RH_slope,
    RH_intercept,
    sw_bias,      # 追加: SW の平均バイアス
    lw_bias,kappa_field,apply_altitude_corr=False       # 追加: LW の平均バイアス
):
    # 共有メモリ利用例（必要に応じて実装）
    # ※ここでは単純に process_time_step() を呼び出す
    result = process_time_step(
        i, ds_year,
        src_transform_era5, rema_transform,
        W_dem, H_dem, i0, j0,
        lapse_value, lapse_value2, lapse_value3, delta_z_point,
        slope2d, aspect2d, curvature2d,
        zenith_array, azimuth_array, modis_albedo_value,
        rema_dem_data,
        time_interval_s,
        station_data,
        station_data2,
        station_elev, z_orig,
        wind_slope,
        wind_intercept,
        P_slope,
        P_intercept,
        RH_slope,
        RH_intercept,
        sw_bias,      # 追加: SW の平均バイアス
        lw_bias,kappa_field,apply_altitude_corr=False       # 追加: LW の平均バイアス
    )
    return result