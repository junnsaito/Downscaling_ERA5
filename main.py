#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:42:59 2025

@author: junsaito
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import rowcol
from tqdm import tqdm
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import pvlib
from pyproj import Transformer
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from rasterio.transform import array_bounds, from_bounds
import json


# --- JSON設定ファイルの読み込み ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

def get_path(key):
    """設定ファイルのキーが空の場合は None を返す"""
    val = config.get(key, "").strip()
    return os.path.join(BASE_DIR, val) if val != "" else None

# --- 各パス・パラメータの取得 ---
target_coords_csv = get_path("target_coords_csv")
rema_dem_file = get_path("rema_dem_file")
era5_base_dir = get_path("era5_base_dir")
modis_wgs84_file = get_path("modis_wgs84_file")
output_dir = get_path("output_dir")
time_interval_s = config.get("time_interval_s", 3600)
# model_parameters セクションの値で上書き（存在する場合）
model_params = config.get("model_parameters", {})
z0_station = model_params.get("z0_station", config.get("z0_station", 0))
kappa_value = model_params.get("kappa_value", config.get("kappa_value", 0.35))
station_elev = model_params.get("station_elev", config.get("station_elev", 29.0))

station_data_file = get_path("station_data_file")
station_data2_file = get_path("station_data2_file")
vx_tif_path = get_path("vx_tif_path")
vy_tif_path = get_path("vy_tif_path")
vx_error_tif_path = get_path("vx_error_tif_path")
vy_error_tif_path = get_path("vy_error_tif_path")
showa_csv = get_path("showa_csv")

# キャリブレーション（calibration セクション）
calib = config.get("calibration", {})
wind_calibration_method = calib.get("wind_calibration_method", "linear")
pressure_calibration_method = calib.get("pressure_calibration_method", "linear")
rh_calibration_method = calib.get("rh_calibration_method", "linear")

# --- 存在チェック（ファイルが見つからなければ None を返して後続処理でデフォルト値を使用） ---
if not (vx_tif_path and os.path.exists(vx_tif_path)):
    print("[WARN] vx_tif_path not specified or file does not exist. Using default zero array.")
    vx_tif_path = None
if not (vy_tif_path and os.path.exists(vy_tif_path)):
    print("[WARN] vy_tif_path not specified or file does not exist. Using default zero array.")
    vy_tif_path = None
if not (vx_error_tif_path and os.path.exists(vx_error_tif_path)):
    print("[WARN] vx_error_tif_path not specified or file does not exist. Using default zeros.")
    vx_error_tif_path = None
if not (vy_error_tif_path and os.path.exists(vy_error_tif_path)):
    print("[WARN] vy_error_tif_path not specified or file does not exist. Using default zeros.")
    vy_error_tif_path = None
if not (modis_wgs84_file and os.path.exists(modis_wgs84_file)):
    print("[WARN] MODIS file not found or not specified. Using default albedo = 0.8")
    modis_wgs84_file = None

###########################
# 各種関数のインポート
###########################

from dem import load_dem
from era5 import (
    reproject_to_rema, adjust_temperature_with_station,
    adjust_dewpoint_with_station
)
from lapse_rate import (calc_lapse_rate_parallel,calc_precip_kappa_block)
from fluxes import (
    downscale_pressure, compute_specific_humidity, compute_emissivity_iziomon,
    downscale_longwave, calc_snow_ratio, calc_esat, compute_rel_humidity
)
from velocity import (
    load_velocity_components, load_uncertainty_maps, apply_variable_gaussian_filter,
    compute_strain_rate, compute_effective_horizontal_strain_rate
)
from parallel_processing import process_time_steps_chunk

from calibration import calc_wind_model, calc_pressure_model, calc_rh_model, calc_sw_bias, calc_lw_bias



###############################################
# ② メイン処理（並列処理等）
###############################################
def main():
    # --- サブフォルダのパスをここで定義する ---
    force_folder = os.path.join(output_dir, "Force")
    strain_folder = os.path.join(output_dir, "Strain")
    rho_folder = os.path.join(output_dir, "Rho")
    os.makedirs(force_folder, exist_ok=True)
    os.makedirs(strain_folder, exist_ok=True)
    os.makedirs(rho_folder, exist_ok=True)
    
    # -------------------------------------------------
    # 0) 昭和基地実測データとERA5を比較して Wind, Pressure の補正値を計算
    # -------------------------------------------------
    df_showa = pd.read_csv(showa_csv, parse_dates=["Date"], index_col="Date")
    # 風速補正モデル
    mask_w = df_showa["WS"].notnull() & df_showa["WS_ERA"].notnull()
    if mask_w.sum() > 10:
        slope_, intercept_, r_value, _, _ = \
            linregress(df_showa.loc[mask_w, "WS_ERA"],
                       df_showa.loc[mask_w, "WS"])
        wind_slope = slope_
        wind_intercept = intercept_
        print(f"[INFO] Wind slope={wind_slope:.2f}, intercept={wind_intercept:.2f}, r={r_value:.2f}")
    else:
        wind_slope = 1.0
        wind_intercept = 0.0
        print("[WARN] Not enough Wind data. No correction.")
    # # 気圧バイアス
    df_showa["P_pa"] = df_showa["P"] * 100.0  # 1 hPa = 100 Pa
    ...
    # ERA5 が Pa なら df_showa["P_ERA"] も Pa のはず
    mask_p = df_showa["P_pa"].notnull() & df_showa["P_ERA"].notnull()
    if mask_w.sum() > 10:
        slope_, intercept_, r_value, _, _ = \
            linregress(df_showa.loc[mask_w, "P_ERA"],
                       df_showa.loc[mask_w, "P_pa"])
        P_slope = slope_
        P_intercept = intercept_
        print(f"[INFO] P slope={P_slope:.2f}, intercept={P_intercept:.2f}, r={r_value:.2f}")
    else:
        P_slope = 1.0
        P_intercept = 0.0
        print("[WARN] Not enough Wind data. No correction.")
    
    
    mask_r = df_showa["RH"].notnull() & df_showa["RH_ERA"].notnull()
    if mask_w.sum() > 10:
        slope_, intercept_, r_value, _, _ = \
            linregress(df_showa.loc[mask_w, "RH_ERA"],
                       df_showa.loc[mask_w, "RH"])
        RH_slope = slope_
        RH_intercept = intercept_
        print(f"[INFO] RH slope={RH_slope:.2f}, intercept={RH_intercept:.2f}, r={r_value:.2f}")
    else:
        RH_slope = 1.0
        RH_intercept = 0.0
        print("[WARN] Not enough Wind data. No correction.")
        
 # --- SW補正 (Daily Averageによる平均バイアス) ---
    df_showa_daily = df_showa.resample("D").mean(numeric_only=True)
    mask_sw = (df_showa_daily["SW"].notnull() &
               df_showa_daily["SW_ERA"].notnull() &
               (df_showa_daily["SW"] != 0) &
               (df_showa_daily["SW_ERA"] != 0))
    if mask_sw.sum() > 10:
        bias_sw = df_showa_daily.loc[mask_sw, "SW"].mean() - df_showa_daily.loc[mask_sw, "SW_ERA"].mean()
        print(f"[INFO] SW bias (Daily Average): {bias_sw:.2f}")
    else:
        bias_sw = 0.0
        print("[WARN] Not enough SW daily data. No SW correction.")
    
    # --- LW補正 (Daily Averageによる平均バイアス) ---
    df_showa_daily = df_showa.resample("D").mean(numeric_only=True)
    mask_lw = (df_showa_daily["LW"].notnull() &
               df_showa_daily["LW_ERA"].notnull() &
               (df_showa_daily["LW"] != 0) &
               (df_showa_daily["LW_ERA"] != 0))
    if mask_lw.sum() > 10:
        bias_lw = df_showa_daily.loc[mask_lw, "LW"].mean() - df_showa_daily.loc[mask_lw, "LW_ERA"].mean()
        print(f"[INFO] LW bias (Daily Average): {bias_lw:.2f}")
    else:
        bias_lw = 0.0
        print("[WARN] Not enough LW daily data. No LW correction.")

    # DEM読み込み
    rema_dem_data, rema_transform, rema_crs = load_dem(rema_dem_file)
    H_dem, W_dem = rema_dem_data.shape
    print("[INFO] DEM loaded:", rema_dem_file, (H_dem, W_dem))

    # --- MODIS Albedo ---
    if modis_wgs84_file and os.path.exists(modis_wgs84_file):
        with rasterio.open(modis_wgs84_file) as modis_src:
            modis_albedo_data = modis_src.read(1)
            modis_transform = modis_src.transform
        print("[INFO] MODIS loaded:", modis_wgs84_file, modis_albedo_data.shape)
        # MODISから抽出する際に変数modis_albedo_valueを初期化
        # ※後の処理で各座標ごとに抽出するのでここでは特に値を入れなくてもOK
        modis_albedo_value = None
    else:
        print("[WARN] MODIS file not found or not specified. Using default albedo = 0.8")
        # DEMサイズに合わせたデフォルトアルベド配列を作成
        modis_albedo_data = np.full((H_dem, W_dem), 0.8, dtype=np.float32)
        modis_transform = rema_transform
        # ここでもデフォルト値をセット
        modis_albedo_value = 0.8


    # 観測所データ読み込み
    station_data = pd.read_csv(station_data_file, parse_dates=["Date"], index_col="Date")
    station_data2 = pd.read_csv(station_data2_file, parse_dates=["Date"], index_col="Date")
    station_data_1H = station_data.resample("h").mean(numeric_only=True).interpolate(method="linear")
    station_data2_1H = station_data2.resample("h").mean(numeric_only=True).interpolate(method="linear")

    # Velocityと不確実性マップ読み込み
    Vx, Vy, dx, dy, vel_transform, vx_crs = load_velocity_components(vx_tif_path, vy_tif_path)
    uncertainty_map = load_uncertainty_maps(vx_error_tif_path, vy_error_tif_path)
    Vx_smooth, Vy_smooth = apply_variable_gaussian_filter(Vx, Vy, uncertainty_map)

    # 複数座標読み込み
    coords_df = pd.read_csv(target_coords_csv)
    print("[INFO] target coords loaded:", len(coords_df))

    all_results = []
    transformer_vel = Transformer.from_crs("EPSG:4326", vx_crs, always_xy=True)

    # 各座標ループ
    for idx, row in coords_df.iterrows():
        lon_target = row["lon"]
        lat_target = row["lat"]
        # print("\n==============================")
        # print(f"[INFO] Now processing lon={lon_target}, lat={lat_target}")
        
        # すでに処理済みならスキップ（例：Force フォルダ内の出力ファイルが存在するかチェック）
        forcing_filename = f"cfm_forcing_{lon_target:.3f}_{lat_target:.3f}.csv"
        forcing_filepath = os.path.join(force_folder, forcing_filename)
        if os.path.exists(forcing_filepath):
            print(f"[INFO] {forcing_filename} already exists. Skipping this coordinate.")
            continue
        
        print(f"[INFO] Processing coordinate: lon={lon_target}, lat={lat_target}")

        # DEM上のインデックス
        i0_target, j0_target = rowcol(rema_transform, lon_target, lat_target, op=round)
        i0_target = np.clip(i0_target, 0, H_dem - 1)
        j0_target = np.clip(j0_target, 0, W_dem - 1)
        print(f"  DEM index=({i0_target}, {j0_target})")

        i_mod, j_mod = rowcol(modis_transform, lon_target, lat_target, op=round)
        if 0 <= i_mod < modis_albedo_data.shape[0] and 0 <= j_mod < modis_albedo_data.shape[1]:
            modis_albedo_value = float(modis_albedo_data[i_mod, j_mod])
        else:
            modis_albedo_value = 0.8
            print("[INFO] MODIS albedo extraction failed (location out of bounds). Using default albedo = 0.8")
        print(f"  MODIS Albedo= {modis_albedo_value:.3f}")

        
        # Velocity抽出（氷流速）
        tx_vel, ty_vel = transformer_vel.transform(lon_target, lat_target)
        i_vel, j_vel = rowcol(vel_transform, tx_vel, ty_vel, op=round)
        i_vel = np.clip(i_vel, 0, Vx_smooth.shape[0] - 1)
        j_vel = np.clip(j_vel, 0, Vx_smooth.shape[1] - 1)
        vx_val = Vx_smooth[i_vel, j_vel]
        vy_val = Vy_smooth[i_vel, j_vel]
        flow_speed = float(np.sqrt(vx_val**2 + vy_val**2))
        if not flow_speed or np.isnan(flow_speed):
            print("[INFO] Computed flow speed is 0 or NaN. Using default value of 0.1 m/yr.")
            flow_speed = 0.1
        flow_dir = float(np.degrees(np.arctan2(vy_val, vx_val)))
        print(f"  Ice Flow speed= {flow_speed:.2f} m/yr, dir= {flow_dir:.1f} deg")

        # 各年ごとのERA5データを並列実行（例：2021～2023年）
        all_year_forcing = []
        all_year_rho = []
        all_year_time = []
        years = list(range(2023, 2024))
        for year in years:
            print(f"  >> Year={year}")
            accum_file = os.path.join(era5_base_dir, str(year), "data_stream-oper_stepType-accum.nc")
            instant_file = os.path.join(era5_base_dir, str(year), "data_stream-oper_stepType-instant.nc")
            ds_accum = xr.open_dataset(accum_file)
            ds_instant = xr.open_dataset(instant_file)
            ds_merged = xr.merge([ds_accum, ds_instant], compat="override")
            if "time" in ds_merged.coords:
                ds_merged = ds_merged.rename({"time": "valid_time"})
            ds_year = ds_merged.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31")).load()
            nT_year = ds_year.valid_time.size
            print(f"     nT_year= {nT_year}")
            
            
           # ERA5データセットから緯度・経度の最小・最大を取得する例
            lon_min = float(ds_year.longitude.min())
            lon_max = float(ds_year.longitude.max())
            lat_min = float(ds_year.latitude.min())
            lat_max = float(ds_year.latitude.max())
            
            # 経度・緯度のサイズ（ピクセル数）を取得
            nlon = ds_year.longitude.size
            nlat = ds_year.latitude.size
            
            # 自動的に変換行列を作成
            src_transform_era5 = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, nlon, nlat)

            
            # ここで ERA5 の累積降水量（mm）を用いて κ フィールドを計算
            tp_cum = ds_year["tp"].sum(dim="valid_time").values * 1000.0
            indices = [(i, j) for i in range(H_dem) for j in range(W_dem)]
            kappa_results = calc_precip_kappa_block(indices, tp_cum, rema_dem_data)
            kappa_field = np.full((H_dem, W_dem), 0.35, dtype=np.float32)
            for ii, jj, kapp in kappa_results:
                kappa_field[ii, jj] = kapp
                
            t2m_aver = ds_year["t2m"].mean(dim="valid_time")
            t2m_first = reproject_to_rema(t2m_aver, "EPSG:4326", "EPSG:4326",
                                          src_transform_era5, rema_transform, W_dem, H_dem)
            if t2m_first.shape != rema_dem_data.shape:
                t2m_first = t2m_first.T
            d2m_aver = ds_year["d2m"].mean(dim="valid_time")
            d2m_first = reproject_to_rema(d2m_aver, "EPSG:4326", "EPSG:4326",
                                          src_transform_era5, rema_transform, W_dem, H_dem)
            if d2m_first.shape != rema_dem_data.shape:
                d2m_first = d2m_first.T
            skt_aver = ds_year["skt"].mean(dim="valid_time")
            skt_first = reproject_to_rema(skt_aver, "EPSG:4326", "EPSG:4326",
                                          src_transform_era5, rema_transform, W_dem, H_dem)
            if skt_first.shape != rema_dem_data.shape:
                skt_first = skt_first.T

            from lapse_rate import calc_lapse_rate_parallel
            lapse_1 = calc_lapse_rate_parallel(t2m_first, rema_dem_data)
            lapse_2 = calc_lapse_rate_parallel(d2m_first, rema_dem_data)
            lapse_3 = calc_lapse_rate_parallel(skt_first, rema_dem_data)
            lapse_value  = lapse_1[i0_target, j0_target]
            lapse_value2 = lapse_2[i0_target, j0_target]
            lapse_value3 = lapse_3[i0_target, j0_target]
            delta_z_point = rema_dem_data[i0_target, j0_target]

            from scipy.ndimage import laplace
            curvature2d = laplace(rema_dem_data) / (500**2)
            from dem import compute_slope_and_aspect
            slope2d, aspect2d = compute_slope_and_aspect(rema_dem_data, dx=500)
            g = 9.81
            z_era5 = ds_year["z"].mean(dim="valid_time") / g
            z_orig = reproject_to_rema(z_era5, "EPSG:4326", "EPSG:4326",
                                        src_transform_era5, rema_transform, W_dem, H_dem)

            print("  Calculating solar position for all timestamps...")
            times = pd.to_datetime(ds_year.valid_time.values)
            solpos = pvlib.solarposition.get_solarposition(times, lat_target, lon_target)
            zenith_array = solpos["zenith"].values
            azimuth_array = solpos["azimuth"].values

            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(create=True, size=rema_dem_data.nbytes)
            shared_array = np.ndarray(rema_dem_data.shape, dtype=rema_dem_data.dtype, buffer=shm.buf)
            np.copyto(shared_array, rema_dem_data)

            chunk_size = 50
            time_indices = list(range(nT_year))
            year_results = []
            flux_results = []  # 並列処理の結果を格納するリスト
            with ProcessPoolExecutor(max_workers=15) as exe:
                futures = []
                for start_idx in range(0, nT_year, chunk_size):
                    chunk = time_indices[start_idx : start_idx+chunk_size]
                    fut = exe.submit(
                        process_time_steps_chunk,
                        chunk,
                        ds_year,
                        src_transform_era5,
                        rema_transform,
                        W_dem, H_dem,
                        i0_target, j0_target,
                        lapse_value, lapse_value2,
                        lapse_value3, delta_z_point,
                        slope2d, aspect2d, curvature2d,
                        zenith_array, azimuth_array, modis_albedo_value,
                        rema_dem_data,
                        time_interval_s,
                        station_data_1H,
                        station_data2_1H,
                        station_elev, z_orig,
                        wind_slope,  # 追加: 風速スロープ
                        wind_intercept,  # 追加: 風速intercept
                        P_slope,  # 追加: 風速スロープ
                        P_intercept,  # 追加: 風速intercept
                        RH_slope,  # 追加: 風速スロープ
                        RH_intercept,  # 追加: 風速intercept
                        bias_sw,   # 追加: SWバイアス
                        bias_lw,    # 追加: LWバイアス
                        kappa_field
                    )
                    futures.append(fut)
                for ft in tqdm(futures, desc=f"Processing {year} chunks"):
                    year_results.extend(ft.result())
                    # flux_results.extend(ft.result())
            shm.close()
            shm.unlink()
            # flux_results.extend(fut.result())
            flux_results.extend(ft.result())
            year_forcing = year_results

            # year_forcing = [r[:-2] for r in year_results]
            year_rho = [r[10] + r[11] for r in year_results]
            year_time = [r[8] for r in year_results]

            all_year_forcing.extend(year_forcing)
            all_year_rho.extend(year_rho)
            all_year_time.extend(year_time)

        combined = list(zip(all_year_time, all_year_forcing, all_year_rho))
        combined.sort(key=lambda x: x[0])
        time_sorted, forcing_sorted, rho_sorted = zip(*combined)
        
        forcing_csv = os.path.join(force_folder, f"cfm_forcing_{lon_target:.3f}_{lat_target:.3f}.csv")
        df_forcing = pd.DataFrame(list(forcing_sorted),
                                  columns=["SW_d", "LW_d", "T2m", "TSKIN", "QH", "QL",
                "RAIN", "BDOT", "time", "ALBEDO",
                "rho_new", "rho_redeposit",
                "WindSpeed", "Pressure", "RH","Ts_surface", "Melt_rate","G","estimated_albedo","Tsub_est", "Lnet", "Snet" ,"QM","Sublimation"
                ,"U10","V10","BDOT_new"])
        df_forcing["time"] = pd.to_datetime(time_sorted)
        df_forcing.set_index("time", inplace=True)
        df_forcing.to_csv(forcing_csv, float_format="%.3f")
        print("[DONE] Forcing CSV:", forcing_csv)

        strain_csv = os.path.join(output_dir, f"cfm_strain_inputs_{lon_target:.3f}_{lat_target:.3f}.csv")
        t_start = pd.Timestamp("2023-01-01")
        t_end = pd.Timestamp("2023-12-31 23:00:00")
        time_strain = pd.date_range(t_start, t_end, freq="h")
        N_strain = len(time_strain)
        from velocity import compute_strain_rate
        eps_xx, eps_yy, eps_xy, divergence, eps_zz = compute_strain_rate(Vx_smooth, Vy_smooth, dx, dy)
        target_eps_xx = eps_xx[i_vel, j_vel]
        target_eps_yy = eps_yy[i_vel, j_vel]
        target_eps_xy = eps_xy[i_vel, j_vel]
        eps_xx_arr = np.full(N_strain, target_eps_xx)
        eps_yy_arr = np.full(N_strain, target_eps_yy)
        eps_xy_arr = np.full(N_strain, target_eps_xy)
        def to_year_frac(dt):
            y = dt.year
            start = pd.Timestamp(y, 1, 1)
            end = pd.Timestamp(y + 1, 1, 1)
            return y + (dt - start).total_seconds() / ((end - start).total_seconds())
        dec_dates = [to_year_frac(t) for t in time_strain]
        data_strain = np.column_stack([dec_dates, eps_xx_arr, eps_yy_arr, eps_xy_arr])
        df_strain = pd.DataFrame(data_strain, columns=["DecimalDate", "eps_xx", "eps_yy", "eps_xy"])
        df_strain_T = df_strain.T
        strain_csv = os.path.join(strain_folder, f"cfm_strain_inputs_{lon_target:.3f}_{lat_target:.3f}.csv")
        df_strain_T.to_csv(strain_csv, header=False, index=False, float_format="%.8f")

        print("[DONE] Strain CSV:", strain_csv)

        rho_csv = os.path.join(output_dir, f"cfm_rho_inputs_{lon_target:.3f}_{lat_target:.3f}.csv")
        dec_dates_rho = []
        def to_year_frac_rho(dt):
            y = dt.year
            st = pd.Timestamp(y, 1, 1)
            ed = pd.Timestamp(y + 1, 1, 1)
            return y + (dt - st).total_seconds() / ((ed - st).total_seconds())
        for t in time_sorted:
            dec_dates_rho.append(to_year_frac_rho(t))
        df_rho = pd.DataFrame(list(zip(dec_dates_rho, rho_sorted)))
        df_rho_T = df_rho.T
        # df_rho_T.to_csv(rho_folder, header=False, index=False, float_format="%.8f")
        rho_csv = os.path.join(rho_folder, f"cfm_rho_inputs_{lon_target:.3f}_{lat_target:.3f}.csv")
        df_rho_T.to_csv(rho_csv, header=False, index=False, float_format="%.8f")

        print("[DONE] Rho CSV:", rho_csv)

        all_results.append({
            "lon": lon_target,
            "lat": lat_target,
            "forcing_csv": forcing_csv,
            "strain_csv": strain_csv,
            "rho_csv": rho_csv,
            "flow_speed": flow_speed,
            "flow_dir": flow_dir
        })

    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(output_dir, "summary_target_outputs.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[DONE] Summary CSV:", summary_csv)
    
 
    
def summarize_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    sum_series = df.sum(numeric_only=True)

    if "WindSpeed" in df.columns:
        ws_mean = df["WindSpeed"].mean()
        print(f"Wind Speed Mean: {ws_mean:.3f}")
        
    if "estimated_albedo" in df.columns:
        al_mean = df["estimated_albedo"].mean()
        print(f"Albedo Mean: {al_mean:.3f}")

    if "Ts_surface" in df.columns:
        ts_surface_mean = df["Ts_surface"].mean() - 273.15
        print(f"Ts_surface Mean (C): {ts_surface_mean:.3f}")
            
    if "T2m" in df.columns:
        ts_surface_mean = df["T2m"].mean() - 273.15
        print(f"Air Temp Mean (C): {ts_surface_mean:.3f}")

    if "Melt_rate" in df.columns:
        melt_rate_sum = df["Melt_rate"].sum() / 1000
        print(f"Melt_rate SUM (m w.e.): {melt_rate_sum:.3f}")

    if "Sublimation" in df.columns:
        sublimation_sum = df["Sublimation"].sum() / 1000
        print(f"Sublimation SUM (m w.e.): {sublimation_sum:.3f}")
        
    if "BDOT_new" in df.columns:
        Total_precipitation = df["BDOT_new"].sum() / 1000
        print(f"Total Precipitation SUN (m w.e.): {Total_precipitation:.3f}")
        
    if "RAIN" in df.columns:
        Total_Rain = df["RAIN"].sum() / 1000
        print(f"Total Rain SUM (m w.e.): {Total_Rain:.3f}")
        
    if "BDOT_new" in df.columns:
        SMB = df["BDOT_new"].sum() / 1000 - df["Melt_rate"].sum() / 1000 - df["Sublimation"].sum() / 1000
        print(f"SMB Mean (m w.e.): {SMB:.3f}")

    print("\n[Done] CSV completed!")


if __name__ == "__main__":
    main()
    
    # Force サブフォルダ内の CSV ファイル一覧を取得
    force_folder = os.path.join(output_dir, "Force")
    forcing_csv_files = [os.path.join(force_folder, f) for f in os.listdir(force_folder)
                         if f.startswith("cfm_forcing_") and f.endswith(".csv")]

    if forcing_csv_files:
        for csv_file in forcing_csv_files:
            print(f"\n[INFO] Summarizing CSV: {csv_file}")
            summarize_csv(csv_file)
    else:
        print(f"[WARN] forcing CSV ファイルが見つかりません: {force_folder}")