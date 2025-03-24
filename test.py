#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:10:30 2025

@author: junsaito
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from rasterio.transform import from_bounds
from tqdm import tqdm
from scipy.stats import linregress
from scipy.ndimage import sobel, laplace, gaussian_filter, uniform_filter
import math
import pyproj
from rasterio.transform import rowcol
from scipy.signal import convolve2d

############################################################################
# 0. ユーザ設定
# ############################################################################
era5_accum_file   = "/media/junsaito/E4308E2C308E05B0/JS_model/ERA5/2015/data_stream-oper_stepType-accum.nc"
era5_instant_file = "/media/junsaito/E4308E2C308E05B0/JS_model/ERA5/2015/data_stream-oper_stepType-instant.nc"
rema_dem_file     = "/media/junsaito/E4308E2C308E05B0/JS_model/Tiff/rema_mosaic_500m_v2.0_dem_wgs84_shirase.tif"
output_dir        = "/media/junsaito/E4308E2C308E05B0/Antarctica/H128"


# era5_accum_file   = "/media/junsaito/E4308E2C308E05B0/Antarctica/H128/data_stream-oper_stepType-accum.nc"
# era5_instant_file = "/media/junsaito/E4308E2C308E05B0/Antarctica/H128/data_stream-oper_stepType-instant.nc"

# # era5_accum_file   = "/media/junsaito/E4308E2C308E05B0/JS_model/ERA5/2015/data_stream-oper_stepType-accum.nc"
# # era5_instant_file = "/media/junsaito/E4308E2C308E05B0/JS_model/ERA5/2015/data_stream-oper_stepType-instant.nc"


# # rema_dem_file     = "/media/junsaito/E4308E2C308E05B0/JS_model/Tiff/rema_mosaic_500m_v2.0_dem_wgs84_shirase.tif"
# # rema_dem_file     = "/media/junsaito/E4308E2C308E05B0/Antarctica/REMA/H128_ele_wgs84.tif"
# rema_dem_file     = "/media/junsaito/E4308E2C308E05B0/Antarctica/H128/H128_dem2.tif"
# output_dir        = "/media/junsaito/E4308E2C308E05B0/Antarctica/H128"

# 補正方式の選択：True なら風速などその他の因子も含めた補正、False なら標高のみの補正
# use_wind_adjustment = True


use_wind_adjustment = False  # Trueなら風速などその他の因子も利用

############################################################################
# 1. ERA5 データの読み込みとマージ
############################################################################
ds_accum   = xr.open_dataset(era5_accum_file)
ds_instant = xr.open_dataset(era5_instant_file)
ds_all = xr.merge([ds_accum, ds_instant], compat="override")
print("Merged ERA5 dataset:")
print(ds_all)

############################################################################
# 2. 2020年のデータ抽出（valid_time）
############################################################################
# ds_2020 = ds_all.sel(valid_time=slice("2020-01-01", "2020-12-31"))
ds_2020 = ds_all.sel(valid_time=slice("2015-01-01", "2015-12-31"))
print("2020 dataset:")
print(ds_2020)

############################################################################
# 3. REMA DEM の読み込み＆マスク
############################################################################
with rasterio.open(rema_dem_file) as dem_src:
    rema_crs = dem_src.crs
    rema_profile = dem_src.profile
    lat_min, lat_max = float(ds_2020.latitude.min()), float(ds_2020.latitude.max())
    lon_min, lon_max = float(ds_2020.longitude.min()), float(ds_2020.longitude.max())
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    geo = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326").to_crs(rema_crs)
    rema_dem_data, rema_transform = mask(dem_src, geo.geometry, crop=True)
    print("REMA Transform:", rema_transform)
    print("REMA Bounds:", rasterio.transform.array_bounds(rema_dem_data.shape[1],
                                                           rema_dem_data.shape[2],
                                                           rema_transform))
    print("REMA CRS:", rema_crs)

# マスクされたDEMを利用
DEM = rema_dem_data[0]
H_dem, W_dem = DEM.shape
print(f"REMA DEM size (cropped): {H_dem} x {W_dem}")

############################################################################
# 4. ERA5データの再投影（DEMのグリッドに合わせる）
############################################################################
src_crs = "EPSG:4326"
def transform_for_era5(ds):
    nlon = ds.longitude.size
    nlat = ds.latitude.size
    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())
    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())
    return from_bounds(lon_min, lat_min, lon_max, lat_max, nlon, nlat)
src_transform_era5 = transform_for_era5(ds_2020)

def reproject_to_rema_2d(data_2d, src_transform, src_crs, dst_transform, dst_crs, width, height):
    out = np.empty((height, width), dtype=np.float32)
    reproject(source=data_2d.astype(np.float32),
              destination=out,
              src_transform=src_transform,
              src_crs=src_crs,
              dst_transform=dst_transform,
              dst_crs=dst_crs,
              resampling=Resampling.cubic)
    return out

nT = ds_2020.valid_time.size
tp_3d   = np.zeros((nT, H_dem, W_dem), dtype=np.float32)
u10_3d  = np.zeros((nT, H_dem, W_dem), dtype=np.float32)
v10_3d  = np.zeros((nT, H_dem, W_dem), dtype=np.float32)

print("Reprojecting ERA5 variables to REMA DEM domain:")
for i in tqdm(range(nT), desc="Reprojecting ERA5"):
    ds_t = ds_2020.isel(valid_time=i)
    tp_3d[i]  = reproject_to_rema_2d(ds_t["tp"].values, src_transform_era5, src_crs,
                                      rema_transform, rema_crs, W_dem, H_dem)
    u10_3d[i] = reproject_to_rema_2d(ds_t["u10"].values, src_transform_era5, src_crs,
                                      rema_transform, rema_crs, W_dem, H_dem)
    v10_3d[i] = reproject_to_rema_2d(ds_t["v10"].values, src_transform_era5, src_crs,
                                      rema_transform, rema_crs, W_dem, H_dem)

############################################################################
# 5. 降水‐標高回帰から κ の推定（簡易版）
############################################################################
def calc_precip_kappa_block(block_indices, precip2d, elev2d):
    results = []
    for (i, j) in block_indices:
        if i < 2 or j < 2 or i >= precip2d.shape[0]-2 or j >= precip2d.shape[1]-2:
            results.append((i, j, 0.20))
            continue
        p_win = precip2d[i-2:i+3, j-2:j+3].flatten()
        e_win = elev2d[i-2:i+3, j-2:j+3].flatten()
        if np.std(e_win) == 0:
            kappa = 0.20
        else:
            slope, _, _, _, _ = linregress(e_win, p_win)
            kappa = slope * 1000
            kappa = np.clip(kappa, 0.1, 1.0)
        results.append((i, j, kappa))
    return results

indices = [(i, j) for i in range(tp_3d.sum(axis=0).shape[0]) for j in range(tp_3d.sum(axis=0).shape[1])]
kappa_results = calc_precip_kappa_block(indices, tp_3d.sum(axis=0)*1000, DEM)
kappa_field = np.full((H_dem, W_dem), 0.35, dtype=np.float32)
for i, j, kappa_val in kappa_results:
    kappa_field[i, j] = kappa_val

############################################################################
# 6. 各種補正関数
############################################################################
# def adjust_precip_topo(P0, z, z0, kappa):
#     return P0

def adjust_precip_topo(P0, z, z0, kappa):
    # z: 各グリッドセルの真の標高 (m)
    # z0: 基準となる補間された標高 (m)（例：DEM全体の中央値）
    # kappa: 降水ラプス率 (km⁻¹)
    dz_km = (z - z0) / 1000.0  # m -> km
    correction = 1 + kappa * dz_km / (1 + kappa * dz_km)
    return P0 * correction
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

def downscale_precip(P0, z, z0, kappa, wind_speed, slope, curvature, sublimation_mm, wind_direction, aspect):
    P_topo = adjust_precip_topo(P0, z, z0, kappa)
    drift_factor = drift_accumulation_factor(wind_speed, slope, curvature, wind_direction, aspect)
    P_drift = P_topo * drift_factor
    P_final = np.maximum(P_drift - sublimation_mm, 0)
    return P_final

def adjust_wind_speed(W_coarse, u10, v10, slope, curvature, delta_z, roughness, d=0.1, measurement_height=10.0):
    kappa_val = 0.41
    z0_local = np.maximum(roughness, 1e-3)
    u_star = kappa_val * W_coarse / np.log((measurement_height - d) / z0_local)
    U_log = u_star / kappa_val * np.log((measurement_height - d) / z0_local)
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

def smooth_and_replace_outliers(data, high_threshold, window_size=5, min_value=1.0):
    local_mean = uniform_filter(data, size=window_size)
    data_smoothed = np.where(data > high_threshold, local_mean, data)
    data_smoothed = np.where(data_smoothed < min_value, local_mean, data_smoothed)
    return np.maximum(data_smoothed, min_value)

def reproject_to_rema(data, src_crs, dst_crs, src_transform, dst_transform, width, height):
    out = np.empty((height, width), dtype=np.float32)
    reproject(source=data.astype(np.float32),
              destination=out,
              src_transform=src_transform,
              src_crs=src_crs,
              dst_transform=dst_transform,
              dst_crs=dst_crs,
              resampling=Resampling.cubic)
    return out



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
############################################################################
# 7. 補正方式の切替えと降水量計算、出力
############################################################################
TP_target = tp_3d.sum(axis=0) * 1000  # ERA5の積算降水量 (mm)

g = 9.80665
z_era5 = ds_2020["z"].mean(dim="valid_time") / g  # ERA5ジオポテンシャル標高 (m)
z_era5_reproj = reproject_to_rema(z_era5.values, "EPSG:4326", rema_crs,
                                  src_transform_era5, rema_transform, W_dem, H_dem)
delta_z = DEM - z_era5_reproj

indices = [(i, j) for i in range(TP_target.shape[0]) for j in range(TP_target.shape[1])]
kappa_results = calc_precip_kappa_block(indices, TP_target, DEM)
kappa_field = np.full((H_dem, W_dem), 0.35, dtype=np.float32)
for i, j, kappa_val in kappa_results:
    kappa_field[i, j] = kappa_val

u10_mean = u10_3d.mean(axis=0)
v10_mean = v10_3d.mean(axis=0)
W_coarse = np.sqrt(u10_mean**2 + v10_mean**2)

dx = 500
slope_field, aspect_field = compute_slope_and_aspect(DEM, dx)
curvature_field = laplace(DEM) / (500**2)
roughness = 1e-3

adjusted_wind = adjust_wind_speed(W_coarse, u10_mean, v10_mean, slope_field, curvature_field, delta_z, roughness)
sublimation_mm = 0.0

# ERA5の平均風向をスカラー値として利用
# wind_direction = float((np.degrees(np.arctan2(v10_mean, u10_mean)) + 360) % 360)
wind_direction = float((np.degrees(np.arctan2(v10_mean.mean(), u10_mean.mean())) + 360) % 360)


if use_wind_adjustment:
    TP_target_final = downscale_precip(TP_target, DEM, np.median(DEM), kappa_field, adjusted_wind,
                                       slope_field, curvature_field, sublimation_mm,
                                       wind_direction, aspect_field)
else:
    TP_target_final = adjust_precip_topo(TP_target, DEM, np.median(DEM), kappa_field)

TP_target_final = np.maximum(TP_target, 0)
TP_target_final_smoothed = smooth_and_replace_outliers(TP_target, high_threshold=1000, window_size=5, min_value=1.0)

# GeoTIFF出力
output_tif = f"{output_dir}/"
with rasterio.open(output_tif, 'w', driver='GTiff', height=H_dem, width=W_dem, count=1,
                   dtype=TP_target_final_smoothed.dtype, crs=rema_crs, transform=rema_transform) as dst:
    dst.write(TP_target_final_smoothed, 1)
print("Downscaled precipitation TIF has been written to:", output_tif)

############################################################################
# 各時刻ごとに downscale_precip を適用して時系列降水量フィールドを作成
############################################################################
downscaled_tp_ts = np.zeros_like(tp_3d)
for i in range(nT):
    P0_ts = tp_3d[i] * 1000
    downscaled_tp_ts[i] = downscale_precip(P0_ts, DEM, np.median(DEM), kappa_field, adjusted_wind,
                                           slope_field, curvature_field, sublimation_mm,
                                           wind_direction, aspect_field)

############################################################################
# 特定地点の時系列データ（降水量と風速）をCSVで保存
############################################################################
target_lon, target_lat = 38.78, -70.08
# target_lon, target_lat = 41.34, -69.24
transformer = pyproj.Transformer.from_crs("EPSG:4326", rema_crs, always_xy=True)
target_x, target_y = transformer.transform(target_lon, target_lat)
target_row, target_col = rowcol(rema_transform, target_x, target_y)

bdot_timeseries = downscaled_tp_ts[:, target_row, target_col]
wind_speed_timeseries = np.sqrt(u10_3d[:, target_row, target_col]**2 + v10_3d[:, target_row, target_col]**2)
time_series = ds_2020.valid_time.values

df_bdot = pd.DataFrame({
    'Date': pd.to_datetime(time_series),
    'BDOT_mm': bdot_timeseries,
    'Wind_Speed_m_s': wind_speed_timeseries
})
# output_csv_path = f"{output_dir}/BDOT_WindSpeed_timeseries_38.78_-70.08_new2.csv"
output_csv_path = f"{output_dir}/BDOT_WindSpeed_timeseries_2015.csv"
df_bdot.to_csv(output_csv_path, index=False)
print(f"BDOT and Wind Speed timeseries have been written to: {output_csv_path}")

