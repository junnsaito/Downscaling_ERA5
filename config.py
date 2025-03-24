#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:31:58 2025

@author: junsaito
"""

import os

# `config.py` があるディレクトリ（JS_modelの絶対パスを取得）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
CSV file for the location you are interesred in
"""
# target_coords_csv = os.path.join(BASE_DIR,"csv", "target_coords_showa.csv")
target_coords_csv = os.path.join(BASE_DIR,"csv", "target_coords.csv")
# target_coords_csv = os.path.join(BASE_DIR,"csv", "H128.csv")
# target_coords_csv = os.path.join(BASE_DIR,"csv", "shirase_glacier_total.csv")
# target_coords_csv = os.path.join(BASE_DIR,"csv", "target_coords_Lang.csv")
# rema_dem_file = os.path.join(BASE_DIR, "Tiff", "rema_mosaic_500m_v2.0_dem_wgs84_shirase.tif")

"""
DEM file for the location you are interesred in
"""
# rema_dem_file = os.path.join(BASE_DIR, "Tiff", "H128_dem2.tif")
# rema_dem_file = os.path.join(BASE_DIR, "Tiff", "rema_mosaic_500m_v2.0_dem_station_wgs84.tif")
# rema_dem_file = os.path.join(BASE_DIR, "Tiff", "rema_mosaic_500m_v2.0_dem_showa_wgs84_latest.tif")

rema_dem_file = os.path.join(BASE_DIR, "Tiff", "rema_mosaic_500m_v2.0_dem_wgs84_shirase.tif")
"""
Albedo & Showa station AWS data for calibration
"""
era5_base_dir = os.path.join(BASE_DIR, "ERA5")
modis_wgs84_file = os.path.join(BASE_DIR, "Tiff", "MODIS_Albedo_2000-2024_avg_wgs84.tif")
showa_csv = "/media/junsaito/E4308E2C308E05B0/JS_model/csv/Showa_P_WS_LW_SW_hourly_1980_2024.csv"
output_dir = os.path.join(BASE_DIR, "Output_shirase")

# 必要なら `Output` フォルダを作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

z0_station = 0
kappa_value = 0.35
time_interval_s = 3600  # 1時間 = 3600秒

# # 月別ラプスレート（単位: °C/km）
# monthly_beta = {
#     1: 7.44, 2: 8.70, 3: 9.59, 4: 9.68, 5: 10.00, 6: 9.12,
#     7: 8.23, 8: 8.83, 9: 8.06, 10: 9.83, 11: 9.69, 12: 9.04
# }

# 観測所データ
station_data_file = os.path.join(BASE_DIR, "csv", "Showa_AT_hourly_1980_2024.csv")
station_data2_file = os.path.join(BASE_DIR, "csv", "Showa_DT_hourly_1980_2024.csv")
station_elev = 29.0

# 氷流速度（ベクトル）データ
vx_tif_path = os.path.join(BASE_DIR, "Tiff", "measure_velocity_450m_vx_shirase_lang_new.tif")
vy_tif_path = os.path.join(BASE_DIR, "Tiff", "measure_velocity_450m_vy_shirase_lang_new.tif")
vx_error_tif_path = os.path.join(BASE_DIR, "Tiff", "measure_velocity_450m_errvx_shirase_lang_new.tif")
vy_error_tif_path = os.path.join(BASE_DIR, "Tiff", "measure_velocity_450m_errvy_shirase_lang_new.tif")
