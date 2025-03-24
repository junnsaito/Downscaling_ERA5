#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:41:26 2025

@author: junsaito
"""

# ==========================================
# era5.py
# ==========================================
import xarray as xr
import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import reproject, Resampling

def load_era5_dataset(file_path):
    ds = xr.open_dataset(file_path)
    if "time" in ds.coords:
        ds = ds.rename({"time": "valid_time"})
    return ds

def reproject_to_rema(data, src_crs, dst_crs, src_transform, dst_transform, width, height):
    out = np.empty((height, width), dtype=np.float32)
    reproject(
        source=data.astype(np.float32),
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.cubic
    )
    return out

def get_station_temperature(date, station_data_1H):
    """
    1時間刻みの観測データ (°C) から、同じ時刻 date の気温を取得
    """
    return station_data_1H.loc[date, "AT"]  # カラム名 "AT" と想定

def get_station_dewpoint(date, station_data_1H):
    """
    同様に露点温度 (°C) を取得
    """
    return station_data_1H.loc[date, "DT"]

def adjust_temperature_with_station(
    era5_temp,       # ERA5の2m温度 (K)
    z_coarse,        # ERA5の粗い標高 (m)
    grid_elev,       # 高解像度DEM（REMA）の標高 (m)
    station_elev,    # AWS観測所の標高 (m)
    date,            # 対象日時 (pandas.Timestamp)
    station_data_1H, # (1Hリサンプリング済み観測温度 [°C])
    lapse_rate,      # ラプスレート (K/m)（負値想定）
    method="lapse_rate"
):
    """
    ERA5温度を補正する関数 (ラプスレート or 単純に観測値を使う)。
    
    method='lapse_rate': 観測所温度 (T_obs) をもとに
      1) station_elev -> z_coarse
      2) z_coarse -> grid_elev
    という形で補正
    """
    # 観測所温度(°C)->K
    T_obs = get_station_temperature(date, station_data_1H) + 273.15

    if method == "lapse_rate":
        # (1) 観測所 -> ERA5標高
        correct0 = T_obs - abs(lapse_rate)*(z_coarse - station_elev)
        # (2) ERA5標高 -> grid_elev
        correct1 = correct0 - abs(lapse_rate)*(grid_elev - z_coarse)
        return correct1
    elif method == "linear":
        # 線形補正ではなく、単に観測温度をターゲット地点とみなす
        return T_obs
    else:
        raise ValueError(f"Unknown method: {method}")

def adjust_dewpoint_with_station(
    era5_dew,
    z_coarse,
    grid_elev,
    station_elev,
    date,
    station_data_1H,
    dew_lapse_rate,
    method="lapse_rate"
):
    T_obs_dew = get_station_dewpoint(date, station_data_1H) + 273.15
    if method=="lapse_rate":
        correct0 = T_obs_dew - abs(dew_lapse_rate)*(z_coarse - station_elev)
        correct1 = correct0 - abs(dew_lapse_rate)*(grid_elev - z_coarse)
        return correct1
    elif method=="linear":
        return T_obs_dew
    else:
        raise ValueError(f"Unknown method: {method}")
