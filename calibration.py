#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:25:30 2025

@author: junsaito
"""

# calibration.py
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calc_wind_model(df_showa):
    """
    ERA5と昭和基地実測の風速を比較し、線形回帰モデル {slope, intercept} を返す。
    
    Parameters:
      df_showa: DataFrame
        "Wind_era5" 列（ERA5の10m風速）および "Wind_obs" 列（実測10m風速）を含む。
    
    Returns:
      dict: {"slope": ..., "intercept": ..., "r_value": ...}
    """
    mask = df_showa["Wind_obs"].notnull() & df_showa["Wind_era5"].notnull()
    if mask.sum() < 10:
        return {"slope": 1.0, "intercept": 0.0, "r_value": 0.0}
    slope, intercept, r_value, _, _ = linregress(df_showa.loc[mask, "Wind_era5"],
                                                  df_showa.loc[mask, "Wind_obs"])
    return {"slope": slope, "intercept": intercept, "r_value": r_value}

def calc_pressure_model(df_showa):
    """
    観測された気圧（df_showa["P"]：hPa→Pa換算済み）とERA5の気圧（df_showa["P_ERA"]）の間で
    線形回帰を行い、補正パラメータ {slope, intercept} を返す。
    
    Returns:
      dict: {"slope": ..., "intercept": ..., "r_value": ...}
    """
    mask = df_showa["P_pa"].notnull() & df_showa["P_ERA"].notnull()
    if mask.sum() < 10:
        return {"slope": 1.0, "intercept": 0.0, "r_value": 0.0}
    slope, intercept, r_value, _, _ = linregress(df_showa.loc[mask, "P_ERA"],
                                                  df_showa.loc[mask, "P_pa"])
    return {"slope": slope, "intercept": intercept, "r_value": r_value}

def calc_rh_model(df_showa):
    """
    ERA5と昭和基地実測の相対湿度（"RH_era5" と "RH_obs"）の間で
    線形回帰を行い、補正パラメータ {slope, intercept} を返す。
    
    Returns:
      dict: {"slope": ..., "intercept": ..., "r_value": ...}
    """
    mask = df_showa["RH_obs"].notnull() & df_showa["RH_era5"].notnull()
    if mask.sum() < 10:
        return {"slope": 1.0, "intercept": 0.0, "r_value": 0.0}
    slope, intercept, r_value, _, _ = linregress(df_showa.loc[mask, "RH_era5"],
                                                  df_showa.loc[mask, "RH_obs"])
    return {"slope": slope, "intercept": intercept, "r_value": r_value}

def calc_sw_bias(df_showa, resample_period="D"):
    """
    ERA5と昭和基地実測の短波放射（"SW_ERA" と "SW"）のバイアスを、指定した期間（デフォルトは1日）
    の平均値として算出する。LocalまたはERA5が0の日は除外する。
    
    Returns:
      bias (float): SWのバイアス [W/m²]
    """
    df_daily = df_showa.resample(resample_period).mean(numeric_only=True)
    mask = (df_daily["SW"].notnull() & df_daily["SW_ERA"].notnull() &
            (df_daily["SW"] != 0) & (df_daily["SW_ERA"] != 0))
    if mask.sum() < 10:
        return 0.0
    bias = df_daily.loc[mask, "SW"].mean() - df_daily.loc[mask, "SW_ERA"].mean()
    return bias

def calc_lw_bias(df_showa, resample_period="D"):
    """
    ERA5と昭和基地実測の長波放射（"LW_ERA" と "LW"）のバイアスを、指定した期間（デフォルトは1日）
    の平均値として算出する。LocalまたはERA5が0の日は除外する。
    
    Returns:
      bias (float): LWのバイアス [W/m²]
    """
    df_daily = df_showa.resample(resample_period).mean(numeric_only=True)
    mask = (df_daily["LW"].notnull() & df_daily["LW_ERA"].notnull() &
            (df_daily["LW"] != 0) & (df_daily["LW_ERA"] != 0))
    if mask.sum() < 10:
        return 0.0
    bias = df_daily.loc[mask, "LW"].mean() - df_daily.loc[mask, "LW_ERA"].mean()
    return bias
