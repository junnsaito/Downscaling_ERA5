#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:02:15 2025

@author: junsaito
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
import rasterio
from pyproj import Transformer
import geopandas as gpd
from shapely.vectorized import contains

# --- 1. CSVファイルから各地点のデータを抽出 ---
csv_dir  = "/media/junsaito/E4308E2C308E05B0/JS_model/Output_final_shirase_2024/Force"
csv_dir1 = "/media/junsaito/E4308E2C308E05B0/JS_model/Output_final_shirase_high_2024/Force"

file_pattern1 = os.path.join(csv_dir, "cfm_forcing_*.csv")
file_pattern2 = os.path.join(csv_dir1, "cfm_forcing_*.csv")
csv_files = glob.glob(file_pattern1) + glob.glob(file_pattern2)
print(f"Found {len(csv_files)} CSV files.")

data_list = []
for file in csv_files:
    basename = os.path.basename(file)
    parts = basename.replace("cfm_forcing_", "").replace(".csv", "").split("_")
    lon = float(parts[0])
    lat = float(parts[1])
    df = pd.read_csv(file)
    total_melt = df["Melt_rate"].sum()         # 融解量合計（m water equivalent）
    total_BDOT = df["BDOT"].sum()               # 降雪量合計
    total_Sub = df["Sublimation"].sum()          # 昇華量合計
    ave_albedo = df["estimated_albedo"][0:1441].mean()  # 夏季平均アルベド
    ave_Ts = df["Ts_surface"].mean()  
    ave_WS = df["WindSpeed"].mean()           # 平均表面温度 (K)
    data_list.append({
        "lon": lon,
        "lat": lat,
        "total_melt": total_melt,
        "total_BDOT": total_BDOT,
        "total_Sub": total_Sub,
        "ave_albedo": ave_albedo,
        "Ts_surface": ave_Ts,
        "WS": ave_WS
    })

points_df = pd.DataFrame(data_list)
print(points_df.head())

# --- 2. 座標変換とグリッド作成 ---
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
points_df["x_3031"], points_df["y_3031"] = zip(*points_df.apply(lambda row: transformer.transform(row["lon"], row["lat"]), axis=1))

buffer_m = 5000
x_min, x_max = points_df["x_3031"].min() - buffer_m, points_df["x_3031"].max() + buffer_m
y_min, y_max = points_df["y_3031"].min() - buffer_m, points_df["y_3031"].max() + buffer_m
grid_x_vals = np.arange(x_min, x_max, 500)
grid_y_vals = np.arange(y_min, y_max, 500)
grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

# --- 3. 補間 ---
points_xy = points_df[["x_3031", "y_3031"]].values
values_melt = points_df["total_melt"].values
values_BDOT = points_df["total_BDOT"].values
values_Sub = points_df["total_Sub"].values
values_albedo = points_df["ave_albedo"].values
values_Ts = points_df["Ts_surface"].values - 273.15
values_WS = points_df["WS"].values

grid_total_melt = griddata(points_xy, values_melt, (grid_x, grid_y), method="linear") / 1000
grid_total_BDOT = griddata(points_xy, values_BDOT, (grid_x, grid_y), method="linear") / 1000
grid_total_Sub = griddata(points_xy, values_Sub, (grid_x, grid_y), method="linear") / 1000
grid_ave_albedo = griddata(points_xy, values_albedo, (grid_x, grid_y), method="linear")
grid_ave_Ts = griddata(points_xy, values_Ts, (grid_x, grid_y), method="linear")
grid_ave_WS = griddata(points_xy, values_WS, (grid_x, grid_y), method="linear")
SMB = grid_total_BDOT - grid_total_melt - grid_total_Sub

# --- 4. シェープファイルによるクリップ ---
shp_file = "/media/junsaito/E4308E2C308E05B0/JS_model/shp/shirase_glacier_total.shp"
gdf = gpd.read_file(shp_file)
gdf_proj = gdf.to_crs("EPSG:3031")
poly = gdf_proj.unary_union
mask = contains(poly, grid_x, grid_y)
grid_total_melt_clipped = np.where(mask, grid_total_melt, np.nan)
grid_total_BDOT_clipped = np.where(mask, grid_total_BDOT, np.nan)
SMB_clipped = np.where(mask, SMB, np.nan)
grid_ave_Ts_clipped = np.where(mask, grid_ave_Ts, np.nan)
grid_ave_WS_clipped = np.where(mask, grid_ave_WS, np.nan)
grid_ave_albedo_clipped = np.where(mask, grid_ave_albedo, np.nan)

# --- 5. Landsat画像の読み込み ---
landsat_file = "/media/junsaito/E4308E2C308E05B0/JS_model/Tiff/LC08_L2SR_149109_20241124_20241127_02_T2_SR_B1.TIF"
with rasterio.open(landsat_file) as src:
    landsat_img = src.read(1)
    landsat_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
print("Landsat extent:", landsat_extent)

# --- 6. プロット設定 ---
proj_plot = ccrs.SouthPolarStereo()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 16), subplot_kw={'projection': proj_plot})

bbox = gdf_proj.total_bounds  # [minx, miny, maxx, maxy]
buffer = 5000
extent_proj = [bbox[0]-buffer, bbox[2]+buffer, bbox[1]-buffer, bbox[3]+buffer]
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_extent(extent_proj, crs=proj_plot)

# --- 7. 背景の追加 ---
for ax in [ax1, ax2, ax3, ax4]:
    ax.imshow(landsat_img, origin='upper', cmap='gray',
              transform=proj_plot, extent=landsat_extent, alpha=0.5)
    ax.add_geometries(gdf_proj.geometry, crs=proj_plot,
                      facecolor='none', edgecolor='blue', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")

# # --- 8. pcolormeshプロットとカラーバー作成 ---
# mesh1 = ax1.pcolormesh(grid_x, grid_y, grid_total_melt_clipped*1000, cmap="plasma_r", shading="auto",
#                         transform=proj_plot, vmin=0, vmax=1000, edgecolors="none")
# cbar1 = plt.colorbar(mesh1, ax=ax1, orientation="vertical", shrink=0.4)
# cbar1.set_label("Total Melt (mm w.e.)", fontsize=20)
# cbar1.ax.tick_params(labelsize=18)

import matplotlib.colors as colors

# PowerNormを利用して、gamma=0.5でコントラストを強調
norm = colors.PowerNorm(gamma=1.5, vmin=0, vmax=1000)

mesh1 = ax1.pcolormesh(grid_x, grid_y, grid_total_melt_clipped*1000, cmap="plasma_r", norm=norm, shading="auto",
                        transform=proj_plot, edgecolors="none")
cbar1 = plt.colorbar(mesh1, ax=ax1, orientation="vertical", shrink=0.4)
cbar1.set_label("Total Melt (mm w.e.)", fontsize=20)
cbar1.ax.tick_params(labelsize=18)


mesh2 = ax2.pcolormesh(grid_x, grid_y, grid_ave_albedo_clipped, cmap="Spectral", shading="auto",
                        transform=proj_plot, vmin=0.68, vmax=0.75, edgecolors="none")
cbar2 = plt.colorbar(mesh2,ax=ax2, orientation="vertical", shrink=0.4)
cbar2.set_label("Albedo", fontsize=20)
cbar2.ax.tick_params(labelsize=18)

mesh3 = ax3.pcolormesh(grid_x, grid_y, grid_total_BDOT_clipped, cmap="bwr_r", shading="auto",
                        transform=proj_plot, vmin=0.5, vmax=0.8, edgecolors="none")
cbar3 = plt.colorbar(mesh3, ax=ax3, orientation="vertical", shrink=0.4)
cbar3.set_label("Precipitation (m)", fontsize=20)
cbar3.ax.tick_params(labelsize=18)

mesh4 = ax4.pcolormesh(grid_x, grid_y, grid_ave_Ts_clipped, cmap="viridis", shading="auto",
                        transform=proj_plot, vmin=-16, vmax=-12, edgecolors="none")
cbar4 = plt.colorbar(mesh4, ax=ax4, orientation="vertical", shrink=0.4)
cbar4.set_label("Surface Temperature (C)", fontsize=20)
cbar4.ax.tick_params(labelsize=18)


# norm1 = colors.PowerNorm(gamma=1.5, vmin=2, vmax=6)
# mesh4 = ax4.pcolormesh(grid_x, grid_y, grid_ave_WS_clipped, cmap="viridis", norm=norm1, shading="auto",
#                         transform=proj_plot, edgecolors="none")
# cbar4 = plt.colorbar(mesh4, ax=ax4, orientation="vertical", shrink=0.4)
# cbar4.set_label("WS (m/s)", fontsize=20)
# cbar4.ax.tick_params(labelsize=18)


# --- 9. 緯度・経度ラベルの設定 ---（
# # 非矩形投影では set_xticks/set_yticks は使えないため、gridlines() を利用
# for ax in [ax1, ax2, ax3, ax4]:
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                       linewidth=0.5, color='gray', alpha=0.7, linestyle='--',
#                       x_inline=False, y_inline=False, rotate_labels=False)
#     # 上側と右側のラベルは非表示、左側と下側は表示
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.left_labels = True
#     gl.bottom_labels = True
#     # tick の位置とフォーマッタを PlateCarree 座標系で設定
#     gl.xlocator = mticker.FixedLocator(np.arange(38.5, 40.1, 0.1))
#     gl.ylocator = mticker.FixedLocator(np.arange(-70.5, -69.4, 0.1))
#     gl.xformatter = cticker.LongitudeFormatter()
#     gl.yformatter = cticker.LatitudeFormatter()
#     gl.xlabel_style = {'size': 12, 'color': 'black'}
#     gl.ylabel_style = {'size': 12, 'color': 'black'}

# --- 10. スケールバーの追加 ---
def add_scalebar(ax, length, location=(0.8, 0.05), linewidth=3):
    length_m = length * 1000  # km → m
    x0, x1, y0, y1 = ax.get_extent(crs=proj_plot)
    sb_x = x0 + (x1 - x0) * location[0]
    sb_y = y0 + (y1 - y0) * location[1]
    ax.plot([sb_x, sb_x + length_m], [sb_y, sb_y],
            transform=proj_plot, color="black", linewidth=linewidth)
    ax.text(sb_x + length_m / 2, sb_y - (y1 - y0) * 0.02, f"{length} km",
            transform=proj_plot, ha="center", va="top", fontsize=12, color="black")

add_scalebar(ax1, length=5, location=(0.8, 0.05), linewidth=5)
add_scalebar(ax2, length=5, location=(0.8, 0.05), linewidth=5)
add_scalebar(ax3, length=5, location=(0.8, 0.05), linewidth=5)
add_scalebar(ax4, length=5, location=(0.8, 0.05), linewidth=5)

# --- 11. 追加の2点の座標をプロット ---
extra_points = pd.DataFrame({
    "lon": [39.41, 38.78],
    "lat": [-70.28, -70.08]
})
extra_points["x"], extra_points["y"] = zip(*extra_points.apply(lambda row: transformer.transform(row["lon"], row["lat"]), axis=1))
for ax in [ax1, ax2, ax3, ax4]:
    ax.scatter(extra_points["x"], extra_points["y"], facecolors='none', edgecolors='k', s=200,
               marker="o", transform=proj_plot, linewidths=2, zorder=10)

plt.subplots_adjust(wspace=0.01, hspace=0.05)
plt.tight_layout()
plt.show()
