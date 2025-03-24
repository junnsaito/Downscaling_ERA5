#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 21:03:09 2025

@author: junsaito
"""
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from rasterio.plot import reshape_as_image
from rasterio.warp import calculate_default_transform, reproject, Resampling

# 出力ディレクトリと各ファイルのパス（適宜変更してください）
output_dir = "/media/junsaito/E4308E2C308E05B0/Antarctica/H128"
output_tif = f"{output_dir}/downscaled_TP_wind_2015raw.tif"  # 降水量データ（mm/hr）
landsat_tif = "/media/junsaito/E4308E2C308E05B0/JS_model/Tiff/LC08_L2SR_149109_20241124_20241127_02_T2_SR_B1.TIF"  # Landsat画像
shapefile_path = "/media/junsaito/E4308E2C308E05B0/JS_model/shp/shirase_glacier_total.shp"  # シェープファイル

# 目的の投影系（南極用：極面ステレオ EPSG:3031）
dst_crs = "EPSG:3031"

# ① 降水量データ（TIF）の読み込みと再投影
with rasterio.open(output_tif) as src:
    precip_data = src.read(1)
    # 再投影パラメータの計算
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    precip_reproj = np.empty((height, width), dtype=src.meta['dtype'])
    reproject(
        source=rasterio.band(src, 1),
        destination=precip_reproj,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest)
    
    # NaN および 0 以下の値を NaN に置換
    precip_reproj = np.where(np.isnan(precip_reproj) | (precip_reproj <= 0), np.nan, precip_reproj)
    
    precip_extent = [transform[2], transform[2] + transform[0] * width,
                     transform[5] + transform[4] * height, transform[5]]
    print("[INFO] 降水量TIF の再投影完了。")


# ② Landsat 画像の読み込みと再投影（RGB 画像として利用）
with rasterio.open(landsat_tif) as l_src:
    # 再投影パラメータの計算
    transform_l, width_l, height_l = calculate_default_transform(l_src.crs, dst_crs, l_src.width, l_src.height, *l_src.bounds)
    kwargs_l = l_src.meta.copy()
    kwargs_l.update({
        'crs': dst_crs,
        'transform': transform_l,
        'width': width_l,
        'height': height_l
    })
    if l_src.count >= 3:
        landsat_arr = np.empty((3, height_l, width_l), dtype=l_src.meta['dtype'])
        for i in range(3):
            reproject(
                source=rasterio.band(l_src, i+1),
                destination=landsat_arr[i],
                src_transform=l_src.transform,
                src_crs=l_src.crs,
                dst_transform=transform_l,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
    else:
        band1 = np.empty((height_l, width_l), dtype=l_src.meta['dtype'])
        reproject(
            source=rasterio.band(l_src, 1),
            destination=band1,
            src_transform=l_src.transform,
            src_crs=l_src.crs,
            dst_transform=transform_l,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        landsat_arr = np.stack([band1, band1, band1])

    landsat_rgb = reshape_as_image(landsat_arr)

    def normalize_band(band):
        vmin, vmax = np.percentile(band, (2, 98))
        norm = (band - vmin) / (vmax - vmin)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    landsat_rgb = np.stack([normalize_band(landsat_rgb[:, :, i]) for i in range(3)], axis=-1)
    landsat_extent = [transform_l[2], transform_l[2] + transform_l[0] * width_l,
                       transform_l[5] + transform_l[4] * height_l, transform_l[5]]
    print("[INFO] Landsat画像 の再投影完了。")

# ③ シェープファイルの読み込みと再投影（WGS84→EPSG:3031）
gdf = gpd.read_file(shapefile_path)
print("[INFO] シェープファイルの元のCRS:", gdf.crs)
if gdf.crs != dst_crs:
    gdf = gdf.to_crs(dst_crs)
    print("[INFO] シェープファイルを", dst_crs, "に再投影しました。")

# ④ プロット
fig, ax = plt.subplots(figsize=(12, 10))

# Landsat画像を背景に表示
ax.imshow(landsat_rgb, extent=landsat_extent)

# 降水量データを、vmin=200, vmax=1200 の範囲で半透明で上に重ねる
precip_im = ax.imshow(precip_reproj, cmap='viridis', alpha=0.8, extent=precip_extent, vmin=100, vmax=600)
cbar = fig.colorbar(precip_im, ax=ax, fraction=0.036, pad=0.04)
cbar.set_label('Precipitation (mm/hr)')

# シェープファイルの境界線を赤色でプロット
gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2)

ax.set_title('Downscaled Precipitation with Landsat and Shapefile Overlay (Antarctica, EPSG:3031)')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')

plt.tight_layout()
plt.show()
