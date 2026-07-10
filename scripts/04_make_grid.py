# -*- coding: utf-8 -*-
"""Grilla de predicción de 100 m sobre el distrito de Cali con las variables
estáticas precalculadas. Se ejecuta una sola vez; el pronóstico diario solo
le agrega el clima del día."""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(__file__))
import paths

RES = 100.0  # metros


def main():
    cali = gpd.read_file(paths.LOCALIDADES).to_crs(paths.CRS_M).union_all()
    minx, miny, maxx, maxy = cali.bounds
    xs = np.arange(minx + RES / 2, maxx, RES)
    ys = np.arange(miny + RES / 2, maxy, RES)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()),
                           crs=paths.CRS_M)
    pts = pts[pts.within(cali)].reset_index(drop=True)
    print(f"celdas dentro del distrito: {len(pts)}")

    # terreno
    for name, col in [("dem_10m.tif", "elevacion"), ("slope_10m.tif", "pendiente")]:
        with rasterio.open(os.path.join(paths.DATA, name)) as src:
            vals = np.array([v[0] for v in src.sample(
                [(g.x, g.y) for g in pts.geometry])], dtype="float64")
            vals[vals == src.nodata] = np.nan
        pts[col] = vals

    # susceptibilidad de la vegetación
    sus = gpd.read_file(paths.SUSCEPTIBILIDAD).to_crs(paths.CRS_M)
    sus = sus[["ctc", "cdc", "ccc", "SUSC", "geometry"]]
    pts = gpd.sjoin(pts, sus, how="left", predicate="within").drop(columns="index_right")
    pts = pts[~pts.index.duplicated(keep="first")]
    for c in ["ctc", "cdc", "ccc", "SUSC"]:
        pts[c] = pts[c].fillna(0)

    # distancia a vías
    vias = gpd.read_file(paths.VIAS).to_crs(paths.CRS_M)
    vias = vias[vias["estado_act"] == "EXISTENTE"]
    prin = vias[vias["tipo_via"].isin(
        ["Via Arteria Principal", "Via Arteria Secundaria", "Via Interegional",
         "Via Colectora", "Via Colectora Rural"])]
    j = gpd.sjoin_nearest(pts[["geometry"]], vias[["geometry"]],
                          distance_col="dist_vias")
    pts["dist_vias"] = j[~j.index.duplicated(keep="first")]["dist_vias"]
    j = gpd.sjoin_nearest(pts[["geometry"]], prin[["geometry"]],
                          distance_col="dist_vias_prin")
    pts["dist_vias_prin"] = j[~j.index.duplicated(keep="first")]["dist_vias_prin"]

    # rellenos: terreno faltante (bordes) con vecinos ya es NaN -> mediana
    for c in ["elevacion", "pendiente"]:
        pts[c] = pts[c].fillna(pts[c].median())

    # zona (corregimiento o zona urbana) para el resumen diario del geoportal;
    # se precalcula aquí para que 05_predict_daily.py no dependa de shapefiles
    corr = gpd.read_file(paths.CORREGIMIENTOS).to_crs(paths.CRS_M)
    corr = corr[["corregimie", "geometry"]].rename(columns={"corregimie": "nombre"})
    urb = gpd.read_file(paths.LOCALIDADES).to_crs(paths.CRS_M)
    urb = urb[urb["zona"] == "Urbana"][["geometry"]].dissolve()
    urb["nombre"] = "Zona Urbana"
    zonas = pd.concat([corr, urb[["nombre", "geometry"]]], ignore_index=True)
    jz = gpd.sjoin(pts[["geometry"]], zonas, predicate="within")
    pts["zona"] = jz[~jz.index.duplicated(keep="first")]["nombre"]

    webdata = os.path.join(paths.WEB, "data")
    os.makedirs(webdata, exist_ok=True)
    zb = zonas.to_crs(paths.CRS_GEO)
    zb["geometry"] = zb.geometry.simplify(0.0004)
    zb.to_file(os.path.join(webdata, "zonas_base.geojson"), driver="GeoJSON")

    wgs = pts.to_crs(paths.CRS_GEO)
    pts["lat"] = wgs.geometry.y
    pts["lon"] = wgs.geometry.x
    pts["x"] = pts.geometry.x
    pts["y"] = pts.geometry.y

    out = os.path.join(paths.DATA, "grid_static.parquet")
    pts.drop(columns="geometry").to_parquet(out)
    print(f"OK: {out}  columnas: {[c for c in pts.columns if c != 'geometry']}")


if __name__ == "__main__":
    main()
