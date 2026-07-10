# -*- coding: utf-8 -*-
"""Parsea la matriz de registros históricos 2010-2023 (coordenadas en formatos
inconsistentes) y produce data/incendios_historicos.geojson."""
import os
import re
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
import paths

MESES = {'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
         'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11,
         'DICIEMBRE': 12}


def parse_lat(v):
    """Latitudes de Cali ~3.27-3.55. Vienen como 33328 -> 3.3328, 3532881 -> 3.532881."""
    if pd.isna(v):
        return None
    try:
        f = float(v)
        if 3.0 <= f <= 3.7:
            return f
    except (TypeError, ValueError):
        pass
    s = re.sub(r"\D", "", str(v))
    if not s or s[0] != "3":
        return None
    return float(s[0] + "." + s[1:])


def parse_lon(v):
    """Longitudes ~ -76.71 a -76.45. Vienen como -764714 o 76545486 (sin signo)."""
    if pd.isna(v):
        return None
    try:
        f = float(v)
        if -77.0 <= f <= -76.0:
            return f
    except (TypeError, ValueError):
        pass
    s = re.sub(r"\D", "", str(v))
    if not s.startswith("76"):
        return None
    return -float(s[:2] + "." + s[2:])


def main():
    df = pd.read_excel(paths.MATRIZ_XLSX, header=3)
    df.columns = ['ANO', 'MES', 'DIA', 'LAT', 'LON', 'HORA', 'VEREDA', 'CORREG',
                  'SATELITE', 'OBS'][:len(df.columns)]
    df = df.dropna(subset=['ANO', 'LAT', 'LON']).copy()

    df['lat'] = df['LAT'].map(parse_lat)
    df['lon'] = df['LON'].map(parse_lon)
    df['mes_num'] = df['MES'].astype(str).str.strip().str.upper().map(MESES)
    df['dia_num'] = pd.to_numeric(df['DIA'], errors='coerce')

    bad = df[df['lat'].isna() | df['lon'].isna() | df['mes_num'].isna() | df['dia_num'].isna()]
    if len(bad):
        print(f"ADVERTENCIA: {len(bad)} registros descartados por coordenadas/fecha ilegibles:")
        print(bad[['ANO', 'MES', 'DIA', 'LAT', 'LON']].to_string())
    df = df.dropna(subset=['lat', 'lon', 'mes_num', 'dia_num'])

    df['fecha'] = pd.to_datetime(dict(year=df['ANO'].astype(int),
                                      month=df['mes_num'].astype(int),
                                      day=df['dia_num'].astype(int)), errors='coerce')
    df = df.dropna(subset=['fecha'])

    gdf = gpd.GeoDataFrame(
        df[['fecha', 'SATELITE', 'VEREDA']],
        geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])], crs=paths.CRS_GEO)

    # Validación espacial contra el límite municipal (+1 km de tolerancia)
    cali = gpd.read_file(paths.LOCALIDADES).to_crs(paths.CRS_M).union_all()
    gdf_m = gdf.to_crs(paths.CRS_M)
    dentro = gdf_m.within(cali.buffer(1000))
    fuera = gdf[~dentro]
    if len(fuera):
        print(f"ADVERTENCIA: {len(fuera)} puntos fuera de Cali (+1km), se descartan:")
        for _, r in fuera.iterrows():
            print("  ", r['fecha'].date(), r.geometry.y, r.geometry.x)
    gdf = gdf[dentro.values].copy()

    out = os.path.join(paths.DATA, "incendios_historicos.geojson")
    gdf['fecha'] = gdf['fecha'].dt.strftime('%Y-%m-%d')
    gdf.to_file(out, driver="GeoJSON")
    print(f"\nOK: {len(gdf)} incendios 2010-2023 -> {out}")
    print(gdf['fecha'].str[:4].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
