# -*- coding: utf-8 -*-
"""Construye el dataset de entrenamiento del SAT de incendios de Cali.

Presencias  : incendios históricos 2010-2023 (matriz FIRMS curada).
Ausencias   : puntos y fechas aleatorios dentro del distrito, lejos (espacio-tiempo)
              de los incendios registrados.
Estáticas   : elevación y pendiente (DTM 2.5 m), susceptibilidad de la vegetación
              (ctc, cdc, ccc, SUSC), distancia a vías (todas y principales).
Dinámicas   : clima diario ERA5 (Open-Meteo archive): temperatura, lluvia (día y
              acumulados 3/7/30 días), días sin lluvia, viento, radiación, ET0.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
import interp
import paths

RNG = np.random.default_rng(42)
N_ABS_RATIO = 4          # ausencias por cada presencia
WEATHER_CACHE = os.path.join(paths.DATA, "weather_cache")
os.makedirs(WEATHER_CACHE, exist_ok=True)

DAILY_VARS = ("temperature_2m_max,temperature_2m_mean,precipitation_sum,"
              "wind_speed_10m_max,shortwave_radiation_sum,et0_fao_evapotranspiration,"
              "relative_humidity_2m_min,relative_humidity_2m_mean")


# ---------------------------------------------------------------- clima ERA5
def snap(v):
    """Celda ERA5-Land de 0.1 grados."""
    return round(round(v * 10) / 10, 1)


def fetch_cell(lat, lon, start="2009-12-01", end="2023-12-31"):
    key = f"cell_{lat:.1f}_{lon:.1f}_v2.json"
    fp = os.path.join(WEATHER_CACHE, key)
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily={DAILY_VARS}&timezone=America%2FBogota")
    print("  descargando", key)
    for intento in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                espera = 30 * (intento + 1)
                print(f"    429 rate-limit, esperando {espera}s...")
                time.sleep(espera)
            else:
                raise
    else:
        raise RuntimeError(f"Open-Meteo: demasiados 429 para {key}")
    time.sleep(5)
    if "daily" not in data:
        raise RuntimeError(f"Open-Meteo sin datos para {key}: {data}")
    with open(fp, "w") as f:
        json.dump(data, f)
    return data


def cell_dataframe(lat, lon):
    d = fetch_cell(lat, lon)["daily"]
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").astype(float)
    p = df["precipitation_sum"].fillna(0.0)
    df["precip_3d"] = p.rolling(3, min_periods=1).sum()
    df["precip_7d"] = p.rolling(7, min_periods=1).sum()
    df["precip_30d"] = p.rolling(30, min_periods=1).sum()
    # días desde la última lluvia >= 1 mm (incluye el propio día como 0)
    rain = (p >= 1.0).to_numpy()
    days = np.zeros(len(rain))
    c = 60.0
    for i, r in enumerate(rain):
        c = 0.0 if r else min(c + 1.0, 60.0)
        days[i] = c
    df["dias_sin_lluvia"] = days
    return df


_CELLS = {}
_VARMAP = {
    "tmax": "temperature_2m_max", "tmean": "temperature_2m_mean",
    "precip": "precipitation_sum", "precip_3d": "precip_3d",
    "precip_7d": "precip_7d", "precip_30d": "precip_30d",
    "dias_sin_lluvia": "dias_sin_lluvia", "viento_max": "wind_speed_10m_max",
    "radiacion": "shortwave_radiation_sum", "et0": "et0_fao_evapotranspiration",
    "humedad_min": "relative_humidity_2m_min",
    "humedad_media": "relative_humidity_2m_mean",
}


def _cell(key):
    if key not in _CELLS:
        _CELLS[key] = cell_dataframe(*key)
    return _CELLS[key]


def weather_at(lat, lon, fecha):
    """Clima interpolado bilinealmente entre las 4 celdas ERA5 circundantes
    (evita bordes rectos entre celdas de 0.1 grados)."""
    ts = pd.Timestamp(fecha)
    out = {}
    for feat, col in _VARMAP.items():
        vals = {}
        for la in interp.LATS:
            for lo in interp.LONS:
                df = _cell((la, lo))
                try:
                    vals[(la, lo)] = float(df.loc[ts, col])
                except KeyError:
                    return None
        out[feat] = float(interp.bilinear(vals, lat, lon))
    return out


# ------------------------------------------------------------ capas estáticas
def build_terrain():
    """DEM y pendiente a 10 m derivados del DTM 2.5 m; se guardan en data/."""
    dem_fp = os.path.join(paths.DATA, "dem_10m.tif")
    slope_fp = os.path.join(paths.DATA, "slope_10m.tif")
    if os.path.exists(dem_fp) and os.path.exists(slope_fp):
        return dem_fp, slope_fp
    # Nota: los .ovr del DTM están corruptos (devuelven 0), así que se lee a
    # resolución nativa por franjas y se promedia manualmente 4x4 (2.5 -> 10 m).
    factor = 4
    with rasterio.open(paths.DTM) as src:
        h, w = src.height // factor, src.width // factor
        dem = np.empty((h, w), dtype="float32")
        strip = 500  # filas de salida por franja
        for r0 in range(0, h, strip):
            r1 = min(r0 + strip, h)
            win = ((r0 * factor, r1 * factor), (0, w * factor))
            blk = src.read(1, window=win)
            blk = np.where(blk == src.nodata, np.nan, blk).astype("float32")
            with np.errstate(invalid="ignore"):
                dem[r0:r1] = np.nanmean(
                    blk.reshape(r1 - r0, factor, w, factor), axis=(1, 3))
        tr = src.transform * src.transform.scale(factor, factor)
        crs = src.crs
    gy, gx = np.gradient(dem, 10.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    prof = dict(driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
                dtype="float32", crs=crs, transform=tr, nodata=-9999,
                compress="deflate")
    for fp, arr in [(dem_fp, dem), (slope_fp, slope)]:
        with rasterio.open(fp, "w", **prof) as dst:
            dst.write(np.where(np.isnan(arr), -9999, arr), 1)
    print("OK terreno:", dem_fp, slope_fp)
    return dem_fp, slope_fp


def sample_raster(fp, gdf_m):
    with rasterio.open(fp) as src:
        pts = [(g.x, g.y) for g in gdf_m.geometry]
        vals = np.array([v[0] for v in src.sample(pts)], dtype="float64")
        vals[vals == src.nodata] = np.nan
    return vals


def add_static_features(gdf):
    """gdf en EPSG:4326 -> DataFrame con columnas estáticas."""
    gdf_m = gdf.to_crs(paths.CRS_M)

    dem_fp, slope_fp = build_terrain()
    elev = sample_raster(dem_fp, gdf_m)
    slope = sample_raster(slope_fp, gdf_m)

    sus = gpd.read_file(paths.SUSCEPTIBILIDAD).to_crs(paths.CRS_M)
    sus = sus[["ctc", "cdc", "ccc", "SUSC", "geometry"]]
    joined = gpd.sjoin(gdf_m, sus, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    vias = gpd.read_file(paths.VIAS).to_crs(paths.CRS_M)
    vias = vias[vias["estado_act"] == "EXISTENTE"]
    principales = vias[vias["tipo_via"].isin(
        ["Via Arteria Principal", "Via Arteria Secundaria", "Via Interegional",
         "Via Colectora", "Via Colectora Rural"])]
    todas_u = vias.union_all()
    prin_u = principales.union_all()
    dist_vias = gdf_m.geometry.apply(lambda g: g.distance(todas_u))
    dist_prin = gdf_m.geometry.apply(lambda g: g.distance(prin_u))

    out = pd.DataFrame(index=gdf.index)
    out["elevacion"] = elev
    out["pendiente"] = slope
    for c in ["ctc", "cdc", "ccc", "SUSC"]:
        out[c] = joined[c].fillna(0).values  # fuera de polígonos = no combustible
    out["dist_vias"] = dist_vias.values
    out["dist_vias_prin"] = dist_prin.values
    return out


# ------------------------------------------------------------------ muestreo
def sample_absences(cali_m, fires, n):
    """Puntos aleatorios en el distrito con fechas aleatorias 2010-2023,
    alejados >2 km o >7 días de cualquier incendio registrado."""
    minx, miny, maxx, maxy = cali_m.bounds
    fires_m = fires.to_crs(paths.CRS_M)
    fdates = pd.to_datetime(fires["fecha"]).values
    fxy = np.array([(g.x, g.y) for g in fires_m.geometry])

    all_days = pd.date_range("2010-01-01", "2023-12-31", freq="D")
    pts, dates = [], []
    while len(pts) < n:
        x = RNG.uniform(minx, maxx)
        y = RNG.uniform(miny, maxy)
        p = Point(x, y)
        if not cali_m.contains(p):
            continue
        d = all_days[RNG.integers(0, len(all_days))]
        near_t = np.abs((fdates - np.datetime64(d)) / np.timedelta64(1, "D")) <= 7
        if near_t.any():
            dd = np.hypot(fxy[near_t, 0] - x, fxy[near_t, 1] - y)
            if (dd < 2000).any():
                continue
        pts.append(p)
        dates.append(d)
    g = gpd.GeoDataFrame({"fecha": [d.strftime("%Y-%m-%d") for d in dates]},
                         geometry=pts, crs=paths.CRS_M).to_crs(paths.CRS_GEO)
    return g


def main():
    fires = gpd.read_file(os.path.join(paths.DATA, "incendios_historicos.geojson"))
    cali_m = gpd.read_file(paths.LOCALIDADES).to_crs(paths.CRS_M).union_all()

    absences = sample_absences(cali_m, fires, N_ABS_RATIO * len(fires))
    fires["incendio"] = 1
    absences["incendio"] = 0
    pts = pd.concat([fires[["fecha", "incendio", "geometry"]],
                     absences[["fecha", "incendio", "geometry"]]], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, crs=paths.CRS_GEO)
    print(f"presencias={int(pts['incendio'].sum())}  ausencias={(pts['incendio']==0).sum()}")

    print("Extrayendo variables estáticas...")
    static = add_static_features(pts)

    print("Extrayendo clima diario (Open-Meteo/ERA5)...")
    met_rows = []
    for i, row in pts.iterrows():
        w = weather_at(row.geometry.y, row.geometry.x, row["fecha"])
        met_rows.append(w if w else {})
    met = pd.DataFrame(met_rows, index=pts.index)

    df = pd.concat([pts.drop(columns="geometry"),
                    pd.DataFrame({"lat": pts.geometry.y, "lon": pts.geometry.x}),
                    static, met], axis=1)
    n0 = len(df)
    df = df.dropna(subset=["tmax", "precip"])
    if len(df) < n0:
        print(f"ADVERTENCIA: {n0 - len(df)} filas sin clima, descartadas")

    out = os.path.join(paths.DATA, "dataset_entrenamiento.csv")
    df.to_csv(out, index=False)
    print(f"OK: dataset {df.shape} -> {out}")
    print(df.groupby("incendio")[["pendiente", "SUSC", "dist_vias", "precip_30d",
                                  "tmax", "dias_sin_lluvia"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
