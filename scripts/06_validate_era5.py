# -*- coding: utf-8 -*-
"""Validación cruzada de ERA5/Open-Meteo contra observaciones en tierra.

Tres comparaciones con los datos realmente disponibles:
1. Temperatura diaria: estación GHCN del aeropuerto SKCL (CO000080259,
   3.543 N -76.381 W) vs ERA5 en ese punto, últimos ~5 años.
2. Precipitación anual: estación CALI SEDE IDEAM [26080310] (serie anual
   1966-2022 del archivo del proyecto) vs ERA5 anual en ese punto, 1981-2022.
3. Diario reciente: histórico acumulado de las estaciones IDEAM en tiempo real
   (data/stations_history.csv) vs el modelo (crece día a día).

Escribe models/validacion_era5.json y muestra el reporte.
"""
import glob
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paths
import stations

GHCN_ST = "CO000080259"   # Aeropuerto Alfonso Bonilla Aragón (SKCL)
GHCN_LAT, GHCN_LON = 3.543, -76.381
UNIV_LAT, UNIV_LON = 3.476, -76.523  # CALI SEDE IDEAM [26080310]


def http_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def era5_daily(lat, lon, start, end, var):
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily={var}&timezone=America%2FBogota")
    d = http_json(url)["daily"]
    s = pd.Series(d[var], index=pd.to_datetime(d["time"]), dtype=float)
    return s


def stats_pair(obs, mod):
    j = pd.concat([obs.rename("obs"), mod.rename("mod")], axis=1).dropna()
    if len(j) < 10:
        return None
    r = float(np.corrcoef(j["obs"], j["mod"])[0, 1])
    bias = float((j["mod"] - j["obs"]).mean())
    rmse = float(np.sqrt(((j["mod"] - j["obs"]) ** 2).mean()))
    return {"n": int(len(j)), "correlacion": round(r, 3),
            "sesgo_modelo_menos_obs": round(bias, 2), "rmse": round(rmse, 2)}


def validar_temperatura():
    """GHCN TAVG diaria vs ERA5 temperatura media diaria (5 años)."""
    url = ("https://www.ncei.noaa.gov/access/services/data/v1?"
           "dataset=daily-summaries&stations=" + GHCN_ST +
           "&startDate=2021-01-01&endDate=2026-06-30&dataTypes=TAVG"
           "&format=json&units=metric")
    data = http_json(url)
    obs = pd.Series({pd.Timestamp(d["DATE"]): float(d["TAVG"])
                     for d in data if "TAVG" in d})
    mod = era5_daily(GHCN_LAT, GHCN_LON, "2021-01-01", "2026-06-30",
                     "temperature_2m_mean")
    return stats_pair(obs, mod)


def validar_precip_anual():
    """Serie anual de la estación CALI SEDE IDEAM vs ERA5 (1981-2022)."""
    fp = glob.glob(os.path.join(
        paths.BASE, "Informaci*n Base", "Datos Climaticos", "Datos Lluvia",
        "Univalle Sede IDEAM.csv"))
    if not fp:
        return None
    df = pd.read_csv(fp[0], encoding="latin-1")
    df = df[df["Etiqueta"] == "PTPM_TT_A"]
    df["anio"] = pd.to_datetime(df["Fecha"], format="%m/%d/%Y %H:%M",
                                errors="coerce").dt.year
    if df["anio"].isna().all():
        df["anio"] = pd.to_datetime(df["Fecha"], errors="coerce",
                                    dayfirst=True).dt.year
    obs = (df.dropna(subset=["anio"]).drop_duplicates(subset=["anio"])
           .set_index("anio")["Valor"].astype(float))
    obs = obs[(obs.index >= 1981) & (obs.index <= 2022)]

    mod_d = era5_daily(UNIV_LAT, UNIV_LON, "1981-01-01", "2022-12-31",
                       "precipitation_sum")
    mod = mod_d.groupby(mod_d.index.year).sum()
    obs.index = obs.index.astype(int)
    return stats_pair(obs, mod)


def validar_diario_reciente():
    """Histórico acumulado de estaciones en tiempo real vs modelo registrado."""
    fp = os.path.join(paths.DATA, "model_at_stations.csv")
    h = stations.dias_completos()
    if not len(h) or not os.path.exists(fp):
        return {"nota": "aún sin días completos acumulados; este contraste "
                        "crece automáticamente con cada ejecución diaria"}
    m = pd.read_csv(fp, dtype={"estacion": str})
    m["fecha"] = pd.to_datetime(m["fecha"]).dt.date
    out = {}
    for cod, info in stations.ESTACIONES.items():
        d = h[h["estacion"] == cod]
        if not len(d):
            continue
        obs_t = (d[d["sensor"].isin(["temp", "tmax"])]
                 .groupby("fecha")["valor"].max())
        mm = m[m["estacion"] == cod].set_index("fecha")
        st = stats_pair(obs_t, mm["tmax_mod"])
        if st:
            out[info["nombre"]] = {"tmax": st}
    return out or {"nota": "historia insuficiente todavía"}


def main():
    print("1. Temperatura diaria GHCN (aeropuerto SKCL) vs ERA5, 2021-2026...")
    t = validar_temperatura()
    print("   ", t)

    print("2. Precipitación ANUAL estación CALI SEDE IDEAM vs ERA5, 1981-2022...")
    p = validar_precip_anual()
    print("   ", p)

    print("3. Estaciones IDEAM tiempo real vs modelo (acumulándose)...")
    d = validar_diario_reciente()
    print("   ", d)

    out = {"temperatura_diaria_skcl": t, "precip_anual_cali_ideam": p,
           "diario_reciente_estaciones": d,
           "generado": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    fp = os.path.join(paths.MODELS, "validacion_era5.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("OK ->", fp)


if __name__ == "__main__":
    main()
