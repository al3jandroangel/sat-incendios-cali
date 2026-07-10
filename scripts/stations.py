# -*- coding: utf-8 -*-
"""Estaciones meteorológicas IDEAM de Cali en tiempo casi real.

Fuente: datos.gov.co, dataset 57sv-p2fu «Datos Hidrometeorológicos Crudos»
(ventana móvil de ~12-24 h, latencia ~6-10 h, sin autenticación).

Los códigos de estación llevan ceros iniciales (p. ej. 0026085160).
El sensor de viento de UNIVALLE reporta valores absurdos (>20 m/s sostenidos);
se filtra cualquier viento > 20 m/s como dato defectuoso.

Cada ejecución agrega las observaciones nuevas a data/stations_history.csv;
con >= MIN_DIAS_CORRECCION días completos acumulados se calculan factores de
corrección de sesgo del modelo (delta de temperatura, razón de precipitación)
que 05_predict_daily.py aplica con ponderación espacial gaussiana.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paths

HIST = os.path.join(paths.DATA, "stations_history.csv")
MIN_DIAS_CORRECCION = 7
MAX_VIENTO_MS = 20.0

ESTACIONES = {
    "0026085160": {"nombre": "Siloé", "lat": 3.425278, "lon": -76.561111},
    "0026055120": {"nombre": "Universidad del Valle", "lat": 3.378, "lon": -76.533889},
    "0026085170": {"nombre": "Base Aérea M.F. Suárez", "lat": 3.4543, "lon": -76.4997},
}
SENSORES = {"0240": "precip", "0068": "temp", "0069": "tmax",
            "0070": "tmin", "0103": "viento", "0027": "humedad"}


def _q(params, intentos=4):
    url = ("https://www.datos.gov.co/resource/57sv-p2fu.json?"
           + urllib.parse.urlencode(params))
    ultimo = None
    for i in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read())
        except Exception as e:
            ultimo = e
            time.sleep(8 * (i + 1))
    raise RuntimeError(f"datos.gov.co: fallo persistente: {ultimo}")


def fetch_raw():
    """Observaciones crudas disponibles para las estaciones de Cali."""
    rows = []
    for cod in ESTACIONES:
        for sensor in SENSORES:
            try:
                data = _q({"codigoestacion": cod, "codigosensor": sensor,
                           "$select": "fechaobservacion,valorobservado",
                           "$limit": "500"})
            except Exception as e:
                print(f"  aviso: {cod}/{sensor}: {e}")
                continue
            for d in data:
                rows.append({"estacion": cod, "sensor": SENSORES[sensor],
                             "fecha_hora": d["fechaobservacion"],
                             "valor": float(d["valorobservado"])})
    df = pd.DataFrame(rows)
    if len(df):
        df.loc[(df["sensor"] == "viento") & (df["valor"] > MAX_VIENTO_MS),
               "valor"] = np.nan
    return df


def update_history(df_new):
    """Acumula observaciones sin duplicar (estacion, sensor, fecha_hora)."""
    if os.path.exists(HIST):
        hist = pd.read_csv(HIST)
        df = pd.concat([hist, df_new], ignore_index=True)
    else:
        df = df_new
    df = df.drop_duplicates(subset=["estacion", "sensor", "fecha_hora"])
    df.to_csv(HIST, index=False)
    return df


def resumen_actual(df_raw):
    """Resumen por estación de la ventana disponible (para mostrar en el mapa)."""
    out = []
    for cod, info in ESTACIONES.items():
        d = df_raw[df_raw["estacion"] == cod]
        if not len(d):
            continue
        piv = {}
        for sensor, g in d.groupby("sensor"):
            v = g["valor"].dropna()
            if not len(v):
                continue
            if sensor == "precip":
                piv["precip_mm"] = round(float(v.sum()), 1)
            elif sensor in ("temp", "tmax"):
                piv["tmax_C"] = round(max(piv.get("tmax_C", -99),
                                          float(v.max())), 1)
            elif sensor == "tmin":
                piv["tmin_C"] = round(float(v.min()), 1)
            elif sensor == "viento":
                piv["viento_max_kmh"] = round(float(v.max()) * 3.6, 1)
            elif sensor == "humedad":
                piv["humedad_min"] = round(float(v.min()), 0)
        piv["ultima_obs"] = d["fecha_hora"].max()[:16].replace("T", " ")
        piv["ventana_desde"] = d["fecha_hora"].min()[:16].replace("T", " ")
        out.append({"codigo": cod, "nombre": info["nombre"],
                    "lat": info["lat"], "lon": info["lon"], **piv})
    return out


def dias_completos():
    """Días calendario con cobertura >= 18 h en el histórico acumulado."""
    if not os.path.exists(HIST):
        return pd.DataFrame()
    h = pd.read_csv(HIST)
    h["fecha_hora"] = pd.to_datetime(h["fecha_hora"])
    h["fecha"] = h["fecha_hora"].dt.date
    h["hora"] = h["fecha_hora"].dt.hour
    cov = (h.groupby(["estacion", "fecha"])["hora"].nunique()
           .rename("horas").reset_index())
    ok = cov[cov["horas"] >= 18][["estacion", "fecha"]]
    return h.merge(ok, on=["estacion", "fecha"])


def factores_correccion(model_daily_at_station):
    """Sesgo modelo-vs-estación sobre días completos acumulados.

    model_daily_at_station: DataFrame index=fecha (date) con columnas
    (estacion, tmax_mod, precip_mod) interpoladas del modelo en cada estación.
    Devuelve dict por estación: {delta_tmax, razon_precip, n_dias} o {} si
    todavía no hay historia suficiente.
    """
    h = dias_completos()
    if not len(h):
        return {}
    fact = {}
    for cod in ESTACIONES:
        d = h[h["estacion"] == cod]
        if not len(d):
            continue
        obs_t = (d[d["sensor"].isin(["temp", "tmax"])]
                 .groupby("fecha")["valor"].max().rename("tmax_obs"))
        obs_p = (d[d["sensor"] == "precip"]
                 .groupby("fecha")["valor"].sum().rename("precip_obs"))
        obs = pd.concat([obs_t, obs_p], axis=1).dropna()
        m = model_daily_at_station[model_daily_at_station["estacion"] == cod]
        j = obs.join(m.set_index("fecha"), how="inner").dropna()
        if len(j) < MIN_DIAS_CORRECCION:
            continue
        delta_t = float((j["tmax_obs"] - j["tmax_mod"]).mean())
        # razón de precipitación con suavizado (evita división por ~0)
        razon_p = float((j["precip_obs"].sum() + 1.0) /
                        (j["precip_mod"].sum() + 1.0))
        fact[cod] = {"delta_tmax": round(np.clip(delta_t, -5, 5), 2),
                     "razon_precip": round(np.clip(razon_p, 0.2, 3.0), 2),
                     "n_dias": int(len(j))}
    return fact


if __name__ == "__main__":
    raw = fetch_raw()
    print(f"observaciones descargadas: {len(raw)}")
    update_history(raw)
    for e in resumen_actual(raw):
        print(e)
