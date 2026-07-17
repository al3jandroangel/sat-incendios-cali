# -*- coding: utf-8 -*-
"""Actualización diaria del SAT: clima del día (Open-Meteo) interpolado
bilinealmente + corrección con estaciones IDEAM en tierra, aplica el modelo y
escribe las salidas del geoportal en web/data/.

Salidas:
  web/data/alerta_overlay.png      mapa de alerta (rojo/naranja/verde)
  web/data/alerta_meta.json        fecha, límites, estadísticas, clima del día
  web/data/corregimientos.geojson  resumen por corregimiento (% por nivel)
  web/data/estaciones.geojson      estaciones IDEAM con observaciones vs modelo
  web/data/hotspots.geojson        puntos calientes FIRMS 48 h (si hay MAP_KEY)

La corrección de sesgo modelo-vs-estaciones (delta de temperatura, razón de
precipitación) se activa automáticamente cuando data/stations_history.csv
acumula >= 7 días completos por estación; mientras tanto las observaciones se
muestran en el portal como control de calidad.
"""
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
import interp
import paths
import stations

WEBDATA = os.path.join(paths.WEB, "data")
os.makedirs(WEBDATA, exist_ok=True)
MODEL_AT_ST = os.path.join(paths.DATA, "model_at_stations.csv")

DAILY_VARS = ("temperature_2m_max,temperature_2m_mean,precipitation_sum,"
              "wind_speed_10m_max,shortwave_radiation_sum,et0_fao_evapotranspiration,"
              "relative_humidity_2m_min,relative_humidity_2m_mean")
HOURLY_VARS = "wind_speed_10m,relative_humidity_2m"
COLORS = {"ALTA": (198, 40, 40, 235), "MEDIA": (249, 168, 37, 235),
          "BAJA": (46, 125, 50, 200)}
FEATS_MET = ["tmax", "tmean", "precip", "precip_3d", "precip_7d", "precip_30d",
             "dias_sin_lluvia", "viento_max", "radiacion", "et0",
             "humedad_min", "humedad_media"]
SIGMA_M = 8000.0   # alcance espacial (gaussiano) de la corrección de estaciones
W0 = 0.15          # encogimiento hacia "sin corrección" lejos de las estaciones
LAPSE = -0.0065    # gradiente térmico vertical (°C/m) para desescalado orográfico
WET_MM = 10.0      # umbral de "día con lluvia" en ERA5 calibrado a la frecuencia
                   # real de Cali (~131 días/año IDEAM); ver 02_build_dataset.py


def http_json(url, intentos=8):
    ultimo = None
    for i in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            ultimo = e
            if e.code == 429 or e.code >= 500:
                # rate limit o error transitorio del servidor
                time.sleep(20 * (i + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # errores transitorios de red (SSL handshake, DNS, etc.)
            ultimo = e
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"Open-Meteo: fallo persistente tras {intentos} "
                       f"intentos: {ultimo}")


def weather_cells():
    """Features climáticas de HOY en las 9 celdas de 0.1 grados."""
    cells = {}
    for lat in interp.LATS:
        for lon in interp.LONS:
            url = ("https://api.open-meteo.com/v1/forecast?"
                   f"latitude={lat}&longitude={lon}&past_days=35&forecast_days=1"
                   f"&daily={DAILY_VARS}&hourly={HOURLY_VARS}"
                   "&timezone=America%2FBogota")
            resp = http_json(url)
            d = resp["daily"]
            df = pd.DataFrame(d)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").astype(float)
            p = df["precipitation_sum"].fillna(0.0)
            hoy = df.index[-1]
            rain = (p >= WET_MM)
            dias = 0
            for day in reversed(rain.index):
                if rain[day]:
                    break
                dias += 1

            # patrón vespertino de Cali: viento y sequedad 12:00-18:00 de HOY
            hr = pd.DataFrame(resp["hourly"])
            hr["time"] = pd.to_datetime(hr["time"])
            tarde = hr[(hr["time"].dt.date == hoy.date())
                       & (hr["time"].dt.hour >= 12) & (hr["time"].dt.hour <= 18)]

            cells[(lat, lon)] = {
                "tmax": df.loc[hoy, "temperature_2m_max"],
                "tmean": df.loc[hoy, "temperature_2m_mean"],
                "precip": p.loc[hoy],
                "precip_3d": p.tail(3).sum(),
                "precip_7d": p.tail(7).sum(),
                "precip_30d": p.tail(30).sum(),
                "dias_sin_lluvia": min(dias, 60),
                "viento_max": df.loc[hoy, "wind_speed_10m_max"],
                "radiacion": df.loc[hoy, "shortwave_radiation_sum"],
                "et0": df.loc[hoy, "et0_fao_evapotranspiration"],
                "humedad_min": df.loc[hoy, "relative_humidity_2m_min"],
                "humedad_media": df.loc[hoy, "relative_humidity_2m_mean"],
                "viento_tarde": float(tarde["wind_speed_10m"].max()),
                "humedad_tarde": float(tarde["relative_humidity_2m"].min()),
                "elev_modelo": float(resp.get("elevation", 0.0)),
            }
            time.sleep(3)
    return cells, str(hoy.date())


def interp_met(cells, lats, lons, elev_puntos=None):
    """Interpola bilinealmente cada variable sobre coordenadas arbitrarias.

    Con elev_puntos (m), desescala la temperatura por elevación real: las
    celdas ERA5 difieren hasta 2000 m entre sí y la interpolación horizontal
    sola produce rampas rectas que no siguen la topografía."""
    out = {}
    for f in FEATS_MET:
        vals = {k: v[f] for k, v in cells.items()}
        out[f] = interp.bilinear_frame(vals, lats, lons)
    df = pd.DataFrame(out)
    if elev_puntos is not None:
        elevs = {k: v["elev_modelo"] for k, v in cells.items()}
        elev_mod = interp.bilinear_frame(elevs, lats, lons)
        dT = LAPSE * (np.asarray(elev_puntos, dtype=float) - elev_mod)
        dT = np.where(np.isfinite(dT), dT, 0.0)
        df["tmax"] = df["tmax"] + dT
        df["tmean"] = df["tmean"] + dT
    return df


# --------------------------------------------------- corrección por estaciones
def station_corrections(grid, cells, fecha, resumen):
    """Campos de corrección (delta tmax, razón precip) con peso gaussiano.

    Devuelve (delta_t, razon_p, info) donde delta_t/razon_p son arrays por
    celda de la grilla. Se activa solo con historia suficiente."""
    tr = Transformer.from_crs(paths.CRS_GEO, paths.CRS_M, always_xy=True)

    # 1. registrar los valores del modelo de HOY en cada estación (histórico)
    rows = []
    for cod, info in stations.ESTACIONES.items():
        m = interp_met(cells, np.array([info["lat"]]), np.array([info["lon"]]),
                       elev_puntos=np.array([info["elev"]]))
        rows.append({"fecha": fecha, "estacion": cod,
                     "tmax_mod": round(float(m["tmax"][0]), 2),
                     "precip_mod": round(float(m["precip"][0]), 2)})
    hist_m = pd.DataFrame(rows)
    if os.path.exists(MODEL_AT_ST):
        prev = pd.read_csv(MODEL_AT_ST, dtype={"estacion": str})
        hist_m = (pd.concat([prev, hist_m], ignore_index=True)
                  .drop_duplicates(subset=["fecha", "estacion"], keep="last"))
    hist_m.to_csv(MODEL_AT_ST, index=False)

    # 2. factores de sesgo por estación (requiere días completos acumulados)
    hm = hist_m.copy()
    hm["fecha"] = pd.to_datetime(hm["fecha"]).dt.date
    fact = stations.factores_correccion(hm)

    delta_t = np.zeros(len(grid))
    razon_p = np.ones(len(grid))
    if fact:
        gx, gy = grid["x"].to_numpy(), grid["y"].to_numpy()
        sw = np.zeros(len(grid))
        sd = np.zeros(len(grid))
        sr = np.zeros(len(grid))
        for cod, f in fact.items():
            info = stations.ESTACIONES[cod]
            ex, ey = tr.transform(info["lon"], info["lat"])
            w = np.exp(-((gx - ex) ** 2 + (gy - ey) ** 2) / (2 * SIGMA_M ** 2))
            sw += w
            sd += w * f["delta_tmax"]
            sr += w * f["razon_precip"]
        delta_t = sd / (sw + W0)
        razon_p = (sr + W0 * 1.0) / (sw + W0)
    return delta_t, razon_p, fact


def write_estaciones_geojson(resumen, cells):
    feats = []
    for e in resumen:
        elev = stations.ESTACIONES.get(e["codigo"], {}).get("elev")
        m = interp_met(cells, np.array([e["lat"]]), np.array([e["lon"]]),
                       elev_puntos=(np.array([elev]) if elev else None))
        props = dict(e)
        props["modelo_tmax"] = round(float(m["tmax"][0]), 1)
        props["modelo_precip_hoy"] = round(float(m["precip"][0]), 1)
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point",
                                   "coordinates": [e["lon"], e["lat"]]},
                      "properties": props})
    with open(os.path.join(WEBDATA, "estaciones.geojson"), "w",
              encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False)


# ------------------------------------------------------------------- salidas
def make_overlay(grid, res=100.0):
    xs = np.sort(grid["x"].unique())
    ys = np.sort(grid["y"].unique())
    ix = ((grid["x"] - xs[0]) / res).round().astype(int)
    iy = ((grid["y"] - ys[0]) / res).round().astype(int)
    h, w = len(ys), len(xs)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    for nivel, rgba in COLORS.items():
        m = grid["nivel"] == nivel
        img[iy[m], ix[m]] = rgba
    img = img[::-1]
    png = Image.fromarray(img, "RGBA")
    png = png.resize((w * 3, h * 3), Image.NEAREST)
    png.save(os.path.join(WEBDATA, "alerta_overlay.png"))

    tr = Transformer.from_crs(paths.CRS_M, paths.CRS_GEO, always_xy=True)
    lon0, lat0 = tr.transform(xs[0] - res / 2, ys[0] - res / 2)
    lon1, lat1 = tr.transform(xs[-1] + res / 2, ys[-1] + res / 2)
    return [[lat0, lon0], [lat1, lon1]]


def archivar_overlay(generado):
    """Guarda una copia del mapa de alerta en historico/ (retención: 7 días).

    El nombre lleva la fecha-hora de Colombia (alerta_2026-07-09_1200.png).
    Los archivos con más de 7 días se eliminan; el commit del workflow
    persiste tanto las adiciones como los borrados en el repositorio."""
    hist_dir = os.path.join(paths.PROJ, "historico")
    os.makedirs(hist_dir, exist_ok=True)

    stamp = generado.replace(" ", "_").replace(":", "")
    src = Image.open(os.path.join(WEBDATA, "alerta_overlay.png")).convert("RGBA")
    fondo = Image.new("RGBA", src.size, (255, 255, 255, 255))
    fondo.alpha_composite(src)
    img = fondo.convert("RGB")
    try:
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        texto = f"SAT Incendios Cali - {generado} (hora Colombia)"
        d.rectangle([0, 0, 8 + 7 * len(texto), 26], fill=(13, 59, 46))
        d.text((6, 6), texto, fill=(255, 255, 255))
    except Exception:
        pass
    img.save(os.path.join(hist_dir, f"alerta_{stamp}.png"))

    limite = (dt.datetime.strptime(generado[:10], "%Y-%m-%d")
              - dt.timedelta(days=7))
    borrados = 0
    for f in os.listdir(hist_dir):
        if f.startswith("alerta_") and f.endswith(".png"):
            try:
                fd = dt.datetime.strptime(f[7:17], "%Y-%m-%d")
            except ValueError:
                continue
            if fd < limite:
                os.remove(os.path.join(hist_dir, f))
                borrados += 1
    print(f"histórico: guardado alerta_{stamp}.png "
          f"({borrados} antiguos eliminados)")


def resumen_corregimientos(grid):
    stats = (grid.dropna(subset=["zona"]).groupby(["zona", "nivel"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["ALTA", "MEDIA", "BAJA"], fill_value=0))
    tot = stats.sum(axis=1).replace(0, 1)
    pct = (stats.div(tot, axis=0) * 100).round(1)
    dominante = stats.idxmax(axis=1)

    with open(os.path.join(WEBDATA, "zonas_base.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        nombre = feat["properties"]["nombre"]
        feat["properties"] = {
            "nombre": nombre,
            "pct_alta": float(pct["ALTA"].get(nombre, 0)),
            "pct_media": float(pct["MEDIA"].get(nombre, 0)),
            "pct_baja": float(pct["BAJA"].get(nombre, 0)),
            "nivel_dominante": str(dominante.get(nombre, "BAJA")),
        }
    with open(os.path.join(WEBDATA, "corregimientos.geojson"), "w",
              encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)
    return stats


def firms_key():
    """MAP_KEY de NASA FIRMS: variable de entorno o data/firms_key.txt
    (el archivo está en .gitignore para no publicar la clave)."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    fp = os.path.join(paths.DATA, "firms_key.txt")
    if os.path.exists(fp):
        with open(fp) as f:
            return f.read().strip()
    return ""


def hotspots_firms():
    key = firms_key()
    fp = os.path.join(WEBDATA, "hotspots.geojson")
    if not key:
        if not os.path.exists(fp):
            with open(fp, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
        return 0
    feats = []
    for producto in ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]:
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
               f"{producto}/-76.75,3.25,-76.44,3.56/2")
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                df = pd.read_csv(io.StringIO(r.read().decode()))
        except Exception as e:
            print(f"  aviso FIRMS {producto}: {e}")
            continue
        for row in df.itertuples():
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [float(row.longitude),
                                             float(row.latitude)]},
                "properties": {"fecha": str(row.acq_date),
                               "hora": f"{int(row.acq_time):04d}",
                               "satelite": str(row.satellite),
                               "frp": float(getattr(row, "frp", 0) or 0)}})
    with open(fp, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return len(feats)


def main():
    grid = pd.read_parquet(os.path.join(paths.DATA, "grid_static.parquet"))
    modelo = joblib.load(os.path.join(paths.MODELS, "modelo_alerta.joblib"))
    with open(os.path.join(paths.MODELS, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    print("Descargando clima (9 celdas)...")
    cells, fecha = weather_cells()

    print("Descargando estaciones IDEAM...")
    try:
        raw = stations.fetch_raw()
        stations.update_history(raw)
        resumen = stations.resumen_actual(raw)
    except Exception as e:
        print(f"  aviso: estaciones no disponibles hoy: {e}")
        resumen = []

    met = interp_met(cells, grid["lat"].to_numpy(), grid["lon"].to_numpy(),
                     elev_puntos=grid["elevacion"].to_numpy())

    # La corrección de sesgo está DESACTIVADA por defecto: el modelo se entrenó
    # con ERA5, así que su sesgo sistemático (validado en 06_validate_era5.py:
    # +2078 mm/año de lluvia, -2.5 °C) ya está absorbido en los umbrales.
    # Corregir solo la predicción rompería esa coherencia. El histórico de
    # estaciones y del modelo se registra siempre; cuando haya serie suficiente
    # para reentrenar con campos corregidos, activar con CORRECCION_ESTACIONES=1.
    delta_t, razon_p, fact = station_corrections(grid, cells, fecha, resumen)
    aplicar = os.environ.get("CORRECCION_ESTACIONES", "0") == "1"
    if aplicar and fact:
        met["tmax"] = met["tmax"] + delta_t
        met["tmean"] = met["tmean"] + delta_t
        for c in ["precip", "precip_3d", "precip_7d", "precip_30d"]:
            met[c] = met[c] * razon_p
        print("Corrección por estaciones ACTIVA:", fact)
    else:
        fact = {}
        print("Corrección por estaciones desactivada (registrando historia; "
              "ver nota metodológica en README).")

    met.index = grid.index
    X = pd.concat([grid, met], axis=1)[meta["features"]]
    prob = modelo.predict_proba(X)[:, 1]
    grid["prob"] = prob
    grid["nivel"] = np.where(prob >= meta["umbral_alta"], "ALTA",
                             np.where(prob >= meta["umbral_media"], "MEDIA",
                                      "BAJA"))

    tz_col = dt.timezone(dt.timedelta(hours=-5))  # Colombia: UTC-5 fijo
    generado = dt.datetime.now(tz_col).strftime("%Y-%m-%d %H:%M")

    bounds = make_overlay(grid)
    archivar_overlay(generado)
    stats = resumen_corregimientos(grid)
    if resumen:
        write_estaciones_geojson(resumen, cells)
    n_hs = hotspots_firms()

    wmean = met.mean()
    dist = grid["nivel"].value_counts()
    lluvia_obs = ([e.get("precip_mm") for e in resumen
                   if e.get("precip_mm") is not None] or [None])
    out_meta = {
        "fecha": fecha,
        "generado": generado,
        "bounds": bounds,
        "celdas": {"total": int(len(grid)),
                   "ALTA": int(dist.get("ALTA", 0)),
                   "MEDIA": int(dist.get("MEDIA", 0)),
                   "BAJA": int(dist.get("BAJA", 0))},
        "clima": {"tmax": round(float(wmean["tmax"]), 1),
                  "precip_hoy": round(float(wmean["precip"]), 1),
                  "precip_30d": round(float(wmean["precip_30d"]), 1),
                  "dias_sin_lluvia": round(float(wmean["dias_sin_lluvia"]), 1),
                  "viento_max": round(float(wmean["viento_max"]), 1),
                  "radiacion": round(float(wmean["radiacion"]), 1),
                  "humedad_min": round(float(wmean["humedad_min"]), 0),
                  "viento_tarde": round(float(np.mean(
                      [c["viento_tarde"] for c in cells.values()])), 1),
                  "humedad_tarde": round(float(np.mean(
                      [c["humedad_tarde"] for c in cells.values()])), 0)},
        "estaciones": {
            "n": len(resumen),
            "lluvia_observada_mm": (round(float(np.mean(
                [v for v in lluvia_obs if v is not None])), 1)
                if any(v is not None for v in lluvia_obs) else None),
            "ventana": resumen[0]["ventana_desde"] if resumen else None,
            "correccion_activa": bool(fact),
        },
        "hotspots_48h": n_hs,
        "modelo": {"auc_cv": meta["auc_cv_5fold"],
                   "auc_holdout": meta["auc_holdout_2021_2023"],
                   "periodo": meta["periodo"], "n_incendios": meta["n_incendios"]},
    }
    with open(os.path.join(WEBDATA, "alerta_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, ensure_ascii=False)

    print(f"OK {fecha}: ALTA={dist.get('ALTA',0)} MEDIA={dist.get('MEDIA',0)} "
          f"BAJA={dist.get('BAJA',0)}  estaciones={len(resumen)}  hotspots={n_hs}")
    print(stats.to_string())


if __name__ == "__main__":
    main()
