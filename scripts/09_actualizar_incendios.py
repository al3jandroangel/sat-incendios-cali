# -*- coding: utf-8 -*-
"""Aprendizaje continuo del SAT: incorpora al Random Forest cada incendio
nuevo (puntos calientes NASA FIRMS y reportes en campo) y reentrena.

  python 09_actualizar_incendios.py                 # busca FIRMS, reentrena si hay nuevos
  python 09_actualizar_incendios.py --agregar 2026-07-13 3.4386 -76.5527 "Reporte"
  python 09_actualizar_incendios.py --auto          # + git commit/push (para el Plan B)
  python 09_actualizar_incendios.py --forzar        # reentrena aunque no haya nuevos

Deduplicación: un candidato se descarta si ya existe un incendio registrado a
menos de 1.5 km y ±2 días (los incendios grandes generan varios píxeles VIIRS).
Se descartan detecciones FIRMS con confianza 'l' (low).
"""
import io
import json
import os
import subprocess
import sys
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
import paths

NUEVOS = os.path.join(paths.DATA, "incendios_nuevos.csv")
VERIF = os.path.join(paths.DATA, "verificacion_predicciones.csv")
DIST_M = 1500.0
DIAS = 2
# Colores de los PNG archivados: el overlay RGBA (alfa 235/200) compuesto
# sobre blanco aclara los tonos, así que se compara por color más cercano.
COLORES_REF = {"ALTA": (202, 57, 57), "MEDIA": (250, 175, 54),
               "BAJA": (91, 153, 94), None: (255, 255, 255)}
ORDEN = {"BAJA": 0, "MEDIA": 1, "ALTA": 2}


def color_a_nivel(px):
    mejor, dmin = None, 1e9
    for nivel, ref in COLORES_REF.items():
        d = sum((int(a) - b) ** 2 for a, b in zip(px, ref))
        if d < dmin:
            mejor, dmin = nivel, d
    return mejor if dmin < 3600 else None  # tolerancia ~60 por canal


def firms_key():
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    fp = os.path.join(paths.DATA, "firms_key.txt")
    if os.path.exists(fp):
        with open(fp) as f:
            return f.read().strip()
    return ""


def registrados():
    """Todos los incendios ya conocidos (históricos + nuevos), en metros."""
    hist = gpd.read_file(os.path.join(paths.DATA, "incendios_historicos.geojson"))
    dfs = [hist[["fecha", "geometry"]]]
    if os.path.exists(NUEVOS):
        n = pd.read_csv(NUEVOS)
        if len(n):
            dfs.append(gpd.GeoDataFrame(
                n[["fecha"]], geometry=[Point(xy) for xy in
                                        zip(n["lon"], n["lat"])],
                crs=paths.CRS_GEO))
    g = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=paths.CRS_GEO)
    g["fecha_dt"] = pd.to_datetime(g["fecha"], format="mixed")
    return g.to_crs(paths.CRS_M)


def es_duplicado(reg, lat, lon, fecha):
    from pyproj import Transformer
    tr = Transformer.from_crs(paths.CRS_GEO, paths.CRS_M, always_xy=True)
    x, y = tr.transform(lon, lat)
    dt = np.abs((reg["fecha_dt"] - pd.Timestamp(fecha)).dt.days)
    cerca_t = reg[dt <= DIAS]
    if not len(cerca_t):
        return False
    d = np.hypot(cerca_t.geometry.x - x, cerca_t.geometry.y - y)
    return bool((d < DIST_M).any())


def buscar_firms(dias=10):
    """Hotspots VIIRS de los últimos `dias` dentro del distrito."""
    key = firms_key()
    if not key:
        print("sin FIRMS_MAP_KEY; solo reportes manuales")
        return []
    zonas = gpd.read_file(os.path.join(paths.WEB, "data", "zonas_base.geojson"))
    distrito = zonas.union_all()
    reg = registrados()
    hallados = []
    for prod in ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]:
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
               f"{prod}/-76.75,3.25,-76.44,3.56/{dias}")
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                df = pd.read_csv(io.StringIO(r.read().decode()))
        except Exception as e:
            print(f"  aviso FIRMS {prod}: {e}")
            continue
        for row in df.itertuples():
            if str(getattr(row, "confidence", "n")).lower().startswith("l"):
                continue  # confianza baja
            if not distrito.contains(Point(row.longitude, row.latitude)):
                continue
            if es_duplicado(reg, row.latitude, row.longitude, row.acq_date):
                continue
            hallados.append({"fecha": str(row.acq_date),
                             "lat": float(row.latitude),
                             "lon": float(row.longitude),
                             "fuente": f"NASA FIRMS {row.satellite}"})
            # evitar duplicados dentro de la misma pasada
            reg = pd.concat([reg, gpd.GeoDataFrame(
                {"fecha": [row.acq_date],
                 "fecha_dt": [pd.Timestamp(row.acq_date)]},
                geometry=gpd.points_from_xy([row.longitude], [row.latitude],
                                            crs=paths.CRS_GEO).to_crs(paths.CRS_M)
            )], ignore_index=True)
    return hallados


def _bounds_overlay():
    """Límites geográficos del overlay (constantes, derivados de la grilla)."""
    from pyproj import Transformer
    g = pd.read_parquet(os.path.join(paths.DATA, "grid_static.parquet"))
    tr = Transformer.from_crs(paths.CRS_M, paths.CRS_GEO, always_xy=True)
    lon0, lat0 = tr.transform(g["x"].min() - 50, g["y"].min() - 50)
    lon1, lat1 = tr.transform(g["x"].max() + 50, g["y"].max() + 50)
    return lat0, lon0, lat1, lon1


def verificar_prediccion(fecha, lat, lon, hora="14:00"):
    """¿Qué nivel de alerta estaba PUBLICADO en ese lugar cuando inició el
    incendio? Lee el mapa archivado más reciente anterior al evento
    (historico/, retención 7 días). Devuelve (mapa, nivel_punto, nivel_anillo)."""
    from PIL import Image
    hist = os.path.join(paths.PROJ, "historico")
    objetivo = f"{fecha}_{hora.replace(':', '')}"
    candidatos = sorted(f for f in os.listdir(hist)
                        if f.startswith("alerta_") and f.endswith(".png")
                        and f[7:22].replace(".png", "") <= objetivo)
    if not candidatos:
        return None, None, None
    mapa = candidatos[-1]
    img = np.array(Image.open(os.path.join(hist, mapa)).convert("RGB"))
    lat0, lon0, lat1, lon1 = _bounds_overlay()
    h, w = img.shape[:2]
    # el PNG archivado lleva una franja de título de 26 px arriba sobre el
    # mismo lienzo del overlay; la geometría del mapa no cambia
    col = int((lon - lon0) / (lon1 - lon0) * w)
    fila = int((lat1 - lat) / (lat1 - lat0) * h)

    def nivel_px(f, c):
        if not (0 <= f < h and 0 <= c < w):
            return None
        return color_a_nivel(tuple(img[f, c]))

    punto = nivel_px(fila, col)
    anillo = [n for df in (-3, 0, 3) for dc in (-3, 0, 3)
              if (n := nivel_px(fila + df, col + dc))]
    anillo_max = (max(anillo, key=lambda n: ORDEN[n]) if anillo else None)
    return mapa, punto, anillo_max


def registrar_verificacion(filas):
    """Anexa el contraste predicción-vs-evento y publica el resumen web."""
    reg = (pd.read_csv(VERIF) if os.path.exists(VERIF)
           else pd.DataFrame(columns=["fecha", "lat", "lon", "fuente", "mapa",
                                      "nivel_punto", "nivel_anillo"]))
    for f in filas:
        mapa, punto, anillo = verificar_prediccion(
            f["fecha"], f["lat"], f["lon"], f.get("hora", "14:00"))
        reg = pd.concat([reg, pd.DataFrame([{**{k: f[k] for k in
                        ("fecha", "lat", "lon", "fuente")},
                        "mapa": mapa, "nivel_punto": punto,
                        "nivel_anillo": anillo}])], ignore_index=True)
    reg = reg.drop_duplicates(subset=["fecha", "lat", "lon"])
    reg.to_csv(VERIF, index=False)
    publicar_verificacion(reg)
    return reg


def publicar_verificacion(reg=None):
    """web/data/verificacion.json: desempeño del sistema en operación."""
    if reg is None:
        if not os.path.exists(VERIF):
            return
        reg = pd.read_csv(VERIF)
    con_mapa = reg.dropna(subset=["nivel_punto"])
    resumen = {
        "desde": "2026-07-09",
        "total_eventos": int(len(reg)),
        "con_mapa_archivado": int(len(con_mapa)),
        "punto_alta": int((con_mapa["nivel_punto"] == "ALTA").sum()),
        "punto_media_o_mas": int(con_mapa["nivel_punto"].isin(["MEDIA", "ALTA"]).sum()),
        "anillo_media_o_mas": int(con_mapa["nivel_anillo"].isin(["MEDIA", "ALTA"]).sum()),
        "eventos": reg.where(pd.notna(reg), None).to_dict("records"),
    }
    with open(os.path.join(paths.WEB, "data", "verificacion.json"), "w",
              encoding="utf-8") as fo:
        json.dump(resumen, fo, ensure_ascii=False, indent=1)
    print(f"verificación: {resumen['punto_media_o_mas']}/{len(con_mapa)} "
          "eventos con alerta MEDIA+ en el punto")


def agregar(filas):
    df = (pd.read_csv(NUEVOS) if os.path.exists(NUEVOS)
          else pd.DataFrame(columns=["fecha", "lat", "lon", "fuente"]))
    df = pd.concat([df, pd.DataFrame(filas)], ignore_index=True)
    df = df.drop_duplicates(subset=["fecha", "lat", "lon"])
    df.to_csv(NUEVOS, index=False)
    return df


def reentrenar():
    sdir = os.path.dirname(os.path.abspath(__file__))
    for script in ["02_build_dataset.py", "03_train_model.py",
                   "08_figuras.py", "05_predict_daily.py"]:
        print(f"== {script}")
        r = subprocess.run([sys.executable, os.path.join(sdir, script)],
                           cwd=sdir)
        if r.returncode != 0:
            raise RuntimeError(f"{script} falló (código {r.returncode})")


def git_publicar():
    cwd = paths.PROJ
    def g(*args):
        return subprocess.run(["git"] + list(args), cwd=cwd).returncode
    g("add", "-A")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd).returncode:
        g("commit", "-m", "Aprendizaje continuo: nuevos incendios incorporados y modelo reentrenado")
        g("pull", "--rebase")
        g("push")


def main():
    args = sys.argv[1:]
    nuevos = []
    if "--agregar" in args:
        i = args.index("--agregar")
        nuevos.append({"fecha": args[i + 1], "lat": float(args[i + 2]),
                       "lon": float(args[i + 3]),
                       "fuente": args[i + 4] if len(args) > i + 4 else "manual"})
    nuevos += buscar_firms()
    if nuevos:
        agregar(nuevos)
        registrar_verificacion(nuevos)
        print(f"nuevos incendios registrados: {len(nuevos)}")
        for n in nuevos:
            print("  +", n["fecha"], n["lat"], n["lon"], "-", n["fuente"])
    else:
        publicar_verificacion()  # mantiene fresco el resumen web
    if nuevos or "--forzar" in args:
        reentrenar()
        if "--auto" in args:
            git_publicar()
        print("OK: modelo reentrenado con el registro actualizado")
    else:
        print("sin incendios nuevos; el modelo vigente sigue válido")


if __name__ == "__main__":
    main()
