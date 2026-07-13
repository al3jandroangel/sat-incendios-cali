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
DIST_M = 1500.0
DIAS = 2


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
        print(f"nuevos incendios registrados: {len(nuevos)}")
        for n in nuevos:
            print("  +", n["fecha"], n["lat"], n["lon"], "-", n["fuente"])
    if nuevos or "--forzar" in args:
        reentrenar()
        if "--auto" in args:
            git_publicar()
        print("OK: modelo reentrenado con el registro actualizado")
    else:
        print("sin incendios nuevos; el modelo vigente sigue válido")


if __name__ == "__main__":
    main()
