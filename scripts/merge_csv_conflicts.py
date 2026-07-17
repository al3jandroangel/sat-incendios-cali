# -*- coding: utf-8 -*-
"""Resuelve los conflictos de rebase en los CSV acumulativos (histórico de
estaciones, modelo en estaciones, hotspots): une ambas versiones y deduplica.
Correr desde la raíz del repo cuando `git pull --rebase` reporte conflicto,
y luego: git rebase --continue"""
import io
import subprocess
import sys

import pandas as pd

CLAVES = {
    "data/stations_history.csv": ["estacion", "sensor", "fecha_hora"],
    "data/model_at_stations.csv": ["fecha", "estacion"],
    "data/hotspots_recientes.csv": ["fecha", "hora", "lat", "lon", "satelite"],
    "data/verificacion_predicciones.csv": ["fecha", "lat", "lon"],
    "data/incendios_nuevos.csv": ["fecha", "lat", "lon"],
}


def main():
    resueltos = 0
    for f, claves in CLAVES.items():
        r = subprocess.run(["git", "ls-files", "-u", f],
                           capture_output=True, text=True)
        if not r.stdout.strip():
            continue
        ours = subprocess.run(["git", "show", f":2:{f}"],
                              capture_output=True, text=True).stdout
        theirs = subprocess.run(["git", "show", f":3:{f}"],
                                capture_output=True, text=True).stdout
        a = pd.read_csv(io.StringIO(ours), dtype=str)
        b = pd.read_csv(io.StringIO(theirs), dtype=str)
        m = (pd.concat([a, b], ignore_index=True)
             .drop_duplicates(subset=[c for c in claves if c in a.columns],
                              keep="last"))
        m.to_csv(f, index=False)
        subprocess.run(["git", "add", f])
        print(f"{f}: {len(a)} + {len(b)} -> {len(m)} filas")
        resueltos += 1
    if not resueltos:
        print("sin conflictos CSV pendientes")
        sys.exit(1)


if __name__ == "__main__":
    main()
