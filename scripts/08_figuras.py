# -*- coding: utf-8 -*-
"""Figuras estáticas para la página de metodología (web/metodologia.html):
mapas de las variables estáticas, mapa de incendios históricos, y el JSON
público con métricas del modelo. Se corre tras cada reentrenamiento."""
import json
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paths

IMG = os.path.join(paths.WEB, "img")
os.makedirs(IMG, exist_ok=True)
RES = 100.0

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#8a949e", "text.color": "#1c2430",
                     "axes.labelcolor": "#4a5560", "xtick.color": "#8a949e",
                     "ytick.color": "#8a949e"})


def grid_to_image(grid, col):
    xs = np.sort(grid["x"].unique())
    ys = np.sort(grid["y"].unique())
    ix = ((grid["x"] - xs[0]) / RES).round().astype(int)
    iy = ((grid["y"] - ys[0]) / RES).round().astype(int)
    img = np.full((len(ys), len(xs)), np.nan)
    img[iy, ix] = grid[col]
    return img[::-1], (xs[0], xs[-1], ys[0], ys[-1])


def mapa(grid, col, titulo, cmap, unidad, fname, vmax=None):
    img, ext = grid_to_image(grid, col)
    fig, ax = plt.subplots(figsize=(5.4, 5.6), dpi=115)
    im = ax.imshow(img, cmap=cmap, extent=ext, vmax=vmax, interpolation="nearest")
    ax.set_title(titulo, fontsize=11, loc="left", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(unidad)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, fname), transparent=False,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("OK", fname)


def mapa_incendios():
    zonas = gpd.read_file(os.path.join(paths.WEB, "data", "zonas_base.geojson"))
    fuegos = gpd.read_file(os.path.join(paths.DATA, "incendios_historicos.geojson"))
    fig, ax = plt.subplots(figsize=(5.4, 5.6), dpi=115)
    zonas.plot(ax=ax, facecolor="#eef1f4", edgecolor="#8a949e", linewidth=0.7)
    fuegos.plot(ax=ax, color="#c62828", markersize=14, alpha=0.85,
                edgecolor="white", linewidth=0.4)
    ax.set_title(f"Incendios históricos 2010–2023 ({len(fuegos)} puntos calientes)",
                 fontsize=11, loc="left", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "mapa_incendios.png"), facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("OK mapa_incendios.png")


def main():
    grid = pd.read_parquet(os.path.join(paths.DATA, "grid_static.parquet"))
    mapa(grid, "elevacion", "Elevación del terreno (DTM radar 2.5 m)",
         "Blues", "m s.n.m.", "mapa_elevacion.png")
    mapa(grid, "pendiente", "Pendiente del terreno",
         "Oranges", "grados", "mapa_pendiente.png", vmax=45)
    mapa(grid, "SUSC", "Susceptibilidad de la vegetación (tipo × duración × carga)",
         "Reds", "calificación IDEAM", "mapa_susceptibilidad.png")
    grid["dist_vias_km"] = grid["dist_vias"] / 1000
    mapa(grid, "dist_vias_km", "Distancia a vías (accesibilidad)",
         "Purples", "km", "mapa_vias.png", vmax=3)
    mapa_incendios()

    # métricas públicas para que metodologia.html se mantenga sincronizada
    with open(os.path.join(paths.MODELS, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    val_fp = os.path.join(paths.MODELS, "validacion_era5.json")
    val = {}
    if os.path.exists(val_fp):
        with open(val_fp, encoding="utf-8") as f:
            val = json.load(f)
    with open(os.path.join(paths.WEB, "data", "modelo_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({"modelo": meta, "validacion_era5": val}, f,
                  ensure_ascii=False, indent=1)
    print("OK modelo_meta.json")


if __name__ == "__main__":
    main()
