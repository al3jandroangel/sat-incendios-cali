# -*- coding: utf-8 -*-
"""Interpolación bilineal de los campos de clima ERA5/Open-Meteo (celdas de
0.1 grados) para eliminar los bordes rectos entre celdas.

La malla local que cubre Cali es 3x3: lat {3.3, 3.4, 3.5}, lon {-76.7, -76.6,
-76.5}. Las coordenadas se recortan a esa malla (extrapolación plana en los
bordes del distrito, diferencia < 0.05 grados).
"""
import numpy as np

LATS = (3.3, 3.4, 3.5)
LONS = (-76.7, -76.6, -76.5)


def _clip(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def bilinear(values, lat, lon):
    """Interpola sobre la malla 3x3.

    values: dict {(lat_celda, lon_celda): valor} con las 9 celdas.
    lat, lon: escalares o arrays numpy.
    """
    lat = _clip(np.asarray(lat, dtype=float), LATS[0], LATS[-1])
    lon = _clip(np.asarray(lon, dtype=float), LONS[0], LONS[-1])

    la0 = _clip(np.floor(lat * 10) / 10, LATS[0], LATS[-2])
    lo0 = _clip(np.floor(lon * 10) / 10, LONS[0], LONS[-2])
    la1, lo1 = la0 + 0.1, lo0 + 0.1
    t = (lat - la0) / 0.1
    u = (lon - lo0) / 0.1

    def v(la, lo):
        la_r = np.round(la, 1)
        lo_r = np.round(lo, 1)
        if np.isscalar(la_r) or la_r.ndim == 0:
            return values[(float(la_r), float(lo_r))]
        return np.array([values[(a, o)] for a, o in zip(la_r.ravel(), lo_r.ravel())
                         ]).reshape(la_r.shape)

    return ((1 - t) * (1 - u) * v(la0, lo0) + (1 - t) * u * v(la0, lo1)
            + t * (1 - u) * v(la1, lo0) + t * u * v(la1, lo1))


def bilinear_frame(cell_values, lats, lons):
    """Versión vectorizada para muchas coordenadas.

    cell_values: dict {(lat, lon): valor escalar}.
    lats, lons: arrays. Devuelve array interpolado.
    """
    lut = np.full((len(LATS), len(LONS)), np.nan)
    for i, la in enumerate(LATS):
        for j, lo in enumerate(LONS):
            if (la, lo) in cell_values:
                lut[i, j] = cell_values[(la, lo)]

    lat = _clip(np.asarray(lats, dtype=float), LATS[0], LATS[-1])
    lon = _clip(np.asarray(lons, dtype=float), LONS[0], LONS[-1])
    fi = _clip((lat - LATS[0]) / 0.1, 0, len(LATS) - 1)
    fj = _clip((lon - LONS[0]) / 0.1, 0, len(LONS) - 1)
    i0 = _clip(np.floor(fi).astype(int), 0, len(LATS) - 2)
    j0 = _clip(np.floor(fj).astype(int), 0, len(LONS) - 2)
    t = fi - i0
    u = fj - j0
    return ((1 - t) * (1 - u) * lut[i0, j0] + (1 - t) * u * lut[i0, j0 + 1]
            + t * (1 - u) * lut[i0 + 1, j0] + t * u * lut[i0 + 1, j0 + 1])
