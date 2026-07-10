# -*- coding: utf-8 -*-
"""Genera web/data/bomberos.geojson: estaciones de bomberos de OpenStreetMap
(bbox Cali + vecinos) más las adiciones manuales que OSM no tiene."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import paths

# estaciones que faltan en OSM (nombre, lat, lon, operador)
MANUALES = [
    ("Bomberos Cali - Estación 11", 3.379875, -76.544324,
     "Benemérito Cuerpo de Bomberos Voluntarios de Cali"),
]

QUERY = """
[out:json][timeout:60];
(
  node["amenity"="fire_station"](3.20,-76.80,3.62,-76.40);
  way["amenity"="fire_station"](3.20,-76.80,3.62,-76.40);
);
out center tags;
"""


SERVIDORES = ["https://overpass-api.de/api/interpreter",
              "https://overpass.kumi.systems/api/interpreter"]


def consultar():
    ultimo = None
    for _ in range(2):
        for url in SERVIDORES:
            req = urllib.request.Request(
                url, data=urllib.parse.urlencode({"data": QUERY}).encode(),
                headers={"User-Agent":
                         "SAT-Incendios-Cali/1.0 (proyecto academico)"})
            try:
                return json.loads(urllib.request.urlopen(req, timeout=120).read())
            except Exception as e:
                ultimo = e
                print(f"  aviso {url}: {e}")
                time.sleep(10)
    raise RuntimeError(f"Overpass no disponible: {ultimo}")


def main():
    d = consultar()

    feats = []
    for el in d["elements"]:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None:
            continue
        t = el.get("tags", {})
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": {
                "nombre": t.get("name", "Estación de Bomberos"),
                "operador": t.get("operator", ""),
                "direccion": " ".join(filter(None, [t.get("addr:street", ""),
                                                    t.get("addr:housenumber", "")]))}})

    for nombre, lat, lon, oper in MANUALES:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"nombre": nombre, "operador": oper, "direccion": ""}})

    fp = os.path.join(paths.WEB, "data", "bomberos.geojson")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False)
    print(f"OK: {len(feats)} estaciones de bomberos -> {fp}")


if __name__ == "__main__":
    main()
