# SAT Incendios Forestales — Santiago de Cali

Sistema de Alertas Tempranas de incendios forestales para el Distrito Especial de
Santiago de Cali. Combina un modelo de machine learning entrenado con los
incendios históricos 2010–2023 y clima diario en tiempo real, publicado como
geoportal web (similar a satic.cali.gov.co).

## Arquitectura

```
Datos fuente (D:\Proyectos Personales\SAT Incendios\Zonificacion Incendios Forestales)
        │
        ▼  (una sola vez, en esta máquina)
01_parse_fires.py      → data/incendios_historicos.geojson  (106 incendios 2010-2023)
02_build_dataset.py    → data/dataset_entrenamiento.csv     (presencias + ausencias
                                                             + estáticas + clima ERA5)
03_train_model.py      → models/modelo_alerta.joblib + metadata.json
04_make_grid.py        → data/grid_static.parquet           (grilla 100 m, 56.262 celdas)
        │
        ▼  (todos los días — portable, no necesita los shapefiles fuente)
05_predict_daily.py    → web/data/alerta_overlay.png        (mapa alta/media/baja)
                         web/data/corregimientos.geojson    (resumen por zona)
                         web/data/alerta_meta.json          (fecha, clima, estadísticas)
                         web/data/hotspots.geojson          (FIRMS 48 h, opcional)
        │
        ▼
web/index.html         → geoportal Leaflet (archivos 100% estáticos)
```

## Variables del modelo (según componentes de la amenaza)

| Componente        | Variables                                                        | Fuente |
|-------------------|------------------------------------------------------------------|--------|
| Clima (dinámico)  | tmax, tmedia, lluvia día/3d/7d/30d, días sin lluvia, viento máx, radiación, ET0, **humedad relativa mín/media** | Open-Meteo (ERA5 histórico, pronóstico diario) |
| Relieve           | elevación, pendiente                                             | DTM radar 2.5 m (DTMCaliOK.tif) |
| Vegetación        | tipo (ctc), duración (cdc), carga (ccc), susceptibilidad (SUSC)  | Estudio Urrutia 2018 / protocolo IDEAM |
| Accesibilidad     | distancia a vías (todas y principales)                           | Jerarquización vial IDESC |
| Factor histórico  | 106 puntos calientes FIRMS/VIIRS 2010-2023 (etiquetas de entrenamiento y validación temporal) | Matriz Reg. Históricos |

## Desempeño

- Random Forest calibrado (isotónico), 529 muestras (105 incendios / 424 ausencias).
- **AUC 0.876** validación cruzada 5-fold; **AUC 0.897** holdout temporal
  (entrenado 2010-2020, probado contra los incendios reales 2021-2023).
- Umbrales de alerta (sobre probabilidad out-of-fold): MEDIA = sensibilidad 90 %,
  ALTA = máxima J de Youden. El 82 % de los incendios históricos cae en ALTA.

## Uso

```bash
pip install -r requirements.txt

# Reconstruir todo (solo si cambian los datos fuente)
python scripts/01_parse_fires.py
python scripts/02_build_dataset.py     # descarga clima histórico (caché en data/weather_cache)
python scripts/03_train_model.py
python scripts/04_make_grid.py

# Actualización diaria (rápida, ~1 min)
python scripts/05_predict_daily.py

# Ver el geoportal en local
python -m http.server 8123 --directory web
```

## Automatización

El sistema se actualiza **dos veces al día: 6:00 am y 12:00 m** (hora Colombia).
La corrida del mediodía es importante porque en Cali el viento aumenta
considerablemente después del mediodía, elevando el riesgo de propagación; el
panel muestra explícitamente el viento máximo y la humedad mínima de la franja
12:00–18:00.

- **En línea (GitHub Pages)**: repositorio
  https://github.com/al3jandroangel/sat-incendios-cali con workflow programado
  que regenera las alertas, guarda los históricos y publica en
  https://al3jandroangel.github.io/sat-incendios-cali/. Los crons corren a las
  05:43/06:13 y 11:43/12:13 hora Colombia (principal + respaldo): GitHub
  retrasa u omite con frecuencia los crons del minuto :00, así que se programa
  antes de la hora objetivo con un segundo intento después. Si aun así un día
  no corre, se puede lanzar a mano: pestaña Actions → "Actualizar alertas SAT"
  → Run workflow.
- **Local (Windows)**: `run_daily.bat` + Programador de tareas, dos tareas
  (6:00 y 12:00; instrucciones dentro del .bat).
- **Puntos calientes NASA FIRMS**: activos (VIIRS Suomi-NPP y NOAA-20, últimas
  48 h). El MAP_KEY se lee de `data/firms_key.txt` en local (excluido del
  repositorio por .gitignore) y del secret `FIRMS_MAP_KEY` en GitHub Actions.

## Estaciones en tierra y validación de ERA5 (v2)

`06_validate_era5.py` contrasta ERA5/Open-Meteo con observaciones en tierra
(resultados en `models/validacion_era5.json`):

| Contraste | n | Correlación | Sesgo (modelo − obs) |
|---|---|---|---|
| Temperatura diaria, aeropuerto SKCL (GHCN), 2021-2026 | 1684 | 0.61 | −2.5 °C |
| Precipitación anual, est. CALI SEDE IDEAM, 1981-2022 | 42 | 0.40 | **+2078 mm/año** |

ERA5 sobreestima fuertemente la lluvia en el piedemonte de Cali. Por eso el
portal muestra también las observaciones en tiempo casi real de las 3
estaciones IDEAM del distrito (Siloé, Univalle, Base Aérea; dataset abierto
57sv-p2fu de datos.gov.co, latencia ~6-10 h) y cada corrida acumula su
histórico en `data/stations_history.csv`.

**Nota metodológica**: la corrección de sesgo con estaciones
(`CORRECCION_ESTACIONES=1`) está desactivada por defecto. El modelo se entrenó
con ERA5, de modo que el sesgo sistemático ya está absorbido en los umbrales de
alerta; corregir solo la predicción rompería la coherencia
entrenamiento/predicción. La ruta correcta (v3) es conseguir series diarias
históricas de estaciones (exportación DHIME del IDEAM o red telemétrica de la
CVC, que no tienen API pública) y reentrenar con campos corregidos. Con solo 3
estaciones urbanas tampoco es viable una interpolación espacial pura: dejaría
sin control precisamente los Farallones. El sensor de viento de Univalle
reporta >20 m/s sostenidos y se filtra como defectuoso.

## Notas y limitaciones

- Los .ovr del DTM original están corruptos (devuelven 0); los scripts leen a
  resolución nativa. `DTM_Cali.tif` solo cubre la mitad norte; se usa `DTMCaliOK.tif`.
- El clima ERA5 tiene resolución ~10 km (9 celdas cubren el distrito): las alertas
  capturan la variación día a día, y el detalle espacial fino lo aportan las capas
  estáticas (pendiente, combustibles, vías). Los campos se interpolan
  bilinealmente entre celdas para evitar bordes artificiales rectos.
- Capas informativas del portal: estaciones meteorológicas IDEAM (observado vs
  modelo en el popup) y estaciones de bomberos (OpenStreetMap,
  `web/data/bomberos.geojson`).
- La muestra de entrenamiento es pequeña (105 incendios); conviene reentrenar cada
  año agregando los nuevos eventos a la matriz histórica.
- 6 registros de la matriz histórica caen fuera del distrito (lado Yumbo/La Cumbre)
  y 3 tienen coordenadas/fechas ilegibles; se descartan con advertencia en el log.
