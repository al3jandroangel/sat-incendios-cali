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

El sistema se actualiza **cada 6 horas** (listo hacia las 00:00, 06:00, 12:00
y 18:00 hora Colombia), con tres capas de garantía porque los crons de GitHub
se retrasan u omiten con frecuencia:

1. **Cron principal de GitHub**: 23:43/05:43/11:43/17:43 hora Colombia
   (minutos fuera de pico, antes de la hora objetivo).
2. **Cron de respaldo de GitHub**: 00:13/06:13/12:13/18:13, por si el
   principal se omite (el workflow es idempotente y con grupo de concurrencia).
3. **Plan B — tarea programada de Windows** «SAT Cali Plan B»: ejecuta
   `plan_b_dispatch.bat` cada 6 h (05:50/11:50/17:50/23:50 hora Colombia) y
   dispara el workflow directamente por la API de GitHub, inmune a los
   retrasos de sus crons. Requiere el PC encendido; usa el token del
   Administrador de credenciales de Windows. Nota: la tarea corre en hora del
   PC (Los Ángeles), así que con el cambio de horario de EE.UU. se desplaza
   1 h respecto a Colombia. Para eliminarla:
   `schtasks /delete /tn "SAT Cali Plan B"`.

La actualización del mediodía importa especialmente: en Cali el viento aumenta
después del mediodía y el panel muestra viento máximo y humedad mínima de la
franja 12:00–18:00. Fallback manual: pestaña Actions → "Actualizar alertas
SAT" → Run workflow. El portal vive en
https://al3jandroangel.github.io/sat-incendios-cali/
- **Puntos calientes NASA FIRMS**: activos (VIIRS Suomi-NPP y NOAA-20, últimas
  48 h). El MAP_KEY se lee de `data/firms_key.txt` en local (excluido del
  repositorio por .gitignore) y del secret `FIRMS_MAP_KEY` en GitHub Actions.

## Aprendizaje continuo (v5)

Cada incendio nuevo se incorpora al entrenamiento del Random Forest:

- `data/incendios_nuevos.csv` es el registro incremental (reportes en campo +
  puntos calientes FIRMS). `scripts/09_actualizar_incendios.py` consulta FIRMS
  (VIIRS SNPP y NOAA-20, confianza nominal/alta, dentro del distrito), descarta
  duplicados (<1.5 km y ±2 días de un incendio ya registrado) y, si hay nuevos,
  reconstruye el dataset (con clima ERA5/pronóstico también para fechas
  posteriores a 2023), reentrena, regenera figuras y mapa, y publica.
- El Plan B ejecuta este ciclo **automáticamente cada 6 horas** antes de cada
  actualización, así que todo punto caliente FIRMS nuevo queda involucrado en
  el modelo a más tardar 6 h después de ser detectado.
- Reporte manual: `python scripts/09_actualizar_incendios.py --agregar
  AAAA-MM-DD lat lon "fuente"`.

## Verificación en operación y visitas (v6)

- **Aciertos del sistema**: cada incendio nuevo se contrasta automáticamente con
  el mapa que estaba publicado al momento de su inicio (usando el archivo
  `historico/`): se registra el nivel de alerta en el punto exacto y en un
  anillo de 300 m en `data/verificacion_predicciones.csv` (inmutable, sin
  edición) y el resumen se publica en la sección «Desempeño en operación» de
  metodologia.html.
- **Visitas a la página**: GitHub Pages no trae analítica. La integración con
  GoatCounter (gratuito, sin cookies, cumple privacidad) está lista en ambas
  páginas: crear cuenta en https://www.goatcounter.com/signup, elegir código de
  sitio y descomentar la línea `data-goatcounter` en `web/index.html` y
  `web/metodologia.html`. Complemento: GitHub → Insights → Traffic muestra las
  visitas al repositorio (últimos 14 días).

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
  bilinealmente entre celdas y la temperatura se **desescala orográficamente**
  (gradiente −6.5 °C/km entre la elevación real del DEM de 10 m y la elevación
  del modelo ERA5 interpolada — las celdas vecinas difieren hasta 2000 m), de
  modo que el mapa sigue la topografía y no la malla del modelo.
- Capas informativas del portal: estaciones meteorológicas IDEAM (observado vs
  modelo en el popup) y estaciones de bomberos (OpenStreetMap,
  `web/data/bomberos.geojson`).
- La muestra de entrenamiento es pequeña (105 incendios); conviene reentrenar cada
  año agregando los nuevos eventos a la matriz histórica.
- 6 registros de la matriz histórica caen fuera del distrito (lado Yumbo/La Cumbre)
  y 3 tienen coordenadas/fechas ilegibles; se descartan con advertencia en el log.
