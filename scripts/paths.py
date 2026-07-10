"""Rutas centralizadas del proyecto SAT Incendios Cali."""
import glob
import os

# Carpeta del proyecto: relativa a este archivo, para que el repositorio
# funcione tanto en esta máquina como en un servidor (GitHub Actions, etc.).
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
MODELS = os.path.join(PROJ, "models")
WEB = os.path.join(PROJ, "web")

os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
os.makedirs(WEB, exist_ok=True)

# Datos fuente originales: solo existen en la máquina local. Los scripts de
# preparación (01-04) los necesitan; el diario (05) no.
BASE = r"D:\Proyectos Personales\SAT Incendios\Zonificacion Incendios Forestales"


def _g(pattern):
    m = glob.glob(pattern)
    return m[0] if m else None


MATRIZ_XLSX = _g(os.path.join(BASE, "Matriz Reg. Hist*ricos - PUNTOS CALIENTES.xlsx"))
CORREGIMIENTOS = _g(os.path.join(BASE, "Informaci*n Base", "SHAPES CARTOGRAF*A BASE",
                                 "BASE_CATASTRAL_2022.gdb", "Corregimientos.shp"))
LOCALIDADES = _g(os.path.join(BASE, "Informaci*n Base", "SHAPES CARTOGRAF*A BASE",
                              "BASE_CATASTRAL_2022.gdb",
                              "Modelo_localidades_cali_distrito.shp"))
# DTMCaliOK cubre todo el distrito (DTM_Cali.tif solo la mitad norte)
DTM = _g(os.path.join(BASE, "Informaci*n Base", "Datos Climaticos", "SIG",
                      "DTMCaliOK.tif"))
SUSCEPTIBILIDAD = _g(os.path.join(BASE, "SUSCEPTIBILIDAD DE LA COBERTURA VEGETAL",
                                  "Datos", "Susceptibilidad.shp"))
VIAS = _g(os.path.join(BASE, "Informaci*n Base", "SHAPES CARTOGRAF*A BASE",
                       "JERARQUIZACI*N VIAL CALI", "*Jerarquizacion_vial.shp"))
INCENDIOS_2023 = _g(os.path.join(BASE, "Shape Incendios 2023", "Incendios_Cali.shp"))

# CRS de trabajo: MAGNA-SIRGAS / Colombia West (metros), igual que el DTM
CRS_M = "EPSG:3115"
CRS_GEO = "EPSG:4326"
