# -*- coding: utf-8 -*-
"""Entrena el modelo de alerta de incendios forestales de Cali.

- Clasificador: RandomForest (probabilidad de ocurrencia de incendio).
- Evaluación: validación cruzada estratificada (5 folds) + holdout temporal
  (entrena 2010-2020, prueba 2021-2023) que actúa como validación con el
  "factor histórico" reciente.
- Niveles de alerta: BAJA / MEDIA / ALTA a partir de la probabilidad calibrada:
    * umbral MEDIA: probabilidad que retiene el 90% de los incendios (sensibilidad 0.9)
    * umbral ALTA : punto de máxima J de Youden (mejor separación)
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(__file__))
import paths

FEATURES = ["elevacion", "pendiente", "ctc", "cdc", "ccc", "SUSC",
            "dist_vias", "dist_vias_prin",
            "tmax", "tmean", "precip", "precip_3d", "precip_7d", "precip_30d",
            "dias_sin_lluvia", "viento_max", "radiacion", "et0",
            "humedad_min", "humedad_media"]


def main():
    df = pd.read_csv(os.path.join(paths.DATA, "dataset_entrenamiento.csv"))
    df = df.dropna(subset=FEATURES)
    X, y = df[FEATURES], df["incendio"].astype(int)
    anio = pd.to_datetime(df["fecha"], format="mixed").dt.year
    print(f"muestras: {len(df)} (incendios={y.sum()}, no-incendio={(y==0).sum()})")

    rf = RandomForestClassifier(
        n_estimators=600, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1)

    # --- 1. Validación cruzada estratificada -------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob_cv = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc_cv = roc_auc_score(y, prob_cv)
    print(f"\nAUC validación cruzada (5-fold): {auc_cv:.3f}")

    # --- 2. Holdout temporal (train<=2020, test 2021-2023) -----------------
    tr, te = anio <= 2020, anio >= 2021
    rf_t = RandomForestClassifier(**rf.get_params())
    rf_t.fit(X[tr], y[tr])
    prob_te = rf_t.predict_proba(X[te])[:, 1]
    auc_te = roc_auc_score(y[te], prob_te)
    print(f"AUC holdout temporal 2021-2023 (n={te.sum()}, "
          f"incendios={y[te].sum()}): {auc_te:.3f}")

    # --- 3. Modelo final calibrado sobre todos los datos -------------------
    modelo = CalibratedClassifierCV(rf, method="isotonic", cv=5)
    modelo.fit(X, y)
    prob_all = modelo.predict_proba(X)[:, 1]

    # Umbrales de alerta sobre las probabilidades out-of-fold (sin sobreajuste):
    # MEDIA retiene el 90% de los incendios, ALTA el 66% (dos tercios) en la
    # menor área posible.
    fpr, tpr, thr = roc_curve(y, prob_cv)
    thr_media = float(thr[np.argmax(tpr >= 0.90)])
    thr_alta = float(thr[np.argmax(tpr >= 0.66)])
    print(f"\nUmbrales de alerta -> MEDIA: p>={thr_media:.3f}  ALTA: p>={thr_alta:.3f}")

    nivel = np.where(prob_cv >= thr_alta, "ALTA",
                     np.where(prob_cv >= thr_media, "MEDIA", "BAJA"))
    tabla = pd.crosstab(nivel, y.values, rownames=["alerta"],
                        colnames=["incendio_real"])
    print("\nDistribución de niveles (probabilidades out-of-fold):")
    print(tabla.to_string())

    pred_bin = (prob_cv >= thr_alta).astype(int)
    print("\nMatriz de confusión (corte ALTA):")
    print(confusion_matrix(y, pred_bin))
    print(classification_report(y, pred_bin, target_names=["no incendio", "incendio"]))

    imp = pd.Series(rf_t.feature_importances_, index=FEATURES).sort_values(
        ascending=False)
    print("Importancia de variables:")
    print(imp.round(3).to_string())

    joblib.dump(modelo, os.path.join(paths.MODELS, "modelo_alerta.joblib"))
    meta = {
        "features": FEATURES,
        "umbral_media": thr_media,
        "umbral_alta": thr_alta,
        "auc_cv_5fold": round(auc_cv, 4),
        "auc_holdout_2021_2023": round(auc_te, 4),
        "n_entrenamiento": int(len(df)),
        "n_incendios": int(y.sum()),
        "periodo": "2010-2023",
        "importancias": {k: round(float(v), 4) for k, v in imp.items()},
    }
    with open(os.path.join(paths.MODELS, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    imp.to_csv(os.path.join(paths.MODELS, "importancia_variables.csv"))
    print(f"\nOK: modelo -> {paths.MODELS}\\modelo_alerta.joblib")


if __name__ == "__main__":
    main()
