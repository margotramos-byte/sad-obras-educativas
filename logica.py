"""Lógica del SAD: construcción de predictores, validación de entradas y predicción.
Los modelos (modelo_costo.joblib, modelo_plazo.joblib) y config_niveles.json se generan en el cuaderno de entrenamiento."""
import json
import os

import joblib
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))

NUM = ['monto_adjudicado', 'reduccion_pct', 'n_ofertantes', 'dias_conv_bp', 'dias_bp_cons',
       'plazo_original_dias', 'anio_inicio']
CAT = ['nivel_gobierno', 'sistema_contratacion', 'tipo_procedimiento_2', 'tipo_proveedor_2', 'provincia']
PREDICTORES = NUM + CAT

# Nombres de las columnas de entrada (formulario y plantilla de carga masiva)
CAMPOS_FECHA = ['fecha_convocatoria', 'fecha_buena_pro', 'fecha_consentimiento', 'fecha_inicio', 'fecha_fin_original']
CAMPOS_ENTRADA = (['monto_referencial', 'monto_adjudicado', 'n_ofertantes'] + CAMPOS_FECHA +
                  ['nivel_gobierno', 'sistema_contratacion', 'tipo_procedimiento', 'tipo_proveedor', 'provincia'])
ENTRADA_A_MODELO = {'tipo_procedimiento': 'tipo_procedimiento_2', 'tipo_proveedor': 'tipo_proveedor_2'}


def cargar():
    with open(os.path.join(AQUI, 'config_niveles.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    modelos = {t: joblib.load(os.path.join(AQUI, f'modelo_{t}.joblib')) for t in ('costo', 'plazo')}
    return cfg, modelos


def _fecha(s):
    """Convierte a fecha aceptando datetime, AAAA-MM-DD y DD/MM/AAAA (formatos explícitos, sin ambigüedad)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    t = s.astype('string').str.strip().str.split(r'[ T]').str[0]
    iso = pd.to_datetime(t.where(t.str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)), format='%Y-%m-%d', errors='coerce')
    dmy = pd.to_datetime(t.where(t.str.match(r'^\d{1,2}/\d{1,2}/\d{4}$', na=False)), format='%d/%m/%Y', errors='coerce')
    return iso.fillna(dmy)


def construir_predictores(raw):
    """De los datos de entrada del contrato a los 12 predictores del modelo."""
    f = {c: _fecha(raw[c]) for c in CAMPOS_FECHA}
    ref = pd.to_numeric(raw['monto_referencial'], errors='coerce')
    adj = pd.to_numeric(raw['monto_adjudicado'], errors='coerce')
    X = pd.DataFrame(index=raw.index)
    X['monto_adjudicado'] = adj
    X['reduccion_pct'] = np.where(ref > 0, (ref - adj) / ref * 100, np.nan)
    X['n_ofertantes'] = pd.to_numeric(raw['n_ofertantes'], errors='coerce')
    X['dias_conv_bp'] = (f['fecha_buena_pro'] - f['fecha_convocatoria']).dt.days
    X['dias_bp_cons'] = (f['fecha_consentimiento'] - f['fecha_buena_pro']).dt.days
    X['plazo_original_dias'] = (f['fecha_fin_original'] - f['fecha_inicio']).dt.days
    X['anio_inicio'] = f['fecha_inicio'].dt.year
    for c in CAT:
        origen = next((k for k, v in ENTRADA_A_MODELO.items() if v == c), c)
        X[c] = raw[origen].astype(str).str.strip()
    return X[PREDICTORES]


def validar(raw, cfg):
    """Devuelve, por fila, (errores, avisos). Los errores impiden evaluar; los avisos solo advierten."""
    f = {c: _fecha(raw[c]) for c in CAMPOS_FECHA}
    ref = pd.to_numeric(raw['monto_referencial'], errors='coerce')
    adj = pd.to_numeric(raw['monto_adjudicado'], errors='coerce')
    nof = pd.to_numeric(raw['n_ofertantes'], errors='coerce')
    X = construir_predictores(raw)
    salida = []
    for i in raw.index:
        err, av = [], []
        if not (ref[i] > 0): err.append('El monto referencial debe ser mayor que cero.')
        if not (adj[i] > 0): err.append('El monto adjudicado debe ser mayor que cero.')
        if not (nof[i] >= 1 and float(nof[i]).is_integer()): err.append('El número de ofertantes debe ser un entero mayor o igual a 1.')
        for c in CAMPOS_FECHA:
            if pd.isna(f[c][i]): err.append(f'Fecha no válida o vacía: {c} (use AAAA-MM-DD o DD/MM/AAAA).')
        if not any(e.startswith('Fecha') for e in err):
            if f['fecha_buena_pro'][i] < f['fecha_convocatoria'][i]: err.append('La buena pro no puede ser anterior a la convocatoria.')
            if f['fecha_consentimiento'][i] < f['fecha_buena_pro'][i]: err.append('El consentimiento no puede ser anterior a la buena pro.')
            if f['fecha_fin_original'][i] <= f['fecha_inicio'][i]: err.append('La fecha de fin original debe ser posterior a la de inicio.')
        for c in CAT:
            if X.at[i, c] not in cfg['categorias'][c]: err.append(f'Categoría no admitida en {c}: "{X.at[i, c]}".')
        if not err:
            for c in NUM:
                lo, hi = cfg['rangos'][c]
                if not (lo <= X.at[i, c] <= hi):
                    av.append(f'{c} = {X.at[i, c]:,.1f} está fuera del rango histórico ({lo:,.0f} a {hi:,.0f}); el puntaje es menos confiable.')
        salida.append((err, av))
    return salida


def niveles_de(cfg, tarea):
    n = cfg['niveles']
    return n[tarea] if isinstance(n, dict) else n


def nivel_indice(puntaje, cortes):
    return int(sum(puntaje >= c for c in cortes))


def predecir(raw, cfg, modelos):
    """Puntajes y niveles de costo y plazo para cada fila de entrada (ya validada)."""
    X = construir_predictores(raw)
    res = pd.DataFrame(index=raw.index)
    for t in ('costo', 'plazo'):
        p = modelos[t]['pipeline'].predict_proba(X[modelos[t]['predictores']])[:, 1]
        res[f'puntaje_{t}'] = np.round(p, 3)
        res[f'nivel_{t}'] = [niveles_de(cfg, t)[nivel_indice(v, cfg['cortes'][t])] for v in p]
    return res, X
