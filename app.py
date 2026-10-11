import datetime as dt

import pandas as pd
import streamlit as st

import logica as lg

st.set_page_config(page_title='SAD - Riesgo contractual en obras educativas', page_icon='🏫', layout='wide')


@st.cache_resource
def recursos():
    return lg.cargar()


cfg, modelos = recursos()

def caja_nivel(idx, total, texto):
    if idx == 0:
        st.success(texto)
    elif idx == total - 1:
        st.error(texto)
    else:
        st.warning(texto)


def frecuencia(tarea, nivel):
    f = next(x for x in cfg['frecuencia_observada'][tarea] if x['nivel'] == nivel)
    if f['frecuencia_desviacion'] is None:
        return ''
    return (f'En los contratos históricos de este nivel, {f["frecuencia_desviacion"]:.0%} presentó desviación '
            f'(n = {f["n"]}, validación externa).')


def tarjeta(tarea, titulo, puntaje, nivel):
    st.subheader(titulo)
    st.metric('Puntaje de riesgo (0 a 1)', f'{puntaje:.2f}')
    st.progress(float(min(max(puntaje, 0), 1)))
    niveles = lg.niveles_de(cfg, tarea)
    caja_nivel(niveles.index(nivel), len(niveles), f'**{nivel}**')
    st.caption(frecuencia(tarea, nivel))


st.title('🏫 Sistema de Apoyo a la Decisión')
st.markdown('**Riesgo de desviación contractual de costo y plazo** en obras de infraestructura educativa '
            '(educación básica regular) del departamento de Lima.')
st.info('Los puntajes sirven para **priorizar la revisión** de contratos. No son probabilidades exactas ni '
        'predicen con certeza: el apoyo es moderado y no reemplaza el criterio técnico de la entidad.')

tab1, tab2, tab3 = st.tabs(['Evaluar un contrato', 'Priorizar varios contratos', 'Acerca del modelo'])

# ---------------------------------------------------------------- Evaluar un contrato
with tab1:
    st.write('Ingrese los datos disponibles **al inicio de la ejecución** del contrato.')
    hoy = dt.date.today()
    lim = dict(min_value=dt.date(2015, 1, 1), max_value=dt.date(2035, 12, 31))
    with st.form('form_contrato'):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('**Montos y competencia**')
            monto_ref = st.number_input('Monto referencial (S/)', min_value=0.0, value=1_000_000.0, step=10_000.0)
            monto_adj = st.number_input('Monto adjudicado (S/)', min_value=0.0, value=950_000.0, step=10_000.0)
            n_of = st.number_input('Número de ofertantes', min_value=1, value=3, step=1)
        with c2:
            st.markdown('**Fechas del proceso y del contrato**')
            f_conv = st.date_input('Convocatoria', value=hoy - dt.timedelta(days=60), **lim)
            f_bp = st.date_input('Buena pro', value=hoy - dt.timedelta(days=40), **lim)
            f_cons = st.date_input('Consentimiento de la buena pro', value=hoy - dt.timedelta(days=30), **lim)
            f_ini = st.date_input('Inicio del plazo contractual', value=hoy - dt.timedelta(days=15), **lim)
            f_fin = st.date_input('Fin original del plazo contractual', value=hoy + dt.timedelta(days=165), **lim)
        with c3:
            st.markdown('**Características**')
            nivel_g = st.selectbox('Nivel de gobierno', cfg['categorias']['nivel_gobierno'])
            sistema = st.selectbox('Sistema de contratación', cfg['categorias']['sistema_contratacion'])
            proced = st.selectbox('Tipo de procedimiento', cfg['categorias']['tipo_procedimiento_2'])
            provee = st.selectbox('Tipo de proveedor', cfg['categorias']['tipo_proveedor_2'])
            prov = st.selectbox('Provincia', cfg['categorias']['provincia'])
        enviar = st.form_submit_button('Calcular riesgo', type='primary')

    if enviar:
        raw = pd.DataFrame([{
            'monto_referencial': monto_ref, 'monto_adjudicado': monto_adj, 'n_ofertantes': n_of,
            'fecha_convocatoria': f_conv, 'fecha_buena_pro': f_bp, 'fecha_consentimiento': f_cons,
            'fecha_inicio': f_ini, 'fecha_fin_original': f_fin, 'nivel_gobierno': nivel_g,
            'sistema_contratacion': sistema, 'tipo_procedimiento': proced, 'tipo_proveedor': provee,
            'provincia': prov}])
        errores, avisos = lg.validar(raw, cfg)[0]
        if errores:
            st.session_state.pop('ultimo', None)
            for e in errores:
                st.error(e)
        else:
            res, X = lg.predecir(raw, cfg, modelos)
            st.session_state['ultimo'] = {'raw': raw, 'res': res, 'X': X, 'avisos': avisos}

    u = st.session_state.get('ultimo')
    if u:
        for a in u['avisos']:
            st.warning(a)
        r = u['res'].iloc[0]
        col_a, col_b = st.columns(2)
        with col_a:
            tarjeta('costo', 'Riesgo de desviación de costo', r['puntaje_costo'], r['nivel_costo'])
        with col_b:
            tarjeta('plazo', 'Riesgo de desviación de plazo', r['puntaje_plazo'], r['nivel_plazo'])
        with st.expander('Variables calculadas que recibe el modelo'):
            st.dataframe(u['X'].T.rename(columns={u['X'].index[0]: 'valor'}).astype(str))
        salida = pd.concat([u['raw'], u['res']], axis=1)
        st.download_button('Descargar resultado (CSV)', salida.to_csv(index=False).encode('utf-8-sig'),
                           file_name='resultado_contrato.csv', mime='text/csv')

# ---------------------------------------------------------------- Priorizar varios contratos
with tab2:
    st.write('Cargue un archivo CSV o Excel con un contrato por fila para obtener una lista priorizada.')
    plantilla = pd.DataFrame([{
        'id_contrato': 'EJEMPLO-1', 'monto_referencial': 1000000, 'monto_adjudicado': 950000, 'n_ofertantes': 3,
        'fecha_convocatoria': '2025-01-10', 'fecha_buena_pro': '2025-02-05', 'fecha_consentimiento': '2025-02-14',
        'fecha_inicio': '2025-03-01', 'fecha_fin_original': '2025-08-28',
        'nivel_gobierno': cfg['categorias']['nivel_gobierno'][0],
        'sistema_contratacion': cfg['categorias']['sistema_contratacion'][0],
        'tipo_procedimiento': cfg['categorias']['tipo_procedimiento_2'][0],
        'tipo_proveedor': cfg['categorias']['tipo_proveedor_2'][0],
        'provincia': cfg['categorias']['provincia'][0]}])
    st.download_button('Descargar plantilla (CSV)', plantilla.to_csv(index=False).encode('utf-8-sig'),
                       file_name='plantilla_contratos.csv', mime='text/csv')
    st.caption('Fechas en formato AAAA-MM-DD o DD/MM/AAAA. Categorías admitidas: ' +
               '; '.join(f'{k}: {", ".join(v)}' for k, v in cfg['categorias'].items()))
    archivo = st.file_uploader('Archivo de contratos', type=['csv', 'xlsx'])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo, dtype=str) if archivo.name.lower().endswith('.csv') else pd.read_excel(archivo, dtype=str)
        except Exception as e:
            st.error(f'No se pudo leer el archivo: {e}')
            lote = None
        if lote is not None:
            falta = [c for c in lg.CAMPOS_ENTRADA if c not in lote.columns]
            if falta:
                st.error('Faltan columnas: ' + ', '.join(falta))
            else:
                lote = lote.reset_index(drop=True)
                if 'id_contrato' not in lote.columns:
                    lote.insert(0, 'id_contrato', [f'fila {i + 2}' for i in lote.index])
                val = lg.validar(lote, cfg)
                ok = [i for i, (e, _) in enumerate(val) if not e]
                if ok:
                    res, _ = lg.predecir(lote.loc[ok], cfg, modelos)
                    tabla = pd.concat([lote.loc[ok, ['id_contrato']], res], axis=1)
                    tabla['avisos'] = [' | '.join(val[i][1]) for i in ok]
                    tabla = tabla.sort_values(['puntaje_costo', 'puntaje_plazo'], ascending=False).reset_index(drop=True)
                    st.session_state['lote'] = tabla
                else:
                    st.session_state.pop('lote', None)
                malos = [(lote.at[i, 'id_contrato'], ' '.join(val[i][0])) for i, (e, _) in enumerate(val) if e]
                if malos:
                    st.warning(f'{len(malos)} contrato(s) no se evaluaron por errores en los datos.')
                    st.dataframe(pd.DataFrame(malos, columns=['id_contrato', 'problema']))
    t = st.session_state.get('lote')
    if t is not None:
        st.success(f'{len(t)} contrato(s) evaluados, ordenados por puntaje de costo y luego de plazo.')
        st.dataframe(t)
        st.download_button('Descargar lista priorizada (CSV)', t.to_csv(index=False).encode('utf-8-sig'),
                           file_name='lista_priorizada.csv', mime='text/csv')

# ---------------------------------------------------------------- Acerca del modelo
with tab3:
    st.markdown('''
**Qué hace.** Estima dos riesgos por separado, con modelos independientes: desviación de **costo**
(adicionales mayores que las reducciones) y desviación de **plazo** (fin del plazo posterior al original).

**Datos de entrenamiento.** Contratos de obras de infraestructura de inicial, primaria y secundaria del
departamento de Lima, de fuentes públicas (SEACE-OSCE). Se usan solo variables disponibles al inicio del contrato.

**Validación.** Anidada y agrupada por proceso (5 particiones externas, 3 internas), de modo que ningún proceso
aparece a la vez en entrenamiento y prueba.
''')
    d = cfg['desempeno']
    st.dataframe(pd.DataFrame(d).T.rename(columns={
        'modelo': 'Modelo', 'n': 'Contratos', 'positivos': 'Con desviación', 'bal_acc': 'Exactitud balanceada',
        'roc_auc': 'ROC-AUC', 'pr_auc': 'PR-AUC', 'pr_auc_azar': 'PR-AUC por azar'}))
    st.markdown('''
**Cómo leer los niveles.** Cada nivel resume la frecuencia de desviación observada históricamente en contratos
con puntajes similares (ver cada resultado). Un nivel prioritario no asegura una desviación, y uno estándar no la descarta.

**Limitaciones.**
- La muestra es pequeña, sobre todo para plazo; el desempeño es moderado y varía entre particiones.
- Los puntajes no están calibrados como probabilidades.
- Las desviaciones son modificaciones contractuales registradas, no costo o plazo finales liquidados.
- Los contratos de 2022 no tienen fecha de fin actualizada registrada y no entran al modelo de plazo.
- El modelo se entrenó con contratos de 2020 a 2025; fuera de ese rango, o con valores atípicos, el puntaje es menos confiable.
''')
