import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from sklearn.linear_model import LinearRegression
import uuid

# --- Funciones Auxiliares ---
# FunciÃ³n de presiÃ³n de vapor usando Antoine ---
def get_vapor_pressure_mca(temperature_celsius):
    # Antoine para agua: 1Â°C a 100Â°C
    # log10(P_mmHg) = A - B / (C + T)
    A = 8.07131
    B = 1730.63
    C = 233.426
    T = temperature_celsius
    P_mmHg = 10 ** (A - (B / (C + T)))
    # Convertir mmHg a mca (1 mmHg = 0.0136 mca)
    P_mca = P_mmHg * 0.0136
    return P_mca


# FunciÃ³n para calcular pÃ©rdidas por fricciÃ³n (ejemplo con Hazen-Williams)
# Para una app real, se considerarÃ­an otras fÃ³rmulas y tipos de tuberÃ­a.
def calculate_friction_losses_hw(flow_rate_m3s, diameter_m, length_m, C_hw):
    if diameter_m == 0: return float('inf')
    Q_lps = flow_rate_m3s * 1000 # Convertir m3/s a L/s
    if Q_lps == 0: return 0
    # FÃ³rmula de Hazen-Williams (vÃ¡lida solo para agua a 15Â°C, pero Ãºtil para demostraciÃ³n)
    # Hf = (10.67 * L * Q^1.852) / (C^1.852 * D^4.87)
    hf = (10.67 * length_m * (Q_lps / 1000)**1.852) / ((C_hw)**1.852 * diameter_m**4.87)
    return hf

# FunciÃ³n para calcular pÃ©rdidas locales (ejemplo con coeficiente K)
def calculate_local_losses(flow_rate_m3s, diameter_m, k_value):
    if diameter_m == 0: return float('inf')
    area = np.pi * (diameter_m / 2)**2
    if area == 0: return float('inf')
    velocity = flow_rate_m3s / area if area > 0 else 0
    g = 9.81 # m/s^2
    hl = k_value * (velocity**2) / (2 * g)
    return hl

# --- Clases de CÃ¡lculo ---

class CavitationCalculator:
    def __init__(self, fluid_temperature_celsius):
        self.fluid_temperature_celsius = fluid_temperature_celsius
        self.Pv = get_vapor_pressure_mca(fluid_temperature_celsius)

    def calculate_cavitation_index(self, P_upstream_mca, P_downstream_mca):
        # Asegurar que P_upstream_mca > P_downstream_mca para evitar divisiÃ³n por cero o negativos irracionales
        if P_upstream_mca <= P_downstream_mca:
            st.error("Error: La presiÃ³n aguas arriba debe ser mayor que la presiÃ³n aguas abajo para calcular un delta de presiÃ³n.")
            return None
        
        delta_P = P_upstream_mca - P_downstream_mca
        if delta_P == 0:
            return float('inf') # Sin caÃ­da de presiÃ³n, sin riesgo de cavitaciÃ³n por delta P
        
        numerator = P_downstream_mca - self.Pv
        
        # El Ã­ndice de cavitaciÃ³n para vÃ¡lvulas y componentes es (P2 - Pv) / (P1 - P2)
        # La interpretaciÃ³n es: si P2 cae cerca de Pv, el numerador se acerca a cero
        # Si la caÃ­da de presiÃ³n (P1-P2) es grande, el denominador es grande
        # Un valor bajo de theta indica riesgo.
        
        theta = numerator / delta_P
        return theta

    def get_cavitation_risk_description(self, theta):
        if theta is None:
            return "No calculado"
        elif theta < 0.5:
            return "ðŸš¨ Â¡Riesgo CRÃTICO de daÃ±o por cavitaciÃ³n! ðŸš¨"
        elif theta < 0.8:
            return "âš ï¸ Riesgo ALTO de ruido por cavitaciÃ³n. âš ï¸"
        else:
            return "âœ… Riesgo de cavitaciÃ³n BAJO o NULO."

class NPSHCalculator:
    def __init__(self, fluid_temperature_celsius):
        self.fluid_temperature_celsius = fluid_temperature_celsius
        self.Pv = get_vapor_pressure_mca(fluid_temperature_celsius)

    def calculate_npsha(self, H_atm_mca, H_static_suction_mca, H_losses_friction_suction_mca):
        npsha = H_atm_mca + H_static_suction_mca - H_losses_friction_suction_mca - self.Pv
        return npsha

    def get_npsh_risk_description(self, npsha, npshr):
        if npsha is None or npshr is None:
            return "No calculado"
        if npsha < npshr * 1.1: # Considerar un margen de seguridad del 10%
            return "ðŸš¨ Â¡Riesgo CRÃTICO de cavitaciÃ³n en bomba (NPSH Insuficiente)! ðŸš¨"
        else:
            return "âœ… NPSH Suficiente. Riesgo de cavitaciÃ³n en bomba BAJO."

# --- ConfiguraciÃ³n de la PÃ¡gina de Streamlit ---
st.set_page_config(layout="wide", page_title="Calculadora de Ãndice de CavitaciÃ³n")

st.title("ðŸ§® CÃ¡lculo de Ãndice de CavitaciÃ³n")
st.markdown("Herramienta de anÃ¡lisis del riesgo de cavitaciÃ³n en diferentes componentes del sistema de bombeo.")

# --- Barra Lateral para ParÃ¡metros Comunes ---
# Tabla de presiÃ³n de vapor y densidad de agua
tabla_temp = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
tabla_dens = [999.9, 1000.0, 999.7, 999.1, 998.2, 997.0, 995.7, 994.1, 992.2, 990.2, 988.1, 985.7, 983.2, 980.6, 977.8, 974.9, 971.8, 968.6, 965.3, 961.9, 958.4]
tabla_pv = [0.06, 0.09, 0.12, 0.17, 0.25, 0.33, 0.44, 0.58, 0.76, 0.98, 1.25, 1.61, 2.03, 2.56, 3.20, 3.96, 4.85, 5.93, 7.18, 8.62, 10.33]
def interpola_tabla(temp, tabla_x, tabla_y):
    if temp <= tabla_x[0]:
        return tabla_y[0]
    if temp >= tabla_x[-1]:
        return tabla_y[-1]
    for i in range(1, len(tabla_x)):
        if temp < tabla_x[i]:
            x0, x1 = tabla_x[i-1], tabla_x[i]
            y0, y1 = tabla_y[i-1], tabla_y[i]
            return y0 + (y1 - y0) * (temp - x0) / (x1 - x0)
    return tabla_y[-1]

st.sidebar.header("âš™ï¸ ParÃ¡metros Generales")
# SincronizaciÃ³n de temperatura
if 'fluid_temp' not in st.session_state:
    st.session_state['fluid_temp'] = 20
if 'fluid_temp_input' not in st.session_state:
    st.session_state['fluid_temp_input'] = 20

def on_slider_change():
    st.session_state['fluid_temp_input'] = st.session_state['fluid_temp']

def on_input_change():
    st.session_state['fluid_temp'] = st.session_state['fluid_temp_input']

st.sidebar.slider(
    "Temperatura del Fluido (Â°C)",
    min_value=0, max_value=100, value=st.session_state['fluid_temp'], step=5,
    key='fluid_temp', on_change=on_slider_change
)
st.sidebar.number_input(
    "Temperatura precisa (Â°C)",
    min_value=0, max_value=100, value=st.session_state['fluid_temp_input'], step=1,
    key='fluid_temp_input', on_change=on_input_change
)
fluid_temp = st.session_state['fluid_temp']
fluid_temp_input = st.session_state['fluid_temp_input']
Pv_calculated = interpola_tabla(fluid_temp_input, tabla_temp, tabla_pv)
densidad_calculada = interpola_tabla(fluid_temp_input, tabla_temp, tabla_dens)
st.sidebar.info(f"PresiÃ³n de Vapor (Pv) a {fluid_temp_input}Â°C: **{Pv_calculated:.3f} mca**\nDensidad del agua a {fluid_temp_input}Â°C: **{densidad_calculada:.1f} kg/mÂ³**")
# Usar fluid_temp_input en los cÃ¡lculos
cav_calc = CavitationCalculator(fluid_temp_input)
npsh_calc = NPSHCalculator(fluid_temp_input)
with st.sidebar.expander("PresiÃ³n absoluta de vapor de agua y densidad segÃºn temperatura"):
    st.markdown('''
| Temperatura (Â°C) | Densidad (kg/mÂ³) | PresiÃ³n de vapor H_vap (m) |
|:---------------:|:----------------:|:-------------------------:|
| 0   | 999,90 | 0,06 |
| 5   | 1000,00 | 0,09 |
| 10  | 999,70 | 0,12 |
| 15  | 999,10 | 0,17 |
| 20  | 998,20 | 0,25 |
| 25  | 997,00 | 0,33 |
| 30  | 995,70 | 0,44 |
| 35  | 994,10 | 0,58 |
| 40  | 992,20 | 0,76 |
| 45  | 990,20 | 0,98 |
| 50  | 988,10 | 1,25 |
| 55  | 985,70 | 1,61 |
| 60  | 983,20 | 2,03 |
| 65  | 980,60 | 2,56 |
| 70  | 977,80 | 3,20 |
| 75  | 974,90 | 3,96 |
| 80  | 971,80 | 4,85 |
| 85  | 968,60 | 5,93 |
| 90  | 965,30 | 7,18 |
| 95  | 961,90 | 8,62 |
| 100 | 958,40 | 10,33 |
''')

# Instancia del calculador de cavitaciÃ³n para usar en todas las pestaÃ±as
# cav_calc = CavitationCalculator(fluid_temp) # This line is now redundant as cav_calc is defined above
# npsh_calc = NPSHCalculator(fluid_temp_input) # This line is now redundant as npsh_calc is defined above

# --- NUEVA PESTAÃ‘A: Coeficiente de Descarga (C) ---
tab_c, tab1, tab_trans, tab2, tab3, tab4, tab5 = st.tabs([
    "ðŸ”‘ Coeficiente de Descarga (C)",
    "ðŸ’§ VÃ¡lvula",
    "ðŸŒŠ Flujos Transitorios",
    "ðŸš° Sistema de Bombeo (NPSH)",
    "â¬†ï¸ LÃ­nea de ImpulsiÃ³n",
    "â¬‡ï¸ LÃ­nea de SucciÃ³n",
    "ðŸ”„ PÃ©rdidas por FricciÃ³n"
])

with tab_c:
    st.header("Coeficiente de Descarga (C)")
    col1, col2, col3, col4 = st.columns([0.18, 0.42, 0.24, 0.16])
    with col1:
        st.subheader("CÃ¡lculo interactivo")
        tipos_valvula = [
            ("Globo", "Lineal"),
            ("Ãngulo", "Lineal"),
            ("Compuerta (incl. Y)", "Lineal"),
            ("Obturador CilÃ­ndrico ExcÃ©ntrico", "Rotativo"),
            ("Mariposa", "Rotativo"),
            ("Bola", "Rotativo"),
            ("Macho", "Rotativo"),
            ("Orificio Ajustable", "Rotativo"),
            ("Flujo Axial", "Rotativo")
        ]
        tipo_valvula = st.selectbox("Tipo de vÃ¡lvula", [v[0] for v in tipos_valvula])
        movimiento_permitido = [v[1] for v in tipos_valvula if v[0] == tipo_valvula][0]
        movimiento = st.selectbox("Movimiento", [movimiento_permitido], disabled=True)
        estado = st.selectbox("Estado de apertura", ["Totalmente abierta", "Parcialmente abierta"])
        valores_C = {
            ("Globo", "Lineal", "Totalmente abierta"): (0.85, 0.95),
            ("Globo", "Lineal", "Parcialmente abierta"): (0.60, 0.85),
            ("Ãngulo", "Lineal", "Totalmente abierta"): (0.90, 0.98),
            ("Ãngulo", "Lineal", "Parcialmente abierta"): (0.70, 0.90),
            ("Compuerta (incl. Y)", "Lineal", "Totalmente abierta"): (0.95, 1.00),
            ("Compuerta (incl. Y)", "Lineal", "Parcialmente abierta"): (0.75, 0.95),
            ("Obturador CilÃ­ndrico ExcÃ©ntrico", "Rotativo", "Totalmente abierta"): (0.90, 0.98),
            ("Obturador CilÃ­ndrico ExcÃ©ntrico", "Rotativo", "Parcialmente abierta"): (0.70, 0.90),
            ("Mariposa", "Rotativo", "Totalmente abierta"): (0.95, 1.00),
            ("Mariposa", "Rotativo", "Parcialmente abierta"): (0.75, 0.95),
            ("Bola", "Rotativo", "Totalmente abierta"): (0.98, 1.00),
            ("Bola", "Rotativo", "Parcialmente abierta"): (0.80, 0.98),
            ("Macho", "Rotativo", "Totalmente abierta"): (0.90, 0.98),
            ("Macho", "Rotativo", "Parcialmente abierta"): (0.70, 0.90),
            ("Orificio Ajustable", "Rotativo", "Totalmente abierta"): (0.90, 0.98),
            ("Orificio Ajustable", "Rotativo", "Parcialmente abierta"): (0.70, 0.90),
            ("Flujo Axial", "Rotativo", "Totalmente abierta"): (0.95, 1.00),
            ("Flujo Axial", "Rotativo", "Parcialmente abierta"): (0.75, 0.95),
        }
        c_min, c_max = valores_C.get((tipo_valvula, movimiento_permitido, estado), (0.7, 1.0))
        C = st.number_input(f"Coeficiente de descarga C [{c_min:.2f} - {c_max:.2f}]", min_value=0.5, max_value=1.2, value=float(f"{(c_min+c_max)/2:.2f}"), step=0.01, format="%.2f")
        D = st.number_input("DiÃ¡metro de la tuberÃ­a D (mm)", min_value=1.0, max_value=2000.0, value=100.0, step=1.0) / 1000  # convertir a metros
        delta_h = st.number_input("Diferencia de altura Î”h (m)", min_value=0.01, max_value=100.0, value=5.0, step=0.01)
        g = 9.81
        Q = C * ((2 * g * delta_h) ** 0.5) * (np.pi * D ** 2 / 4)
        st.markdown(f"**Caudal calculado Q = {Q:.4f} mÂ³/s ({Q*3600:.2f} mÂ³/h, {Q*1000:.2f} L/s)**")
        st.markdown("---")
        st.markdown("### Datos de grÃ¡fico")
        diam_ini = st.number_input("DiÃ¡metro inicial (mm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.25)
        diam_fin = st.number_input("DiÃ¡metro final (mm)", min_value=0.0, max_value=2000.0, value=100.0, step=0.25)
        diam_paso = st.number_input("Paso de diÃ¡metro (mm)", min_value=0.01, max_value=500.0, value=0.25, step=0.01)
        altura_max = st.number_input("Altura mÃ¡xima en grÃ¡fico (m)", min_value=1.0, max_value=200.0, value=40.0, step=0.5)
    with col2:
        st.subheader("Curvas de igual caudal Q")
        st.markdown("#### DiÃ¡metro de tuberÃ­a D (mm)")
        st.markdown("""
**SimbologÃ­a:**  
- **Q**: Caudal (mÂ³/s)  
- **D**: DiÃ¡metro de la tuberÃ­a (mm)  
- **Î”h**: Diferencia de altura (m)  
- **C**: Coeficiente de descarga (adimensional)  
- **g**: Gravedad (9.81 m/sÂ²)
""")
        # Usar los valores de diam_ini, diam_fin, diam_paso para el rango de D
        D_range = np.arange(diam_ini, diam_fin + diam_paso, diam_paso) / 1000  # convertir a metros
        D_range = D_range[D_range > 0]  # Elimina cualquier valor cero
        C_graf = C  # Usar el coeficiente seleccionado
        g = 9.81
        Q_targets = [0.005, 0.01, 0.02, 0.05]  # Caudales objetivo en mÂ³/s (5, 10, 20, 50 L/s)
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Azul, naranja, verde, rojo
        fig = go.Figure()
        tabla_dict = {"D (mm)": D_range*1000}
        for idx, Qg in enumerate(Q_targets):
            delta_h_curve = (Qg / (C_graf * (np.pi * D_range**2 / 4)))**2 / (2 * g)
            fig.add_trace(go.Scatter(
                x=D_range*1000, y=delta_h_curve, mode='lines', name=f'Q={Qg*1000:.0f} L/s',
                hovertemplate='<b>Î”h = %{y:.2f} m</b><extra></extra>',
                line=dict(color=colores[idx % len(colores)]),
                hoverlabel=dict(bgcolor=colores[idx % len(colores)], font=dict(color='white', family='Arial Black'))
            ))
            tabla_dict[f"Î”h (m) Q={Qg*1000:.0f} L/s"] = delta_h_curve
        # Agregar punto rojo de los inputs
        fig.add_trace(go.Scatter(
            x=[D*1000], y=[delta_h], mode='markers',
            marker=dict(color='red', size=9, symbol='circle'),
            name='Punto de entrada',
            hovertemplate='<b>Entrada:<br>D = %{x:.2f} mm<br>Î”h = %{y:.2f} m</b><extra></extra>',
            hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
        ))
        fig.update_layout(
            title='Curvas de igual caudal Q en el plano DiÃ¡metro vs Diferencia de altura',
            xaxis=dict(
                title='DiÃ¡metro de tuberÃ­a D (mm)',
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title='Diferencia de altura Î”h (m)',
                tickformat='.2f',
                range=[0, altura_max],
            ),
            hovermode="x",
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
            template='simple_white',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        import pandas as pd
        df_tabla = pd.DataFrame(tabla_dict)
        st.markdown('<h4 style="text-align:center;">Tabla de Î”h para cada Q</h4>', unsafe_allow_html=True)
        st.dataframe(df_tabla, use_container_width=True)
        with st.expander("Valores tÃ­picos del coeficiente de descarga (C)"):
            st.markdown("""
| Tipo de VÃ¡lvula | Movimiento | Estado | C (tÃ­pico) |
|:-------------------------------|:----------:|:---------------------:|:----------:|
| Globo                          | Lineal     | Totalmente abierta    | 0.85 - 0.95|
| Globo                          | Lineal     | Parcialmente abierta  | 0.60 - 0.85|
| Ãngulo                         | Lineal     | Totalmente abierta    | 0.90 - 0.98|
| Ãngulo                         | Lineal     | Parcialmente abierta  | 0.70 - 0.90|
| Compuerta (incl. Y)            | Lineal     | Totalmente abierta    | 0.95 - 1.00|
| Compuerta (incl. Y)            | Lineal     | Parcialmente abierta  | 0.75 - 0.95|
| Obturador CilÃ­ndrico ExcÃ©ntrico| Rotativo   | Totalmente abierta    | 0.90 - 0.98|
| Obturador CilÃ­ndrico ExcÃ©ntrico| Rotativo   | Parcialmente abierta  | 0.70 - 0.90|
| Mariposa                       | Rotativo   | Totalmente abierta    | 0.95 - 1.00|
| Mariposa                       | Rotativo   | Parcialmente abierta  | 0.75 - 0.95|
| Bola                           | Rotativo   | Totalmente abierta    | 0.98 - 1.00|
| Bola                           | Rotativo   | Parcialmente abierta  | 0.80 - 0.98|
| Macho                          | Rotativo   | Totalmente abierta    | 0.90 - 0.98|
| Macho                          | Rotativo   | Parcialmente abierta  | 0.70 - 0.90|
| Orificio Ajustable             | Rotativo   | Totalmente abierta    | 0.90 - 0.98|
| Orificio Ajustable             | Rotativo   | Parcialmente abierta  | 0.70 - 0.90|
| Flujo Axial                    | Rotativo   | Totalmente abierta    | 0.95 - 1.00|
| Flujo Axial                    | Rotativo   | Parcialmente abierta  | 0.75 - 0.95|
""")
        with st.expander("Conceptos y teorÃ­a"):
            st.markdown("""
El coeficiente de descarga es un factor adimensional caracterÃ­stico de la vÃ¡lvula, que permite calcular el caudal (**Q**) que pasa a travÃ©s de la vÃ¡lvula en funciÃ³n del nivel del fluido en el embalse o reserva (**Î”h**).

La ecuaciÃ³n general es:
""")
            st.latex(r"Q = C \sqrt{2g\Delta h} \frac{\pi D^2}{4}")
            st.markdown("""
Donde:
- **Q**: Caudal (mÂ³/s)
- **Î”h**: Diferencia de altura (m) Î”h
- **D**: DiÃ¡metro de la tuberÃ­a (m)
- **C**: Coeficiente de descarga (adimensional)
- **g**: Gravedad (9.81 m/sÂ²)

Un valor alto de **C** indica que la vÃ¡lvula permite un flujo eficiente con mÃ­nima resistencia, mientras que un valor bajo indica mayor restricciÃ³n al flujo.
""")
        with st.expander("Factores principales en el daÃ±o por cavitaciÃ³n"):
            st.markdown("""
- **Intensidad de la cavitaciÃ³n**
- **Materiales utilizados**
- **Tiempo de exposiciÃ³n a la cavitaciÃ³n**
- **TamaÃ±o de la vÃ¡lvula**
- **DiseÃ±o de la vÃ¡lvula y sus componentes**
- **Fugas cuando la vÃ¡lvula estÃ¡ cerrada**
""")
    with col4:
        pass
    st.markdown("---")
    st.markdown(f"**Caudal calculado Q = {Q:.4f} mÂ³/s ({Q*3600:.2f} mÂ³/h, {Q*1000:.2f} L/s)**")
    st.caption("La fÃ³rmula asume flujo libre y condiciones ideales. Para condiciones reales, considerar pÃ©rdidas adicionales y coeficientes de seguridad.")
    st.markdown("**Importancia del coeficiente de descarga (C):** Un valor alto de C indica que la vÃ¡lvula permite un flujo eficiente con mÃ­nima resistencia, mientras que un valor bajo indica mayor restricciÃ³n al flujo. La elecciÃ³n del tipo de vÃ¡lvula depende del caudal requerido, presiÃ³n del sistema y necesidad de control fino del flujo.")

# --- PestaÃ±a 1: VÃ¡lvula ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de VÃ¡lvula
# --------------------------------------------------
with tab1:
    st.header("AnÃ¡lisis de CavitaciÃ³n en VÃ¡lvula")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    # Definir P2_range y theta_values antes de los bloques de columnas para uso global en la pestaÃ±a
    P2_min_default = 0.0
    P2_max_default = 8.0
    P2_step_default = 0.25
    # Se usan los valores por defecto, pero luego se actualizan con los inputs
    P2_min = P2_min_default
    P2_max = P2_max_default
    P2_step = P2_step_default
    with col_input:
        st.subheader("Datos de la VÃ¡lvula")
        P1_valve = st.number_input("PresiÃ³n de Entrada (P1, mca)", min_value=0.0, value=10.0, step=0.1)
        P2_valve = st.number_input("PresiÃ³n de Salida (P2, mca)", min_value=0.0, value=5.0, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        P2_min = st.number_input("P2 mÃ­nimo (mca)", min_value=0.0, value=P2_min_default, step=0.1)
        P2_max = st.number_input("P2 mÃ¡ximo (mca)", min_value=0.0, value=P2_max_default, step=0.1)
        P2_step = st.number_input("Paso de P2 (mca)", min_value=0.001, max_value=10.0, value=P2_step_default, step=0.1)
        st.subheader("Resultados")
        theta_valve = cav_calc.calculate_cavitation_index(P1_valve, P2_valve)
        if theta_valve is not None:
            st.metric(label="Ãndice de CavitaciÃ³n (Ï‘)", value=f"{theta_valve:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_valve))
        else:
            st.write("Ingrese valores vÃ¡lidos para calcular el Ã­ndice de cavitaciÃ³n.")
        st.info("""
**InterpretaciÃ³n:**
- Si Ï‘ < 0.5: Riesgo crÃ­tico de daÃ±o por cavitaciÃ³n.
- Si 0.5 â‰¤ Ï‘ < 0.8: Riesgo alto de ruido por cavitaciÃ³n.
- Si Ï‘ â‰¥ 0.8: Riesgo bajo o nulo de cavitaciÃ³n.
""")
        # ExplicaciÃ³n Kv ...
    # Calcular P2_range y theta_values despuÃ©s de obtener los inputs
    P2_range = np.arange(P2_min, P2_max+P2_step, P2_step)
    theta_values = []
    for p2 in P2_range:
        if P1_valve > p2:
            theta = cav_calc.calculate_cavitation_index(P1_valve, p2)
            theta_values.append(theta if theta is not None else np.nan)
        else:
            theta_values.append(np.nan)

    with col_graph:
        st.subheader("GrÃ¡fica: Ãndice de CavitaciÃ³n vs. PresiÃ³n de Salida")
        fig_valve = go.Figure()
        fig_valve.add_trace(go.Scatter(x=P2_range, y=theta_values, mode='lines+markers',
                                       line=dict(color='#2ca02c'),
                                       marker=dict(symbol='diamond-open', color='blue'),
                                       hovertemplate='<b>P2 = %{x:.2f} mca<br>Ï‘ = %{y:.2f}</b><extra></extra>',
                                       hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))))
        fig_valve.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="LÃ­mite DaÃ±o (0.5)", secondary_y=False)
        fig_valve.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="LÃ­mite Ruido (0.8)", secondary_y=False)
        fig_valve.update_layout(
            title="Ãndice de CavitaciÃ³n vs. PresiÃ³n de Salida en VÃ¡lvula",
            xaxis=dict(
                title="PresiÃ³n de Salida (mca)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="Ãndice de CavitaciÃ³n (Ï‘)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_valve, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Ãndice de CavitaciÃ³n</h4>', unsafe_allow_html=True)
        df_valve = pd.DataFrame({
            'P2 (mca)': np.round(P2_range, 2),
            'Ï‘': np.round(theta_values, 2)
        })
        st.dataframe(df_valve, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"): 
            st.markdown(r"""
**Ãndice de CavitaciÃ³n:**

$$
\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}
$$

**Donde:**
- **P1**: PresiÃ³n de entrada (mca)
- **P2**: PresiÃ³n de salida (mca)
- **Pvapor**: PresiÃ³n de vapor del fluido (mca)

Un valor bajo de **Ï‘** indica mayor riesgo de cavitaciÃ³n.
""")
        # Panel independiente de explicaciÃ³n de presiones
        with st.expander("Â¿QuÃ© significan la presiÃ³n de entrada y salida?"):
            st.markdown("""
- **PresiÃ³n de entrada (Pâ‚):** Es la presiÃ³n del fluido justo antes de entrar a la vÃ¡lvula. Representa la energÃ­a por unidad de peso del fluido en el punto aguas arriba de la vÃ¡lvula, normalmente medida en metros de columna de agua (mca). Esta presiÃ³n depende de la altura, el caudal y las condiciones del sistema antes de la vÃ¡lvula.

- **PresiÃ³n de salida (Pâ‚‚):** Es la presiÃ³n del fluido justo despuÃ©s de pasar por la vÃ¡lvula. Representa la energÃ­a por unidad de peso del fluido en el punto aguas abajo de la vÃ¡lvula, tambiÃ©n en metros de columna de agua (mca). Esta presiÃ³n suele ser menor que la de entrada debido a la pÃ©rdida de energÃ­a (caÃ­da de presiÃ³n) que ocurre al atravesar la vÃ¡lvula.

**En resumen:**
- **Pâ‚** indica cuÃ¡nta presiÃ³n tiene el fluido antes de la vÃ¡lvula.
- **Pâ‚‚** indica cuÃ¡nta presiÃ³n queda despuÃ©s de la vÃ¡lvula.
""")
        with st.expander("Criterios de cavitaciÃ³n segÃºn sigma (Ïƒ)"):
            st.markdown("""
**Criterios de cavitaciÃ³n segÃºn el Ã­ndice sigma (Ïƒ):**

| Rango de Ïƒ         | InterpretaciÃ³n                                      |
|:------------------:|:---------------------------------------------------|
| Ïƒ â‰¥ 2.0            | <span style='color:green'><b>No hay cavitaciÃ³n</b></span> |
| 1.7 < Ïƒ < 2.0      | <span style='color:#7ca300'><b>ProtecciÃ³n suficiente con materiales endurecidos</b></span> |
| 1.5 < Ïƒ < 1.7      | <span style='color:orange'><b>Algo de cavitaciÃ³n, puede funcionar un solo escalÃ³n</b></span> |
| 1.0 < Ïƒ < 1.5      | <span style='color:#ff6600'><b>Potencial de cavitaciÃ³n severa, se requiere reducciÃ³n en varias etapas</b></span> |
| Ïƒ < 1.0            | <span style='color:red'><b>Flashing (vaporizaciÃ³n instantÃ¡nea)</b></span> |

> **Ïƒ = (Pâ‚ - P_v) / (Pâ‚ - Pâ‚‚)**

- **SUPER CAVITACIÃ“N:** Ïƒ bajo, aceleraciÃ³n alta, daÃ±o severo.
- **CAVITACIÃ“N PLENA:** Ïƒ intermedio, daÃ±o considerable.
- **CAVITACIÃ“N INCIPIENTE:** Ïƒ cerca de 1.5-1.7, inicio de daÃ±o.
- **SUBCRÃTICO:** Ïƒ alto, sin daÃ±o.

Estos criterios ayudan a seleccionar el diseÃ±o y materiales adecuados para evitar daÃ±os por cavitaciÃ³n en vÃ¡lvulas.
""", unsafe_allow_html=True)

    # --- SEPARADOR A TODO EL ANCHO ---
    st.markdown("---")
    # --- NUEVA FILA DE COLUMNAS PARA Kv SOLO EN ESTA PESTAÃ‘A ---
    col_kv_exp, col_kv_graf, col_kv_tabla, col_kv_extra = st.columns([0.22, 0.4, 0.28, 0.1])
    with col_kv_exp:
        st.subheader("Coeficiente de caudal (Kv)")
        st.markdown("""
Las vÃ¡lvulas de control son conceptualmente orificios de Ã¡rea variable. Se las puede considerar simplemente como una restricciÃ³n que cambia su tamaÃ±o de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relaciÃ³n de diferencia de altura (Î”h) o presiÃ³n (Î”P) entre la entrada y salida de la vÃ¡lvula con el caudal (Q).
""")
        st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
        st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (mÂ³/h)
- $Q$: Caudal volumÃ©trico (mÂ³/h)
- $\rho$: Densidad (kg/mÂ³)
- $\Delta p$: Diferencia de presiÃ³n (bar)
- $P_1$: PresiÃ³n de entrada (bar)
- $P_2$: PresiÃ³n de salida (bar)
""")
        mca_a_bar = 0.0980665
        P1_bar = P1_valve * mca_a_bar
        P2_bar = P2_valve * mca_a_bar
        delta_p_bar = np.abs(P1_bar - P2_bar)
        densidad = densidad_calculada
        st.markdown(f"**PresiÃ³n de entrada (P1):** {P1_bar:.2f} bar  ")
        st.markdown(f"**PresiÃ³n de salida (P2):** {P2_bar:.2f} bar  ")
        st.markdown(f"**Diferencia de presiÃ³n (Î”p):** {delta_p_bar:.2f} bar  ")
        st.markdown(f"**Densidad (Ï):** {densidad:.1f} kg/mÂ³  ")
    with col_kv_graf:
        st.subheader("Ãndice de caudal Kv")
        # Corregido: GrÃ¡fico Î”p vs Kv con Q fijo
        Q_m3h = 10  # Caudal fijo tÃ­pico para la curva
        Kv_range = np.linspace(1, 15, 30)
        delta_p = (densidad / 1000) * (Q_m3h / Kv_range) ** 2
        fig_kv = go.Figure()
        fig_kv.add_trace(go.Scatter(
            x=Kv_range, y=delta_p, mode='markers+lines',
            marker=dict(symbol='diamond-open', color='blue'),
            line=dict(dash='solid', color='blue'),
            name='Î”p vs Kv'
        ))
        fig_kv.update_layout(
            title="VariaciÃ³n de Î”p con Kv",
            xaxis=dict(
                title="Kv (mÂ³/h)", 
                tickformat='.1f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            yaxis=dict(
                title="Î”p (bar)", 
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            template="simple_white",
            height=300
        )
        st.plotly_chart(fig_kv, use_container_width=True, key=f"fig_kv_kv_{uuid.uuid4()}")
        # Mantener el segundo grÃ¡fico y tablas igual
        delta_p_graf = np.linspace(0.1, 4.5, 30)
        Kv_alto = 15
        Kv_bajo = 3
        Q_kv_alto = Kv_alto / np.sqrt(densidad / (1000 * delta_p_graf))
        Q_kv_bajo = Kv_bajo / np.sqrt(densidad / (1000 * delta_p_graf))
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=delta_p_graf, y=Q_kv_alto, mode='markers+lines',
                                   marker=dict(symbol='diamond-open', color='blue'),
                                   line=dict(dash='solid', color='blue'),
                                   name='Kv >>'))
        fig_q.add_trace(go.Scatter(x=delta_p_graf, y=Q_kv_bajo, mode='markers+lines',
                                   marker=dict(symbol='square-open', color='orange'),
                                   line=dict(dash='solid', color='orange'),
                                   name='Kv <<'))
        fig_q.update_layout(
            title="Q vs. PÃ©rdida de carga para Kv alto y bajo",
            xaxis=dict(
                title="PÃ©rdida de carga (bar)", 
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            yaxis=dict(
                title="Q (mÂ³/h)", 
                tickformat='.1f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            template="simple_white",
            height=300
        )
        st.plotly_chart(fig_q, use_container_width=True, key=f"fig_q_kv_{uuid.uuid4()}")
    with col_kv_tabla:
        st.markdown('<h4 style="text-align:center;">Tabla Kv calculado</h4>', unsafe_allow_html=True)
        df_kv = pd.DataFrame({
            'Kv (mÂ³/h)': np.round(Kv_range, 2),
            'Î”p (bar)': np.round(delta_p, 2)
        })
        st.dataframe(df_kv, use_container_width=True)
        st.markdown('<h4 style="text-align:center;">Tabla Q vs Î”p para Kv</h4>', unsafe_allow_html=True)
        df_qkv = pd.DataFrame({
            'Î”p (bar)': np.round(delta_p_graf, 2),
            'Q (mÂ³/h) Kv alto': np.round(Q_kv_alto, 2),
            'Q (mÂ³/h) Kv bajo': np.round(Q_kv_bajo, 2)
        })
        st.dataframe(df_qkv, use_container_width=True)
    with col_kv_extra:
        pass

# --- PestaÃ±a 2: Sistema de Bombeo (NPSH) ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de Sistema de Bombeo (NPSH)
# --------------------------------------------------
with tab2:
    st.header("AnÃ¡lisis de NPSH Disponible para Bomba")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos del Sistema de Bombeo")
        # Tabla de presiÃ³n atmosfÃ©rica segÃºn altura sobre el nivel del mar
        alturas_msnm = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
        presiones_mca = [10.3, 9.7, 9.1, 8.6, 8.1, 7.6, 7.1, 6.7, 6.3, 5.9, 5.5, 5.2, 4.9]
        def interpolar_presion(cota):
            if cota <= alturas_msnm[0]:
                return presiones_mca[0]
            if cota >= alturas_msnm[-1]:
                return presiones_mca[-1]
            for i in range(1, len(alturas_msnm)):
                if cota < alturas_msnm[i]:
                    x0, x1 = alturas_msnm[i-1], alturas_msnm[i]
                    y0, y1 = presiones_mca[i-1], presiones_mca[i]
                    return y0 + (y1 - y0) * (cota - x0) / (x1 - x0)
            return presiones_mca[-1]
        cota_bomba = st.number_input("Cota de la bomba (msnm)", min_value=0.0, max_value=6000.0, value=0.0, step=1.0)
        H_atm = interpolar_presion(cota_bomba)
        st.number_input("PresiÃ³n AtmosfÃ©rica (H_atm, mca)", value=H_atm, disabled=True)
        H_static_suction = st.number_input("Altura EstÃ¡tica de SucciÃ³n (H_s, mca, + si es sobre nivel bomba, - si es bajo)", value=2.0, step=0.1)
        NPSHr_pump = st.number_input("NPSH Requerido de la Bomba (NPSH_necesaria, mca)", min_value=0.0, value=3.0, step=0.1)
        H_losses_suction = st.number_input("PÃ©rdidas por FricciÃ³n y Accesorios en SucciÃ³n (Î”H_s, mca)", min_value=0.0, value=1.5, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        losses_min = st.number_input("PÃ©rdidas mÃ­nimas (Î”H_s, mca)", min_value=0.0, value=0.0, step=0.01)
        losses_max = st.number_input("PÃ©rdidas mÃ¡ximas (Î”H_s, mca)", min_value=0.0, value=5.0, step=0.01)
        losses_step = st.number_input("Paso de pÃ©rdidas (Î”H_s, mca)", min_value=0.001, max_value=10.0, value=0.25, step=0.1)
        st.subheader("Resultados")
        npsha_calculated = npsh_calc.calculate_npsha(H_atm, H_static_suction, H_losses_suction)
        st.metric(label="NPSH Disponible (NPSHA, mca)", value=f"{npsha_calculated:.3f}")
        st.write(npsh_calc.get_npsh_risk_description(npsha_calculated, NPSHr_pump))
        st.info(f"**Criterio:** NPSHA debe ser mayor a {NPSHr_pump:.1f} mca (NPSHR). Se recomienda un margen de seguridad (ej. 1.1 * NPSHR).")

    with col_graph:
        st.subheader("GrÃ¡fica: NPSHA vs. PÃ©rdidas en SucciÃ³n")
        losses_range = np.arange(losses_min, losses_max+losses_step, losses_step)
        npsha_values = [npsh_calc.calculate_npsha(H_atm, H_static_suction, l) for l in losses_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.round(losses_range,2), y=npsha_values, mode='lines', name='NPSH Disponible',
                                 line=dict(color='#2ca02c'),
                                 hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                 hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))))
        # Punto rojo para el valor de entrada actual
        if H_losses_suction is not None and npsha_calculated is not None:
            fig.add_trace(go.Scatter(x=[H_losses_suction], y=[npsha_calculated], mode='markers',
                                     marker=dict(color='red', size=12, symbol='circle'),
                                     name='Punto de entrada',
                                     hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))))
        fig.add_hline(y=NPSHr_pump, line_dash="dash", line_color="red", annotation_text=f"NPSHR ({NPSHr_pump:.1f})")
        fig.add_hline(y=NPSHr_pump * 1.1, line_dash="dash", line_color="orange", annotation_text=f"NPSHR + Margen ({NPSHr_pump*1.1:.1f})")
        fig.update_layout(
            title=f"NPSHA vs. PÃ©rdidas en SucciÃ³n (H_estÃ¡tica={H_static_suction:.1f}m)",
            xaxis=dict(
                title="PÃ©rdidas por FricciÃ³n y Accesorios en SucciÃ³n (mca)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="NPSH Disponible (mca)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de NPSHA vs. PÃ©rdidas</h4>', unsafe_allow_html=True)
        df_npsh = pd.DataFrame({
            'PÃ©rdidas (mca)': np.round(losses_range, 2),
            'NPSHA (mca)': np.round(npsha_values, 2)
        })
        st.dataframe(df_npsh, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**NPSH Disponible:**

$$
NPSH_{disponible} = H_{atm} - (P_{v} + H_{s} + \Delta H_{s})
$$

**Donde:**
- **NPSHdisponible**: Carga neta positiva de succiÃ³n disponible (m)
- **Hatm**: PresiÃ³n atmosfÃ©rica (m)
- **Pv**: PresiÃ³n de vapor (m)
- **Hs**: Altura estÃ¡tica de succiÃ³n (m)
- **Î”Hs**: PÃ©rdida de presiÃ³n por fricciÃ³n (m)
""")
        with st.expander("PresiÃ³n AtmosfÃ©rica segÃºn altura"):
            st.markdown('''
| Altura sobre el nivel del mar (m) | PresiÃ³n atmosfÃ©rica (m.c.a.) |
|:-------------------------------:|:----------------------------:|
| 0     | 10,3 |
| 500   | 9,7  |
| 1 000 | 9,1  |
| 1 500 | 8,6  |
| 2 000 | 8,1  |
| 2 500 | 7,6  |
| 3 000 | 7,1  |
| 3 500 | 6,7  |
| 4 000 | 6,3  |
| 4 500 | 5,9  |
| 5 000 | 5,5  |
| 5 500 | 5,2  |
| 6 000 | 4,9  |
''')
        
        with st.expander("InterpretaciÃ³n y criterios"):
            st.markdown(r"""
La NPSH necesaria y la disponible son parÃ¡metros de control de la cavitaciÃ³n en los impulsores de las bombas.

La NPSH disponible depende del diseÃ±o del bombeo y representa la diferencia entre la carga absoluta y la presiÃ³n de vapor del lÃ­quido a temperatura constante.

**NPSH necesaria:** Es la carga exigida por la bomba entre la presiÃ³n de succiÃ³n y la presiÃ³n de vapor del lÃ­quido para que la bomba no cavite.

$$
NPSH_{disponible} \geq NPSH_{necesaria} + 0.5
$$
""")

# --- PestaÃ±a 3: LÃ­nea de ImpulsiÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de LÃ­nea de ImpulsiÃ³n
# --------------------------------------------------
with tab3:
    st.header("AnÃ¡lisis de CavitaciÃ³n en LÃ­nea de ImpulsiÃ³n")
    st.write("Se analiza un segmento o punto crÃ­tico de la lÃ­nea de impulsiÃ³n.")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos de la LÃ­nea de ImpulsiÃ³n")
        pipe_roughness_C = st.slider("Coeficiente Hazen-Williams (C)", min_value=80, max_value=150, value=120, step=5)
        flow_rate_imp = st.number_input("Caudal (L/s)", min_value=0.0, value=5.0, step=0.1)
        reference_diameter_mm = st.number_input("DiÃ¡metro (mm)", min_value=0.0, value=50.0, step=1.0)
        pipe_length_imp = st.number_input("Longitud del Tramo (m)", min_value=1.0, value=100.0, step=1.0)
        P_upstream_imp = st.number_input("PresiÃ³n Aguas Arriba (P_upstream, mca)", min_value=0.0, value=25.0, step=0.1)
        P_downstream_imp = st.number_input("PresiÃ³n Aguas Abajo (P_downstream, mca)", min_value=0.0, value=20.0, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        diam_min = st.number_input("DiÃ¡metro mÃ­nimo (mm)", min_value=0.0, value=0.0, step=1.0)
        diam_max = st.number_input("DiÃ¡metro mÃ¡ximo (mm)", min_value=0.0, value=100.0, step=1.0)
        diam_step = st.number_input("Paso de diÃ¡metro (mm)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        # Ajustar cÃ¡lculos para mm y L/s
        diameter_range_m = np.arange(diam_min, diam_max+diam_step, diam_step) / 1000
        flow_rate_imp_m3s = flow_rate_imp / 1000
        reference_diameter_m = reference_diameter_mm / 1000
        pressure_drop_values = []
        downstream_pressure_values = []
        theta_values_imp_graph = []
        P_initial_graph = P_upstream_imp
        for d in diameter_range_m:
            friction_loss = calculate_friction_losses_hw(flow_rate_imp_m3s, d, pipe_length_imp, pipe_roughness_C)
            P_downstream_calc_graph = P_initial_graph - friction_loss
            pressure_drop_values.append(friction_loss)
            downstream_pressure_values.append(P_downstream_calc_graph)
            if P_initial_graph > P_downstream_calc_graph:
                theta = cav_calc.calculate_cavitation_index(P_initial_graph, P_downstream_calc_graph)
                theta_values_imp_graph.append(theta if theta is not None else np.nan)
            else:
                theta_values_imp_graph.append(np.nan)
        # Punto rojo para el diÃ¡metro de referencia
        friction_loss_ref = calculate_friction_losses_hw(flow_rate_imp_m3s, reference_diameter_m, pipe_length_imp, pipe_roughness_C)
        P_downstream_ref = P_upstream_imp - friction_loss_ref
        theta_ref = cav_calc.calculate_cavitation_index(P_upstream_imp, P_downstream_ref)
        st.subheader("Resultados")
        theta_imp = cav_calc.calculate_cavitation_index(P_upstream_imp, P_downstream_imp)
        if theta_imp is not None:
            st.metric(label="Ãndice de CavitaciÃ³n (Ï‘)", value=f"{theta_imp:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_imp))
        else:
            st.write("Ingrese valores vÃ¡lidos para calcular el Ã­ndice de cavitaciÃ³n.")

    with col_graph:
        st.subheader("GrÃ¡fica: Impacto del DiÃ¡metro en PÃ©rdidas y Presiones")
        fig_imp = make_subplots(specs=[[{"secondary_y": True}]])
        # Convertir a mm para el eje x
        diameter_range_mm = diameter_range_m * 1000
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=pressure_drop_values, mode='lines', name='PÃ©rdida de PresiÃ³n',
                                     line=dict(color='#d62728'),
                                     hovertemplate='<b>PÃ©rdida = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#d62728', font=dict(color='white', family='Arial Black'))), secondary_y=False)
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=downstream_pressure_values, mode='lines', name='PresiÃ³n Aguas Abajo',
                                     line=dict(color='#1f77b4'),
                                     hovertemplate='<b>PresiÃ³n = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))), secondary_y=False)
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=theta_values_imp_graph, mode='lines', name='Ãndice de CavitaciÃ³n (Ï‘)',
                                     line=dict(color='#2ca02c'),
                                     hovertemplate='<b>Ï‘ = %{y:.2f}</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))), secondary_y=True)
        fig_imp.add_hline(y=cav_calc.Pv, line_dash="dot", line_color="blue", annotation_text="Pv", secondary_y=False)
        fig_imp.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="LÃ­mite DaÃ±o (0.5)", secondary_y=True)
        fig_imp.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="LÃ­mite Ruido (0.8)", secondary_y=True)
        fig_imp.update_layout(
            title="Impacto del DiÃ¡metro en PÃ©rdidas, PresiÃ³n y Ï‘ en LÃ­nea de ImpulsiÃ³n",
            xaxis=dict(
                title="DiÃ¡metro de TuberÃ­a (mm)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="PresiÃ³n / PÃ©rdida (mca)",
                tickformat='.2f',
            ),
            yaxis2=dict(
                title="Ãndice de CavitaciÃ³n (Ï‘)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Resultados de ImpulsiÃ³n</h4>', unsafe_allow_html=True)
        df_imp = pd.DataFrame({
            'DiÃ¡metro (m)': np.round(diameter_range_m, 3),
            'PÃ©rdida (mca)': np.round(pressure_drop_values, 2),
            'PresiÃ³n Abajo (mca)': np.round(downstream_pressure_values, 2),
            'Ï‘': np.round(theta_values_imp_graph, 2)
        })
        st.dataframe(df_imp, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**PÃ©rdida por fricciÃ³n (Hazen-Williams):**

$$
H_f = \frac{10.67 \cdot L \cdot Q^{1.852}}{C^{1.852} \cdot D^{4.87}}
$$

**Ãndice de CavitaciÃ³n:**

$$
\vartheta = \frac{P_{abajo} - P_{vapor}}{P_{arriba} - P_{abajo}}
$$

**Donde:**
- **Parriba**: PresiÃ³n aguas arriba (mca)
- **Pabajo**: PresiÃ³n aguas abajo (mca)
- **Pvapor**: PresiÃ³n de vapor (mca)
""")

# --- PestaÃ±a 4: LÃ­nea de SucciÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de LÃ­nea de SucciÃ³n
# --------------------------------------------------
with tab4:
    st.header("AnÃ¡lisis de CavitaciÃ³n en LÃ­nea de SucciÃ³n")
    st.write("La lÃ­nea de succiÃ³n es crÃ­tica para la cavitaciÃ³n en bombas.")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos de la LÃ­nea de SucciÃ³n")
        diam_min_s = st.number_input("DiÃ¡metro mÃ­nimo (m) succiÃ³n", min_value=0.01, value=0.05, step=0.01)
        diam_max_s = st.number_input("DiÃ¡metro mÃ¡ximo (m) succiÃ³n", min_value=0.01, value=0.4, step=0.01)
        diam_step_s = st.number_input("Paso de diÃ¡metro (m) succiÃ³n", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        P_source_suction = st.number_input("PresiÃ³n en Fuente (P_fuente, mca, ej. AtmosfÃ©rica)", min_value=0.0, value=10.33, step=0.1)
        H_static_suction_tab4 = st.number_input("Altura EstÃ¡tica al Eje de Bomba (mca, + si lÃ­quido arriba, - si abajo)", value=2.0, step=0.1)
        flow_rate_suction = st.number_input("Caudal (mÂ³/s)", min_value=0.0, value=0.05, step=0.001, format="%.3f")
        pipe_length_suction = st.number_input("Longitud de la LÃ­nea de SucciÃ³n (m)", min_value=1.0, value=10.0, step=1.0)
        pipe_roughness_C_suction = st.slider("Coeficiente Hazen-Williams (C) SucciÃ³n", min_value=80, max_value=150, value=120, step=5)
        k_fitting_suction = st.number_input("Coeficiente K de Accesorios Global (succiÃ³n)", min_value=0.0, value=2.0, step=0.1)
        st.subheader("Resultados")
        reference_diameter_suction = st.number_input("DiÃ¡metro de Referencia para CÃ¡lculo (m)", min_value=0.05, value=0.2, step=0.01)
        friction_loss_suction = calculate_friction_losses_hw(flow_rate_suction, reference_diameter_suction, pipe_length_suction, pipe_roughness_C_suction)
        local_loss_suction = calculate_local_losses(flow_rate_suction, reference_diameter_suction, k_fitting_suction)
        total_loss_suction = friction_loss_suction + local_loss_suction
        P_brida_succion_calc = P_source_suction + H_static_suction_tab4 - total_loss_suction
        npsha_suction_calc = npsh_calc.calculate_npsha(P_source_suction, H_static_suction_tab4, total_loss_suction)
        st.metric(label="PresiÃ³n en Brida de SucciÃ³n (mca)", value=f"{P_brida_succion_calc:.3f}")
        st.metric(label="NPSH Disponible (NPSHA, mca)", value=f"{npsha_suction_calc:.3f}")
        NPSHr_dummy_for_desc = 3.0
        st.write(npsh_calc.get_npsh_risk_description(npsha_suction_calc, NPSHr_dummy_for_desc))
        st.info("Nota: Para una evaluaciÃ³n completa, compare el NPSH Disponible con el NPSH Requerido de su bomba.")

    with col_graph:
        st.subheader("GrÃ¡fica: NPSHA vs. DiÃ¡metro de SucciÃ³n")
        diameter_range_suction_graph = np.arange(diam_min_s, diam_max_s+diam_step_s, diam_step_s)
        npsha_values_suction_graph = []
        for d in diameter_range_suction_graph:
            friction_loss_g = calculate_friction_losses_hw(flow_rate_suction, d, pipe_length_suction, pipe_roughness_C_suction)
            local_loss_g = calculate_local_losses(flow_rate_suction, d, k_fitting_suction)
            total_loss_g = friction_loss_g + local_loss_g
            npsha_g = npsh_calc.calculate_npsha(P_source_suction, H_static_suction_tab4, total_loss_g)
            npsha_values_suction_graph.append(npsha_g)
        fig_suction = go.Figure()
        fig_suction.add_trace(go.Scatter(x=np.round(diameter_range_suction_graph,3), y=npsha_values_suction_graph, mode='lines', name='NPSH Disponible',
                                         line=dict(color='#9467bd'),
                                         hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                         hoverlabel=dict(bgcolor='#9467bd', font=dict(color='white', family='Arial Black'))))
        fig_suction.add_hline(y=NPSHr_dummy_for_desc, line_dash="dash", line_color="red", annotation_text=f"NPSHR de Ejemplo ({NPSHr_dummy_for_desc:.1f})")
        fig_suction.update_layout(
            title="NPSH Disponible vs. DiÃ¡metro de TuberÃ­a de SucciÃ³n",
            xaxis=dict(
                title="DiÃ¡metro de TuberÃ­a (m)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="NPSH Disponible (mca)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_suction, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de NPSHA vs. DiÃ¡metro</h4>', unsafe_allow_html=True)
        df_suction = pd.DataFrame({
            'DiÃ¡metro (m)': np.round(diameter_range_suction_graph, 3),
            'NPSHA (mca)': np.round(npsha_values_suction_graph, 2)
        })
        st.dataframe(df_suction, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**PÃ©rdida por fricciÃ³n (Hazen-Williams):**

$$
H_f = \frac{10.67 \cdot L \cdot Q^{1.852}}{C^{1.852} \cdot D^{4.87}}
$$

**PÃ©rdida local:**

$$
H_{local} = K \cdot \frac{V^2}{2g}
$$

Donde:
- \(K\): Coeficiente global de accesorios
- \(V\): Velocidad del fluido (m/s)
- \(g\): Gravedad (9.81 m/sÂ²)

**NPSH Disponible:**

$$
NPSH_{A} = H_{atm} + H_{estÃ¡tica\ succiÃ³n} - H_{pÃ©rdidas\ succiÃ³n} - P_{vapor}
$$
""")

# --- PestaÃ±a 5: PÃ©rdidas por FricciÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de PÃ©rdidas por FricciÃ³n
# --------------------------------------------------
def mougnie_velocity(diametro_m):
    # V = 1.5 * sqrt(D) + 0.05
    return 1.5 * np.sqrt(diametro_m) + 0.05

with tab5:
    st.header("GrÃ¡fico de Caudal vs. PÃ©rdidas por FricciÃ³n (Hazen-Williams)")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("ParÃ¡metros de la TuberÃ­a")
        diametro_mm = st.number_input("DiÃ¡metro de tuberÃ­a (mm)", min_value=10.0, max_value=1000.0, value=63.0, step=0.1, format="%.2f")
        materiales_c = [
            {"nombre": "PVC o plÃ¡stico (nuevo)", "C": 150, "rango": "140 - 150"},
            {"nombre": "Polietileno de alta densidad (HDPE)", "C": 150, "rango": "140 - 150"},
            {"nombre": "Asbesto-cemento (nuevo)", "C": 140, "rango": "130 - 140"},
            {"nombre": "Hierro fundido (nuevo)", "C": 130, "rango": "120 - 130"},
            {"nombre": "Hierro fundido (usado)", "C": 110, "rango": "80 - 130"},
            {"nombre": "Acero nuevo (sin costura)", "C": 145, "rango": "130 - 150"},
            {"nombre": "Acero galvanizado (nuevo)", "C": 120, "rango": "110 - 120"},
            {"nombre": "Acero galvanizado (viejo)", "C": 90, "rango": "70 - 110"},
            {"nombre": "HormigÃ³n (buen acabado)", "C": 130, "rango": "120 - 140"},
            {"nombre": "HormigÃ³n (viejo o rugoso)", "C": 110, "rango": "90 - 130"},
            {"nombre": "Cobre o latÃ³n", "C": 145, "rango": "130 - 150"},
            {"nombre": "Fibrocemento", "C": 135, "rango": "130 - 140"},
        ]
        nombres_materiales = [m["nombre"] for m in materiales_c]
        material_idx = st.selectbox("Material de la tuberÃ­a", nombres_materiales, index=0)
        material_seleccionado = materiales_c[nombres_materiales.index(material_idx)]
        # Permitir modificar C dentro del rango sugerido
        rango_c = material_seleccionado["rango"].replace(" ", "").split("-")
        c_min = int(rango_c[0])
        c_max = int(rango_c[1])
        C_hw = st.number_input(
            "Coeficiente Hazen-Williams (C)",
            min_value=c_min,
            max_value=c_max,
            value=material_seleccionado["C"],
            step=1
        )
        st.info(f"Material simulado: **{material_seleccionado['nombre']}**\n\nC tÃ­pico: **{material_seleccionado['C']}**\n\nRango de C permitido: {material_seleccionado['rango']}")
        st.subheader("Rango de Caudal (L/s)")
        caudal_min = st.number_input("Caudal mÃ­nimo (L/s)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)
        caudal_max = st.number_input("Caudal mÃ¡ximo (L/s)", min_value=0.01, max_value=100.0, value=5.0, step=0.01)
        paso_caudal = st.number_input("Paso de caudal (L/s)", min_value=0.01, max_value=10.0, value=0.25, step=0.01)
        mostrar_umbral = False
        umbral_pend = None
        with st.expander("Opciones avanzadas: Umbral de pendiente para zona segura", expanded=False):
            st.caption("El umbral de pendiente define hasta quÃ© valor de pendiente (variaciÃ³n de pÃ©rdida por variaciÃ³n de caudal) se considera la zona segura. Si la pendiente local supera este valor, se considera que la curva ya no es segura para diseÃ±o.")
            umbral_pend = st.number_input("Umbral de pendiente para zona segura (m/Km por L/s)", min_value=0.01, max_value=100.0, value=1.0, step=0.25, key="umbral_pend")
            col1, col2 = st.columns([1,1])
            with col1:
                aplicar_umbral = st.button("Aplicar umbral de pendiente", key="btn_umbral")
            with col2:
                borrar_umbral = st.button("Borrar", key="btn_borrar_umbral")
            if aplicar_umbral:
                mostrar_umbral = True
            if borrar_umbral:
                mostrar_umbral = False

    with col_graph:
        st.subheader(f"GrÃ¡fica: PÃ©rdida de carga por fricciÃ³n vs. Caudal  ", divider="gray")
        st.markdown(f"**Material simulado:** {material_seleccionado['nombre']}  |  C tÃ­pico: {material_seleccionado['C']}  |  Rango de C: {material_seleccionado['rango']}")
        Q_lps_graf = np.arange(caudal_min, caudal_max + paso_caudal, paso_caudal)
        diametro_m = diametro_mm / 1000
        Q_m3s_graf = Q_lps_graf / 1000
        L = 1  # longitud en metros
        j_graf = [10.67 * L * (q ** 1.852) / (C_hw ** 1.852 * diametro_m ** 4.87) for q in Q_m3s_graf]
        hf_km_graf = [j * 1000 for j in j_graf]

        # --- AnÃ¡lisis: punto donde la pendiente supera el umbral ---
        hf_km_graf = np.array(hf_km_graf)
        Q_lps_graf = np.array(Q_lps_graf)
        if mostrar_umbral and umbral_pend is not None:
            pendiente = np.diff(hf_km_graf) / np.diff(Q_lps_graf)
            idx_cambio = next((i for i, p in enumerate(pendiente) if p > umbral_pend), None)
            if idx_cambio is not None:
                # El punto de cambio es el primer Q donde la pendiente supera el umbral
                caudal_div = float(Q_lps_graf[idx_cambio+1])
                cambio_detectado = True
            else:
                caudal_div = None
                cambio_detectado = False
            # Ãrea bajo la curva segura (antes del punto de cambio)
            if cambio_detectado:
                area_segura = np.trapz(hf_km_graf[:idx_cambio+2], Q_lps_graf[:idx_cambio+2])
            else:
                area_segura = np.trapz(hf_km_graf, Q_lps_graf)
        else:
            cambio_detectado = False
            caudal_div = None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Q_lps_graf,
            y=hf_km_graf,
            mode='lines',
            name='PÃ©rdida de carga (m/Km)',
            line=dict(color='#ff7f0e'),
            hovertemplate='<b>Q = %{x:.2f} L/s<br>hf/Km = %{y:.2f} m/Km</b><extra></extra>'
        ))
        # LÃ­nea vertical en el punto de cambio solo si se aplica el umbral
        if mostrar_umbral and cambio_detectado and caudal_div is not None:
            fig.add_vline(x=caudal_div, line_dash="dot", line_color="#0080ff", line_width=2,
                          annotation_text=f"Q={caudal_div:.2f} L/s", annotation_position="top right")
        fig.update_layout(
            title=f"PÃ©rdida de carga por fricciÃ³n vs. Caudal  |  Material: {material_seleccionado['nombre']} (C={C_hw})",
            xaxis=dict(
                title="Caudal (L/s)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="PÃ©rdida de carga (m/Km)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        if mostrar_umbral:
            if cambio_detectado and caudal_div is not None:
                st.info(f"El punto de cambio de pendiente se estima en Q = {caudal_div:.2f} L/s")
            else:
                st.warning(f"No se detectÃ³ un cambio de pendiente dentro del rango y umbral seleccionados. La lÃ­nea se muestra al final del rango.")

        # GrÃ¡fico adicional: Caudal vs Velocidad
        V_graf = Q_m3s_graf / (np.pi * (diametro_m / 2) ** 2) if diametro_m > 0 else np.zeros_like(Q_m3s_graf)
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=Q_lps_graf,
            y=V_graf,
            mode='lines',
            name='Velocidad (m/s)',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>Q = %{x:.2f} L/s<br>V = %{y:.2f} m/s</b><extra></extra>'
        ))
        fig_vel.update_layout(
            title="Caudal vs. Velocidad en la TuberÃ­a",
            xaxis=dict(
                title="Caudal (L/s)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="Velocidad (m/s)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de PÃ©rdidas por FricciÃ³n</h4>', unsafe_allow_html=True)
        df_friction = pd.DataFrame({
            'Caudal (L/s)': np.round(Q_lps_graf, 2),
            'PÃ©rdida (m/Km)': np.round(hf_km_graf, 2),
            'Velocidad (m/s)': np.round(V_graf, 2)
        })
        st.dataframe(df_friction, use_container_width=True)
        
        with st.expander("ðŸ“š Coeficiente de caudal (Kv)"):
            st.markdown("""
Las vÃ¡lvulas de control son conceptualmente orificios de Ã¡rea variable. Se las puede considerar simplemente como una restricciÃ³n que cambia su tamaÃ±o de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relaciÃ³n de diferencia de altura (Î”h) o presiÃ³n (Î”P) entre la entrada y salida de la vÃ¡lvula con el caudal (Q).
""")
            st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
            st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (mÂ³/h)
- $Q$: Caudal volumÃ©trico (mÂ³/h)
- $\rho$: Densidad (kg/mÂ³)
- $\Delta p$: Diferencia de presiÃ³n (bar)
- $P_1$: PresiÃ³n de entrada (bar)
- $P_2$: PresiÃ³n de salida (bar)
""")
            mca_a_bar = 0.0980665
            P1_bar = P1_valve * mca_a_bar
            P2_bar = P2_valve * mca_a_bar
            delta_p_bar = np.abs(P1_bar - P2_bar)
            densidad = densidad_calculada
            st.markdown(f"**PresiÃ³n de entrada (P1):** {P1_bar:.2f} bar  ")
            st.markdown(f"**PresiÃ³n de salida (P2):** {P2_bar:.2f} bar  ")
            st.markdown(f"**Diferencia de presiÃ³n (Î”p):** {delta_p_bar:.2f} bar  ")
            st.markdown(f"**Densidad (Ï):** {densidad:.1f} kg/mÂ³  ")

# --- NUEVA PESTAÃ‘A: Flujos Transitorios ---
with tab_trans:
    st.header("ðŸŒŠ Flujos Transitorios")
    # Primer bloque: CÃ¡lculo de celeridad y espesor de tuberÃ­a
    st.subheader("CÃ¡lculo de velocidad de propagaciÃ³n de onda (celeridad) y espesor de tuberÃ­a")
    col1, col2, col3, col4 = st.columns([0.22, 0.4, 0.28, 0.1])
    with col1:
        st.markdown("#### Datos de entrada")
        materiales = {
            "Acero": {"E": (200e9, 212e9), "a": (1000, 1250)},
            "Fibro Cemento": {"E": (23.5e9, 23.5e9), "a": (900, 1200)},
            "Concreto": {"E": (39e9, 39e9), "a": (1050, 1150)},
            "Hierro DÃºctil": {"E": (166e9, 166e9), "a": (1000, 1350)},
            "Polietileno alta densidad": {"E": (0.59e9, 1.67e9), "a": (230, 430)},
            "PVC": {"E": (2.4e9, 2.75e9), "a": (300, 500)},
        }
        material = st.selectbox("Material de la tuberÃ­a", list(materiales.keys()))
        E_range = materiales[material]["E"]
        a_range = materiales[material]["a"]
        E_prom = sum(E_range) / 2
        a_prom = sum(a_range) / 2
        st.markdown(f"**Rango de mÃ³dulo de elasticidad (E):** {E_range[0]/1e9:.2f} - {E_range[1]/1e9:.2f} GPa")
        st.markdown(f"**Rango de celeridad (a):** {a_range[0]:.0f} - {a_range[1]:.0f} m/s")
        # Input en GPa, conversiÃ³n a Pa para cÃ¡lculos
        E_GPa = st.number_input("MÃ³dulo de elasticidad E (GPa)", min_value=0.01, max_value=300.0, value=float(E_prom/1e9), step=0.01, format="%.2f")
        E = E_GPa * 1e9
        a = st.number_input("Celeridad a (m/s)", min_value=100.0, max_value=2000.0, value=float(a_prom), step=1.0, key="a_material")
        densidad = st.number_input("Densidad del agua (kg/mÂ³)", min_value=800.0, max_value=1200.0, value=float(densidad_calculada), step=1.0)
        D = st.number_input("DiÃ¡metro de la tuberÃ­a (mm)", min_value=10.0, max_value=2000.0, value=100.0, step=1.0) / 1000
        K = st.number_input("MÃ³dulo de elasticidad del fluido K (N/mÂ²)", min_value=1e8, max_value=3e9, value=2.2e9, step=1e7, format="%.0f")
        
        # CÃ¡lculo de delta
        denominador = K - densidad * a**2
        if denominador <= 0:
            st.error("Error: El denominador (K - ÏaÂ²) debe ser positivo. Revise los valores de K, densidad y celeridad.")
            delta_calc = None
            delta_mm = None
        else:
            delta_calc = (K * D * densidad * a**2) / (E * denominador)
            delta_mm = delta_calc * 1000
            # --- NUEVO: Mostrar datos PEAD si corresponde ---
            if material == "Polietileno alta densidad":
                # Tabla extendida PEAD: cada fila es un diÃ¡metro nominal, con sus espesores, series, SDR, diÃ¡metro interno y presiÃ³n
                pead_tabla = [
                    {"DN": 20, "series": ["S5", "S4"], "SDR": [11, 9], "esp": [2.0, 2.3], "dint": [16.0, 15.4], "pres": [1.6, 2.0]},
                    {"DN": 25, "series": ["S5", "S4"], "SDR": [11, 9], "esp": [2.3, 2.8], "dint": [20.4, 19.4], "pres": [1.6, 2.0]},
                    {"DN": 32, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [2.4, 3.0, 3.6], "dint": [27.2, 26.0, 24.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 40, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [3.0, 3.7, 4.5], "dint": [34.0, 32.6, 31.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 50, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [3.7, 4.6, 5.6], "dint": [42.6, 40.8, 38.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 63, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [4.7, 5.8, 7.1], "dint": [53.6, 51.4, 48.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 75, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [5.6, 6.8, 8.4], "dint": [63.8, 61.4, 58.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 90, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [6.7, 8.2, 10.1], "dint": [76.6, 73.6, 69.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 110, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [8.1, 10.0, 12.3], "dint": [93.8, 90.0, 85.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 125, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [9.2, 11.4, 14.0], "dint": [106.6, 102.2, 97.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 140, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [10.3, 12.7, 15.7], "dint": [119.4, 114.6, 108.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 160, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [11.8, 14.6, 18.0], "dint": [136.4, 130.8, 124.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 180, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [13.3, 16.4, 20.1], "dint": [153.4, 147.2, 139.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 200, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [14.7, 18.2, 22.7], "dint": [167.6, 163.6, 154.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 225, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [16.6, 20.5, 25.6], "dint": [191.8, 184.0, 173.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 250, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [18.4, 22.7, 28.4], "dint": [213.2, 204.6, 193.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 280, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [20.6, 25.4, 31.8], "dint": [238.8, 229.2, 217.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 315, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [23.2, 28.6, 35.7], "dint": [268.6, 257.8, 243.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 355, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [26.1, 32.2, 40.2], "dint": [302.8, 290.6, 274.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 400, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [29.2, 36.3, 45.4], "dint": [341.6, 327.4, 309.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 450, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [32.8, 41.0, 51.0], "dint": [384.4, 368.2, 349.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 500, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [36.3, 45.4, 56.8], "dint": [427.4, 409.2, 386.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 560, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [40.7, 51.0, 63.5], "dint": [479.6, 458.4, 433.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 630, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [46.3, 57.2, 71.0], "dint": [555.2, 514.6, 489.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 710, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [52.2, 64.7, 80.3], "dint": [622.2, 580.6, 549.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 800, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [58.1, 72.7, 90.8], "dint": [705.2, 654.8, 618.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 900, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [65.1, 81.7, 102.3], "dint": [793.4, 738.8, 695.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 1000, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [73.5, 90.8, 115.0], "dint": [881.4, 818.4, 770.0], "pres": [1.0, 1.6, 2.0]},
                ]
                DN_input = int(round(D*1000))
                DN_candidatos = [fila["DN"] for fila in pead_tabla]
                DN_cercano = min(DN_candidatos, key=lambda x: abs(x - DN_input))
                fila = next(f for f in pead_tabla if f["DN"] == DN_cercano)
                if delta_mm is not None:
                    # Buscar el primer espesor nominal igual o superior al calculado
                    idx = next((i for i, e in enumerate(fila["esp"]) if e >= delta_mm), len(fila["esp"]) - 1)
                    serie = fila["series"][idx]
                    SDR = fila["SDR"][idx]
                    esp_nom = fila["esp"][idx]
                    dint = fila["dint"][idx]
                    pres = fila["pres"][idx]
                    st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm**\n\n**PEAD:**\n- DiÃ¡metro nominal: {DN_cercano} mm\n- DiÃ¡metro interior: {dint:.1f} mm\n- Serie: {serie}\n- SDR: {SDR}\n- Espesor nominal: {esp_nom:.2f} mm\n- PresiÃ³n de trabajo: {pres} MPa")
                else:
                    st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm**\n\n**PEAD:**\n- DiÃ¡metro nominal: {DN_cercano} mm\n- (No se pudo determinar serie ni presiÃ³n de trabajo)")
            else:
                st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm")
            if delta_mm > 50:
                st.warning("El espesor calculado es muy grande. Revise los parÃ¡metros o considere que para materiales plÃ¡sticos este cÃ¡lculo puede no ser aplicable para diseÃ±o real.")
        # Mostrar resultados de celeridad y espesor
        st.info(f"**Resultados:**\n- Celeridad (a): {a:.2f} m/s\n- Espesor calculado (Î´): {delta_mm:.2f} mm")
        
        st.markdown("---")
        st.markdown("### Datos de grÃ¡fico")
        espesor_inicial = st.number_input("Espesor inicial (mm)", min_value=0.0, max_value=50.0, value=0.1, step=0.1)
        espesor_final = st.number_input("Espesor final (mm)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        paso_espesor = st.number_input("Paso de espesor (mm)", min_value=0.1, max_value=5.0, value=0.1, step=0.1)
    with col2:
        st.markdown("#### GrÃ¡fico de Celeridad vs Espesor de TuberÃ­a")
        st.markdown("Este grÃ¡fico muestra cÃ³mo varÃ­a la celeridad en funciÃ³n del espesor de la tuberÃ­a para diferentes materiales.")
        
        # Generar datos para el grÃ¡fico
        espesores_mm = np.arange(espesor_inicial, espesor_final + paso_espesor, paso_espesor)
        espesores_m = espesores_mm / 1000
        
        # Calcular celeridad para cada material
        fig_celeridad = go.Figure()
        
        for mat_name, mat_props in materiales.items():
            E_mat = sum(mat_props["E"]) / 2  # Usar valor promedio
            celeridades = []
            
            for delta_m in espesores_m:
                if delta_m > 0:  # Evitar divisiÃ³n por cero
                    # FÃ³rmula de celeridad: a = sqrt(K/Ï / (1 + K*D/(E*Î´)))
                    celeridad = np.sqrt((K / densidad) / (1 + (K * D) / (E_mat * delta_m)))
                    celeridades.append(celeridad)
                else:
                    celeridades.append(np.sqrt(K / densidad))  # Celeridad sin tuberÃ­a
            
            # Color segÃºn el material
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color_idx = list(materiales.keys()).index(mat_name) % len(colors)
            
            fig_celeridad.add_trace(go.Scatter(
                x=espesores_mm,
                y=celeridades,
                mode='lines',
                name=mat_name,
                line=dict(color=colors[color_idx], width=2),
                hovertemplate='<b>%{fullData.name}: %{y:.0f} m/s</b><extra></extra>',
                hoverlabel=dict(bgcolor=colors[color_idx], font=dict(color='white', family='Arial Black'))
            ))
        
        # Agregar punto del material seleccionado
        if delta_mm is not None and delta_mm > 0:
            E_selected = E
            celeridad_selected = np.sqrt((K / densidad) / (1 + (K * D) / (E_selected * delta_calc)))
            fig_celeridad.add_trace(go.Scatter(
                x=[delta_mm],
                y=[celeridad_selected],
                mode='markers',
                name=f'{material} (calculado)',
                marker=dict(color='red', size=10, symbol='star'),
                hovertemplate='<b>%{fullData.name}: %{y:.0f} m/s</b><extra></extra>',
                hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
            ))
        
        fig_celeridad.update_layout(
            title="Celeridad vs Espesor de TuberÃ­a",
            xaxis=dict(
                title="Espesor de tuberÃ­a (mm)",
                showgrid=True,
                gridcolor='#cccccc',
                gridwidth=0.7,
                zeroline=False,
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                hoverformat='.1f'
            ),
            yaxis=dict(
                title="Celeridad (m/s)",
                showgrid=True,
                gridcolor='#cccccc',
                gridwidth=0.7,
                zeroline=False,
                showspikes=False,
                showline=True,
                hoverformat='.0f'
            ),
            hovermode="x",
            legend=dict(orientation='h', yanchor='bottom', y=-0.6, xanchor='center', x=0.5)
        )
        
        st.plotly_chart(fig_celeridad, use_container_width=True)

        # --- GrÃ¡fico Conceptual de AceleraciÃ³n vs Ãndice de CavitaciÃ³n (Ïƒ) ---
        st.markdown('#### Ãndice de CavitaciÃ³n vs AceleraciÃ³n (grÃ¡fico conceptual)')
        import plotly.graph_objects as go
        # Curva conceptual (valores aproximados para ilustrar el comportamiento)
        sigma_concept = [1, 1.2, 1.5, 1.7, 2, 4, 8, 20]
        aceleracion_concept = [a*1.15, a*1.2, a*1.1, a*0.9, a*0.7, a*0.5, a*0.3, a*0.2]
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(
            x=sigma_concept,
            y=aceleracion_concept,
            mode='lines+markers',
            name='Curva conceptual',
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black'),
            hovertemplate='<b>Ïƒ = %{x:.2f}<br>AceleraciÃ³n = %{y:.0f} m/s</b><extra></extra>'
        ))
        # Punto del material seleccionado (en Ïƒ=2, por ejemplo)
        fig_sigma.add_trace(go.Scatter(
            x=[2],
            y=[a],
            mode='markers',
            name=f'Material seleccionado (a = {a:.0f} m/s)',
            marker=dict(color='red', size=7, symbol='star'),
            hovertemplate='<b>Ïƒ = 2.00<br>AceleraciÃ³n = %{y:.0f} m/s</b><extra></extra>'
        ))
        # Zonas de colores
        fig_sigma.add_vrect(x0=2, x1=20, fillcolor='green', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=1.7, x1=2, fillcolor='#7ca300', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=1.5, x1=1.7, fillcolor='orange', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=1, x1=1.5, fillcolor='#ff6600', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=0, x1=1, fillcolor='red', opacity=0.2, layer='below', line_width=0)
        # LÃ­mites y etiquetas
        fig_sigma.update_layout(
            title='Ãndice de CavitaciÃ³n (Ïƒ) vs AceleraciÃ³n (Curva Conceptual)',
            xaxis=dict(title='Ãndice de CavitaciÃ³n Ïƒ', range=[0.8, 4], tickvals=[1, 1.5, 1.7, 2, 3, 4]),
            yaxis=dict(title='AceleraciÃ³n (m/s)', range=[0, a*1.3]),
            hovermode='x',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        # Etiquetas de zonas en vertical, en negrita y al inicio de cada zona
        fig_sigma.add_annotation(x=1, y=0, text='<b>Flashing</b>', showarrow=False, font=dict(color='red', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=1.5, y=0, text='<b>CavitaciÃ³n severa</b>', showarrow=False, font=dict(color='#ff6600', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=1.7, y=0, text='<b>CavitaciÃ³n incipiente</b>', showarrow=False, font=dict(color='orange', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=2, y=0, text='<b>ProtecciÃ³n suficiente</b>', showarrow=False, font=dict(color='#7ca300', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=2.5, y=0, text='<b>No cavitaciÃ³n</b>', showarrow=False, font=dict(color='green', size=12), textangle=-90, xanchor='left', yanchor='bottom')
        st.plotly_chart(fig_sigma, use_container_width=True)
    with col3:
        with st.expander("ðŸ“š FÃ³rmulas y TeorÃ­a"):
            st.markdown("#### FÃ³rmula de celeridad")
            st.latex(r"a = \frac{K}{\rho} \frac{1}{\sqrt{1 + \frac{K D}{E \delta}}}")
            st.markdown("#### FÃ³rmula de espesor de tuberÃ­a")
            st.latex(r"\delta = \frac{K D \rho a^2}{E (K - \rho a^2)}")
            st.markdown("""
**Fundamento teÃ³rico:**
Esta fÃ³rmula se deriva de la ecuaciÃ³n de celeridad del golpe de ariete, despejando el espesor Î´.
La celeridad a estÃ¡ relacionada con las propiedades del fluido y la tuberÃ­a mediante:

a = \sqrt{\frac{K/\rho}{1 + \frac{K D}{E \delta}}}

Despejando Î´ se obtiene la fÃ³rmula mostrada.
""")
            # Agregar fÃ³rmulas de Ã­ndice de cavitaciÃ³n y concepto
            st.markdown("---")
            st.markdown("#### Ãndice de cavitaciÃ³n")
            st.latex(r"\sigma = \frac{P_1 - P_v}{P_1 - P_2}")
            st.latex(r"\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}")
            st.markdown("""
El **Ã­ndice de cavitaciÃ³n** es un parÃ¡metro adimensional que permite evaluar el riesgo de formaciÃ³n de cavitaciÃ³n en sistemas hidrÃ¡ulicos. Compara la presiÃ³n disponible en el sistema con la presiÃ³n de vapor del fluido. Un valor bajo indica mayor riesgo de cavitaciÃ³n, mientras que valores altos indican condiciones seguras.

- **Pâ‚:** PresiÃ³n de entrada (aguas arriba)
- **Pâ‚‚:** PresiÃ³n de salida (aguas abajo)
- **P_v / P_{vapor}:** PresiÃ³n de vapor del fluido
""")
        # Panel desplegable con definiciÃ³n de grÃ¡fico conceptual
        with st.expander("Â¿QuÃ© significa 'grÃ¡fico conceptual'?"):
            st.markdown("""
Un **grÃ¡fico conceptual** es una representaciÃ³n idealizada o esquemÃ¡tica que ilustra tendencias generales, relaciones teÃ³ricas o comportamientos tÃ­picos de un fenÃ³meno fÃ­sico. No se basa en datos experimentales ni en cÃ¡lculos especÃ­ficos para un caso real, sino que ayuda a visualizar de manera clara y didÃ¡ctica cÃ³mo suelen comportarse las variables bajo ciertas condiciones ideales o teÃ³ricas. Su objetivo es facilitar la comprensiÃ³n de conceptos clave, no mostrar resultados numÃ©ricos exactos.
""")

        with st.expander("ðŸ“‹ Tabla de materiales y celeridad"):
            import pandas as pd
            df_mat = pd.DataFrame([
                {
                    "Material": k,
                    "E (GPa)": f"{v['E'][0]/1e9:.2f} - {v['E'][1]/1e9:.2f}" if v['E'][0] != v['E'][1] else f"{v['E'][0]/1e9:.2f}",
                    "a (m/s)": f"{v['a'][0]:.0f} - {v['a'][1]:.0f}"
                }
                for k, v in materiales.items()
            ])
            st.dataframe(df_mat, use_container_width=True)
        
        st.markdown("#### Tabla de datos del grÃ¡fico")
        # Crear tabla con los datos del grÃ¡fico
        espesores_mm_tabla = np.arange(espesor_inicial, espesor_final + paso_espesor, paso_espesor)
        espesores_m_tabla = espesores_mm_tabla / 1000
        
        # Calcular celeridades para la tabla
        datos_tabla = {}
        for mat_name, mat_props in materiales.items():
            E_mat = sum(mat_props["E"]) / 2
            celeridades_mat = []
            
            for delta_m in espesores_m_tabla:
                if delta_m > 0:
                    celeridad = np.sqrt((K / densidad) / (1 + (K * D) / (E_mat * delta_m)))
                else:
                    celeridad = np.sqrt(K / densidad)
                celeridades_mat.append(celeridad)
            
            datos_tabla[mat_name] = celeridades_mat
        
        # Crear DataFrame con materiales en columnas y espesores en filas
        df_celeridad = pd.DataFrame(datos_tabla, index=espesores_mm_tabla)
        df_celeridad.index.name = "Espesor (mm)"
        
        # Formatear valores de celeridad
        for col in df_celeridad.columns:
            df_celeridad[col] = df_celeridad[col].apply(lambda x: f"{x:.0f}")
        
        st.dataframe(df_celeridad, use_container_width=True)
        # Panel desplegable con definiciÃ³n de grÃ¡fico conceptual debajo de la tabla
        with st.expander("GrÃ¡fico conceptual"):
            st.markdown("""
Un **grÃ¡fico conceptual** es una representaciÃ³n idealizada o esquemÃ¡tica que ilustra tendencias generales, relaciones teÃ³ricas o comportamientos tÃ­picos de un fenÃ³meno fÃ­sico. No se basa en datos experimentales ni en cÃ¡lculos especÃ­ficos para un caso real, sino que ayuda a visualizar de manera clara y didÃ¡ctica cÃ³mo suelen comportarse las variables bajo ciertas condiciones ideales o teÃ³ricas. Su objetivo es facilitar la comprensiÃ³n de conceptos clave, no mostrar resultados numÃ©ricos exactos.
""")
        with st.expander("Tabla PEAD (Polietileno de alta densidad)"):
            st.markdown("""
| DiÃ¡metro Nominal (mm) | SDR 26 | SDR 21 | SDR 17 | SDR 13.6 | SDR 11 | SDR 9 |
|----------------------|--------|--------|--------|----------|--------|-------|
| 20  | 2.0 | 2.0 | 2.3 | 2.8 | 3.0 | 3.6 |
| 25  | -   | 2.0 | 2.3 | 2.8 | 3.0 | 3.6 |
| 32  | -   | 2.0 | 2.4 | 3.0 | 3.7 | 4.4 |
| 40  | 2.0 | 2.4 | 3.0 | 3.7 | 4.6 | 5.5 |
| 50  | 2.0 | 2.4 | 3.7 | 4.6 | 5.6 | 6.9 |
| 63  | 2.5 | 3.0 | 4.7 | 5.8 | 7.1 | 8.6 |
| 75  | 2.9 | 3.6 | 5.6 | 6.8 | 8.4 | 10.3 |
| 90  | 3.5 | 4.3 | 6.7 | 8.2 | 10.1 | 12.3 |
| 110 | 4.2 | 5.3 | 8.1 | 10.0 | 12.3 | 15.0 |
| 125 | 4.8 | 6.0 | 9.2 | 11.4 | 14.0 | 17.1 |
| 140 | 5.4 | 6.7 | 10.3 | 12.7 | 15.7 | 19.2 |
| 160 | 6.2 | 7.7 | 11.8 | 14.6 | 18.0 | 22.0 |
| 180 | 6.9 | 8.6 | 13.3 | 16.4 | 20.1 | 24.7 |
| 200 | 7.7 | 9.6 | 14.7 | 18.2 | 22.7 | 27.4 |
| 225 | 8.6 | 10.8 | 16.6 | 20.5 | 25.6 | 30.8 |
| 250 | 9.6 | 12.3 | 18.4 | 22.7 | 28.4 | 34.2 |
| 280 | 10.7 | 13.7 | 20.6 | 25.4 | 31.8 | 38.3 |
| 315 | 12.1 | 15.4 | 23.2 | 28.6 | 35.7 | 43.1 |
| 355 | 13.6 | 17.6 | 26.1 | 32.2 | 40.2 | 48.3 |
| 400 | 15.3 | 19.6 | 29.2 | 36.3 | 45.4 | 54.5 |
| 450 | 17.2 | 21.5 | 32.8 | 41.0 | 51.0 | 61.4 |
| 500 | 19.1 | 23.9 | 36.3 | 45.4 | 56.8 | 68.2 |
| 560 | 21.4 | 26.7 | 40.7 | 51.0 | 63.5 | 76.4 |
| 630 | 24.1 | 30.0 | 46.3 | 57.2 | 71.0 | 85.7 |
| 710 | 27.2 | 34.0 | 52.2 | 64.7 | 80.3 | 97.0 |
| 800 | 30.6 | 38.1 | 58.1 | 72.7 | 90.8 | 109.7 |
| 900 | 34.4 | 42.9 | 65.1 | 81.7 | 102.3 | 123.4 |
| 1000 | 38.2 | 47.7 | 73.5 | 90.8 | 115.0 | 138.1 |

**PresiÃ³n nominal de trabajo (Mpa):**
- SDR 26: 0.63
- SDR 21: 0.8
- SDR 17: 1.0
- SDR 13.6: 1.25
- SDR 11: 1.6
- SDR 9: 2.0
""")
    with col4:
        pass
    st.markdown("---")
    st.markdown(f"**Caudal calculado Q = {Q:.4f} mÂ³/s ({Q*3600:.2f} mÂ³/h, {Q*1000:.2f} L/s)**")
    st.caption("La fÃ³rmula asume flujo libre y condiciones ideales. Para condiciones reales, considerar pÃ©rdidas adicionales y coeficientes de seguridad.")
    st.markdown("**Importancia del coeficiente de descarga (C):** Un valor alto de C indica que la vÃ¡lvula permite un flujo eficiente con mÃ­nima resistencia, mientras que un valor bajo indica mayor restricciÃ³n al flujo. La elecciÃ³n del tipo de vÃ¡lvula depende del caudal requerido, presiÃ³n del sistema y necesidad de control fino del flujo.")

# --- PestaÃ±a 1: VÃ¡lvula ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de VÃ¡lvula
# --------------------------------------------------
with tab1:
    st.header("AnÃ¡lisis de CavitaciÃ³n en VÃ¡lvula")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    # Definir P2_range y theta_values antes de los bloques de columnas para uso global en la pestaÃ±a
    P2_min_default = 0.0
    P2_max_default = 8.0
    P2_step_default = 0.25
    # Se usan los valores por defecto, pero luego se actualizan con los inputs
    P2_min = P2_min_default
    P2_max = P2_max_default
    P2_step = P2_step_default
    with col_input:
        st.subheader("Datos de la VÃ¡lvula")
        P1_valve = st.number_input("PresiÃ³n de Entrada (P1, mca)", min_value=0.0, value=10.0, step=0.1)
        P2_valve = st.number_input("PresiÃ³n de Salida (P2, mca)", min_value=0.0, value=5.0, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        P2_min = st.number_input("P2 mÃ­nimo (mca)", min_value=0.0, value=P2_min_default, step=0.1)
        P2_max = st.number_input("P2 mÃ¡ximo (mca)", min_value=0.0, value=P2_max_default, step=0.1)
        P2_step = st.number_input("Paso de P2 (mca)", min_value=0.001, max_value=10.0, value=P2_step_default, step=0.1)
        st.subheader("Resultados")
        theta_valve = cav_calc.calculate_cavitation_index(P1_valve, P2_valve)
        if theta_valve is not None:
            st.metric(label="Ãndice de CavitaciÃ³n (Ï‘)", value=f"{theta_valve:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_valve))
        else:
            st.write("Ingrese valores vÃ¡lidos para calcular el Ã­ndice de cavitaciÃ³n.")
        st.info("""
**InterpretaciÃ³n:**
- Si Ï‘ < 0.5: Riesgo crÃ­tico de daÃ±o por cavitaciÃ³n.
- Si 0.5 â‰¤ Ï‘ < 0.8: Riesgo alto de ruido por cavitaciÃ³n.
- Si Ï‘ â‰¥ 0.8: Riesgo bajo o nulo de cavitaciÃ³n.
""")
        # ExplicaciÃ³n Kv ...
    # Calcular P2_range y theta_values despuÃ©s de obtener los inputs
    P2_range = np.arange(P2_min, P2_max+P2_step, P2_step)
    theta_values = []
    for p2 in P2_range:
        if P1_valve > p2:
            theta = cav_calc.calculate_cavitation_index(P1_valve, p2)
            theta_values.append(theta if theta is not None else np.nan)
        else:
            theta_values.append(np.nan)

    with col_graph:
        st.subheader("GrÃ¡fica: Ãndice de CavitaciÃ³n vs. PresiÃ³n de Salida")
        fig_valve = go.Figure()
        fig_valve.add_trace(go.Scatter(x=P2_range, y=theta_values, mode='lines+markers',
                                       line=dict(color='#2ca02c'),
                                       marker=dict(symbol='diamond-open', color='blue'),
                                       hovertemplate='<b>P2 = %{x:.2f} mca<br>Ï‘ = %{y:.2f}</b><extra></extra>',
                                       hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))))
        fig_valve.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="LÃ­mite DaÃ±o (0.5)", secondary_y=False)
        fig_valve.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="LÃ­mite Ruido (0.8)", secondary_y=False)
        fig_valve.update_layout(
            title="Ãndice de CavitaciÃ³n vs. PresiÃ³n de Salida en VÃ¡lvula",
            xaxis=dict(
                title="PresiÃ³n de Salida (mca)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="Ãndice de CavitaciÃ³n (Ï‘)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_valve, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Ãndice de CavitaciÃ³n</h4>', unsafe_allow_html=True)
        df_valve = pd.DataFrame({
            'P2 (mca)': np.round(P2_range, 2),
            'Ï‘': np.round(theta_values, 2)
        })
        st.dataframe(df_valve, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"): 
            st.markdown(r"""
**Ãndice de CavitaciÃ³n:**

$$
\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}
$$

**Donde:**
- **P1**: PresiÃ³n de entrada (mca)
- **P2**: PresiÃ³n de salida (mca)
- **Pvapor**: PresiÃ³n de vapor del fluido (mca)

Un valor bajo de **Ï‘** indica mayor riesgo de cavitaciÃ³n.
""")
        # Panel independiente de explicaciÃ³n de presiones
        with st.expander("Â¿QuÃ© significan la presiÃ³n de entrada y salida?"):
            st.markdown("""
- **PresiÃ³n de entrada (Pâ‚):** Es la presiÃ³n del fluido justo antes de entrar a la vÃ¡lvula. Representa la energÃ­a por unidad de peso del fluido en el punto aguas arriba de la vÃ¡lvula, normalmente medida en metros de columna de agua (mca). Esta presiÃ³n depende de la altura, el caudal y las condiciones del sistema antes de la vÃ¡lvula.

- **PresiÃ³n de salida (Pâ‚‚):** Es la presiÃ³n del fluido justo despuÃ©s de pasar por la vÃ¡lvula. Representa la energÃ­a por unidad de peso del fluido en el punto aguas abajo de la vÃ¡lvula, tambiÃ©n en metros de columna de agua (mca). Esta presiÃ³n suele ser menor que la de entrada debido a la pÃ©rdida de energÃ­a (caÃ­da de presiÃ³n) que ocurre al atravesar la vÃ¡lvula.

**En resumen:**
- **Pâ‚** indica cuÃ¡nta presiÃ³n tiene el fluido antes de la vÃ¡lvula.
- **Pâ‚‚** indica cuÃ¡nta presiÃ³n queda despuÃ©s de la vÃ¡lvula.
""")
        with st.expander("Criterios de cavitaciÃ³n segÃºn sigma (Ïƒ)"):
            st.markdown("""
**Criterios de cavitaciÃ³n segÃºn el Ã­ndice sigma (Ïƒ):**

| Rango de Ïƒ         | InterpretaciÃ³n                                      |
|:------------------:|:---------------------------------------------------|
| Ïƒ â‰¥ 2.0            | <span style='color:green'><b>No hay cavitaciÃ³n</b></span> |
| 1.7 < Ïƒ < 2.0      | <span style='color:#7ca300'><b>ProtecciÃ³n suficiente con materiales endurecidos</b></span> |
| 1.5 < Ïƒ < 1.7      | <span style='color:orange'><b>Algo de cavitaciÃ³n, puede funcionar un solo escalÃ³n</b></span> |
| 1.0 < Ïƒ < 1.5      | <span style='color:#ff6600'><b>Potencial de cavitaciÃ³n severa, se requiere reducciÃ³n en varias etapas</b></span> |
| Ïƒ < 1.0            | <span style='color:red'><b>Flashing (vaporizaciÃ³n instantÃ¡nea)</b></span> |

> **Ïƒ = (Pâ‚ - P_v) / (Pâ‚ - Pâ‚‚)**

- **SUPER CAVITACIÃ“N:** Ïƒ bajo, aceleraciÃ³n alta, daÃ±o severo.
- **CAVITACIÃ“N PLENA:** Ïƒ intermedio, daÃ±o considerable.
- **CAVITACIÃ“N INCIPIENTE:** Ïƒ cerca de 1.5-1.7, inicio de daÃ±o.
- **SUBCRÃTICO:** Ïƒ alto, sin daÃ±o.

Estos criterios ayudan a seleccionar el diseÃ±o y materiales adecuados para evitar daÃ±os por cavitaciÃ³n en vÃ¡lvulas.
""", unsafe_allow_html=True)

    # --- SEPARADOR A TODO EL ANCHO ---
    st.markdown("---")
    # --- NUEVA FILA DE COLUMNAS PARA Kv SOLO EN ESTA PESTAÃ‘A ---
    col_kv_exp, col_kv_graf, col_kv_tabla, col_kv_extra = st.columns([0.22, 0.4, 0.28, 0.1])
    with col_kv_exp:
        st.subheader("Coeficiente de caudal (Kv)")
        st.markdown("""
Las vÃ¡lvulas de control son conceptualmente orificios de Ã¡rea variable. Se las puede considerar simplemente como una restricciÃ³n que cambia su tamaÃ±o de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relaciÃ³n de diferencia de altura (Î”h) o presiÃ³n (Î”P) entre la entrada y salida de la vÃ¡lvula con el caudal (Q).
""")
        st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
        st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (mÂ³/h)
- $Q$: Caudal volumÃ©trico (mÂ³/h)
- $\rho$: Densidad (kg/mÂ³)
- $\Delta p$: Diferencia de presiÃ³n (bar)
- $P_1$: PresiÃ³n de entrada (bar)
- $P_2$: PresiÃ³n de salida (bar)
""")
        mca_a_bar = 0.0980665
        P1_bar = P1_valve * mca_a_bar
        P2_bar = P2_valve * mca_a_bar
        delta_p_bar = np.abs(P1_bar - P2_bar)
        densidad = densidad_calculada
        st.markdown(f"**PresiÃ³n de entrada (P1):** {P1_bar:.2f} bar  ")
        st.markdown(f"**PresiÃ³n de salida (P2):** {P2_bar:.2f} bar  ")
        st.markdown(f"**Diferencia de presiÃ³n (Î”p):** {delta_p_bar:.2f} bar  ")
        st.markdown(f"**Densidad (Ï):** {densidad:.1f} kg/mÂ³  ")
    with col_kv_graf:
        st.subheader("Ãndice de caudal Kv")
        # Corregido: GrÃ¡fico Î”p vs Kv con Q fijo
        Q_m3h = 10  # Caudal fijo tÃ­pico para la curva
        Kv_range = np.linspace(1, 15, 30)
        delta_p = (densidad / 1000) * (Q_m3h / Kv_range) ** 2
        fig_kv = go.Figure()
        fig_kv.add_trace(go.Scatter(
            x=Kv_range, y=delta_p, mode='markers+lines',
            marker=dict(symbol='diamond-open', color='blue'),
            line=dict(dash='solid', color='blue'),
            name='Î”p vs Kv'
        ))
        fig_kv.update_layout(
            title="VariaciÃ³n de Î”p con Kv",
            xaxis=dict(
                title="Kv (mÂ³/h)", 
                tickformat='.1f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            yaxis=dict(
                title="Î”p (bar)", 
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            template="simple_white",
            height=300
        )
        st.plotly_chart(fig_kv, use_container_width=True, key=f"fig_kv_kv_{uuid.uuid4()}")
        # Mantener el segundo grÃ¡fico y tablas igual
        delta_p_graf = np.linspace(0.1, 4.5, 30)
        Kv_alto = 15
        Kv_bajo = 3
        Q_kv_alto = Kv_alto / np.sqrt(densidad / (1000 * delta_p_graf))
        Q_kv_bajo = Kv_bajo / np.sqrt(densidad / (1000 * delta_p_graf))
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=delta_p_graf, y=Q_kv_alto, mode='markers+lines',
                                   marker=dict(symbol='diamond-open', color='blue'),
                                   line=dict(dash='solid', color='blue'),
                                   name='Kv >>'))
        fig_q.add_trace(go.Scatter(x=delta_p_graf, y=Q_kv_bajo, mode='markers+lines',
                                   marker=dict(symbol='square-open', color='orange'),
                                   line=dict(dash='solid', color='orange'),
                                   name='Kv <<'))
        fig_q.update_layout(
            title="Q vs. PÃ©rdida de carga para Kv alto y bajo",
            xaxis=dict(
                title="PÃ©rdida de carga (bar)", 
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            yaxis=dict(
                title="Q (mÂ³/h)", 
                tickformat='.1f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            template="simple_white",
            height=300
        )
        st.plotly_chart(fig_q, use_container_width=True, key=f"fig_q_kv_{uuid.uuid4()}")
    with col_kv_tabla:
        st.markdown('<h4 style="text-align:center;">Tabla Kv calculado</h4>', unsafe_allow_html=True)
        df_kv = pd.DataFrame({
            'Kv (mÂ³/h)': np.round(Kv_range, 2),
            'Î”p (bar)': np.round(delta_p, 2)
        })
        st.dataframe(df_kv, use_container_width=True)
        st.markdown('<h4 style="text-align:center;">Tabla Q vs Î”p para Kv</h4>', unsafe_allow_html=True)
        df_qkv = pd.DataFrame({
            'Î”p (bar)': np.round(delta_p_graf, 2),
            'Q (mÂ³/h) Kv alto': np.round(Q_kv_alto, 2),
            'Q (mÂ³/h) Kv bajo': np.round(Q_kv_bajo, 2)
        })
        st.dataframe(df_qkv, use_container_width=True)
    with col_kv_extra:
        pass

# --- PestaÃ±a 2: Sistema de Bombeo (NPSH) ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de Sistema de Bombeo (NPSH)
# --------------------------------------------------
with tab2:
    st.header("AnÃ¡lisis de NPSH Disponible para Bomba")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos del Sistema de Bombeo")
        # Tabla de presiÃ³n atmosfÃ©rica segÃºn altura sobre el nivel del mar
        alturas_msnm = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
        presiones_mca = [10.3, 9.7, 9.1, 8.6, 8.1, 7.6, 7.1, 6.7, 6.3, 5.9, 5.5, 5.2, 4.9]
        def interpolar_presion(cota):
            if cota <= alturas_msnm[0]:
                return presiones_mca[0]
            if cota >= alturas_msnm[-1]:
                return presiones_mca[-1]
            for i in range(1, len(alturas_msnm)):
                if cota < alturas_msnm[i]:
                    x0, x1 = alturas_msnm[i-1], alturas_msnm[i]
                    y0, y1 = presiones_mca[i-1], presiones_mca[i]
                    return y0 + (y1 - y0) * (cota - x0) / (x1 - x0)
            return presiones_mca[-1]
        cota_bomba = st.number_input("Cota de la bomba (msnm)", min_value=0.0, max_value=6000.0, value=0.0, step=1.0)
        H_atm = interpolar_presion(cota_bomba)
        st.number_input("PresiÃ³n AtmosfÃ©rica (H_atm, mca)", value=H_atm, disabled=True)
        H_static_suction = st.number_input("Altura EstÃ¡tica de SucciÃ³n (H_s, mca, + si es sobre nivel bomba, - si es bajo)", value=2.0, step=0.1)
        NPSHr_pump = st.number_input("NPSH Requerido de la Bomba (NPSH_necesaria, mca)", min_value=0.0, value=3.0, step=0.1)
        H_losses_suction = st.number_input("PÃ©rdidas por FricciÃ³n y Accesorios en SucciÃ³n (Î”H_s, mca)", min_value=0.0, value=1.5, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        losses_min = st.number_input("PÃ©rdidas mÃ­nimas (Î”H_s, mca)", min_value=0.0, value=0.0, step=0.01)
        losses_max = st.number_input("PÃ©rdidas mÃ¡ximas (Î”H_s, mca)", min_value=0.0, value=5.0, step=0.01)
        losses_step = st.number_input("Paso de pÃ©rdidas (Î”H_s, mca)", min_value=0.001, max_value=10.0, value=0.25, step=0.1)
        st.subheader("Resultados")
        npsha_calculated = npsh_calc.calculate_npsha(H_atm, H_static_suction, H_losses_suction)
        st.metric(label="NPSH Disponible (NPSHA, mca)", value=f"{npsha_calculated:.3f}")
        st.write(npsh_calc.get_npsh_risk_description(npsha_calculated, NPSHr_pump))
        st.info(f"**Criterio:** NPSHA debe ser mayor a {NPSHr_pump:.1f} mca (NPSHR). Se recomienda un margen de seguridad (ej. 1.1 * NPSHR).")

    with col_graph:
        st.subheader("GrÃ¡fica: NPSHA vs. PÃ©rdidas en SucciÃ³n")
        losses_range = np.arange(losses_min, losses_max+losses_step, losses_step)
        npsha_values = [npsh_calc.calculate_npsha(H_atm, H_static_suction, l) for l in losses_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.round(losses_range,2), y=npsha_values, mode='lines', name='NPSH Disponible',
                                 line=dict(color='#2ca02c'),
                                 hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                 hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))))
        # Punto rojo para el valor de entrada actual
        if H_losses_suction is not None and npsha_calculated is not None:
            fig.add_trace(go.Scatter(x=[H_losses_suction], y=[npsha_calculated], mode='markers',
                                     marker=dict(color='red', size=12, symbol='circle'),
                                     name='Punto de entrada',
                                     hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))))
        fig.add_hline(y=NPSHr_pump, line_dash="dash", line_color="red", annotation_text=f"NPSHR ({NPSHr_pump:.1f})")
        fig.add_hline(y=NPSHr_pump * 1.1, line_dash="dash", line_color="orange", annotation_text=f"NPSHR + Margen ({NPSHr_pump*1.1:.1f})")
        fig.update_layout(
            title=f"NPSHA vs. PÃ©rdidas en SucciÃ³n (H_estÃ¡tica={H_static_suction:.1f}m)",
            xaxis=dict(
                title="PÃ©rdidas por FricciÃ³n y Accesorios en SucciÃ³n (mca)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="NPSH Disponible (mca)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de NPSHA vs. PÃ©rdidas</h4>', unsafe_allow_html=True)
        df_npsh = pd.DataFrame({
            'PÃ©rdidas (mca)': np.round(losses_range, 2),
            'NPSHA (mca)': np.round(npsha_values, 2)
        })
        st.dataframe(df_npsh, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**NPSH Disponible:**

$$
NPSH_{disponible} = H_{atm} - (P_{v} + H_{s} + \Delta H_{s})
$$

**Donde:**
- **NPSHdisponible**: Carga neta positiva de succiÃ³n disponible (m)
- **Hatm**: PresiÃ³n atmosfÃ©rica (m)
- **Pv**: PresiÃ³n de vapor (m)
- **Hs**: Altura estÃ¡tica de succiÃ³n (m)
- **Î”Hs**: PÃ©rdida de presiÃ³n por fricciÃ³n (m)
""")
        with st.expander("PresiÃ³n AtmosfÃ©rica segÃºn altura"):
            st.markdown('''
| Altura sobre el nivel del mar (m) | PresiÃ³n atmosfÃ©rica (m.c.a.) |
|:-------------------------------:|:----------------------------:|
| 0     | 10,3 |
| 500   | 9,7  |
| 1 000 | 9,1  |
| 1 500 | 8,6  |
| 2 000 | 8,1  |
| 2 500 | 7,6  |
| 3 000 | 7,1  |
| 3 500 | 6,7  |
| 4 000 | 6,3  |
| 4 500 | 5,9  |
| 5 000 | 5,5  |
| 5 500 | 5,2  |
| 6 000 | 4,9  |
''')
        
        with st.expander("InterpretaciÃ³n y criterios"):
            st.markdown(r"""
La NPSH necesaria y la disponible son parÃ¡metros de control de la cavitaciÃ³n en los impulsores de las bombas.

La NPSH disponible depende del diseÃ±o del bombeo y representa la diferencia entre la carga absoluta y la presiÃ³n de vapor del lÃ­quido a temperatura constante.

**NPSH necesaria:** Es la carga exigida por la bomba entre la presiÃ³n de succiÃ³n y la presiÃ³n de vapor del lÃ­quido para que la bomba no cavite.

$$
NPSH_{disponible} \geq NPSH_{necesaria} + 0.5
$$
""")

# --- PestaÃ±a 3: LÃ­nea de ImpulsiÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de LÃ­nea de ImpulsiÃ³n
# --------------------------------------------------
with tab3:
    st.header("AnÃ¡lisis de CavitaciÃ³n en LÃ­nea de ImpulsiÃ³n")
    st.write("Se analiza un segmento o punto crÃ­tico de la lÃ­nea de impulsiÃ³n.")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos de la LÃ­nea de ImpulsiÃ³n")
        pipe_roughness_C = st.slider("Coeficiente Hazen-Williams (C)", min_value=80, max_value=150, value=120, step=5)
        flow_rate_imp = st.number_input("Caudal (L/s)", min_value=0.0, value=5.0, step=0.1)
        reference_diameter_mm = st.number_input("DiÃ¡metro (mm)", min_value=0.0, value=50.0, step=1.0)
        pipe_length_imp = st.number_input("Longitud del Tramo (m)", min_value=1.0, value=100.0, step=1.0)
        P_upstream_imp = st.number_input("PresiÃ³n Aguas Arriba (P_upstream, mca)", min_value=0.0, value=25.0, step=0.1)
        P_downstream_imp = st.number_input("PresiÃ³n Aguas Abajo (P_downstream, mca)", min_value=0.0, value=20.0, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        diam_min = st.number_input("DiÃ¡metro mÃ­nimo (mm)", min_value=0.0, value=0.0, step=1.0)
        diam_max = st.number_input("DiÃ¡metro mÃ¡ximo (mm)", min_value=0.0, value=100.0, step=1.0)
        diam_step = st.number_input("Paso de diÃ¡metro (mm)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        # Ajustar cÃ¡lculos para mm y L/s
        diameter_range_m = np.arange(diam_min, diam_max+diam_step, diam_step) / 1000
        flow_rate_imp_m3s = flow_rate_imp / 1000
        reference_diameter_m = reference_diameter_mm / 1000
        pressure_drop_values = []
        downstream_pressure_values = []
        theta_values_imp_graph = []
        P_initial_graph = P_upstream_imp
        for d in diameter_range_m:
            friction_loss = calculate_friction_losses_hw(flow_rate_imp_m3s, d, pipe_length_imp, pipe_roughness_C)
            P_downstream_calc_graph = P_initial_graph - friction_loss
            pressure_drop_values.append(friction_loss)
            downstream_pressure_values.append(P_downstream_calc_graph)
            if P_initial_graph > P_downstream_calc_graph:
                theta = cav_calc.calculate_cavitation_index(P_initial_graph, P_downstream_calc_graph)
                theta_values_imp_graph.append(theta if theta is not None else np.nan)
            else:
                theta_values_imp_graph.append(np.nan)
        # Punto rojo para el diÃ¡metro de referencia
        friction_loss_ref = calculate_friction_losses_hw(flow_rate_imp_m3s, reference_diameter_m, pipe_length_imp, pipe_roughness_C)
        P_downstream_ref = P_upstream_imp - friction_loss_ref
        theta_ref = cav_calc.calculate_cavitation_index(P_upstream_imp, P_downstream_ref)
        st.subheader("Resultados")
        theta_imp = cav_calc.calculate_cavitation_index(P_upstream_imp, P_downstream_imp)
        if theta_imp is not None:
            st.metric(label="Ãndice de CavitaciÃ³n (Ï‘)", value=f"{theta_imp:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_imp))
        else:
            st.write("Ingrese valores vÃ¡lidos para calcular el Ã­ndice de cavitaciÃ³n.")

    with col_graph:
        st.subheader("GrÃ¡fica: Impacto del DiÃ¡metro en PÃ©rdidas y Presiones")
        fig_imp = make_subplots(specs=[[{"secondary_y": True}]])
        # Convertir a mm para el eje x
        diameter_range_mm = diameter_range_m * 1000
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=pressure_drop_values, mode='lines', name='PÃ©rdida de PresiÃ³n',
                                     line=dict(color='#d62728'),
                                     hovertemplate='<b>PÃ©rdida = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#d62728', font=dict(color='white', family='Arial Black'))), secondary_y=False)
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=downstream_pressure_values, mode='lines', name='PresiÃ³n Aguas Abajo',
                                     line=dict(color='#1f77b4'),
                                     hovertemplate='<b>PresiÃ³n = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))), secondary_y=False)
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=theta_values_imp_graph, mode='lines', name='Ãndice de CavitaciÃ³n (Ï‘)',
                                     line=dict(color='#2ca02c'),
                                     hovertemplate='<b>Ï‘ = %{y:.2f}</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))), secondary_y=True)
        fig_imp.add_hline(y=cav_calc.Pv, line_dash="dot", line_color="blue", annotation_text="Pv", secondary_y=False)
        fig_imp.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="LÃ­mite DaÃ±o (0.5)", secondary_y=True)
        fig_imp.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="LÃ­mite Ruido (0.8)", secondary_y=True)
        fig_imp.update_layout(
            title="Impacto del DiÃ¡metro en PÃ©rdidas, PresiÃ³n y Ï‘ en LÃ­nea de ImpulsiÃ³n",
            xaxis=dict(
                title="DiÃ¡metro de TuberÃ­a (mm)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="PresiÃ³n / PÃ©rdida (mca)",
                tickformat='.2f',
            ),
            yaxis2=dict(
                title="Ãndice de CavitaciÃ³n (Ï‘)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Resultados de ImpulsiÃ³n</h4>', unsafe_allow_html=True)
        df_imp = pd.DataFrame({
            'DiÃ¡metro (m)': np.round(diameter_range_m, 3),
            'PÃ©rdida (mca)': np.round(pressure_drop_values, 2),
            'PresiÃ³n Abajo (mca)': np.round(downstream_pressure_values, 2),
            'Ï‘': np.round(theta_values_imp_graph, 2)
        })
        st.dataframe(df_imp, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**PÃ©rdida por fricciÃ³n (Hazen-Williams):**

$$
H_f = \frac{10.67 \cdot L \cdot Q^{1.852}}{C^{1.852} \cdot D^{4.87}}
$$

**Ãndice de CavitaciÃ³n:**

$$
\vartheta = \frac{P_{abajo} - P_{vapor}}{P_{arriba} - P_{abajo}}
$$

**Donde:**
- **Parriba**: PresiÃ³n aguas arriba (mca)
- **Pabajo**: PresiÃ³n aguas abajo (mca)
- **Pvapor**: PresiÃ³n de vapor (mca)
""")

# --- PestaÃ±a 4: LÃ­nea de SucciÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de LÃ­nea de SucciÃ³n
# --------------------------------------------------
with tab4:
    st.header("AnÃ¡lisis de CavitaciÃ³n en LÃ­nea de SucciÃ³n")
    st.write("La lÃ­nea de succiÃ³n es crÃ­tica para la cavitaciÃ³n en bombas.")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos de la LÃ­nea de SucciÃ³n")
        diam_min_s = st.number_input("DiÃ¡metro mÃ­nimo (m) succiÃ³n", min_value=0.01, value=0.05, step=0.01)
        diam_max_s = st.number_input("DiÃ¡metro mÃ¡ximo (m) succiÃ³n", min_value=0.01, value=0.4, step=0.01)
        diam_step_s = st.number_input("Paso de diÃ¡metro (m) succiÃ³n", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        P_source_suction = st.number_input("PresiÃ³n en Fuente (P_fuente, mca, ej. AtmosfÃ©rica)", min_value=0.0, value=10.33, step=0.1)
        H_static_suction_tab4 = st.number_input("Altura EstÃ¡tica al Eje de Bomba (mca, + si lÃ­quido arriba, - si abajo)", value=2.0, step=0.1)
        flow_rate_suction = st.number_input("Caudal (mÂ³/s)", min_value=0.0, value=0.05, step=0.001, format="%.3f")
        pipe_length_suction = st.number_input("Longitud de la LÃ­nea de SucciÃ³n (m)", min_value=1.0, value=10.0, step=1.0)
        pipe_roughness_C_suction = st.slider("Coeficiente Hazen-Williams (C) SucciÃ³n", min_value=80, max_value=150, value=120, step=5)
        k_fitting_suction = st.number_input("Coeficiente K de Accesorios Global (succiÃ³n)", min_value=0.0, value=2.0, step=0.1)
        st.subheader("Resultados")
        reference_diameter_suction = st.number_input("DiÃ¡metro de Referencia para CÃ¡lculo (m)", min_value=0.05, value=0.2, step=0.01)
        friction_loss_suction = calculate_friction_losses_hw(flow_rate_suction, reference_diameter_suction, pipe_length_suction, pipe_roughness_C_suction)
        local_loss_suction = calculate_local_losses(flow_rate_suction, reference_diameter_suction, k_fitting_suction)
        total_loss_suction = friction_loss_suction + local_loss_suction
        P_brida_succion_calc = P_source_suction + H_static_suction_tab4 - total_loss_suction
        npsha_suction_calc = npsh_calc.calculate_npsha(P_source_suction, H_static_suction_tab4, total_loss_suction)
        st.metric(label="PresiÃ³n en Brida de SucciÃ³n (mca)", value=f"{P_brida_succion_calc:.3f}")
        st.metric(label="NPSH Disponible (NPSHA, mca)", value=f"{npsha_suction_calc:.3f}")
        NPSHr_dummy_for_desc = 3.0
        st.write(npsh_calc.get_npsh_risk_description(npsha_suction_calc, NPSHr_dummy_for_desc))
        st.info("Nota: Para una evaluaciÃ³n completa, compare el NPSH Disponible con el NPSH Requerido de su bomba.")

    with col_graph:
        st.subheader("GrÃ¡fica: NPSHA vs. DiÃ¡metro de SucciÃ³n")
        diameter_range_suction_graph = np.arange(diam_min_s, diam_max_s+diam_step_s, diam_step_s)
        npsha_values_suction_graph = []
        for d in diameter_range_suction_graph:
            friction_loss_g = calculate_friction_losses_hw(flow_rate_suction, d, pipe_length_suction, pipe_roughness_C_suction)
            local_loss_g = calculate_local_losses(flow_rate_suction, d, k_fitting_suction)
            total_loss_g = friction_loss_g + local_loss_g
            npsha_g = npsh_calc.calculate_npsha(P_source_suction, H_static_suction_tab4, total_loss_g)
            npsha_values_suction_graph.append(npsha_g)
        fig_suction = go.Figure()
        fig_suction.add_trace(go.Scatter(x=np.round(diameter_range_suction_graph,3), y=npsha_values_suction_graph, mode='lines', name='NPSH Disponible',
                                         line=dict(color='#9467bd'),
                                         hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                         hoverlabel=dict(bgcolor='#9467bd', font=dict(color='white', family='Arial Black'))))
        fig_suction.add_hline(y=NPSHr_dummy_for_desc, line_dash="dash", line_color="red", annotation_text=f"NPSHR de Ejemplo ({NPSHr_dummy_for_desc:.1f})")
        fig_suction.update_layout(
            title="NPSH Disponible vs. DiÃ¡metro de TuberÃ­a de SucciÃ³n",
            xaxis=dict(
                title="DiÃ¡metro de TuberÃ­a (m)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="NPSH Disponible (mca)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_suction, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de NPSHA vs. DiÃ¡metro</h4>', unsafe_allow_html=True)
        df_suction = pd.DataFrame({
            'DiÃ¡metro (m)': np.round(diameter_range_suction_graph, 3),
            'NPSHA (mca)': np.round(npsha_values_suction_graph, 2)
        })
        st.dataframe(df_suction, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**PÃ©rdida por fricciÃ³n (Hazen-Williams):**

$$
H_f = \frac{10.67 \cdot L \cdot Q^{1.852}}{C^{1.852} \cdot D^{4.87}}
$$

**PÃ©rdida local:**

$$
H_{local} = K \cdot \frac{V^2}{2g}
$$

Donde:
- \(K\): Coeficiente global de accesorios
- \(V\): Velocidad del fluido (m/s)
- \(g\): Gravedad (9.81 m/sÂ²)

**NPSH Disponible:**

$$
NPSH_{A} = H_{atm} + H_{estÃ¡tica\ succiÃ³n} - H_{pÃ©rdidas\ succiÃ³n} - P_{vapor}
$$
""")

# --- PestaÃ±a 5: PÃ©rdidas por FricciÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de PÃ©rdidas por FricciÃ³n
# --------------------------------------------------
def mougnie_velocity(diametro_m):
    # V = 1.5 * sqrt(D) + 0.05
    return 1.5 * np.sqrt(diametro_m) + 0.05

with tab5:
    st.header("GrÃ¡fico de Caudal vs. PÃ©rdidas por FricciÃ³n (Hazen-Williams)")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("ParÃ¡metros de la TuberÃ­a")
        diametro_mm = st.number_input("DiÃ¡metro de tuberÃ­a (mm)", min_value=10.0, max_value=1000.0, value=63.0, step=0.1, format="%.2f")
        materiales_c = [
            {"nombre": "PVC o plÃ¡stico (nuevo)", "C": 150, "rango": "140 - 150"},
            {"nombre": "Polietileno de alta densidad (HDPE)", "C": 150, "rango": "140 - 150"},
            {"nombre": "Asbesto-cemento (nuevo)", "C": 140, "rango": "130 - 140"},
            {"nombre": "Hierro fundido (nuevo)", "C": 130, "rango": "120 - 130"},
            {"nombre": "Hierro fundido (usado)", "C": 110, "rango": "80 - 130"},
            {"nombre": "Acero nuevo (sin costura)", "C": 145, "rango": "130 - 150"},
            {"nombre": "Acero galvanizado (nuevo)", "C": 120, "rango": "110 - 120"},
            {"nombre": "Acero galvanizado (viejo)", "C": 90, "rango": "70 - 110"},
            {"nombre": "HormigÃ³n (buen acabado)", "C": 130, "rango": "120 - 140"},
            {"nombre": "HormigÃ³n (viejo o rugoso)", "C": 110, "rango": "90 - 130"},
            {"nombre": "Cobre o latÃ³n", "C": 145, "rango": "130 - 150"},
            {"nombre": "Fibrocemento", "C": 135, "rango": "130 - 140"},
        ]
        nombres_materiales = [m["nombre"] for m in materiales_c]
        material_idx = st.selectbox("Material de la tuberÃ­a", nombres_materiales, index=0)
        material_seleccionado = materiales_c[nombres_materiales.index(material_idx)]
        # Permitir modificar C dentro del rango sugerido
        rango_c = material_seleccionado["rango"].replace(" ", "").split("-")
        c_min = int(rango_c[0])
        c_max = int(rango_c[1])
        C_hw = st.number_input(
            "Coeficiente Hazen-Williams (C)",
            min_value=c_min,
            max_value=c_max,
            value=material_seleccionado["C"],
            step=1
        )
        st.info(f"Material simulado: **{material_seleccionado['nombre']}**\n\nC tÃ­pico: **{material_seleccionado['C']}**\n\nRango de C permitido: {material_seleccionado['rango']}")
        st.subheader("Rango de Caudal (L/s)")
        caudal_min = st.number_input("Caudal mÃ­nimo (L/s)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)
        caudal_max = st.number_input("Caudal mÃ¡ximo (L/s)", min_value=0.01, max_value=100.0, value=5.0, step=0.01)
        paso_caudal = st.number_input("Paso de caudal (L/s)", min_value=0.01, max_value=10.0, value=0.25, step=0.01)
        mostrar_umbral = False
        umbral_pend = None
        with st.expander("Opciones avanzadas: Umbral de pendiente para zona segura", expanded=False):
            st.caption("El umbral de pendiente define hasta quÃ© valor de pendiente (variaciÃ³n de pÃ©rdida por variaciÃ³n de caudal) se considera la zona segura. Si la pendiente local supera este valor, se considera que la curva ya no es segura para diseÃ±o.")
            umbral_pend = st.number_input("Umbral de pendiente para zona segura (m/Km por L/s)", min_value=0.01, max_value=100.0, value=1.0, step=0.25, key="umbral_pend")
            col1, col2 = st.columns([1,1])
            with col1:
                aplicar_umbral = st.button("Aplicar umbral de pendiente", key="btn_umbral")
            with col2:
                borrar_umbral = st.button("Borrar", key="btn_borrar_umbral")
            if aplicar_umbral:
                mostrar_umbral = True
            if borrar_umbral:
                mostrar_umbral = False

    with col_graph:
        st.subheader(f"GrÃ¡fica: PÃ©rdida de carga por fricciÃ³n vs. Caudal  ", divider="gray")
        st.markdown(f"**Material simulado:** {material_seleccionado['nombre']}  |  C tÃ­pico: {material_seleccionado['C']}  |  Rango de C: {material_seleccionado['rango']}")
        Q_lps_graf = np.arange(caudal_min, caudal_max + paso_caudal, paso_caudal)
        diametro_m = diametro_mm / 1000
        Q_m3s_graf = Q_lps_graf / 1000
        L = 1  # longitud en metros
        j_graf = [10.67 * L * (q ** 1.852) / (C_hw ** 1.852 * diametro_m ** 4.87) for q in Q_m3s_graf]
        hf_km_graf = [j * 1000 for j in j_graf]

        # --- AnÃ¡lisis: punto donde la pendiente supera el umbral ---
        hf_km_graf = np.array(hf_km_graf)
        Q_lps_graf = np.array(Q_lps_graf)
        if mostrar_umbral and umbral_pend is not None:
            pendiente = np.diff(hf_km_graf) / np.diff(Q_lps_graf)
            idx_cambio = next((i for i, p in enumerate(pendiente) if p > umbral_pend), None)
            if idx_cambio is not None:
                # El punto de cambio es el primer Q donde la pendiente supera el umbral
                caudal_div = float(Q_lps_graf[idx_cambio+1])
                cambio_detectado = True
            else:
                caudal_div = None
                cambio_detectado = False
            # Ãrea bajo la curva segura (antes del punto de cambio)
            if cambio_detectado:
                area_segura = np.trapz(hf_km_graf[:idx_cambio+2], Q_lps_graf[:idx_cambio+2])
            else:
                area_segura = np.trapz(hf_km_graf, Q_lps_graf)
        else:
            cambio_detectado = False
            caudal_div = None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Q_lps_graf,
            y=hf_km_graf,
            mode='lines',
            name='PÃ©rdida de carga (m/Km)',
            line=dict(color='#ff7f0e'),
            hovertemplate='<b>Q = %{x:.2f} L/s<br>hf/Km = %{y:.2f} m/Km</b><extra></extra>'
        ))
        # LÃ­nea vertical en el punto de cambio solo si se aplica el umbral
        if mostrar_umbral and cambio_detectado and caudal_div is not None:
            fig.add_vline(x=caudal_div, line_dash="dot", line_color="#0080ff", line_width=2,
                          annotation_text=f"Q={caudal_div:.2f} L/s", annotation_position="top right")
        fig.update_layout(
            title=f"PÃ©rdida de carga por fricciÃ³n vs. Caudal  |  Material: {material_seleccionado['nombre']} (C={C_hw})",
            xaxis=dict(
                title="Caudal (L/s)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="PÃ©rdida de carga (m/Km)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        if mostrar_umbral:
            if cambio_detectado and caudal_div is not None:
                st.info(f"El punto de cambio de pendiente se estima en Q = {caudal_div:.2f} L/s")
            else:
                st.warning(f"No se detectÃ³ un cambio de pendiente dentro del rango y umbral seleccionados. La lÃ­nea se muestra al final del rango.")

        # GrÃ¡fico adicional: Caudal vs Velocidad
        V_graf = Q_m3s_graf / (np.pi * (diametro_m / 2) ** 2) if diametro_m > 0 else np.zeros_like(Q_m3s_graf)
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=Q_lps_graf,
            y=V_graf,
            mode='lines',
            name='Velocidad (m/s)',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>Q = %{x:.2f} L/s<br>V = %{y:.2f} m/s</b><extra></extra>'
        ))
        fig_vel.update_layout(
            title="Caudal vs. Velocidad en la TuberÃ­a",
            xaxis=dict(
                title="Caudal (L/s)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="Velocidad (m/s)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de PÃ©rdidas por FricciÃ³n</h4>', unsafe_allow_html=True)
        df_friction = pd.DataFrame({
            'Caudal (L/s)': np.round(Q_lps_graf, 2),
            'PÃ©rdida (m/Km)': np.round(hf_km_graf, 2),
            'Velocidad (m/s)': np.round(V_graf, 2)
        })
        st.dataframe(df_friction, use_container_width=True)
        
        with st.expander("ðŸ“š Coeficiente de caudal (Kv)"):
            st.markdown("""
Las vÃ¡lvulas de control son conceptualmente orificios de Ã¡rea variable. Se las puede considerar simplemente como una restricciÃ³n que cambia su tamaÃ±o de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relaciÃ³n de diferencia de altura (Î”h) o presiÃ³n (Î”P) entre la entrada y salida de la vÃ¡lvula con el caudal (Q).
""")
            st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
            st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (mÂ³/h)
- $Q$: Caudal volumÃ©trico (mÂ³/h)
- $\rho$: Densidad (kg/mÂ³)
- $\Delta p$: Diferencia de presiÃ³n (bar)
- $P_1$: PresiÃ³n de entrada (bar)
- $P_2$: PresiÃ³n de salida (bar)
""")
            mca_a_bar = 0.0980665
            P1_bar = P1_valve * mca_a_bar
            P2_bar = P2_valve * mca_a_bar
            delta_p_bar = np.abs(P1_bar - P2_bar)
            densidad = densidad_calculada
            st.markdown(f"**PresiÃ³n de entrada (P1):** {P1_bar:.2f} bar  ")
            st.markdown(f"**PresiÃ³n de salida (P2):** {P2_bar:.2f} bar  ")
            st.markdown(f"**Diferencia de presiÃ³n (Î”p):** {delta_p_bar:.2f} bar  ")
            st.markdown(f"**Densidad (Ï):** {densidad:.1f} kg/mÂ³  ")

# --- NUEVA PESTAÃ‘A: Flujos Transitorios ---
with tab_trans:
    st.header("ðŸŒŠ Flujos Transitorios")
    # Primer bloque: CÃ¡lculo de celeridad y espesor de tuberÃ­a
    st.subheader("CÃ¡lculo de velocidad de propagaciÃ³n de onda (celeridad) y espesor de tuberÃ­a")
    col1, col2, col3, col4 = st.columns([0.22, 0.4, 0.28, 0.1])
    with col1:
        st.markdown("#### Datos de entrada")
        materiales = {
            "Acero": {"E": (200e9, 212e9), "a": (1000, 1250)},
            "Fibro Cemento": {"E": (23.5e9, 23.5e9), "a": (900, 1200)},
            "Concreto": {"E": (39e9, 39e9), "a": (1050, 1150)},
            "Hierro DÃºctil": {"E": (166e9, 166e9), "a": (1000, 1350)},
            "Polietileno alta densidad": {"E": (0.59e9, 1.67e9), "a": (230, 430)},
            "PVC": {"E": (2.4e9, 2.75e9), "a": (300, 500)},
        }
        material = st.selectbox("Material de la tuberÃ­a", list(materiales.keys()))
        E_range = materiales[material]["E"]
        a_range = materiales[material]["a"]
        E_prom = sum(E_range) / 2
        a_prom = sum(a_range) / 2
        st.markdown(f"**Rango de mÃ³dulo de elasticidad (E):** {E_range[0]/1e9:.2f} - {E_range[1]/1e9:.2f} GPa")
        st.markdown(f"**Rango de celeridad (a):** {a_range[0]:.0f} - {a_range[1]:.0f} m/s")
        # Input en GPa, conversiÃ³n a Pa para cÃ¡lculos
        E_GPa = st.number_input("MÃ³dulo de elasticidad E (GPa)", min_value=0.01, max_value=300.0, value=float(E_prom/1e9), step=0.01, format="%.2f")
        E = E_GPa * 1e9
        a = st.number_input("Celeridad a (m/s)", min_value=100.0, max_value=2000.0, value=float(a_prom), step=1.0, key="a_material")
        densidad = st.number_input("Densidad del agua (kg/mÂ³)", min_value=800.0, max_value=1200.0, value=float(densidad_calculada), step=1.0)
        D = st.number_input("DiÃ¡metro de la tuberÃ­a (mm)", min_value=10.0, max_value=2000.0, value=100.0, step=1.0) / 1000
        K = st.number_input("MÃ³dulo de elasticidad del fluido K (N/mÂ²)", min_value=1e8, max_value=3e9, value=2.2e9, step=1e7, format="%.0f")
        
        # CÃ¡lculo de delta
        denominador = K - densidad * a**2
        if denominador <= 0:
            st.error("Error: El denominador (K - ÏaÂ²) debe ser positivo. Revise los valores de K, densidad y celeridad.")
            delta_calc = None
            delta_mm = None
        else:
            delta_calc = (K * D * densidad * a**2) / (E * denominador)
            delta_mm = delta_calc * 1000
            # --- NUEVO: Mostrar datos PEAD si corresponde ---
            if material == "Polietileno alta densidad":
                # Tabla extendida PEAD: cada fila es un diÃ¡metro nominal, con sus espesores, series, SDR, diÃ¡metro interno y presiÃ³n
                pead_tabla = [
                    {"DN": 20, "series": ["S5", "S4"], "SDR": [11, 9], "esp": [2.0, 2.3], "dint": [16.0, 15.4], "pres": [1.6, 2.0]},
                    {"DN": 25, "series": ["S5", "S4"], "SDR": [11, 9], "esp": [2.3, 2.8], "dint": [20.4, 19.4], "pres": [1.6, 2.0]},
                    {"DN": 32, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [2.4, 3.0, 3.6], "dint": [27.2, 26.0, 24.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 40, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [3.0, 3.7, 4.5], "dint": [34.0, 32.6, 31.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 50, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [3.7, 4.6, 5.6], "dint": [42.6, 40.8, 38.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 63, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [4.7, 5.8, 7.1], "dint": [53.6, 51.4, 48.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 75, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [5.6, 6.8, 8.4], "dint": [63.8, 61.4, 58.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 90, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [6.7, 8.2, 10.1], "dint": [76.6, 73.6, 69.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 110, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [8.1, 10.0, 12.3], "dint": [93.8, 90.0, 85.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 125, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [9.2, 11.4, 14.0], "dint": [106.6, 102.2, 97.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 140, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [10.3, 12.7, 15.7], "dint": [119.4, 114.6, 108.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 160, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [11.8, 14.6, 18.0], "dint": [136.4, 130.8, 124.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 180, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [13.3, 16.4, 20.1], "dint": [153.4, 147.2, 139.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 200, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [14.7, 18.2, 22.7], "dint": [167.6, 163.6, 154.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 225, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [16.6, 20.5, 25.6], "dint": [191.8, 184.0, 173.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 250, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [18.4, 22.7, 28.4], "dint": [213.2, 204.6, 193.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 280, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [20.6, 25.4, 31.8], "dint": [238.8, 229.2, 217.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 315, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [23.2, 28.6, 35.7], "dint": [268.6, 257.8, 243.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 355, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [26.1, 32.2, 40.2], "dint": [302.8, 290.6, 274.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 400, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [29.2, 36.3, 45.4], "dint": [341.6, 327.4, 309.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 450, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [32.8, 41.0, 51.0], "dint": [384.4, 368.2, 349.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 500, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [36.3, 45.4, 56.8], "dint": [427.4, 409.2, 386.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 560, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [40.7, 51.0, 63.5], "dint": [479.6, 458.4, 433.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 630, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [46.3, 57.2, 71.0], "dint": [555.2, 514.6, 489.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 710, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [52.2, 64.7, 80.3], "dint": [622.2, 580.6, 549.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 800, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [58.1, 72.7, 90.8], "dint": [705.2, 654.8, 618.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 900, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [65.1, 81.7, 102.3], "dint": [793.4, 738.8, 695.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 1000, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [73.5, 90.8, 115.0], "dint": [881.4, 818.4, 770.0], "pres": [1.0, 1.6, 2.0]},
                ]
                DN_input = int(round(D*1000))
                DN_candidatos = [fila["DN"] for fila in pead_tabla]
                DN_cercano = min(DN_candidatos, key=lambda x: abs(x - DN_input))
                fila = next(f for f in pead_tabla if f["DN"] == DN_cercano)
                if delta_mm is not None:
                    # Buscar el primer espesor nominal igual o superior al calculado
                    idx = next((i for i, e in enumerate(fila["esp"]) if e >= delta_mm), len(fila["esp"]) - 1)
                    serie = fila["series"][idx]
                    SDR = fila["SDR"][idx]
                    esp_nom = fila["esp"][idx]
                    dint = fila["dint"][idx]
                    pres = fila["pres"][idx]
                    st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm**\n\n**PEAD:**\n- DiÃ¡metro nominal: {DN_cercano} mm\n- DiÃ¡metro interior: {dint:.1f} mm\n- Serie: {serie}\n- SDR: {SDR}\n- Espesor nominal: {esp_nom:.2f} mm\n- PresiÃ³n de trabajo: {pres} MPa")
                else:
                    st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm**\n\n**PEAD:**\n- DiÃ¡metro nominal: {DN_cercano} mm\n- (No se pudo determinar serie ni presiÃ³n de trabajo)")
            else:
                st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm")
            if delta_mm > 50:
                st.warning("El espesor calculado es muy grande. Revise los parÃ¡metros o considere que para materiales plÃ¡sticos este cÃ¡lculo puede no ser aplicable para diseÃ±o real.")
        # Mostrar resultados de celeridad y espesor
        st.info(f"**Resultados:**\n- Celeridad (a): {a:.2f} m/s\n- Espesor calculado (Î´): {delta_mm:.2f} mm")
        
        st.markdown("---")
        st.markdown("### Datos de grÃ¡fico")
        espesor_inicial = st.number_input("Espesor inicial (mm)", min_value=0.0, max_value=50.0, value=0.1, step=0.1)
        espesor_final = st.number_input("Espesor final (mm)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        paso_espesor = st.number_input("Paso de espesor (mm)", min_value=0.1, max_value=5.0, value=0.1, step=0.1)
    with col2:
        st.markdown("#### GrÃ¡fico de Celeridad vs Espesor de TuberÃ­a")
        st.markdown("Este grÃ¡fico muestra cÃ³mo varÃ­a la celeridad en funciÃ³n del espesor de la tuberÃ­a para diferentes materiales.")
        
        # Generar datos para el grÃ¡fico
        espesores_mm = np.arange(espesor_inicial, espesor_final + paso_espesor, paso_espesor)
        espesores_m = espesores_mm / 1000
        
        # Calcular celeridad para cada material
        fig_celeridad = go.Figure()
        
        for mat_name, mat_props in materiales.items():
            E_mat = sum(mat_props["E"]) / 2  # Usar valor promedio
            celeridades = []
            
            for delta_m in espesores_m:
                if delta_m > 0:  # Evitar divisiÃ³n por cero
                    # FÃ³rmula de celeridad: a = sqrt(K/Ï / (1 + K*D/(E*Î´)))
                    celeridad = np.sqrt((K / densidad) / (1 + (K * D) / (E_mat * delta_m)))
                    celeridades.append(celeridad)
                else:
                    celeridades.append(np.sqrt(K / densidad))  # Celeridad sin tuberÃ­a
            
            # Color segÃºn el material
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color_idx = list(materiales.keys()).index(mat_name) % len(colors)
            
            fig_celeridad.add_trace(go.Scatter(
                x=espesores_mm,
                y=celeridades,
                mode='lines',
                name=mat_name,
                line=dict(color=colors[color_idx], width=2),
                hovertemplate='<b>%{fullData.name}: %{y:.0f} m/s</b><extra></extra>',
                hoverlabel=dict(bgcolor=colors[color_idx], font=dict(color='white', family='Arial Black'))
            ))
        
        # Agregar punto del material seleccionado
        if delta_mm is not None and delta_mm > 0:
            E_selected = E
            celeridad_selected = np.sqrt((K / densidad) / (1 + (K * D) / (E_selected * delta_calc)))
            fig_celeridad.add_trace(go.Scatter(
                x=[delta_mm],
                y=[celeridad_selected],
                mode='markers',
                name=f'{material} (calculado)',
                marker=dict(color='red', size=10, symbol='star'),
                hovertemplate='<b>%{fullData.name}: %{y:.0f} m/s</b><extra></extra>',
                hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
            ))
        
        fig_celeridad.update_layout(
            title="Celeridad vs Espesor de TuberÃ­a",
            xaxis=dict(
                title="Espesor de tuberÃ­a (mm)",
                showgrid=True,
                gridcolor='#cccccc',
                gridwidth=0.7,
                zeroline=False,
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                hoverformat='.1f'
            ),
            yaxis=dict(
                title="Celeridad (m/s)",
                showgrid=True,
                gridcolor='#cccccc',
                gridwidth=0.7,
                zeroline=False,
                showspikes=False,
                showline=True,
                hoverformat='.0f'
            ),
            hovermode="x",
            legend=dict(orientation='h', yanchor='bottom', y=-0.6, xanchor='center', x=0.5)
        )
        
        st.plotly_chart(fig_celeridad, use_container_width=True)

        # --- GrÃ¡fico Conceptual de AceleraciÃ³n vs Ãndice de CavitaciÃ³n (Ïƒ) ---
        st.markdown('#### Ãndice de CavitaciÃ³n vs AceleraciÃ³n (grÃ¡fico conceptual)')
        import plotly.graph_objects as go
        # Curva conceptual (valores aproximados para ilustrar el comportamiento)
        sigma_concept = [1, 1.2, 1.5, 1.7, 2, 4, 8, 20]
        aceleracion_concept = [a*1.15, a*1.2, a*1.1, a*0.9, a*0.7, a*0.5, a*0.3, a*0.2]
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(
            x=sigma_concept,
            y=aceleracion_concept,
            mode='lines+markers',
            name='Curva conceptual',
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black'),
            hovertemplate='<b>Ïƒ = %{x:.2f}<br>AceleraciÃ³n = %{y:.0f} m/s</b><extra></extra>'
        ))
        # Punto del material seleccionado (en Ïƒ=2, por ejemplo)
        fig_sigma.add_trace(go.Scatter(
            x=[2],
            y=[a],
            mode='markers',
            name=f'Material seleccionado (a = {a:.0f} m/s)',
            marker=dict(color='red', size=7, symbol='star'),
            hovertemplate='<b>Ïƒ = 2.00<br>AceleraciÃ³n = %{y:.0f} m/s</b><extra></extra>'
        ))
        # Zonas de colores
        fig_sigma.add_vrect(x0=2, x1=20, fillcolor='green', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=1.7, x1=2, fillcolor='#7ca300', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=1.5, x1=1.7, fillcolor='orange', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=1, x1=1.5, fillcolor='#ff6600', opacity=0.2, layer='below', line_width=0)
        fig_sigma.add_vrect(x0=0, x1=1, fillcolor='red', opacity=0.2, layer='below', line_width=0)
        # LÃ­mites y etiquetas
        fig_sigma.update_layout(
            title='Ãndice de CavitaciÃ³n (Ïƒ) vs AceleraciÃ³n (Curva Conceptual)',
            xaxis=dict(title='Ãndice de CavitaciÃ³n Ïƒ', range=[0.8, 4], tickvals=[1, 1.5, 1.7, 2, 3, 4]),
            yaxis=dict(title='AceleraciÃ³n (m/s)', range=[0, a*1.3]),
            hovermode='x',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        # Etiquetas de zonas en vertical, en negrita y al inicio de cada zona
        fig_sigma.add_annotation(x=1, y=0, text='<b>Flashing</b>', showarrow=False, font=dict(color='red', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=1.5, y=0, text='<b>CavitaciÃ³n severa</b>', showarrow=False, font=dict(color='#ff6600', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=1.7, y=0, text='<b>CavitaciÃ³n incipiente</b>', showarrow=False, font=dict(color='orange', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=2, y=0, text='<b>ProtecciÃ³n suficiente</b>', showarrow=False, font=dict(color='#7ca300', size=12), textangle=-90, xanchor='right', yanchor='bottom')
        fig_sigma.add_annotation(x=2.5, y=0, text='<b>No cavitaciÃ³n</b>', showarrow=False, font=dict(color='green', size=12), textangle=-90, xanchor='left', yanchor='bottom')
        st.plotly_chart(fig_sigma, use_container_width=True)
    with col3:
        with st.expander("ðŸ“š FÃ³rmulas y TeorÃ­a"):
            st.markdown("#### FÃ³rmula de celeridad")
            st.latex(r"a = \frac{K}{\rho} \frac{1}{\sqrt{1 + \frac{K D}{E \delta}}}")
            st.markdown("#### FÃ³rmula de espesor de tuberÃ­a")
            st.latex(r"\delta = \frac{K D \rho a^2}{E (K - \rho a^2)}")
            st.markdown("""
**Fundamento teÃ³rico:**
Esta fÃ³rmula se deriva de la ecuaciÃ³n de celeridad del golpe de ariete, despejando el espesor Î´.
La celeridad a estÃ¡ relacionada con las propiedades del fluido y la tuberÃ­a mediante:

a = \sqrt{\frac{K/\rho}{1 + \frac{K D}{E \delta}}}

Despejando Î´ se obtiene la fÃ³rmula mostrada.
""")
            # Agregar fÃ³rmulas de Ã­ndice de cavitaciÃ³n y concepto
            st.markdown("---")
            st.markdown("#### Ãndice de cavitaciÃ³n")
            st.latex(r"\sigma = \frac{P_1 - P_v}{P_1 - P_2}")
            st.latex(r"\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}")
            st.markdown("""
El **Ã­ndice de cavitaciÃ³n** es un parÃ¡metro adimensional que permite evaluar el riesgo de formaciÃ³n de cavitaciÃ³n en sistemas hidrÃ¡ulicos. Compara la presiÃ³n disponible en el sistema con la presiÃ³n de vapor del fluido. Un valor bajo indica mayor riesgo de cavitaciÃ³n, mientras que valores altos indican condiciones seguras.

- **Pâ‚:** PresiÃ³n de entrada (aguas arriba)
- **Pâ‚‚:** PresiÃ³n de salida (aguas abajo)
- **P_v / P_{vapor}:** PresiÃ³n de vapor del fluido
""")
        # Panel desplegable con definiciÃ³n de grÃ¡fico conceptual
        with st.expander("Â¿QuÃ© significa 'grÃ¡fico conceptual'?"):
            st.markdown("""
Un **grÃ¡fico conceptual** es una representaciÃ³n idealizada o esquemÃ¡tica que ilustra tendencias generales, relaciones teÃ³ricas o comportamientos tÃ­picos de un fenÃ³meno fÃ­sico. No se basa en datos experimentales ni en cÃ¡lculos especÃ­ficos para un caso real, sino que ayuda a visualizar de manera clara y didÃ¡ctica cÃ³mo suelen comportarse las variables bajo ciertas condiciones ideales o teÃ³ricas. Su objetivo es facilitar la comprensiÃ³n de conceptos clave, no mostrar resultados numÃ©ricos exactos.
""")

        with st.expander("ðŸ“‹ Tabla de materiales y celeridad"):
            import pandas as pd
            df_mat = pd.DataFrame([
                {
                    "Material": k,
                    "E (GPa)": f"{v['E'][0]/1e9:.2f} - {v['E'][1]/1e9:.2f}" if v['E'][0] != v['E'][1] else f"{v['E'][0]/1e9:.2f}",
                    "a (m/s)": f"{v['a'][0]:.0f} - {v['a'][1]:.0f}"
                }
                for k, v in materiales.items()
            ])
            st.dataframe(df_mat, use_container_width=True)
        
        st.markdown("#### Tabla de datos del grÃ¡fico")
        # Crear tabla con los datos del grÃ¡fico
        espesores_mm_tabla = np.arange(espesor_inicial, espesor_final + paso_espesor, paso_espesor)
        espesores_m_tabla = espesores_mm_tabla / 1000
        
        # Calcular celeridades para la tabla
        datos_tabla = {}
        for mat_name, mat_props in materiales.items():
            E_mat = sum(mat_props["E"]) / 2
            celeridades_mat = []
            
            for delta_m in espesores_m_tabla:
                if delta_m > 0:
                    celeridad = np.sqrt((K / densidad) / (1 + (K * D) / (E_mat * delta_m)))
                else:
                    celeridad = np.sqrt(K / densidad)
                celeridades_mat.append(celeridad)
            
            datos_tabla[mat_name] = celeridades_mat
        
        # Crear DataFrame con materiales en columnas y espesores en filas
        df_celeridad = pd.DataFrame(datos_tabla, index=espesores_mm_tabla)
        df_celeridad.index.name = "Espesor (mm)"
        
        # Formatear valores de celeridad
        for col in df_celeridad.columns:
            df_celeridad[col] = df_celeridad[col].apply(lambda x: f"{x:.0f}")
        
        st.dataframe(df_celeridad, use_container_width=True)
        # Panel desplegable con definiciÃ³n de grÃ¡fico conceptual debajo de la tabla
        with st.expander("GrÃ¡fico conceptual"):
            st.markdown("""
Un **grÃ¡fico conceptual** es una representaciÃ³n idealizada o esquemÃ¡tica que ilustra tendencias generales, relaciones teÃ³ricas o comportamientos tÃ­picos de un fenÃ³meno fÃ­sico. No se basa en datos experimentales ni en cÃ¡lculos especÃ­ficos para un caso real, sino que ayuda a visualizar de manera clara y didÃ¡ctica cÃ³mo suelen comportarse las variables bajo ciertas condiciones ideales o teÃ³ricas. Su objetivo es facilitar la comprensiÃ³n de conceptos clave, no mostrar resultados numÃ©ricos exactos.
""")
        with st.expander("Tabla PEAD (Polietileno de alta densidad)"):
            st.markdown("""
| DiÃ¡metro Nominal (mm) | SDR 26 | SDR 21 | SDR 17 | SDR 13.6 | SDR 11 | SDR 9 |
|----------------------|--------|--------|--------|----------|--------|-------|
| 20  | 2.0 | 2.0 | 2.3 | 2.8 | 3.0 | 3.6 |
| 25  | -   | 2.0 | 2.3 | 2.8 | 3.0 | 3.6 |
| 32  | -   | 2.0 | 2.4 | 3.0 | 3.7 | 4.4 |
| 40  | 2.0 | 2.4 | 3.0 | 3.7 | 4.6 | 5.5 |
| 50  | 2.0 | 2.4 | 3.7 | 4.6 | 5.6 | 6.9 |
| 63  | 2.5 | 3.0 | 4.7 | 5.8 | 7.1 | 8.6 |
| 75  | 2.9 | 3.6 | 5.6 | 6.8 | 8.4 | 10.3 |
| 90  | 3.5 | 4.3 | 6.7 | 8.2 | 10.1 | 12.3 |
| 110 | 4.2 | 5.3 | 8.1 | 10.0 | 12.3 | 15.0 |
| 125 | 4.8 | 6.0 | 9.2 | 11.4 | 14.0 | 17.1 |
| 140 | 5.4 | 6.7 | 10.3 | 12.7 | 15.7 | 19.2 |
| 160 | 6.2 | 7.7 | 11.8 | 14.6 | 18.0 | 22.0 |
| 180 | 6.9 | 8.6 | 13.3 | 16.4 | 20.1 | 24.7 |
| 200 | 7.7 | 9.6 | 14.7 | 18.2 | 22.7 | 27.4 |
| 225 | 8.6 | 10.8 | 16.6 | 20.5 | 25.6 | 30.8 |
| 250 | 9.6 | 12.3 | 18.4 | 22.7 | 28.4 | 34.2 |
| 280 | 10.7 | 13.7 | 20.6 | 25.4 | 31.8 | 38.3 |
| 315 | 12.1 | 15.4 | 23.2 | 28.6 | 35.7 | 43.1 |
| 355 | 13.6 | 17.6 | 26.1 | 32.2 | 40.2 | 48.3 |
| 400 | 15.3 | 19.6 | 29.2 | 36.3 | 45.4 | 54.5 |
| 450 | 17.2 | 21.5 | 32.8 | 41.0 | 51.0 | 61.4 |
| 500 | 19.1 | 23.9 | 36.3 | 45.4 | 56.8 | 68.2 |
| 560 | 21.4 | 26.7 | 40.7 | 51.0 | 63.5 | 76.4 |
| 630 | 24.1 | 30.0 | 46.3 | 57.2 | 71.0 | 85.7 |
| 710 | 27.2 | 34.0 | 52.2 | 64.7 | 80.3 | 97.0 |
| 800 | 30.6 | 38.1 | 58.1 | 72.7 | 90.8 | 109.7 |
| 900 | 34.4 | 42.9 | 65.1 | 81.7 | 102.3 | 123.4 |
| 1000 | 38.2 | 47.7 | 73.5 | 90.8 | 115.0 | 138.1 |

**PresiÃ³n nominal de trabajo (Mpa):**
- SDR 26: 0.63
- SDR 21: 0.8
- SDR 17: 1.0
- SDR 13.6: 1.25
- SDR 11: 1.6
- SDR 9: 2.0
""")
    with col4:
        pass
    st.markdown("---")
    st.markdown(f"**Caudal calculado Q = {Q:.4f} mÂ³/s ({Q*3600:.2f} mÂ³/h, {Q*1000:.2f} L/s)**")
    st.caption("La fÃ³rmula asume flujo libre y condiciones ideales. Para condiciones reales, considerar pÃ©rdidas adicionales y coeficientes de seguridad.")
    st.markdown("**Importancia del coeficiente de descarga (C):** Un valor alto de C indica que la vÃ¡lvula permite un flujo eficiente con mÃ­nima resistencia, mientras que un valor bajo indica mayor restricciÃ³n al flujo. La elecciÃ³n del tipo de vÃ¡lvula depende del caudal requerido, presiÃ³n del sistema y necesidad de control fino del flujo.")

# --- PestaÃ±a 1: VÃ¡lvula ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de VÃ¡lvula
# --------------------------------------------------
with tab1:
    st.header("AnÃ¡lisis de CavitaciÃ³n en VÃ¡lvula")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    # Definir P2_range y theta_values antes de los bloques de columnas para uso global en la pestaÃ±a
    P2_min_default = 0.0
    P2_max_default = 8.0
    P2_step_default = 0.25
    # Se usan los valores por defecto, pero luego se actualizan con los inputs
    P2_min = P2_min_default
    P2_max = P2_max_default
    P2_step = P2_step_default
    with col_input:
        st.subheader("Datos de la VÃ¡lvula")
        P1_valve = st.number_input("PresiÃ³n de Entrada (P1, mca)", min_value=0.0, value=10.0, step=0.1)
        P2_valve = st.number_input("PresiÃ³n de Salida (P2, mca)", min_value=0.0, value=5.0, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        P2_min = st.number_input("P2 mÃ­nimo (mca)", min_value=0.0, value=P2_min_default, step=0.1)
        P2_max = st.number_input("P2 mÃ¡ximo (mca)", min_value=0.0, value=P2_max_default, step=0.1)
        P2_step = st.number_input("Paso de P2 (mca)", min_value=0.001, max_value=10.0, value=P2_step_default, step=0.1)
        st.subheader("Resultados")
        theta_valve = cav_calc.calculate_cavitation_index(P1_valve, P2_valve)
        if theta_valve is not None:
            st.metric(label="Ãndice de CavitaciÃ³n (Ï‘)", value=f"{theta_valve:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_valve))
        else:
            st.write("Ingrese valores vÃ¡lidos para calcular el Ã­ndice de cavitaciÃ³n.")
        st.info("""
**InterpretaciÃ³n:**
- Si Ï‘ < 0.5: Riesgo crÃ­tico de daÃ±o por cavitaciÃ³n.
- Si 0.5 â‰¤ Ï‘ < 0.8: Riesgo alto de ruido por cavitaciÃ³n.
- Si Ï‘ â‰¥ 0.8: Riesgo bajo o nulo de cavitaciÃ³n.
""")
        # ExplicaciÃ³n Kv ...
    # Calcular P2_range y theta_values despuÃ©s de obtener los inputs
    P2_range = np.arange(P2_min, P2_max+P2_step, P2_step)
    theta_values = []
    for p2 in P2_range:
        if P1_valve > p2:
            theta = cav_calc.calculate_cavitation_index(P1_valve, p2)
            theta_values.append(theta if theta is not None else np.nan)
        else:
            theta_values.append(np.nan)

    with col_graph:
        st.subheader("GrÃ¡fica: Ãndice de CavitaciÃ³n vs. PresiÃ³n de Salida")
        fig_valve = go.Figure()
        fig_valve.add_trace(go.Scatter(x=P2_range, y=theta_values, mode='lines+markers',
                                       line=dict(color='#2ca02c'),
                                       marker=dict(symbol='diamond-open', color='blue'),
                                       hovertemplate='<b>P2 = %{x:.2f} mca<br>Ï‘ = %{y:.2f}</b><extra></extra>',
                                       hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))))
        fig_valve.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="LÃ­mite DaÃ±o (0.5)", secondary_y=False)
        fig_valve.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="LÃ­mite Ruido (0.8)", secondary_y=False)
        fig_valve.update_layout(
            title="Ãndice de CavitaciÃ³n vs. PresiÃ³n de Salida en VÃ¡lvula",
            xaxis=dict(
                title="PresiÃ³n de Salida (mca)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="Ãndice de CavitaciÃ³n (Ï‘)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_valve, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Ãndice de CavitaciÃ³n</h4>', unsafe_allow_html=True)
        df_valve = pd.DataFrame({
            'P2 (mca)': np.round(P2_range, 2),
            'Ï‘': np.round(theta_values, 2)
        })
        st.dataframe(df_valve, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"): 
            st.markdown(r"""
**Ãndice de CavitaciÃ³n:**

$$
\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}
$$

**Donde:**
- **P1**: PresiÃ³n de entrada (mca)
- **P2**: PresiÃ³n de salida (mca)
- **Pvapor**: PresiÃ³n de vapor del fluido (mca)

Un valor bajo de **Ï‘** indica mayor riesgo de cavitaciÃ³n.
""")
        # Panel independiente de explicaciÃ³n de presiones
        with st.expander("Â¿QuÃ© significan la presiÃ³n de entrada y salida?"):
            st.markdown("""
- **PresiÃ³n de entrada (Pâ‚):** Es la presiÃ³n del fluido justo antes de entrar a la vÃ¡lvula. Representa la energÃ­a por unidad de peso del fluido en el punto aguas arriba de la vÃ¡lvula, normalmente medida en metros de columna de agua (mca). Esta presiÃ³n depende de la altura, el caudal y las condiciones del sistema antes de la vÃ¡lvula.

- **PresiÃ³n de salida (Pâ‚‚):** Es la presiÃ³n del fluido justo despuÃ©s de pasar por la vÃ¡lvula. Representa la energÃ­a por unidad de peso del fluido en el punto aguas abajo de la vÃ¡lvula, tambiÃ©n en metros de columna de agua (mca). Esta presiÃ³n suele ser menor que la de entrada debido a la pÃ©rdida de energÃ­a (caÃ­da de presiÃ³n) que ocurre al atravesar la vÃ¡lvula.

**En resumen:**
- **Pâ‚** indica cuÃ¡nta presiÃ³n tiene el fluido antes de la vÃ¡lvula.
- **Pâ‚‚** indica cuÃ¡nta presiÃ³n queda despuÃ©s de la vÃ¡lvula.
""")
        with st.expander("Criterios de cavitaciÃ³n segÃºn sigma (Ïƒ)"):
            st.markdown("""
**Criterios de cavitaciÃ³n segÃºn el Ã­ndice sigma (Ïƒ):**

| Rango de Ïƒ         | InterpretaciÃ³n                                      |
|:------------------:|:---------------------------------------------------|
| Ïƒ â‰¥ 2.0            | <span style='color:green'><b>No hay cavitaciÃ³n</b></span> |
| 1.7 < Ïƒ < 2.0      | <span style='color:#7ca300'><b>ProtecciÃ³n suficiente con materiales endurecidos</b></span> |
| 1.5 < Ïƒ < 1.7      | <span style='color:orange'><b>Algo de cavitaciÃ³n, puede funcionar un solo escalÃ³n</b></span> |
| 1.0 < Ïƒ < 1.5      | <span style='color:#ff6600'><b>Potencial de cavitaciÃ³n severa, se requiere reducciÃ³n en varias etapas</b></span> |
| Ïƒ < 1.0            | <span style='color:red'><b>Flashing (vaporizaciÃ³n instantÃ¡nea)</b></span> |

> **Ïƒ = (Pâ‚ - P_v) / (Pâ‚ - Pâ‚‚)**

- **SUPER CAVITACIÃ“N:** Ïƒ bajo, aceleraciÃ³n alta, daÃ±o severo.
- **CAVITACIÃ“N PLENA:** Ïƒ intermedio, daÃ±o considerable.
- **CAVITACIÃ“N INCIPIENTE:** Ïƒ cerca de 1.5-1.7, inicio de daÃ±o.
- **SUBCRÃTICO:** Ïƒ alto, sin daÃ±o.

Estos criterios ayudan a seleccionar el diseÃ±o y materiales adecuados para evitar daÃ±os por cavitaciÃ³n en vÃ¡lvulas.
""", unsafe_allow_html=True)

    # --- SEPARADOR A TODO EL ANCHO ---
    st.markdown("---")
    # --- NUEVA FILA DE COLUMNAS PARA Kv SOLO EN ESTA PESTAÃ‘A ---
    col_kv_exp, col_kv_graf, col_kv_tabla, col_kv_extra = st.columns([0.22, 0.4, 0.28, 0.1])
    with col_kv_exp:
        st.subheader("Coeficiente de caudal (Kv)")
        st.markdown("""
Las vÃ¡lvulas de control son conceptualmente orificios de Ã¡rea variable. Se las puede considerar simplemente como una restricciÃ³n que cambia su tamaÃ±o de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relaciÃ³n de diferencia de altura (Î”h) o presiÃ³n (Î”P) entre la entrada y salida de la vÃ¡lvula con el caudal (Q).
""")
        st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
        st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (mÂ³/h)
- $Q$: Caudal volumÃ©trico (mÂ³/h)
- $\rho$: Densidad (kg/mÂ³)
- $\Delta p$: Diferencia de presiÃ³n (bar)
- $P_1$: PresiÃ³n de entrada (bar)
- $P_2$: PresiÃ³n de salida (bar)
""")
        mca_a_bar = 0.0980665
        P1_bar = P1_valve * mca_a_bar
        P2_bar = P2_valve * mca_a_bar
        delta_p_bar = np.abs(P1_bar - P2_bar)
        densidad = densidad_calculada
        st.markdown(f"**PresiÃ³n de entrada (P1):** {P1_bar:.2f} bar  ")
        st.markdown(f"**PresiÃ³n de salida (P2):** {P2_bar:.2f} bar  ")
        st.markdown(f"**Diferencia de presiÃ³n (Î”p):** {delta_p_bar:.2f} bar  ")
        st.markdown(f"**Densidad (Ï):** {densidad:.1f} kg/mÂ³  ")
    with col_kv_graf:
        st.subheader("Ãndice de caudal Kv")
        # Corregido: GrÃ¡fico Î”p vs Kv con Q fijo
        Q_m3h = 10  # Caudal fijo tÃ­pico para la curva
        Kv_range = np.linspace(1, 15, 30)
        delta_p = (densidad / 1000) * (Q_m3h / Kv_range) ** 2
        fig_kv = go.Figure()
        fig_kv.add_trace(go.Scatter(
            x=Kv_range, y=delta_p, mode='markers+lines',
            marker=dict(symbol='diamond-open', color='blue'),
            line=dict(dash='solid', color='blue'),
            name='Î”p vs Kv'
        ))
        fig_kv.update_layout(
            title="VariaciÃ³n de Î”p con Kv",
            xaxis=dict(
                title="Kv (mÂ³/h)", 
                tickformat='.1f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            yaxis=dict(
                title="Î”p (bar)", 
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            template="simple_white",
            height=300
        )
        st.plotly_chart(fig_kv, use_container_width=True, key=f"fig_kv_kv_{uuid.uuid4()}")
        # Mantener el segundo grÃ¡fico y tablas igual
        delta_p_graf = np.linspace(0.1, 4.5, 30)
        Kv_alto = 15
        Kv_bajo = 3
        Q_kv_alto = Kv_alto / np.sqrt(densidad / (1000 * delta_p_graf))
        Q_kv_bajo = Kv_bajo / np.sqrt(densidad / (1000 * delta_p_graf))
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=delta_p_graf, y=Q_kv_alto, mode='markers+lines',
                                   marker=dict(symbol='diamond-open', color='blue'),
                                   line=dict(dash='solid', color='blue'),
                                   name='Kv >>'))
        fig_q.add_trace(go.Scatter(x=delta_p_graf, y=Q_kv_bajo, mode='markers+lines',
                                   marker=dict(symbol='square-open', color='orange'),
                                   line=dict(dash='solid', color='orange'),
                                   name='Kv <<'))
        fig_q.update_layout(
            title="Q vs. PÃ©rdida de carga para Kv alto y bajo",
            xaxis=dict(
                title="PÃ©rdida de carga (bar)", 
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            yaxis=dict(
                title="Q (mÂ³/h)", 
                tickformat='.1f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1
            ),
            template="simple_white",
            height=300
        )
        st.plotly_chart(fig_q, use_container_width=True, key=f"fig_q_kv_{uuid.uuid4()}")
    with col_kv_tabla:
        st.markdown('<h4 style="text-align:center;">Tabla Kv calculado</h4>', unsafe_allow_html=True)
        df_kv = pd.DataFrame({
            'Kv (mÂ³/h)': np.round(Kv_range, 2),
            'Î”p (bar)': np.round(delta_p, 2)
        })
        st.dataframe(df_kv, use_container_width=True)
        st.markdown('<h4 style="text-align:center;">Tabla Q vs Î”p para Kv</h4>', unsafe_allow_html=True)
        df_qkv = pd.DataFrame({
            'Î”p (bar)': np.round(delta_p_graf, 2),
            'Q (mÂ³/h) Kv alto': np.round(Q_kv_alto, 2),
            'Q (mÂ³/h) Kv bajo': np.round(Q_kv_bajo, 2)
        })
        st.dataframe(df_qkv, use_container_width=True)
    with col_kv_extra:
        pass

# --- PestaÃ±a 2: Sistema de Bombeo (NPSH) ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de Sistema de Bombeo (NPSH)
# --------------------------------------------------
with tab2:
    st.header("AnÃ¡lisis de NPSH Disponible para Bomba")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos del Sistema de Bombeo")
        # Tabla de presiÃ³n atmosfÃ©rica segÃºn altura sobre el nivel del mar
        alturas_msnm = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
        presiones_mca = [10.3, 9.7, 9.1, 8.6, 8.1, 7.6, 7.1, 6.7, 6.3, 5.9, 5.5, 5.2, 4.9]
        def interpolar_presion(cota):
            if cota <= alturas_msnm[0]:
                return presiones_mca[0]
            if cota >= alturas_msnm[-1]:
                return presiones_mca[-1]
            for i in range(1, len(alturas_msnm)):
                if cota < alturas_msnm[i]:
                    x0, x1 = alturas_msnm[i-1], alturas_msnm[i]
                    y0, y1 = presiones_mca[i-1], presiones_mca[i]
                    return y0 + (y1 - y0) * (cota - x0) / (x1 - x0)
            return presiones_mca[-1]
        cota_bomba = st.number_input("Cota de la bomba (msnm)", min_value=0.0, max_value=6000.0, value=0.0, step=1.0)
        H_atm = interpolar_presion(cota_bomba)
        st.number_input("PresiÃ³n AtmosfÃ©rica (H_atm, mca)", value=H_atm, disabled=True)
        H_static_suction = st.number_input("Altura EstÃ¡tica de SucciÃ³n (H_s, mca, + si es sobre nivel bomba, - si es bajo)", value=2.0, step=0.1)
        NPSHr_pump = st.number_input("NPSH Requerido de la Bomba (NPSH_necesaria, mca)", min_value=0.0, value=3.0, step=0.1)
        H_losses_suction = st.number_input("PÃ©rdidas por FricciÃ³n y Accesorios en SucciÃ³n (Î”H_s, mca)", min_value=0.0, value=1.5, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        losses_min = st.number_input("PÃ©rdidas mÃ­nimas (Î”H_s, mca)", min_value=0.0, value=0.0, step=0.01)
        losses_max = st.number_input("PÃ©rdidas mÃ¡ximas (Î”H_s, mca)", min_value=0.0, value=5.0, step=0.01)
        losses_step = st.number_input("Paso de pÃ©rdidas (Î”H_s, mca)", min_value=0.001, max_value=10.0, value=0.25, step=0.1)
        st.subheader("Resultados")
        npsha_calculated = npsh_calc.calculate_npsha(H_atm, H_static_suction, H_losses_suction)
        st.metric(label="NPSH Disponible (NPSHA, mca)", value=f"{npsha_calculated:.3f}")
        st.write(npsh_calc.get_npsh_risk_description(npsha_calculated, NPSHr_pump))
        st.info(f"**Criterio:** NPSHA debe ser mayor a {NPSHr_pump:.1f} mca (NPSHR). Se recomienda un margen de seguridad (ej. 1.1 * NPSHR).")

    with col_graph:
        st.subheader("GrÃ¡fica: NPSHA vs. PÃ©rdidas en SucciÃ³n")
        losses_range = np.arange(losses_min, losses_max+losses_step, losses_step)
        npsha_values = [npsh_calc.calculate_npsha(H_atm, H_static_suction, l) for l in losses_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.round(losses_range,2), y=npsha_values, mode='lines', name='NPSH Disponible',
                                 line=dict(color='#2ca02c'),
                                 hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                 hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))))
        # Punto rojo para el valor de entrada actual
        if H_losses_suction is not None and npsha_calculated is not None:
            fig.add_trace(go.Scatter(x=[H_losses_suction], y=[npsha_calculated], mode='markers',
                                     marker=dict(color='red', size=12, symbol='circle'),
                                     name='Punto de entrada',
                                     hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))))
        fig.add_hline(y=NPSHr_pump, line_dash="dash", line_color="red", annotation_text=f"NPSHR ({NPSHr_pump:.1f})")
        fig.add_hline(y=NPSHr_pump * 1.1, line_dash="dash", line_color="orange", annotation_text=f"NPSHR + Margen ({NPSHr_pump*1.1:.1f})")
        fig.update_layout(
            title=f"NPSHA vs. PÃ©rdidas en SucciÃ³n (H_estÃ¡tica={H_static_suction:.1f}m)",
            xaxis=dict(
                title="PÃ©rdidas por FricciÃ³n y Accesorios en SucciÃ³n (mca)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="NPSH Disponible (mca)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de NPSHA vs. PÃ©rdidas</h4>', unsafe_allow_html=True)
        df_npsh = pd.DataFrame({
            'PÃ©rdidas (mca)': np.round(losses_range, 2),
            'NPSHA (mca)': np.round(npsha_values, 2)
        })
        st.dataframe(df_npsh, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**NPSH Disponible:**

$$
NPSH_{disponible} = H_{atm} - (P_{v} + H_{s} + \Delta H_{s})
$$

**Donde:**
- **NPSHdisponible**: Carga neta positiva de succiÃ³n disponible (m)
- **Hatm**: PresiÃ³n atmosfÃ©rica (m)
- **Pv**: PresiÃ³n de vapor (m)
- **Hs**: Altura estÃ¡tica de succiÃ³n (m)
- **Î”Hs**: PÃ©rdida de presiÃ³n por fricciÃ³n (m)
""")
        with st.expander("PresiÃ³n AtmosfÃ©rica segÃºn altura"):
            st.markdown('''
| Altura sobre el nivel del mar (m) | PresiÃ³n atmosfÃ©rica (m.c.a.) |
|:-------------------------------:|:----------------------------:|
| 0     | 10,3 |
| 500   | 9,7  |
| 1 000 | 9,1  |
| 1 500 | 8,6  |
| 2 000 | 8,1  |
| 2 500 | 7,6  |
| 3 000 | 7,1  |
| 3 500 | 6,7  |
| 4 000 | 6,3  |
| 4 500 | 5,9  |
| 5 000 | 5,5  |
| 5 500 | 5,2  |
| 6 000 | 4,9  |
''')
        
        with st.expander("InterpretaciÃ³n y criterios"):
            st.markdown(r"""
La NPSH necesaria y la disponible son parÃ¡metros de control de la cavitaciÃ³n en los impulsores de las bombas.

La NPSH disponible depende del diseÃ±o del bombeo y representa la diferencia entre la carga absoluta y la presiÃ³n de vapor del lÃ­quido a temperatura constante.

**NPSH necesaria:** Es la carga exigida por la bomba entre la presiÃ³n de succiÃ³n y la presiÃ³n de vapor del lÃ­quido para que la bomba no cavite.

$$
NPSH_{disponible} \geq NPSH_{necesaria} + 0.5
$$
""")

# --- PestaÃ±a 3: LÃ­nea de ImpulsiÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de LÃ­nea de ImpulsiÃ³n
# --------------------------------------------------
with tab3:
    st.header("AnÃ¡lisis de CavitaciÃ³n en LÃ­nea de ImpulsiÃ³n")
    st.write("Se analiza un segmento o punto crÃ­tico de la lÃ­nea de impulsiÃ³n.")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos de la LÃ­nea de ImpulsiÃ³n")
        pipe_roughness_C = st.slider("Coeficiente Hazen-Williams (C)", min_value=80, max_value=150, value=120, step=5)
        flow_rate_imp = st.number_input("Caudal (L/s)", min_value=0.0, value=5.0, step=0.1)
        reference_diameter_mm = st.number_input("DiÃ¡metro (mm)", min_value=0.0, value=50.0, step=1.0)
        pipe_length_imp = st.number_input("Longitud del Tramo (m)", min_value=1.0, value=100.0, step=1.0)
        P_upstream_imp = st.number_input("PresiÃ³n Aguas Arriba (P_upstream, mca)", min_value=0.0, value=25.0, step=0.1)
        P_downstream_imp = st.number_input("PresiÃ³n Aguas Abajo (P_downstream, mca)", min_value=0.0, value=20.0, step=0.1)
        st.markdown('---')
        st.subheader("Datos GrÃ¡fico")
        diam_min = st.number_input("DiÃ¡metro mÃ­nimo (mm)", min_value=0.0, value=0.0, step=1.0)
        diam_max = st.number_input("DiÃ¡metro mÃ¡ximo (mm)", min_value=0.0, value=100.0, step=1.0)
        diam_step = st.number_input("Paso de diÃ¡metro (mm)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        # Ajustar cÃ¡lculos para mm y L/s
        diameter_range_m = np.arange(diam_min, diam_max+diam_step, diam_step) / 1000
        flow_rate_imp_m3s = flow_rate_imp / 1000
        reference_diameter_m = reference_diameter_mm / 1000
        pressure_drop_values = []
        downstream_pressure_values = []
        theta_values_imp_graph = []
        P_initial_graph = P_upstream_imp
        for d in diameter_range_m:
            friction_loss = calculate_friction_losses_hw(flow_rate_imp_m3s, d, pipe_length_imp, pipe_roughness_C)
            P_downstream_calc_graph = P_initial_graph - friction_loss
            pressure_drop_values.append(friction_loss)
            downstream_pressure_values.append(P_downstream_calc_graph)
            if P_initial_graph > P_downstream_calc_graph:
                theta = cav_calc.calculate_cavitation_index(P_initial_graph, P_downstream_calc_graph)
                theta_values_imp_graph.append(theta if theta is not None else np.nan)
            else:
                theta_values_imp_graph.append(np.nan)
        # Punto rojo para el diÃ¡metro de referencia
        friction_loss_ref = calculate_friction_losses_hw(flow_rate_imp_m3s, reference_diameter_m, pipe_length_imp, pipe_roughness_C)
        P_downstream_ref = P_upstream_imp - friction_loss_ref
        theta_ref = cav_calc.calculate_cavitation_index(P_upstream_imp, P_downstream_ref)
        st.subheader("Resultados")
        theta_imp = cav_calc.calculate_cavitation_index(P_upstream_imp, P_downstream_imp)
        if theta_imp is not None:
            st.metric(label="Ãndice de CavitaciÃ³n (Ï‘)", value=f"{theta_imp:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_imp))
        else:
            st.write("Ingrese valores vÃ¡lidos para calcular el Ã­ndice de cavitaciÃ³n.")

    with col_graph:
        st.subheader("GrÃ¡fica: Impacto del DiÃ¡metro en PÃ©rdidas y Presiones")
        fig_imp = make_subplots(specs=[[{"secondary_y": True}]])
        # Convertir a mm para el eje x
        diameter_range_mm = diameter_range_m * 1000
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=pressure_drop_values, mode='lines', name='PÃ©rdida de PresiÃ³n',
                                     line=dict(color='#d62728'),
                                     hovertemplate='<b>PÃ©rdida = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#d62728', font=dict(color='white', family='Arial Black'))), secondary_y=False)
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=downstream_pressure_values, mode='lines', name='PresiÃ³n Aguas Abajo',
                                     line=dict(color='#1f77b4'),
                                     hovertemplate='<b>PresiÃ³n = %{y:.2f} mca</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))), secondary_y=False)
        fig_imp.add_trace(go.Scatter(x=np.round(diameter_range_mm,1), y=theta_values_imp_graph, mode='lines', name='Ãndice de CavitaciÃ³n (Ï‘)',
                                     line=dict(color='#2ca02c'),
                                     hovertemplate='<b>Ï‘ = %{y:.2f}</b><extra></extra>',
                                     hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))), secondary_y=True)
        fig_imp.add_hline(y=cav_calc.Pv, line_dash="dot", line_color="blue", annotation_text="Pv", secondary_y=False)
        fig_imp.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="LÃ­mite DaÃ±o (0.5)", secondary_y=True)
        fig_imp.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="LÃ­mite Ruido (0.8)", secondary_y=True)
        fig_imp.update_layout(
            title="Impacto del DiÃ¡metro en PÃ©rdidas, PresiÃ³n y Ï‘ en LÃ­nea de ImpulsiÃ³n",
            xaxis=dict(
                title="DiÃ¡metro de TuberÃ­a (mm)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="PresiÃ³n / PÃ©rdida (mca)",
                tickformat='.2f',
            ),
            yaxis2=dict(
                title="Ãndice de CavitaciÃ³n (Ï‘)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Resultados de ImpulsiÃ³n</h4>', unsafe_allow_html=True)
        df_imp = pd.DataFrame({
            'DiÃ¡metro (m)': np.round(diameter_range_m, 3),
            'PÃ©rdida (mca)': np.round(pressure_drop_values, 2),
            'PresiÃ³n Abajo (mca)': np.round(downstream_pressure_values, 2),
            'Ï‘': np.round(theta_values_imp_graph, 2)
        })
        st.dataframe(df_imp, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**PÃ©rdida por fricciÃ³n (Hazen-Williams):**

$$
H_f = \frac{10.67 \cdot L \cdot Q^{1.852}}{C^{1.852} \cdot D^{4.87}}
$$

**Ãndice de CavitaciÃ³n:**

$$
\vartheta = \frac{P_{abajo} - P_{vapor}}{P_{arriba} - P_{abajo}}
$$

**Donde:**
- **Parriba**: PresiÃ³n aguas arriba (mca)
- **Pabajo**: PresiÃ³n aguas abajo (mca)
- **Pvapor**: PresiÃ³n de vapor (mca)
""")

# --- PestaÃ±a 4: LÃ­nea de SucciÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de LÃ­nea de SucciÃ³n
# --------------------------------------------------
with tab4:
    st.header("AnÃ¡lisis de CavitaciÃ³n en LÃ­nea de SucciÃ³n")
    st.write("La lÃ­nea de succiÃ³n es crÃ­tica para la cavitaciÃ³n en bombas.")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("Datos de la LÃ­nea de SucciÃ³n")
        diam_min_s = st.number_input("DiÃ¡metro mÃ­nimo (m) succiÃ³n", min_value=0.01, value=0.05, step=0.01)
        diam_max_s = st.number_input("DiÃ¡metro mÃ¡ximo (m) succiÃ³n", min_value=0.01, value=0.4, step=0.01)
        diam_step_s = st.number_input("Paso de diÃ¡metro (m) succiÃ³n", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        P_source_suction = st.number_input("PresiÃ³n en Fuente (P_fuente, mca, ej. AtmosfÃ©rica)", min_value=0.0, value=10.33, step=0.1)
        H_static_suction_tab4 = st.number_input("Altura EstÃ¡tica al Eje de Bomba (mca, + si lÃ­quido arriba, - si abajo)", value=2.0, step=0.1)
        flow_rate_suction = st.number_input("Caudal (mÂ³/s)", min_value=0.0, value=0.05, step=0.001, format="%.3f")
        pipe_length_suction = st.number_input("Longitud de la LÃ­nea de SucciÃ³n (m)", min_value=1.0, value=10.0, step=1.0)
        pipe_roughness_C_suction = st.slider("Coeficiente Hazen-Williams (C) SucciÃ³n", min_value=80, max_value=150, value=120, step=5)
        k_fitting_suction = st.number_input("Coeficiente K de Accesorios Global (succiÃ³n)", min_value=0.0, value=2.0, step=0.1)
        st.subheader("Resultados")
        reference_diameter_suction = st.number_input("DiÃ¡metro de Referencia para CÃ¡lculo (m)", min_value=0.05, value=0.2, step=0.01)
        friction_loss_suction = calculate_friction_losses_hw(flow_rate_suction, reference_diameter_suction, pipe_length_suction, pipe_roughness_C_suction)
        local_loss_suction = calculate_local_losses(flow_rate_suction, reference_diameter_suction, k_fitting_suction)
        total_loss_suction = friction_loss_suction + local_loss_suction
        P_brida_succion_calc = P_source_suction + H_static_suction_tab4 - total_loss_suction
        npsha_suction_calc = npsh_calc.calculate_npsha(P_source_suction, H_static_suction_tab4, total_loss_suction)
        st.metric(label="PresiÃ³n en Brida de SucciÃ³n (mca)", value=f"{P_brida_succion_calc:.3f}")
        st.metric(label="NPSH Disponible (NPSHA, mca)", value=f"{npsha_suction_calc:.3f}")
        NPSHr_dummy_for_desc = 3.0
        st.write(npsh_calc.get_npsh_risk_description(npsha_suction_calc, NPSHr_dummy_for_desc))
        st.info("Nota: Para una evaluaciÃ³n completa, compare el NPSH Disponible con el NPSH Requerido de su bomba.")

    with col_graph:
        st.subheader("GrÃ¡fica: NPSHA vs. DiÃ¡metro de SucciÃ³n")
        diameter_range_suction_graph = np.arange(diam_min_s, diam_max_s+diam_step_s, diam_step_s)
        npsha_values_suction_graph = []
        for d in diameter_range_suction_graph:
            friction_loss_g = calculate_friction_losses_hw(flow_rate_suction, d, pipe_length_suction, pipe_roughness_C_suction)
            local_loss_g = calculate_local_losses(flow_rate_suction, d, k_fitting_suction)
            total_loss_g = friction_loss_g + local_loss_g
            npsha_g = npsh_calc.calculate_npsha(P_source_suction, H_static_suction_tab4, total_loss_g)
            npsha_values_suction_graph.append(npsha_g)
        fig_suction = go.Figure()
        fig_suction.add_trace(go.Scatter(x=np.round(diameter_range_suction_graph,3), y=npsha_values_suction_graph, mode='lines', name='NPSH Disponible',
                                         line=dict(color='#9467bd'),
                                         hovertemplate='<b>NPSHA = %{y:.2f} mca</b><extra></extra>',
                                         hoverlabel=dict(bgcolor='#9467bd', font=dict(color='white', family='Arial Black'))))
        fig_suction.add_hline(y=NPSHr_dummy_for_desc, line_dash="dash", line_color="red", annotation_text=f"NPSHR de Ejemplo ({NPSHr_dummy_for_desc:.1f})")
        fig_suction.update_layout(
            title="NPSH Disponible vs. DiÃ¡metro de TuberÃ­a de SucciÃ³n",
            xaxis=dict(
                title="DiÃ¡metro de TuberÃ­a (m)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="NPSH Disponible (mca)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_suction, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de NPSHA vs. DiÃ¡metro</h4>', unsafe_allow_html=True)
        df_suction = pd.DataFrame({
            'DiÃ¡metro (m)': np.round(diameter_range_suction_graph, 3),
            'NPSHA (mca)': np.round(npsha_values_suction_graph, 2)
        })
        st.dataframe(df_suction, use_container_width=True)

        # Panel de fÃ³rmulas
        with st.expander("Ver fÃ³rmulas utilizadas"):
            st.markdown(r"""
**PÃ©rdida por fricciÃ³n (Hazen-Williams):**

$$
H_f = \frac{10.67 \cdot L \cdot Q^{1.852}}{C^{1.852} \cdot D^{4.87}}
$$

**PÃ©rdida local:**

$$
H_{local} = K \cdot \frac{V^2}{2g}
$$

Donde:
- \(K\): Coeficiente global de accesorios
- \(V\): Velocidad del fluido (m/s)
- \(g\): Gravedad (9.81 m/sÂ²)

**NPSH Disponible:**

$$
NPSH_{A} = H_{atm} + H_{estÃ¡tica\ succiÃ³n} - H_{pÃ©rdidas\ succiÃ³n} - P_{vapor}
$$
""")

# --- PestaÃ±a 5: PÃ©rdidas por FricciÃ³n ---
# --------------------------------------------------
# CÃ³digo correspondiente a la pestaÃ±a de PÃ©rdidas por FricciÃ³n
# --------------------------------------------------
def mougnie_velocity(diametro_m):
    # V = 1.5 * sqrt(D) + 0.05
    return 1.5 * np.sqrt(diametro_m) + 0.05

with tab5:
    st.header("GrÃ¡fico de Caudal vs. PÃ©rdidas por FricciÃ³n (Hazen-Williams)")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    with col_input:
        st.subheader("ParÃ¡metros de la TuberÃ­a")
        diametro_mm = st.number_input("DiÃ¡metro de tuberÃ­a (mm)", min_value=10.0, max_value=1000.0, value=63.0, step=0.1, format="%.2f")
        materiales_c = [
            {"nombre": "PVC o plÃ¡stico (nuevo)", "C": 150, "rango": "140 - 150"},
            {"nombre": "Polietileno de alta densidad (HDPE)", "C": 150, "rango": "140 - 150"},
            {"nombre": "Asbesto-cemento (nuevo)", "C": 140, "rango": "130 - 140"},
            {"nombre": "Hierro fundido (nuevo)", "C": 130, "rango": "120 - 130"},
            {"nombre": "Hierro fundido (usado)", "C": 110, "rango": "80 - 130"},
            {"nombre": "Acero nuevo (sin costura)", "C": 145, "rango": "130 - 150"},
            {"nombre": "Acero galvanizado (nuevo)", "C": 120, "rango": "110 - 120"},
            {"nombre": "Acero galvanizado (viejo)", "C": 90, "rango": "70 - 110"},
            {"nombre": "HormigÃ³n (buen acabado)", "C": 130, "rango": "120 - 140"},
            {"nombre": "HormigÃ³n (viejo o rugoso)", "C": 110, "rango": "90 - 130"},
            {"nombre": "Cobre o latÃ³n", "C": 145, "rango": "130 - 150"},
            {"nombre": "Fibrocemento", "C": 135, "rango": "130 - 140"},
        ]
        nombres_materiales = [m["nombre"] for m in materiales_c]
        material_idx = st.selectbox("Material de la tuberÃ­a", nombres_materiales, index=0)
        material_seleccionado = materiales_c[nombres_materiales.index(material_idx)]
        # Permitir modificar C dentro del rango sugerido
        rango_c = material_seleccionado["rango"].replace(" ", "").split("-")
        c_min = int(rango_c[0])
        c_max = int(rango_c[1])
        C_hw = st.number_input(
            "Coeficiente Hazen-Williams (C)",
            min_value=c_min,
            max_value=c_max,
            value=material_seleccionado["C"],
            step=1
        )
        st.info(f"Material simulado: **{material_seleccionado['nombre']}**\n\nC tÃ­pico: **{material_seleccionado['C']}**\n\nRango de C permitido: {material_seleccionado['rango']}")
        st.subheader("Rango de Caudal (L/s)")
        caudal_min = st.number_input("Caudal mÃ­nimo (L/s)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)
        caudal_max = st.number_input("Caudal mÃ¡ximo (L/s)", min_value=0.01, max_value=100.0, value=5.0, step=0.01)
        paso_caudal = st.number_input("Paso de caudal (L/s)", min_value=0.01, max_value=10.0, value=0.25, step=0.01)
        mostrar_umbral = False
        umbral_pend = None
        with st.expander("Opciones avanzadas: Umbral de pendiente para zona segura", expanded=False):
            st.caption("El umbral de pendiente define hasta quÃ© valor de pendiente (variaciÃ³n de pÃ©rdida por variaciÃ³n de caudal) se considera la zona segura. Si la pendiente local supera este valor, se considera que la curva ya no es segura para diseÃ±o.")
            umbral_pend = st.number_input("Umbral de pendiente para zona segura (m/Km por L/s)", min_value=0.01, max_value=100.0, value=1.0, step=0.25, key="umbral_pend")
            col1, col2 = st.columns([1,1])
            with col1:
                aplicar_umbral = st.button("Aplicar umbral de pendiente", key="btn_umbral")
            with col2:
                borrar_umbral = st.button("Borrar", key="btn_borrar_umbral")
            if aplicar_umbral:
                mostrar_umbral = True
            if borrar_umbral:
                mostrar_umbral = False

    with col_graph:
        st.subheader(f"GrÃ¡fica: PÃ©rdida de carga por fricciÃ³n vs. Caudal  ", divider="gray")
        st.markdown(f"**Material simulado:** {material_seleccionado['nombre']}  |  C tÃ­pico: {material_seleccionado['C']}  |  Rango de C: {material_seleccionado['rango']}")
        Q_lps_graf = np.arange(caudal_min, caudal_max + paso_caudal, paso_caudal)
        diametro_m = diametro_mm / 1000
        Q_m3s_graf = Q_lps_graf / 1000
        L = 1  # longitud en metros
        j_graf = [10.67 * L * (q ** 1.852) / (C_hw ** 1.852 * diametro_m ** 4.87) for q in Q_m3s_graf]
        hf_km_graf = [j * 1000 for j in j_graf]

        # --- AnÃ¡lisis: punto donde la pendiente supera el umbral ---
        hf_km_graf = np.array(hf_km_graf)
        Q_lps_graf = np.array(Q_lps_graf)
        if mostrar_umbral and umbral_pend is not None:
            pendiente = np.diff(hf_km_graf) / np.diff(Q_lps_graf)
            idx_cambio = next((i for i, p in enumerate(pendiente) if p > umbral_pend), None)
            if idx_cambio is not None:
                # El punto de cambio es el primer Q donde la pendiente supera el umbral
                caudal_div = float(Q_lps_graf[idx_cambio+1])
                cambio_detectado = True
            else:
                caudal_div = None
                cambio_detectado = False
            # Ãrea bajo la curva segura (antes del punto de cambio)
            if cambio_detectado:
                area_segura = np.trapz(hf_km_graf[:idx_cambio+2], Q_lps_graf[:idx_cambio+2])
            else:
                area_segura = np.trapz(hf_km_graf, Q_lps_graf)
        else:
            cambio_detectado = False
            caudal_div = None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Q_lps_graf,
            y=hf_km_graf,
            mode='lines',
            name='PÃ©rdida de carga (m/Km)',
            line=dict(color='#ff7f0e'),
            hovertemplate='<b>Q = %{x:.2f} L/s<br>hf/Km = %{y:.2f} m/Km</b><extra></extra>'
        ))
        # LÃ­nea vertical en el punto de cambio solo si se aplica el umbral
        if mostrar_umbral and cambio_detectado and caudal_div is not None:
            fig.add_vline(x=caudal_div, line_dash="dot", line_color="#0080ff", line_width=2,
                          annotation_text=f"Q={caudal_div:.2f} L/s", annotation_position="top right")
        fig.update_layout(
            title=f"PÃ©rdida de carga por fricciÃ³n vs. Caudal  |  Material: {material_seleccionado['nombre']} (C={C_hw})",
            xaxis=dict(
                title="Caudal (L/s)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="PÃ©rdida de carga (m/Km)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        if mostrar_umbral:
            if cambio_detectado and caudal_div is not None:
                st.info(f"El punto de cambio de pendiente se estima en Q = {caudal_div:.2f} L/s")
            else:
                st.warning(f"No se detectÃ³ un cambio de pendiente dentro del rango y umbral seleccionados. La lÃ­nea se muestra al final del rango.")

        # GrÃ¡fico adicional: Caudal vs Velocidad
        V_graf = Q_m3s_graf / (np.pi * (diametro_m / 2) ** 2) if diametro_m > 0 else np.zeros_like(Q_m3s_graf)
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=Q_lps_graf,
            y=V_graf,
            mode='lines',
            name='Velocidad (m/s)',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>Q = %{x:.2f} L/s<br>V = %{y:.2f} m/s</b><extra></extra>'
        ))
        fig_vel.update_layout(
            title="Caudal vs. Velocidad en la TuberÃ­a",
            xaxis=dict(
                title="Caudal (L/s)",
                hoverformat='.2f',
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                showgrid=True,
                zeroline=False,
                tickformat='.2f',
            ),
            yaxis=dict(
                title="Velocidad (m/s)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de PÃ©rdidas por FricciÃ³n</h4>', unsafe_allow_html=True)
        df_friction = pd.DataFrame({
            'Caudal (L/s)': np.round(Q_lps_graf, 2),
            'PÃ©rdida (m/Km)': np.round(hf_km_graf, 2),
            'Velocidad (m/s)': np.round(V_graf, 2)
        })
        st.dataframe(df_friction, use_container_width=True)
        
        with st.expander("ðŸ“š Coeficiente de caudal (Kv)"):
            st.markdown("""
Las vÃ¡lvulas de control son conceptualmente orificios de Ã¡rea variable. Se las puede considerar simplemente como una restricciÃ³n que cambia su tamaÃ±o de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relaciÃ³n de diferencia de altura (Î”h) o presiÃ³n (Î”P) entre la entrada y salida de la vÃ¡lvula con el caudal (Q).
""")
            st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
            st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (mÂ³/h)
- $Q$: Caudal volumÃ©trico (mÂ³/h)
- $\rho$: Densidad (kg/mÂ³)
- $\Delta p$: Diferencia de presiÃ³n (bar)
- $P_1$: PresiÃ³n de entrada (bar)
- $P_2$: PresiÃ³n de salida (bar)
""")
            mca_a_bar = 0.0980665
            P1_bar = P1_valve * mca_a_bar
            P2_bar = P2_valve * mca_a_bar
            delta_p_bar = np.abs(P1_bar - P2_bar)
            densidad = densidad_calculada
            st.markdown(f"**PresiÃ³n de entrada (P1):** {P1_bar:.2f} bar  ")
            st.markdown(f"**PresiÃ³n de salida (P2):** {P2_bar:.2f} bar  ")
            st.markdown(f"**Diferencia de presiÃ³n (Î”p):** {delta_p_bar:.2f} bar  ")
            st.markdown(f"**Densidad (Ï):** {densidad:.1f} kg/mÂ³  ")

# --- NUEVA PESTAÃ‘A: Flujos Transitorios ---
with tab_trans:
    st.header("ðŸŒŠ Flujos Transitorios")
    # Primer bloque: CÃ¡lculo de celeridad y espesor de tuberÃ­a
    st.subheader("CÃ¡lculo de velocidad de propagaciÃ³n de onda (celeridad) y espesor de tuberÃ­a")
    col1, col2, col3, col4 = st.columns([0.22, 0.4, 0.28, 0.1])
    with col1:
        st.markdown("#### Datos de entrada")
        materiales = {
            "Acero": {"E": (200e9, 212e9), "a": (1000, 1250)},
            "Fibro Cemento": {"E": (23.5e9, 23.5e9), "a": (900, 1200)},
            "Concreto": {"E": (39e9, 39e9), "a": (1050, 1150)},
            "Hierro DÃºctil": {"E": (166e9, 166e9), "a": (1000, 1350)},
            "Polietileno alta densidad": {"E": (0.59e9, 1.67e9), "a": (230, 430)},
            "PVC": {"E": (2.4e9, 2.75e9), "a": (300, 500)},
        }
        material = st.selectbox("Material de la tuberÃ­a", list(materiales.keys()))
        E_range = materiales[material]["E"]
        a_range = materiales[material]["a"]
        E_prom = sum(E_range) / 2
        a_prom = sum(a_range) / 2
        st.markdown(f"**Rango de mÃ³dulo de elasticidad (E):** {E_range[0]/1e9:.2f} - {E_range[1]/1e9:.2f} GPa")
        st.markdown(f"**Rango de celeridad (a):** {a_range[0]:.0f} - {a_range[1]:.0f} m/s")
        # Input en GPa, conversiÃ³n a Pa para cÃ¡lculos
        E_GPa = st.number_input("MÃ³dulo de elasticidad E (GPa)", min_value=0.01, max_value=300.0, value=float(E_prom/1e9), step=0.01, format="%.2f")
        E = E_GPa * 1e9
        a = st.number_input("Celeridad a (m/s)", min_value=100.0, max_value=2000.0, value=float(a_prom), step=1.0, key="a_material")
        densidad = st.number_input("Densidad del agua (kg/mÂ³)", min_value=800.0, max_value=1200.0, value=float(densidad_calculada), step=1.0)
        D = st.number_input("DiÃ¡metro de la tuberÃ­a (mm)", min_value=10.0, max_value=2000.0, value=100.0, step=1.0) / 1000
        K = st.number_input("MÃ³dulo de elasticidad del fluido K (N/mÂ²)", min_value=1e8, max_value=3e9, value=2.2e9, step=1e7, format="%.0f")
        
        # CÃ¡lculo de delta
        denominador = K - densidad * a**2
        if denominador <= 0:
            st.error("Error: El denominador (K - ÏaÂ²) debe ser positivo. Revise los valores de K, densidad y celeridad.")
            delta_calc = None
            delta_mm = None
        else:
            delta_calc = (K * D * densidad * a**2) / (E * denominador)
            delta_mm = delta_calc * 1000
            # --- NUEVO: Mostrar datos PEAD si corresponde ---
            if material == "Polietileno alta densidad":
                # Tabla extendida PEAD: cada fila es un diÃ¡metro nominal, con sus espesores, series, SDR, diÃ¡metro interno y presiÃ³n
                pead_tabla = [
                    {"DN": 20, "series": ["S5", "S4"], "SDR": [11, 9], "esp": [2.0, 2.3], "dint": [16.0, 15.4], "pres": [1.6, 2.0]},
                    {"DN": 25, "series": ["S5", "S4"], "SDR": [11, 9], "esp": [2.3, 2.8], "dint": [20.4, 19.4], "pres": [1.6, 2.0]},
                    {"DN": 32, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [2.4, 3.0, 3.6], "dint": [27.2, 26.0, 24.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 40, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [3.0, 3.7, 4.5], "dint": [34.0, 32.6, 31.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 50, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [3.7, 4.6, 5.6], "dint": [42.6, 40.8, 38.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 63, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [4.7, 5.8, 7.1], "dint": [53.6, 51.4, 48.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 75, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [5.6, 6.8, 8.4], "dint": [63.8, 61.4, 58.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 90, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [6.7, 8.2, 10.1], "dint": [76.6, 73.6, 69.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 110, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [8.1, 10.0, 12.3], "dint": [93.8, 90.0, 85.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 125, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [9.2, 11.4, 14.0], "dint": [106.6, 102.2, 97.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 140, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [10.3, 12.7, 15.7], "dint": [119.4, 114.6, 108.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 160, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [11.8, 14.6, 18.0], "dint": [136.4, 130.8, 124.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 180, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [13.3, 16.4, 20.1], "dint": [153.4, 147.2, 139.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 200, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [14.7, 18.2, 22.7], "dint": [167.6, 163.6, 154.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 225, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [16.6, 20.5, 25.6], "dint": [191.8, 184.0, 173.8], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 250, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [18.4, 22.7, 28.4], "dint": [213.2, 204.6, 193.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 280, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [20.6, 25.4, 31.8], "dint": [238.8, 229.2, 217.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 315, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [23.2, 28.6, 35.7], "dint": [268.6, 257.8, 243.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 355, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [26.1, 32.2, 40.2], "dint": [302.8, 290.6, 274.6], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 400, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [29.2, 36.3, 45.4], "dint": [341.6, 327.4, 309.2], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 450, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [32.8, 41.0, 51.0], "dint": [384.4, 368.2, 349.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 500, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [36.3, 45.4, 56.8], "dint": [427.4, 409.2, 386.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 560, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [40.7, 51.0, 63.5], "dint": [479.6, 458.4, 433.0], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 630, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [46.3, 57.2, 71.0], "dint": [555.2, 514.6, 489.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 710, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [52.2, 64.7, 80.3], "dint": [622.2, 580.6, 549.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 800, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [58.1, 72.7, 90.8], "dint": [705.2, 654.8, 618.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 900, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [65.1, 81.7, 102.3], "dint": [793.4, 738.8, 695.4], "pres": [1.0, 1.6, 2.0]},
                    {"DN": 1000, "series": ["S8", "S5", "S4"], "SDR": [17, 11, 9], "esp": [73.5, 90.8, 115.0], "dint": [881.4, 818.4, 770.0], "pres": [1.0, 1.6, 2.0]},
                ]
                DN_input = int(round(D*1000))
                DN_candidatos = [fila["DN"] for fila in pead_tabla]
                DN_cercano = min(DN_candidatos, key=lambda x: abs(x - DN_input))
                fila = next(f for f in pead_tabla if f["DN"] == DN_cercano)
                if delta_mm is not None:
                    # Buscar el primer espesor nominal igual o superior al calculado
                    idx = next((i for i, e in enumerate(fila["esp"]) if e >= delta_mm), len(fila["esp"]) - 1)
                    serie = fila["series"][idx]
                    SDR = fila["SDR"][idx]
                    esp_nom = fila["esp"][idx]
                    dint = fila["dint"][idx]
                    pres = fila["pres"][idx]
                    st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm**\n\n**PEAD:**\n- DiÃ¡metro nominal: {DN_cercano} mm\n- DiÃ¡metro interior: {dint:.1f} mm\n- Serie: {serie}\n- SDR: {SDR}\n- Espesor nominal: {esp_nom:.2f} mm\n- PresiÃ³n de trabajo: {pres} MPa")
                else:
                    st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm**\n\n**PEAD:**\n- DiÃ¡metro nominal: {DN_cercano} mm\n- (No se pudo determinar serie ni presiÃ³n de trabajo)")
            else:
                st.success(f"**Celeridad (a): {a:.2f} m/s**\n\n**Espesor calculado (Î´): {delta_mm:.2f} mm")
            if delta_mm > 50:
                st.warning("El espesor calculado es muy grande. Revise los parÃ¡metros o considere que para materiales plÃ¡sticos este cÃ¡lculo puede no ser aplicable para diseÃ±o real.")
        # Mostrar resultados de celeridad y espesor
        st.info(f"**Resultados:**\n- Celeridad (a): {a:.2f} m/s\n- Espesor calculado (Î´): {delta_mm:.2f} mm")
        
        st.markdown("---")
        st.markdown("### Datos de grÃ¡fico")
        espesor_inicial = st.number_input("Espesor inicial (mm)", min_value=0.0, max_value=50.0, value=0.1, step=0.1)
        espesor_final = st.number_input("Espesor final (mm)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        paso_espesor = st.number_input("Paso de espesor (mm)", min_value=0.1, max_value=5.0, value=0.1, step=0.1)
    with col2:
        st.markdown("#### GrÃ¡fico de Celeridad vs Espesor de TuberÃ­a")
        st.markdown("Este grÃ¡fico muestra cÃ³mo varÃ­a la celeridad en funciÃ³n del espesor de la tuberÃ­a para diferentes materiales.")
        
        # Generar datos para el grÃ¡fico
        espesores_mm = np.arange(espesor_inicial, espesor_final + paso_espesor, paso_espesor)
        espesores_m = espesores_mm / 1000
        
        # Calcular celeridad para cada material
        fig_celeridad = go.Figure()
        
        for mat_name, mat_props in materiales.items():
            E_mat = sum(mat_props["E"]) / 2  # Usar valor promedio
            celeridades = []
            
            for delta_m in espesores_m:
                if delta_m > 0:  # Evitar divisiÃ³n por cero
                    # FÃ³rmula de celeridad: a = sqrt(K/Ï / (1 + K*D/(E*Î´)))
                    celeridad = np.sqrt((K / densidad) / (1 + (K * D) / (E_mat * delta_m)))
                    celeridades.append(celeridad)
                else:
                    celeridades.append(np.sqrt(K / densidad))  # Celeridad sin tuberÃ­a
            
            # Color segÃºn el material
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color_idx = list(materiales.keys()).index(mat_name) % len(colors)
            
            fig_celeridad.add_trace(go.Scatter(
                x=espesores_mm,
                y=celeridades,
                mode='lines',
                name=mat_name,
                line=dict(color=colors[color_idx], width=2),
                hovertemplate='<b>%{fullData.name}: %{y:.0f} m/s</b><extra></extra>',
                hoverlabel=dict(bgcolor=colors[color_idx], font=dict(color='white', family='Arial Black'))
            ))
        
        # Agregar punto del material seleccionado
        if delta_mm is not None and delta_mm > 0:
            E_selected = E
            celeridad_selected = np.sqrt((K / densidad) / (1 + (K * D) / (E_selected * delta_calc)))
            fig_celeridad.add_trace(go.Scatter(
                x=[delta_mm],
                y=[celeridad_selected],
                mode='markers',
                name=f'{material} (calculado)',
                marker=dict(color='red', size=10, symbol='star'),
                hovertemplate='<b>%{fullData.name}: %{y:.0f} m/s</b><extra></extra>',
                hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
            ))
        
        fig_celeridad.update_layout(
            title="Celeridad vs Espesor de TuberÃ­a",
            xaxis=dict(
                title="Espesor de tuberÃ­a (mm)",
                showgrid=True,
                gridcolor='#cccccc',
                gridwidth=0.7,
                zeroline=False,
                showspikes=True,
                spikemode='across',
                spikedash='solid',
                spikecolor='#555',
                spikethickness=1,
                showline=True,
                hoverformat='.1f'
            ),
            yaxis=dict(
                title="Celeridad (m/s)",
                showgrid=True,
                gridcolor='#cccccc',
                gridwidth=0.7,
                zeroline=False,
                showspikes=False,
                showline=True,
                hoverformat='.0f'
            ),
            hovermode="x",
            legend=dict(orientation='h', yanchor='bottom', y=-0.6, xanchor='center', x=0.5)
        )
        
        st.plotly_chart(fig_celeridad, use_container_width=True)

        # --- GrÃ¡fico Conceptual de AceleraciÃ³n vs Ãndice de CavitaciÃ³n (Ïƒ) ---
        st.markdown('#### Ãndice de CavitaciÃ³n vs AceleraciÃ³n (grÃ¡fico conceptual)')
        import plotly.graph_objects as go
        # Curva conceptual (valores aproximados para ilustrar el comportamiento)
        sigma_concept = [1, 1.2, 1.5, 1.7, 2, 4, 8, 20]
        aceleracion_concept = [a*1.15, a*1.2, a*1.1, a*0.9, a*0.7, a*0.5, a*0.3, a*0.2]
        fig_sigma = go.Figure()
