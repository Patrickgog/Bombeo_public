import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from sklearn.linear_model import LinearRegression
import uuid

# --- Funciones Auxiliares ---
# Función de presión de vapor usando Antoine ---
def get_vapor_pressure_mca(temperature_celsius):
    # Antoine para agua: 1°C a 100°C
    # log10(P_mmHg) = A - B / (C + T)
    A = 8.07131
    B = 1730.63
    C = 233.426
    T = temperature_celsius
    P_mmHg = 10 ** (A - (B / (C + T)))
    # Convertir mmHg a mca (1 mmHg = 0.0136 mca)
    P_mca = P_mmHg * 0.0136
    return P_mca


# Función para calcular pérdidas por fricción (ejemplo con Hazen-Williams)
# Para una app real, se considerarían otras fórmulas y tipos de tubería.
def calculate_friction_losses_hw(flow_rate_m3s, diameter_m, length_m, C_hw):
    if diameter_m == 0: return float('inf')
    Q_lps = flow_rate_m3s * 1000 # Convertir m3/s a L/s
    if Q_lps == 0: return 0
    # Fórmula de Hazen-Williams (válida solo para agua a 15°C, pero útil para demostración)
    # Hf = (10.67 * L * Q^1.852) / (C^1.852 * D^4.87)
    hf = (10.67 * length_m * (Q_lps / 1000)**1.852) / ((C_hw)**1.852 * diameter_m**4.87)
    return hf

# Función para calcular pérdidas locales (ejemplo con coeficiente K)
def calculate_local_losses(flow_rate_m3s, diameter_m, k_value):
    if diameter_m == 0: return float('inf')
    area = np.pi * (diameter_m / 2)**2
    if area == 0: return float('inf')
    velocity = flow_rate_m3s / area if area > 0 else 0
    g = 9.81 # m/s^2
    hl = k_value * (velocity**2) / (2 * g)
    return hl

# --- Clases de Cálculo ---

class CavitationCalculator:
    def __init__(self, fluid_temperature_celsius):
        self.fluid_temperature_celsius = fluid_temperature_celsius
        self.Pv = get_vapor_pressure_mca(fluid_temperature_celsius)

    def calculate_cavitation_index(self, P_upstream_mca, P_downstream_mca):
        # Asegurar que P_upstream_mca > P_downstream_mca para evitar división por cero o negativos irracionales
        if P_upstream_mca <= P_downstream_mca:
            st.error("Error: La presión aguas arriba debe ser mayor que la presión aguas abajo para calcular un delta de presión.")
            return None
        
        delta_P = P_upstream_mca - P_downstream_mca
        if delta_P == 0:
            return float('inf') # Sin caída de presión, sin riesgo de cavitación por delta P
        
        numerator = P_downstream_mca - self.Pv
        
        # El índice de cavitación para válvulas y componentes es (P2 - Pv) / (P1 - P2)
        # La interpretación es: si P2 cae cerca de Pv, el numerador se acerca a cero
        # Si la caída de presión (P1-P2) es grande, el denominador es grande
        # Un valor bajo de theta indica riesgo.
        
        theta = numerator / delta_P
        return theta

    def get_cavitation_risk_description(self, theta):
        if theta is None:
            return "No calculado"
        elif theta < 0.5:
            return "🚨 ¡Riesgo CRÍTICO de daño por cavitación! 🚨"
        elif theta < 0.8:
            return "⚠️ Riesgo ALTO de ruido por cavitación. ⚠️"
        else:
            return "✅ Riesgo de cavitación BAJO o NULO."

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
            return "🚨 ¡Riesgo CRÍTICO de cavitación en bomba (NPSH Insuficiente)! 🚨"
        else:
            return "✅ NPSH Suficiente. Riesgo de cavitación en bomba BAJO."

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Calculadora de Índice de Cavitación")

st.title("🔬 Cálculo de Índice de Cavitación")
st.markdown("Herramienta de análisis del riesgo de cavitación en diferentes componentes del sistema de bombeo.")

# --- Barra Lateral para Parámetros Comunes ---
# Tabla de presión de vapor y densidad de agua
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

st.sidebar.header("⚙️ Parámetros Generales")
# Sincronización de temperatura
if 'fluid_temp' not in st.session_state:
    st.session_state['fluid_temp'] = 20
if 'fluid_temp_input' not in st.session_state:
    st.session_state['fluid_temp_input'] = 20

def on_slider_change():
    st.session_state['fluid_temp_input'] = st.session_state['fluid_temp']

def on_input_change():
    st.session_state['fluid_temp'] = st.session_state['fluid_temp_input']

st.sidebar.slider(
    "Temperatura del Fluido (°C)",
    min_value=0, max_value=100, value=st.session_state['fluid_temp'], step=5,
    key='fluid_temp', on_change=on_slider_change
)
st.sidebar.number_input(
    "Temperatura precisa (°C)",
    min_value=0, max_value=100, value=st.session_state['fluid_temp_input'], step=1,
    key='fluid_temp_input', on_change=on_input_change
)
fluid_temp = st.session_state['fluid_temp']
fluid_temp_input = st.session_state['fluid_temp_input']
Pv_calculated = interpola_tabla(fluid_temp_input, tabla_temp, tabla_pv)
densidad_calculada = interpola_tabla(fluid_temp_input, tabla_temp, tabla_dens)
st.sidebar.info(f"Presión de Vapor (Pv) a {fluid_temp_input}°C: **{Pv_calculated:.3f} mca**\nDensidad del agua a {fluid_temp_input}°C: **{densidad_calculada:.1f} kg/m³**")
# Usar fluid_temp_input en los cálculos
cav_calc = CavitationCalculator(fluid_temp_input)
npsh_calc = NPSHCalculator(fluid_temp_input)
# Elimina cualquier reasignación manual de st.session_state['fluid_temp'] después de este bloque.

with st.sidebar.expander("Presión absoluta de vapor de agua y densidad según temperatura"):
    st.markdown('''
| Temperatura (°C) | Densidad (kg/m³) | Presión de vapor H_vap (m) |
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

# --- NUEVA PESTAÑA: Coeficiente de Descarga (C) ---
tab_c, tab1, tab_trans, tab5 = st.tabs([
    "📊 Coeficiente de Descarga (C)",
    "🔧 Válvula",
    "🌊 Flujos Transitorios",
    "📈 Pérdidas por Fricción"
])

with tab_c:
    st.header("Coeficiente de Descarga (C)")
    col1, col2, col3, col4 = st.columns([0.18, 0.42, 0.24, 0.16])
    with col1:
        st.subheader("Cálculo interactivo")
        tipos_valvula = [
            ("Globo", "Lineal"),
            ("Ángulo", "Lineal"),
            ("Compuerta (incl. Y)", "Lineal"),
            ("Obturador Cilíndrico Excéntrico", "Rotativo"),
            ("Mariposa", "Rotativo"),
            ("Bola", "Rotativo"),
            ("Macho", "Rotativo"),
            ("Orificio Ajustable", "Rotativo"),
            ("Flujo Axial", "Rotativo")
        ]
        tipo_valvula = st.selectbox("Tipo de válvula", [v[0] for v in tipos_valvula])
        movimiento_permitido = [v[1] for v in tipos_valvula if v[0] == tipo_valvula][0]
        movimiento = st.selectbox("Movimiento", [movimiento_permitido], disabled=True)
        estado = st.selectbox("Estado de apertura", ["Totalmente abierta", "Parcialmente abierta"])
        valores_C = {
            ("Globo", "Lineal", "Totalmente abierta"): (0.85, 0.95),
            ("Globo", "Lineal", "Parcialmente abierta"): (0.60, 0.85),
            ("Ángulo", "Lineal", "Totalmente abierta"): (0.90, 0.98),
            ("Ángulo", "Lineal", "Parcialmente abierta"): (0.70, 0.90),
            ("Compuerta (incl. Y)", "Lineal", "Totalmente abierta"): (0.95, 1.00),
            ("Compuerta (incl. Y)", "Lineal", "Parcialmente abierta"): (0.75, 0.95),
            ("Obturador Cilíndrico Excéntrico", "Rotativo", "Totalmente abierta"): (0.90, 0.98),
            ("Obturador Cilíndrico Excéntrico", "Rotativo", "Parcialmente abierta"): (0.70, 0.90),
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
        D = st.number_input("Diámetro de la tubería D (mm)", min_value=1.0, max_value=2000.0, value=100.0, step=1.0, key="diametro_coef_desc") / 1000  # convertir a metros
        delta_h = st.number_input("Diferencia de altura Δh (m)", min_value=0.01, max_value=100.0, value=5.0, step=0.01, key="delta_h_coef_desc")
        g = 9.81
        Q = C * ((2 * g * delta_h) ** 0.5) * (np.pi * D ** 2 / 4)
        st.markdown(f"**Caudal calculado Q = {Q:.4f} m³/s ({Q*3600:.2f} m³/h, {Q*1000:.2f} L/s)**")
        st.markdown("---")
        st.subheader("Resultados")
        # Panel desplegable de análisis del coeficiente C
        with st.expander("Análisis del coeficiente C"):
            st.markdown(f"""
**Valores seleccionados:**
- Tipo de válvula: **{tipo_valvula}**
- Movimiento: **{movimiento_permitido}**
- Estado: **{estado}**
- Coeficiente C: **{C:.3f}**
- Diámetro: **{D*1000:.1f} mm**
- Δh: **{delta_h:.2f} m**

**Caudal calculado:**
- Q = **{Q:.4f} m³/s**
- Q = **{Q*3600:.2f} m³/h**
- Q = **{Q*1000:.2f} L/s**

**Fórmula utilizada:**
Q = C × √(2×g×Δh) × (π×D²/4)
            """)
            # Análisis de sensibilidad
            st.markdown("#### Análisis de sensibilidad")
            C_variations = [C*0.8, C*0.9, C, C*1.1, C*1.2]
            Q_variations = [C_var * ((2 * g * delta_h) ** 0.5) * (np.pi * D ** 2 / 4) for C_var in C_variations]
            sens_df = pd.DataFrame({
                'C': C_variations,
                'Q (m³/s)': Q_variations,
                'Q (L/s)': [q*1000 for q in Q_variations],
                'Variación (%)': [(q/Q - 1)*100 for q in Q_variations]
            })
            st.dataframe(sens_df, use_container_width=True)
        # Luego sigue el bloque de datos de gráfico
        st.markdown("### Datos de gráfico")
        diam_ini = st.number_input("Diámetro inicial (mm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.25, key="diam_ini_coef")
        diam_fin = st.number_input("Diámetro final (mm)", min_value=0.0, max_value=2000.0, value=100.0, step=0.25, key="diam_fin_coef")
        diam_paso = st.number_input("Paso de diámetro (mm)", min_value=0.01, max_value=500.0, value=0.25, step=0.01, key="diam_paso_coef")
        altura_max = st.number_input("Altura máxima en gráfico (m)", min_value=1.0, max_value=200.0, value=40.0, step=0.5, key="altura_max_coef")
    with col2:
        st.subheader("Curvas de igual caudal Q")
        st.markdown("#### Diámetro de tubería D (mm)")
        st.markdown("""
**Simbología:**  
- **Q**: Caudal (m³/s)  
- **D**: Diámetro de la tubería (mm)  
- **Δh**: Diferencia de altura (m)  
- **C**: Coeficiente de descarga (adimensional)  
- **g**: Gravedad (9.81 m/s²)
""")
        # Usar los valores de diam_ini, diam_fin, diam_paso para el rango de D
        D_range = np.arange(diam_ini, diam_fin + diam_paso, diam_paso) / 1000  # convertir a metros
        D_range = D_range[D_range > 0]  # Elimina cualquier valor cero
        C_graf = C  # Usar el coeficiente seleccionado
        g = 9.81
        Q_targets = [0.005, 0.01, 0.02, 0.05]  # Caudales objetivo en m³/s (5, 10, 20, 50 L/s)
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Azul, naranja, verde, rojo
        fig = go.Figure()
        tabla_dict = {"D (mm)": D_range*1000}
        for idx, Qg in enumerate(Q_targets):
            delta_h_curve = (Qg / (C_graf * (np.pi * D_range**2 / 4)))**2 / (2 * g)
            fig.add_trace(go.Scatter(
                x=D_range*1000, y=delta_h_curve, mode='lines', name=f'Q={Qg*1000:.0f} L/s',
                hovertemplate='<b>Δh = %{y:.2f} m</b><extra></extra>',
                line=dict(color=colores[idx % len(colores)]),
                hoverlabel=dict(bgcolor=colores[idx % len(colores)], font=dict(color='white', family='Arial Black'))
            ))
            tabla_dict[f"Δh (m) Q={Qg*1000:.0f} L/s"] = delta_h_curve
        # Agregar punto rojo de los inputs
        if D > 0:
            delta_h_point = (Q / (C_graf * (np.pi * D**2 / 4)))**2 / (2 * g)
            fig.add_trace(go.Scatter(
                x=[D*1000], y=[delta_h_point], mode='markers', name='Punto actual',
                marker=dict(color='red', size=10, symbol='diamond'),
                hovertemplate='<b>D = %{x:.1f} mm</b><br><b>Δh = %{y:.2f} m</b><extra></extra>'
            ))
        
        fig.update_layout(
            title='Curvas de igual caudal Q en el plano Diámetro vs Diferencia de altura',
            xaxis_title='Diámetro de tubería D (mm)',
            yaxis_title='Diferencia de altura Δh (m)',
            height=500,
            hovermode="x",  # Hovers independientes y del color de cada línea
            showlegend=False
        )
        fig.update_xaxes(
            range=[diam_ini, diam_fin],
            hoverformat=".2f",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            showgrid=True,
            zeroline=False,
            spikethickness=1,  # Línea vertical más fina
        )
        fig.update_yaxes(range=[0, altura_max])
        st.plotly_chart(fig, use_container_width=True)
        # (Elimina la simbología debajo del gráfico)
        
        # Mostrar tabla de datos
        if len(D_range) > 0:
            df_curvas = pd.DataFrame(tabla_dict)
            # st.markdown("#### Tabla de valores") # Moved to col3
            # st.dataframe(df_curvas, use_container_width=True) # Moved to col3
            
            # Botón para descargar datos
            csv = df_curvas.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="curvas_caudal.csv">📥 Descargar datos CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        st.subheader("Información técnica")
        with st.expander("Valores típicos del coeficiente de descarga (C)"):
            st.markdown("""
| Tipo de Válvula | Movimiento | Estado | C (típico) |
|:---------------:|:----------:|:------:|:-----------:|
| Globo                         | Lineal     | Totalmente abierta    | 0.85 - 0.95|
| Globo                         | Lineal     | Parcialmente abierta  | 0.60 - 0.85|
| Compuerta (incl. Y)           | Lineal     | Totalmente abierta    | 0.95 - 1.00|
| Compuerta (incl. Y)           | Lineal     | Parcialmente abierta  | 0.75 - 0.95|
| Ángulo                         | Lineal     | Totalmente abierta    | 0.90 - 0.98|
| Ángulo                         | Lineal     | Parcialmente abierta  | 0.70 - 0.90|
| Obturador Cilíndrico Excéntrico| Rotativo   | Totalmente abierta    | 0.90 - 0.98|
| Obturador Cilíndrico Excéntrico| Rotativo   | Parcialmente abierta  | 0.70 - 0.90|
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
        with st.expander("Factores que afectan el coeficiente C"):
            st.markdown("""
**Factores principales:**
- **Tipo de válvula**: Cada tipo tiene características de flujo diferentes
- **Estado de apertura**: Válvulas parcialmente abiertas tienen C menor
- **Reynolds**: A altos números de Reynolds, C tiende a ser constante
- **Rugosidad**: Superficies rugosas pueden reducir C
- **Cavitación**: La presencia de cavitación reduce significativamente C
- **Vibraciones**: Pueden afectar el rendimiento y C

**Recomendaciones:**
- Usar valores conservadores para diseño
- Considerar degradación por uso
- Verificar en condiciones reales de operación
""")
        # Tabla de valores debajo de la información técnica
        if len(D_range) > 0:
            st.markdown("#### Tabla de valores")
            st.dataframe(df_curvas, use_container_width=True)
            # Botón para descargar datos
            csv = df_curvas.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="curvas_caudal.csv">📥 Descargar datos CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    with col4:
        st.image(
            "https://i.imgur.com/7yBNPez.png",
            caption="Referencia visual",
            use_container_width=True
        )

# --- Pestaña 1: Válvula ---
# --------------------------------------------------
# Código correspondiente a la pestaña de Válvula
# --------------------------------------------------
with tab1:
    st.header("Análisis de Cavitación en Válvula")
    col_input, col_graph, col_table, col_extra = st.columns([0.22, 0.4, 0.28, 0.1])

    # Definir P2_range y theta_values antes de los bloques de columnas para uso global en la pestaña
    P2_min_default = 0.0
    P2_max_default = 8.0
    P2_step_default = 0.25
    # Se usan los valores por defecto, pero luego se actualizan con los inputs
    P2_min = P2_min_default
    P2_max = P2_max_default
    P2_step = P2_step_default
    with col_input:
        st.subheader("Datos de la Válvula")
        P1_valve = st.number_input("Presión de Entrada (P1, mca)", min_value=0.0, value=10.0, step=0.1, key="P1_valve")
        P2_valve = st.number_input("Presión de Salida (P2, mca)", min_value=0.0, value=5.0, step=0.1, key="P2_valve")
        st.markdown('---')
        st.subheader("Datos Gráfico")
        P2_min = st.number_input("P2 mínimo (mca)", min_value=0.0, value=P2_min_default, step=0.1, key="P2_min_valve")
        P2_max = st.number_input("P2 máximo (mca)", min_value=0.0, value=P2_max_default, step=0.1, key="P2_max_valve")
        P2_step = st.number_input("Paso de P2 (mca)", min_value=0.001, max_value=10.0, value=P2_step_default, step=0.1, key="P2_step_valve")
        st.subheader("Resultados")
        theta_valve = cav_calc.calculate_cavitation_index(P1_valve, P2_valve)
        if theta_valve is not None:
            st.metric(label="Índice de Cavitación (θ)", value=f"{theta_valve:.3f}")
            st.write(cav_calc.get_cavitation_risk_description(theta_valve))
        else:
            st.write("Ingrese valores válidos para calcular el índice de cavitación.")
        st.info("""
**Interpretación:**
- Si θ < 0.5: Riesgo crítico de daño por cavitación.
- Si 0.5 ≤ θ < 0.8: Riesgo alto de ruido por cavitación.
- Si θ ≥ 0.8: Riesgo bajo o nulo de cavitación.
""")
        # Explicación Kv ...
    # Calcular P2_range y theta_values después de obtener los inputs
    P2_range = np.arange(P2_min, P2_max+P2_step, P2_step)
    theta_values = []
    for p2 in P2_range:
        if P1_valve > p2:
            theta = cav_calc.calculate_cavitation_index(P1_valve, p2)
            theta_values.append(theta if theta is not None else np.nan)
        else:
            theta_values.append(np.nan)

    with col_graph:
        st.subheader("Gráfica: Índice de Cavitación vs. Presión de Salida")
        
        # Crear máscaras para los rangos de riesgo
        theta_array = np.array(theta_values)
        mask_rojo = theta_array < 0.5
        mask_amarillo = (theta_array >= 0.5) & (theta_array < 0.8)
        mask_verde = theta_array >= 0.8
        
        # Crear arrays para cada rango de riesgo
        y_rojo = np.where(mask_rojo, theta_array, np.nan)
        y_amarillo = np.where(mask_amarillo, theta_array, np.nan)
        y_verde = np.where(mask_verde, theta_array, np.nan)
        
        fig_valve = go.Figure()
        
        # Agregar áreas pintadas para cada rango de riesgo
        # Área roja (riesgo crítico)
        fig_valve.add_trace(go.Scatter(
            x=P2_range, y=y_rojo, mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            name='Riesgo Crítico (θ < 0.5)',
            showlegend=False,
                                       hovertemplate='<b>P2 = %{x:.2f} mca<br>θ = %{y:.2f}</b><extra></extra>',
            hoverlabel=dict(bgcolor='rgba(255,0,0,0.8)', font=dict(color='white', family='Arial Black'))
        ))
        
        # Área amarilla (riesgo alto)
        fig_valve.add_trace(go.Scatter(
            x=P2_range, y=y_amarillo, mode='lines',
            line=dict(color='rgba(255,255,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255,255,0,0.2)',
            name='Riesgo Alto (0.5 ≤ θ < 0.8)',
            showlegend=False,
            hovertemplate='<b>P2 = %{x:.2f} mca<br>θ = %{y:.2f}</b><extra></extra>',
            hoverlabel=dict(bgcolor='rgba(255,255,0,0.8)', font=dict(color='black', family='Arial Black'))
        ))
        
        # Área verde (riesgo bajo)
        fig_valve.add_trace(go.Scatter(
            x=P2_range, y=y_verde, mode='lines',
            line=dict(color='rgba(0,200,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(0,200,0,0.2)',
            name='Riesgo Bajo (θ ≥ 0.8)',
            showlegend=False,
            hovertemplate='<b>P2 = %{x:.2f} mca<br>θ = %{y:.2f}</b><extra></extra>',
            hoverlabel=dict(bgcolor='rgba(0,200,0,0.8)', font=dict(color='white', family='Arial Black'))
        ))
        
        # Agregar la curva principal encima de las áreas
        fig_valve.add_trace(go.Scatter(
            x=P2_range, y=theta_values, mode='lines',
            line=dict(color='#2ca02c', width=3),
            name='Índice de Cavitación',
            hovertemplate='<b>P2 = %{x:.2f} mca<br>θ = %{y:.2f}</b><extra></extra>',
            hoverlabel=dict(bgcolor='#2ca02c', font=dict(color='white', family='Arial Black'))
        ))
        
        # Agregar punto del valor actual calculado
        theta_actual = cav_calc.calculate_cavitation_index(P1_valve, P2_valve)
        if theta_actual is not None:
            fig_valve.add_trace(go.Scatter(
                x=[P2_valve], y=[theta_actual], mode='markers',
                marker=dict(symbol='circle', size=12, color='blue', line=dict(color='white', width=2)),
                name='Valor Actual',
                                           hovertemplate='<b>P2 = %{x:.2f} mca<br>θ = %{y:.2f}</b><extra></extra>',
                hoverlabel=dict(bgcolor='blue', font=dict(color='white', family='Arial Black'))
            ))
        
        # Agregar líneas horizontales de límites
        fig_valve.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=2, 
                           annotation_text="Límite Daño (0.5)", secondary_y=False)
        fig_valve.add_hline(y=0.8, line_dash="dash", line_color="orange", line_width=2, 
                           annotation_text="Límite Ruido (0.8)", secondary_y=False)
        fig_valve.update_layout(
            title="Índice de Cavitación vs. Presión de Salida en Válvula",
            xaxis=dict(
                title="Presión de Salida (mca)",
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
                title="Índice de Cavitación (θ)",
                tickformat='.2f',
            ),
            hovermode="x",
            hoverlabel=dict(bgcolor='#555', font=dict(color='white', family='Arial Black')),
            showlegend=False
        )
        st.plotly_chart(fig_valve, use_container_width=True)
        # Simbología debajo del gráfico
        st.markdown("""
        <div style="text-align: center;">
            <b>Simbología:</b><br><br>
            <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin: 10px 0; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 30px; height: 3px; background-color: #2ca02c; border-radius: 2px;"></div>
                    <span style="font-size: 0.9em;">Índice de cavitación</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(255,0,0,0.2); border: 1px solid rgba(255,0,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo crítico (θ < 0.5)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(255,255,0,0.2); border: 1px solid rgba(255,255,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo alto (0.5 ≤ θ < 0.8)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(0,200,0,0.2); border: 1px solid rgba(0,200,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo bajo (θ ≥ 0.8)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background-color: blue; border-radius: 50%; border: 2px solid white;"></div>
                    <span style="font-size: 0.9em;">Valor calculado actual</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_table:
        st.markdown('<h4 style="text-align:center;">Tabla de Índice de Cavitación</h4>', unsafe_allow_html=True)
        df_valve = pd.DataFrame({
            'P2 (mca)': np.round(P2_range, 2),
            'θ': np.round(theta_values, 2)
        })
        st.dataframe(df_valve, use_container_width=True)

        # Panel de fórmulas
        with st.expander("Ver fórmulas utilizadas"): 
            st.markdown(r"""
**Índice de Cavitación:**

$$
\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}
$$

**Donde:**
- **P1**: Presión de entrada (mca)
- **P2**: Presión de salida (mca)
- **Pvapor**: Presión de vapor del fluido (mca)

Un valor bajo de **θ** indica mayor riesgo de cavitación.
""")
        # Panel independiente de explicación de presiones
        with st.expander("¿Qué significan la presión de entrada y salida?"):
            st.markdown("""
- **Presión de entrada (P₁):** Es la presión del fluido justo antes de entrar a la válvula. Representa la energía por unidad de peso del fluido en el punto aguas arriba de la válvula, normalmente medida en metros de columna de agua (mca). Esta presión depende de la altura, el caudal y las condiciones del sistema antes de la válvula.

- **Presión de salida (P₂):** Es la presión del fluido justo después de pasar por la válvula. Representa la energía por unidad de peso del fluido en el punto aguas abajo de la válvula, también en metros de columna de agua (mca). Esta presión suele ser menor que la de entrada debido a la pérdida de energía (caída de presión) que ocurre al atravesar la válvula.

**En resumen:**
- **P₁** indica cuánta presión tiene el fluido antes de la válvula.
- **P₂** indica cuánta presión queda después de la válvula.
""")
        with st.expander("Criterios de cavitación según sigma (σ)"):
            st.markdown("""
**Criterios de cavitación según el índice sigma (σ):**

| Rango de σ         | Interpretación                                      |
|:------------------:|:---------------------------------------------------|
| σ ≥ 2.0            | <span style='color:green'><b>No hay cavitación</b></span> |
| 1.7 < σ < 2.0      | <span style='color:#7ca300'><b>Protección suficiente con materiales endurecidos</b></span> |
| 1.5 < σ < 1.7      | <span style='color:orange'><b>Algo de cavitación, puede funcionar un solo escalón</b></span> |
| 1.0 < σ < 1.5      | <span style='color:#ff6600'><b>Potencial de cavitación severa, se requiere reducción en varias etapas</b></span> |
| σ < 1.0            | <span style='color:red'><b>Flashing (vaporización instantánea)</b></span> |

> **σ = (P₁ - P_v) / (P₁ - P₂)**

- **SUPER CAVITACIÓN:** σ bajo, aceleración alta, daño severo.
- **CAVITACIÓN PLENA:** σ intermedio, daño considerable.
- **CAVITACIÓN INCIPIENTE:** σ cerca de 1.5-1.7, inicio de daño.
- **SUBCRÍTICO:** σ alto, sin daño.

Estos criterios ayudan a seleccionar el diseño y materiales adecuados para evitar daños por cavitación en válvulas.
""", unsafe_allow_html=True)

    # --- SEPARADOR A TODO EL ANCHO ---
    st.markdown("---")
    # --- NUEVA FILA DE COLUMNAS PARA Kv SOLO EN ESTA PESTAÑA ---
    col_kv_exp, col_kv_graf, col_kv_tabla, col_kv_extra = st.columns([0.22, 0.4, 0.28, 0.1])
    with col_kv_exp:
        st.subheader("Coeficiente de caudal (Kv)")
        st.markdown("""
Las válvulas de control son conceptualmente orificios de área variable. Se las puede considerar simplemente como una restricción que cambia su tamaño de acuerdo a un pedido por parte del actuador.

El coeficiente de caudal es la relación de diferencia de altura (Δh) o presión (ΔP) entre la entrada y salida de la válvula con el caudal (Q).
""")
        st.latex(r"K_v = Q \sqrt{\frac{\rho}{1000\,\Delta p}}")
        st.markdown("""
Donde:
- $K_v$: Coeficiente de flujo (m³/h)
- $Q$: Caudal volumétrico (m³/h)
- $\rho$: Densidad (kg/m³)
- $\Delta p$: Diferencia de presión (bar)
- $P_1$: Presión de entrada (bar)
- $P_2$: Presión de salida (bar)
""")
        mca_a_bar = 0.0980665
        P1_bar = P1_valve * mca_a_bar
        P2_bar = P2_valve * mca_a_bar
        delta_p_bar = np.abs(P1_bar - P2_bar)
        densidad = densidad_calculada
        st.markdown(f"**Presión de entrada (P1):** {P1_bar:.2f} bar  ")
        st.markdown(f"**Presión de salida (P2):** {P2_bar:.2f} bar  ")
        st.markdown(f"**Diferencia de presión (Δp):** {delta_p_bar:.2f} bar  ")
        st.markdown(f"**Densidad (ρ):** {densidad:.1f} kg/m³  ")
    with col_kv_graf:
        st.subheader("Índice de caudal Kv")
        # Gráfico Δp vs Kv con Q fijo
        Q_m3h = 10  # Caudal fijo típico para la curva
        Kv_range = np.linspace(1, 15, 30)
        delta_p = (densidad / 1000) * (Q_m3h / Kv_range) ** 2

        fig_kv = go.Figure()
        fig_kv.add_trace(go.Scatter(
            x=Kv_range, y=delta_p, mode='lines+markers',
            line=dict(color='#1f77b4'),
            marker=dict(symbol='circle', size=6),
            hovertemplate='<b>Kv = %{x:.1f}</b><br><b>Δp = %{y:.2f} bar</b><extra></extra>',
            hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))
        ))
        
        fig_kv.update_layout(
            title="Diferencia de presión vs Coeficiente Kv",
            xaxis_title="Coeficiente Kv (m³/h)",
            yaxis_title="Diferencia de presión Δp (bar)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_kv, use_container_width=True)

        # Nuevo gráfico: Caudal vs Pérdida de carga para Kv bajo y alto
        st.subheader("Caudal vs Pérdida de carga (Kv alto y bajo)")
        delta_p_range = np.linspace(0.01, 4.5, 30)
        Kv_bajo = 5
        Kv_alto = 15
        Q_bajo = Kv_bajo * np.sqrt(1000 * delta_p_range / densidad)
        Q_alto = Kv_alto * np.sqrt(1000 * delta_p_range / densidad)

        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=delta_p_range, y=Q_bajo, mode='lines+markers',
            name='Kv bajo', line=dict(color='orange'), marker=dict(symbol='square'),
            hovertemplate='<b>Pérdida de carga = %{x:.2f} bar</b><br><b>Q = %{y:.2f} m³/h</b><extra></extra>',
            hoverlabel=dict(bgcolor='orange', font=dict(color='white', family='Arial Black'))
        ))
        fig_q.add_trace(go.Scatter(
            x=delta_p_range, y=Q_alto, mode='lines+markers',
            name='Kv alto', line=dict(color='blue'), marker=dict(symbol='diamond'),
            hovertemplate='<b>Pérdida de carga = %{x:.2f} bar</b><br><b>Q = %{y:.2f} m³/h</b><extra></extra>',
            hoverlabel=dict(bgcolor='blue', font=dict(color='white', family='Arial Black'))
        ))
        fig_q.update_layout(
            xaxis_title="Pérdida de carga (bar)",
            yaxis_title="Q (m³/h)",
            height=270,
            hovermode="x",
            showlegend=False
        )
        fig_q.update_yaxes(range=[0, max(Q_alto)*1.15])
        fig_q.update_xaxes(hoverformat=".2f", showspikes=True, spikemode="across", spikesnap="cursor", showline=True, showgrid=True, zeroline=False, spikethickness=1)
        st.plotly_chart(fig_q, use_container_width=True)

    with col_kv_tabla:
        st.markdown('<h4 style="text-align:center;">Tabla Kv vs Δp</h4>', unsafe_allow_html=True)
        df_kv = pd.DataFrame({
            'Kv (m³/h)': np.round(Kv_range, 1),
            'Δp (bar)': np.round(delta_p, 3)
        })
        st.dataframe(df_kv, use_container_width=True)
        # Nueva tabla: Caudal vs Pérdida de carga para Kv bajo y alto
        st.markdown("#### Tabla Caudal vs Pérdida de carga (Kv alto y bajo)")
        df_q = pd.DataFrame({
            "Pérdida de carga (bar)": np.round(delta_p_range, 2),
            "Q (m³/h) Kv bajo": np.round(Q_bajo, 2),
            "Q (m³/h) Kv alto": np.round(Q_alto, 2)
        })
        st.dataframe(df_q, use_container_width=True)

# --- Pestaña 2: Flujos Transitorios ---
with tab_trans:
    st.header("🌊 Análisis de Flujos Transitorios")
    st.markdown("""
    **Flujos transitorios** son cambios temporales en las condiciones de flujo que pueden causar:
    - Golpes de ariete
    - Cavitación transitoria
    - Vibraciones en el sistema
    - Daños a equipos
    """)
    
    col1, col2, col3, col4 = st.columns([0.22, 0.38, 0.28, 0.12])
    
    with col1:
        st.markdown("## Datos de Entrada")
        # Tabla de materiales y rangos
        tabla_materiales = [
            {"Material": "Acero", "E": (200.0, 212.0), "a": (1000, 1250)},
            {"Material": "Fibro Cemento", "E": (23.5, 23.5), "a": (900, 1200)},
            {"Material": "Concreto", "E": (39.0, 39.0), "a": (1050, 1150)},
            {"Material": "Hierro Dúctil", "E": (166.0, 166.0), "a": (1000, 1350)},
            {"Material": "PEAD", "E": (0.59, 1.67), "a": (230, 430)},
            {"Material": "PVC", "E": (2.40, 2.75), "a": (300, 500)}
        ]
        materiales_lista = [m["Material"] for m in tabla_materiales]
        material_sel = st.selectbox("Material de la tubería", materiales_lista, key="mat_trans")
        mat_info = next(m for m in tabla_materiales if m["Material"] == material_sel)
        # Mostrar información de E y a justo debajo del combobox
        st.info(f"Rango de módulo de elasticidad (E): {mat_info['E'][0]:.2f} - {mat_info['E'][1]:.2f} GPa\nRango de celeridad (a): {mat_info['a'][0]:.0f} - {mat_info['a'][1]:.0f} m/s")
        E_default = (mat_info['E'][0] + mat_info['E'][1]) / 2
        # Estado inicial para E
        if 'E_trans' not in st.session_state:
            st.session_state['E_trans'] = E_default
        # Si el material cambió, actualiza E
        if 'last_material' not in st.session_state or st.session_state['last_material'] != material_sel:
            st.session_state['E_trans'] = E_default
            st.session_state['last_material'] = material_sel
        # Input de E (GPa)
        E_input = st.number_input("Módulo de elasticidad E (GPa)", min_value=0.01, value=st.session_state['E_trans'], step=0.01, key="E_trans")
        # Input editable de celeridad (a)
        a_media = (mat_info['a'][0] + mat_info['a'][1]) / 2
        if 'a_media_val' not in st.session_state:
            st.session_state['a_media_val'] = a_media
        # Si el material cambió, actualiza a_media_val
        if 'last_material_a' not in st.session_state or st.session_state['last_material_a'] != material_sel:
            st.session_state['a_media_val'] = a_media
            st.session_state['last_material_a'] = material_sel
        a_input = st.number_input("Celeridad (a) (m/s)", min_value=1.0, value=st.session_state['a_media_val'], step=1.0, key="a_media_val")
        # Densidad del agua
        st.markdown(f"**Densidad del agua (kg/m³):** {densidad_calculada:.1f}")
        # Diámetro de la tubería (por defecto 100 mm)
        D_input = st.number_input("Diámetro de la tubería (mm)", min_value=1.0, value=100.0, step=1.0, key="D_trans_input")
        # Módulo de elasticidad del fluido (agua)
        K_default = 2.2  # GPa
        K_input = st.number_input("Módulo de elasticidad del fluido (GPa)", min_value=0.1, value=K_default, step=0.01, key="K_trans")
        # Cálculo de espesor usando la celeridad ingresada
        D_m = D_input / 1000.0
        E_Pa = E_input * 1e9
        K_Pa = K_input * 1e9
        a_calc = a_input
        K_rho_a2 = K_Pa / (densidad_calculada * a_calc**2)
        denom = E_Pa * (K_rho_a2 - 1)
        if denom != 0:
            espesor_calc = (K_Pa * D_m) / denom
        else:
            espesor_calc = np.nan
        espesor_calc_mm = espesor_calc * 1000 if espesor_calc is not None else np.nan
        # Panel verde con resultados
        st.success(f"**Espesor calculado (δ):** {espesor_calc_mm:.2f} mm")
        
        # Calcular celeridad para el punto específico usando exactamente los mismos parámetros que se usaron para calcular el espesor
        if espesor_calc is not None and not np.isnan(espesor_calc):
            e_calc = espesor_calc  # ya está en metros
            # Usar exactamente los mismos parámetros que se usaron para calcular el espesor
            K = K_Pa  # Módulo de elasticidad del agua (N/m2) - el mismo que se usó para calcular espesor
            densidad_agua = densidad_calculada  # La misma densidad que se usó para calcular espesor
            D_para_calculo = D_m  # El mismo diámetro que se usó para calcular espesor
            E_material = E_Pa  # El mismo módulo de elasticidad que se usó para calcular espesor
            
            if e_calc <= 0 or E_material <= 0:
                a_calc_point = np.nan
            else:
                num = 1.0 / K + D_para_calculo / (E_material * e_calc)
                if num <= 0:
                    a_calc_point = np.nan
                else:
                    a_calc_point = np.sqrt(1.0 / (densidad_agua * num))
        else:
            a_calc_point = np.nan
    
    # --- SEPARADOR QUE ATRAVIESA TODO EL LAYOUT ---
    st.markdown("---")
    
    with col1:
        st.markdown("---")
        st.subheader("Datos de Gráfico")

        # Panel desplegable para a vs delta
        with st.expander("Datos de a vs δ"):
        e_ini = st.number_input("Espesor inicial (mm)", min_value=0.1, max_value=100.0, value=0.1, step=0.1, key="e_ini")
        e_fin = st.number_input("Espesor final (mm)", min_value=0.1, max_value=100.0, value=6.0, step=0.1, key="e_fin")
        e_paso = st.number_input("Paso (mm)", min_value=0.01, max_value=10.0, value=0.1, step=0.01, key="e_paso")
        e_range = np.arange(e_ini, e_fin + e_paso, e_paso)
        
        # Panel desplegable para Tc vs L
        with st.expander("Datos de Tc vs L"):
            L_ini = st.number_input("Longitud inicial (m)", min_value=0.01, value=5.0, step=1.0, key="L_ini_tc")
            L_fin = st.number_input("Longitud final (m)", min_value=0.01, value=2000.0, step=1.0, key="L_fin_tc")
            L_paso = st.number_input("Paso de longitud (m)", min_value=0.01, value=25.0, step=0.01, key="L_paso_tc")
            if L_fin <= L_ini:
                st.warning("La longitud final debe ser mayor que la inicial.")
                L_range_tc = np.array([])
            else:
                L_range_tc = np.arange(L_ini, L_fin + L_paso, L_paso)

        # Panel desplegable para Lc vs T
        with st.expander("Datos Lc vs T"):
            T_ini_lc = st.number_input("Tiempo inicial (s)", min_value=0.01, value=0.01, step=0.01, key="T_ini_lc")
            T_fin_lc = st.number_input("Tiempo final (s)", min_value=0.01, value=10.0, step=0.01, key="T_fin_lc")
            T_paso_lc = st.number_input("Paso (s)", min_value=0.001, value=0.1, step=0.001, key="T_paso_lc")
            if T_fin_lc <= T_ini_lc:
                st.warning("El tiempo final debe ser mayor que el inicial.")
                T_range_lc = np.array([])
            else:
                T_range_lc = np.arange(T_ini_lc, T_fin_lc + T_paso_lc, T_paso_lc)
    
    with col1:
        st.markdown("---")
        # Todo lo que quede debajo de este separador es Bloque2
        st.subheader("Tiempo de Cierre Crítico")  # Bloque 2
        # Inputs en la columna 1
        L = st.number_input("Longitud de la tubería (m)", min_value=0.01, value=1000.00, step=0.01, format="%.2f", key="L_trans")
        # Inputs de diámetro y espesor, por defecto los valores de arriba
        D_tc = st.number_input("Diámetro de la tubería (mm)", min_value=1.0, value=D_input, step=1.0, key="D_tc_input")
        espesor_tc = st.number_input("Espesor de la tubería (mm)", min_value=0.01, value=espesor_calc_mm if espesor_calc_mm is not None and not np.isnan(espesor_calc_mm) else 1.0, step=0.01, key="espesor_tc_input")
        # Conversión a metros
        D_tc_m = D_tc / 1000.0
        espesor_tc_m = espesor_tc / 1000.0
        # Cálculo de celeridad
        # Usar los mismos valores de K y E que arriba
        E_Pa = E_input * 1e9
        K_Pa = K_input * 1e9
        densidad_agua = densidad_calculada
        if espesor_tc_m > 0 and E_Pa > 0:
            num = 1.0 / K_Pa + D_tc_m / (E_Pa * espesor_tc_m)
            if num > 0:
                a_tc = np.sqrt(1.0 / (densidad_agua * num))
            else:
                a_tc = np.nan
        else:
            a_tc = np.nan
        # Calcular Tc
        Tc = 2 * L / a_tc if a_tc and a_tc > 0 else 0
        st.info(f"**Celeridad calculada:** {a_tc:.2f} m/s")
        st.info(f"**Tiempo de cierre crítico:** {Tc:.2f} s")  # Unidades en segundos
        st.info(f"**Usando:** Longitud = {L:.0f} m, Diámetro = {D_tc:.0f} mm, Espesor = {espesor_tc:.2f} mm")
        # Mostrar tipos de material que podrían tener esa celeridad (ahora en col1)
        tabla_materiales = [
            {"Material": "Acero", "a": (1000, 1250)},
            {"Material": "Fibro Cemento", "a": (900, 1200)},
            {"Material": "Concreto", "a": (1050, 1150)},
            {"Material": "Hierro Dúctil", "a": (1000, 1350)},
            {"Material": "PEAD", "a": (230, 430)},
            {"Material": "PVC", "a": (300, 500)}
        ]
        materiales_candidatos = [m["Material"] for m in tabla_materiales if a_tc is not None and not np.isnan(a_tc) and m["a"][0] <= a_tc <= m["a"][1]]
        if materiales_candidatos:
            st.success(f"**Material(es) compatible(s) con celeridad {a_tc:.0f} m/s:** " + ", ".join(materiales_candidatos))
        else:
            st.warning(f"No hay material típico con celeridad en el rango de {a_tc:.0f} m/s. Revisa los parámetros o consulta tablas técnicas.")
        
        # Cálculo de Longitud Crítica (Lc)
        st.markdown("---")
        st.subheader("Longitud Crítica (Lc)")
        
        # Inputs para calcular celeridad
        D_lc = st.number_input("Diámetro de la tubería (mm)", min_value=1.0, value=D_input, step=1.0, key="D_lc_col1")
        espesor_lc = st.number_input("Espesor de la tubería (mm)", min_value=0.01, value=espesor_calc_mm if espesor_calc_mm is not None and not np.isnan(espesor_calc_mm) else 1.0, step=0.01, key="espesor_lc_col1")
        
        # Cálculo de celeridad usando diámetro y espesor
        D_lc_m = D_lc / 1000.0
        espesor_lc_m = espesor_lc / 1000.0
        E_Pa_lc = E_input * 1e9
        K_Pa_lc = K_input * 1e9
        densidad_agua_lc = densidad_calculada
        
        if espesor_lc_m > 0 and E_Pa_lc > 0:
            num_lc = 1.0 / K_Pa_lc + D_lc_m / (E_Pa_lc * espesor_lc_m)
            if num_lc > 0:
                a_lc_calculada = np.sqrt(1.0 / (densidad_agua_lc * num_lc))
        else:
                a_lc_calculada = np.nan
        else:
            a_lc_calculada = np.nan
        
        # Usar el tiempo de cierre crítico como valor por defecto
        T_maniobra_default = Tc if Tc is not None and Tc > 0 else 5.0
        T_maniobra = st.number_input("Tiempo de maniobra (s)", min_value=0.1, value=T_maniobra_default, step=0.1, key="T_maniobra_col1")
        
        # Cálculo de Lc
        Lc_calculada = a_lc_calculada * T_maniobra / 2 if a_lc_calculada is not None and not np.isnan(a_lc_calculada) and a_lc_calculada > 0 else 0
        
        st.info(f"**Celeridad calculada:** {a_lc_calculada:.2f} m/s")
        st.info(f"**Longitud crítica Lc = {Lc_calculada:.2f} m**")
        st.info(f"**Usando:** Diámetro = {D_lc:.0f} mm, Espesor = {espesor_lc:.2f} mm, Tiempo = {T_maniobra:.2f} s")
        
        # Interpretación del resultado
        if Lc_calculada > 0:
            if Lc_calculada < 100:
                interpretacion = "Tubería CORTA - Cierre rápido recomendado"
            elif Lc_calculada < 500:
                interpretacion = "Tubería MEDIA - Cierre moderado"
            else:
                interpretacion = "Tubería LARGA - Cierre lento aceptable"
            st.success(f"**Interpretación:** {interpretacion}")
    
    with col2:
        st.subheader("Gráfico de Celeridad vs Espesor de Tubería")
        # Usar SIEMPRE los valores del usuario
        E_Pa = E_input * 1e9  # Pa
        K_Pa = K_input * 1e9  # Pa
        D = D_input / 1000.0  # m
        densidad_agua = densidad_calculada  # kg/m3

        celeridad_curve = []
        for e_mm in e_range:
            e = e_mm / 1000.0  # mm a m
            if e > 0:
                # Fórmula correcta de celeridad: a = sqrt(K/(ρ * (1/K + D/(E*e))))
                num = 1.0 / K_Pa + D / (E_Pa * e)
                if num > 0:
                    a_val = np.sqrt(1.0 / (densidad_agua * num))
                else:
                    a_val = np.nan
            else:
                a_val = np.nan
            celeridad_curve.append(a_val)

        fig_cel = go.Figure()
        fig_cel.add_trace(go.Scatter(
            x=e_range, y=celeridad_curve, mode='lines', name="Curva usuario",
            line=dict(color="#1f77b4", width=3),
            hovertemplate='<b>Celeridad</b>: %{y:.1f} m/s<extra></extra>',
            hoverlabel=dict(bgcolor="#1f77b4", font=dict(color='white', family='Arial Black'))
        ))
        # Agregar el punto calculado
        if espesor_calc_mm is not None and not np.isnan(espesor_calc_mm) and a_input is not None and not np.isnan(a_input):
            fig_cel.add_trace(go.Scatter(
                x=[espesor_calc_mm], y=[a_input], mode='markers',
                name='Punto calculado',
                marker=dict(color='red', size=12, symbol='diamond'),
                hovertemplate=f'<b>Punto</b><br>δ = {espesor_calc_mm:.2f} mm<br>a = {a_input:.1f} m/s<extra></extra>',
                hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
            ))
        # Ajustar escala para que el punto siempre se vea
        y_min = min(celeridad_curve) if celeridad_curve else 0
        y_max = max(celeridad_curve) if celeridad_curve else 1000
        if a_input is not None and not np.isnan(a_input):
            y_min = min(y_min, a_input * 0.9)
            y_max = max(y_max, a_input * 1.1)
        x_min = min(e_range) if len(e_range) > 0 else 0
        x_max = max(e_range) if len(e_range) > 0 else 10
        if espesor_calc_mm is not None and not np.isnan(espesor_calc_mm):
            x_min = min(x_min, espesor_calc_mm * 0.9)
            x_max = max(x_max, espesor_calc_mm * 1.1)
        fig_cel.update_layout(
            title="Celeridad vs Espesor de Tubería (parámetros de usuario)",
            xaxis_title="Espesor de tubería (mm)",
            yaxis_title="Celeridad (m/s)",
            height=350,
            showlegend=False,
            hovermode="x"
        )
        fig_cel.update_xaxes(
            range=[x_min, x_max],
            showspikes=True, 
            spikemode="across", 
            spikesnap="cursor", 
            spikethickness=1, 
            spikecolor="#555", 
            hoverformat=".2f"
        )
        fig_cel.update_yaxes(range=[y_min, y_max])
        st.plotly_chart(fig_cel, use_container_width=True)
        # Simbología
        st.markdown(f"""
        <div style="text-align: center; margin-top: -10px; margin-bottom: 10px;">
            <b>Simbología:</b><br>
            <span style="display:inline-block; width:30px; height:3px; background-color:#1f77b4; vertical-align:middle;"></span> Curva usuario
            <span style="display:inline-block; width:12px; height:12px; background-color:red; border-radius:50%; vertical-align:middle; margin-left:15px;"></span> Punto calculado
        </div>
        """, unsafe_allow_html=True)

        # --- NUEVOS GRÁFICOS Tc y Lc ---
        st.markdown("---")
        st.subheader("Tiempo crítico Tc vs Longitud")
        if 'L_range_tc' in locals() and L_range_tc.size > 0:
            Tc_vals = 2 * L_range_tc / a_input if a_input > 0 else np.zeros_like(L_range_tc)
            Tc_vals_log = np.clip(Tc_vals, 1e-3, None)
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Scatter(
                x=L_range_tc, y=Tc_vals_log, mode='lines', name=f"a = {a_input:.0f} m/s",
                hovertemplate='<b>Tc</b>: %{y:.3f} s<extra></extra>',
                hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))
            ))
            
            # Agregar punto del Tc calculado
            if Tc is not None and Tc > 0 and L > 0:
                fig_tc.add_trace(go.Scatter(
                    x=[L], y=[Tc], mode='markers',
                    name='Punto calculado',
                    marker=dict(color='red', size=10, symbol='diamond'),
                    hovertemplate=f'<b>Punto calculado</b><br>L = {L:.0f} m<br>Tc = {Tc:.2f} s<extra></extra>',
                    hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
                ))
            
            fig_tc.update_layout(
                title="Tiempo crítico Tc en función de la longitud de la tubería",
                xaxis_title="Longitud de la tubería L (m)",
                yaxis_title="Tiempo crítico Tc (s)",
                legend_title="Celeridad (a)",
                height=320,
                yaxis_type="log",
                hovermode='x'
            )
            st.plotly_chart(fig_tc, use_container_width=True)
            # Simbología debajo del gráfico Tc
            st.markdown("""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 10px;">
                <b>Simbología:</b><br>
                <span style="display:inline-block; width:30px; height:3px; background-color:#1f77b4; vertical-align:middle;"></span> Curva Tc vs L
                <span style="display:inline-block; width:12px; height:12px; background-color:red; border-radius:50%; vertical-align:middle; margin-left:15px;"></span> Punto calculado
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Configura correctamente el rango de longitudes para visualizar el gráfico Tc vs L.")

        st.markdown("---")
        st.subheader("Longitud crítica Lc vs Tiempo de maniobra")
        if 'T_range_lc' in locals() and T_range_lc.size > 0:
            fig_lc = go.Figure()
            Lc_vals = a_input * T_range_lc / 2 if a_input > 0 else np.zeros_like(T_range_lc)
            fig_lc.add_trace(go.Scatter(
                x=T_range_lc, y=Lc_vals, mode='lines', name=f"a = {a_input:.0f} m/s",
                hovertemplate='<b>Lc</b>: %{y:.2f} m<extra></extra>',
                hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))
            ))
            # Agregar punto de la Lc calculada
            if Lc_calculada > 0 and T_maniobra > 0:
                fig_lc.add_trace(go.Scatter(
                    x=[T_maniobra], y=[Lc_calculada], mode='markers',
                    name='Punto calculado',
                    marker=dict(color='red', size=10, symbol='diamond'),
                    hovertemplate=f'<b>Punto calculado</b><br>T = {T_maniobra:.2f} s<br>Lc = {Lc_calculada:.2f} m<extra></extra>',
                    hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
                ))
            fig_lc.update_layout(
                title="Longitud crítica Lc en función del tiempo de maniobra",
                xaxis_title="Tiempo de maniobra T (s)",
                yaxis_title="Longitud crítica Lc (m)",
                height=320,
                hovermode='x',
                xaxis=dict(range=[min(T_range_lc), max(T_range_lc)]),
                yaxis=dict(range=[min(Lc_vals), max(Lc_vals)])
            )
            st.plotly_chart(fig_lc, use_container_width=True)
            # Simbología debajo del gráfico Lc
            st.markdown("""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 10px;">
                <b>Simbología:</b><br>
                <span style="display:inline-block; width:30px; height:3px; background-color:#1f77b4; vertical-align:middle;"></span> Curva Lc vs T
                <span style="display:inline-block; width:12px; height:12px; background-color:red; border-radius:50%; vertical-align:middle; margin-left:15px;"></span> Punto calculado
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Configura correctamente el rango de tiempos para visualizar el gráfico Lc vs T.")

    with col3:
        st.subheader("Tabla de datos del gráfico")
        data = {"Espesor (mm)": np.round(e_range, 2), "Celeridad (m/s)": np.round(celeridad_curve, 1)}
        df_cel = pd.DataFrame(data)
        st.dataframe(df_cel, use_container_width=True)
        # (Asegúrate de que aquí NO haya ningún panel de tiempo de cierre crítico)
        # Panel desplegable con tabla de materiales y celeridad
        with st.expander("Tabla de materiales y celeridad típica"):
            df_mat = pd.DataFrame({
                "Material": ["Acero", "Fibro Cemento", "Concreto", "Hierro Dúctil", "PEAD", "PVC"],
                "E (GPa)": ["200.00 - 212.00", "23.50", "39.00", "166.00", "0.59 - 1.67", "2.40 - 2.75"],
                "a (m/s)": ["1000 - 1250", "900 - 1200", "1050 - 1150", "1000 - 1350", "230 - 430", "300 - 500"]
            })
            st.dataframe(df_mat, use_container_width=True)
        # Panel desplegable con fórmulas y teoría
        with st.expander("Fórmulas y Teoría"):
            st.markdown(r'''
**¿Qué es la celeridad (a)?**

La **celeridad (a)** es la velocidad a la que se propaga una onda de presión a través de un fluido contenido en una tubería. Es fundamental en el análisis de fenómenos transitorios, como el **golpe de ariete**.

- Físicamente, representa la rapidez con la que una perturbación (por ejemplo, un cierre rápido de válvula) se transmite a lo largo de la tubería.
- Depende de las propiedades del fluido (densidad y módulo de elasticidad) y de las características de la tubería (módulo de elasticidad del material, diámetro y espesor).

**Importancia:**
- Una celeridad alta implica que los cambios de presión se transmiten muy rápido, lo que puede generar sobrepresiones peligrosas en eventos transitorios.
- Es clave para calcular el golpe de ariete y diseñar sistemas hidráulicos seguros.

**Fórmula de celeridad:**

$$
a = \sqrt{\frac{K/\rho}{1 + \frac{K D}{E \delta}}}
$$

**Donde:**
- $a$: celeridad de la onda (m/s)
- $K$: módulo de elasticidad del fluido (N/m²)
- $\rho$: densidad del fluido (kg/m³)
- $D$: diámetro interior de la tubería (m)
- $E$: módulo de elasticidad del material de la tubería (N/m²)
- $\delta$: espesor de la tubería (m)

**Fórmula de espesor de tubería:**

$$
\delta = \frac{E(K-\rho a^2)}{K D \rho a^2}
$$

**Fundamento teórico:**
Estas fórmulas se derivan de la ecuación de celeridad del golpe de ariete, despejando el espesor $\delta$. La celeridad $a$ está relacionada con las propiedades del fluido y la tubería mediante la ecuación mostrada arriba.

---

**Índice de cavitación:**

$$
\sigma = \frac{P_1 - P_v}{P_1 - P_2}
$$

$$
\vartheta = \frac{P_2 - P_{vapor}}{P_1 - P_2}
$$

El índice de cavitación es un parámetro adimensional que permite evaluar el riesgo de formación de cavitación en sistemas hidráulicos. Compara la presión disponible en el sistema con la presión de vapor del fluido. Un valor bajo indica mayor riesgo de cavitación, mientras que valores altos indican condiciones seguras.

- $P_1$: Presión de entrada (aguas arriba)
- $P_2$: Presión de salida (aguas abajo)
- $P_v$ / $P_{vapor}$: Presión de vapor del fluido
''')

        # Panel desplegable con concepto y fórmula del tiempo de cierre crítico
        with st.expander("Concepto y Fórmula del Tiempo de Cierre Crítico"):
            st.markdown(r'''
**¿Qué es el tiempo de cierre crítico?**

El **tiempo de cierre crítico (Tc)** es el tiempo mínimo necesario para que una onda de presión recorra la longitud total de la tubería y regrese al punto de origen. Es fundamental en el análisis de fenómenos transitorios como el **golpe de ariete**.

**Importancia:**
- Si el tiempo de cierre es menor que el crítico, se produce un cierre **rápido** con sobrepresiones máximas
- Si el tiempo de cierre es mayor que el crítico, se produce un cierre **lento** con sobrepresiones reducidas
- Es clave para diseñar sistemas hidráulicos seguros y evitar daños por sobrepresión

**Fórmula del tiempo de cierre crítico:**

$$
T_c = \frac{2L}{a}
$$

**Donde:**
- $T_c$: tiempo de cierre crítico (s)
- $L$: longitud de la tubería (m)
- $a$: celeridad de la onda (m/s)

**Interpretación:**
- El factor 2 representa el viaje de ida y vuelta de la onda de presión
- Una celeridad alta implica un tiempo crítico corto
- Una tubería larga requiere más tiempo para el cierre seguro

**Tipos de cierre según el tiempo:**
- **Cierre Rápido** ($t \leq T_c$): Sobrepresión máxima
- **Cierre Lento** ($t > T_c$): Sobrepresión reducida
''')

        # Panel desplegable con teoría de golpe de ariete y consecuencias
        with st.expander("Teoría y consecuencias del golpe de ariete"):
            st.markdown(r'''
**Transitorio rápido o golpe de ariete**

Magnitudes características del flujo varían en el tiempo con una escala temporal de segundos.

Es un repentino aumento de presión que se produce por la maniobra de un elemento de control cuando se detiene de forma severa el fluido que circula por este punto.

---

**Consecuencias del golpe de ariete:**

- **Sobrepresiones**  
  Las altas presiones que se producen como resultado del golpe de ariete pueden llegar a ser superiores a la resistencia de la tubería y provocar su rotura.

- **Depresiones**  
  Si la tubería no es suficientemente rígida en su sección transversal, la diferencia de presiones puede hacer que la sección pierda su estabilidad y colapse la tubería.

- **Fatiga del material**  
  La acción repetida de cargas dinámicas fuertes durante un período de tiempo prolongado disminuye la resistencia del material.

- **Sobrevelocidad en los equipos de bombeo**  
  En caso de que se invierta el flujo y entre al equipo de bombeo, éste puede producir que las bombas giren a una mayor velocidad en sentido contrario de la especificada por el fabricante.
''')

        # Panel desplegable con teoría, fórmulas y cálculo de Tc y Lc
        with st.expander("Cálculo de Tiempo Crítico (Tc) y Longitud Crítica (Lc)"):
            st.markdown(r'''
### 2. Tiempo crítico (Tc)
Las máximas sobrepresiones son producidas por maniobras de cierre menores al tiempo que tarda la onda en su viaje de ida y vuelta al elemento de control.

**Ecuación:**
$$
T_c = \frac{2L}{a}
$$

- $T_c$: tiempo crítico del sistema (s)
- $L$: longitud del sistema (m)
- $a$: velocidad de propagación o celeridad $\left(\frac{m}{s}\right)$

**Criterios:**
- Cierre rápido: $0 < T_c < \frac{2L}{a}$
- Cierre lento: $T_c > \frac{2L}{a}$

---
### 3. Longitud crítica (Lc)
Diferencia entre una conducción corta y una larga con relación a la duración de la maniobra del elemento de control ($T$).

**Ecuación:**
$$
L_c = \frac{aT}{2}
$$
- $L_c$: longitud crítica (m)
- $a$: celeridad $\left(\frac{m}{s}\right)$
- $T$: tiempo de maniobra del elemento de control (s)
''')

            st.markdown("---")
            st.markdown("#### Calculadora interactiva")
            # Inputs para Tc
            L_tc = st.number_input("Longitud del sistema L (m) [Tc]", min_value=1.0, value=1000.0, step=1.0, key="L_tc_col3")
            a_tc = st.number_input("Celeridad a (m/s) [Tc]", min_value=1.0, value=1000.0, step=1.0, key="a_tc_col3")
            Tc_calc = 2 * L_tc / a_tc if a_tc > 0 else 0
            st.info(f"**Tiempo crítico Tc = {Tc_calc:.2f} s**  (para L = {L_tc:.0f} m, a = {a_tc:.0f} m/s)")

            st.markdown("---")
            # Inputs para Lc
            a_lc = st.number_input("Celeridad a (m/s) [Lc]", min_value=1.0, value=1000.0, step=1.0, key="a_lc_col3")
            T_lc = st.number_input("Tiempo de maniobra T (s) [Lc]", min_value=0.1, value=5.0, step=0.1, key="T_lc_col3")
            Lc_calc = a_lc * T_lc / 2
            st.info(f"**Longitud crítica Lc = {Lc_calc:.2f} m**  (para a = {a_lc:.0f} m/s, T = {T_lc:.2f} s)")

        # Mostrar tipos de material que podrían tener esa celeridad
        tabla_materiales = [
            {"Material": "Acero", "a": (1000, 1250)},
            {"Material": "Fibro Cemento", "a": (900, 1200)},
            {"Material": "Concreto", "a": (1050, 1150)},
            {"Material": "Hierro Dúctil", "a": (1000, 1350)},
            {"Material": "PEAD", "a": (230, 430)},
            {"Material": "PVC", "a": (300, 500)}
        ]
        materiales_candidatos = [m["Material"] for m in tabla_materiales if a_tc is not None and not np.isnan(a_tc) and m["a"][0] <= a_tc <= m["a"][1]]
        if materiales_candidatos:
            st.success(f"**Material(es) compatible(s) con celeridad {a_tc:.0f} m/s:** " + ", ".join(materiales_candidatos))
        else:
            st.warning(f"No hay material típico con celeridad en el rango de {a_tc:.0f} m/s. Revisa los parámetros o consulta tablas técnicas.")
        
# --- Pestaña 5: Pérdidas por Fricción ---
with tab5:
    st.header("📈 Análisis de Pérdidas por Fricción")
    
    col_fric1, col_fric2 = st.columns(2)
    
    with col_fric1:
        st.subheader("Datos del Sistema")
        L_fric = st.number_input("Longitud de la tubería (m)", min_value=1.0, value=1000.0, step=10.0, key="L_fric")
        D_fric = st.number_input("Diámetro de la tubería (mm)", min_value=10.0, value=200.0, step=1.0, key="D_fric") / 1000
        Q_fric = st.number_input("Caudal (L/s)", min_value=0.1, value=100.0, step=0.1, key="Q_fric") / 1000  # convertir a m³/s
        C_hw_fric = st.number_input("Coeficiente Hazen-Williams", min_value=50.0, max_value=150.0, value=100.0, step=5.0, key="C_hw_fric")
        
        # Cálculo de pérdidas
        Hf_fric = calculate_friction_losses_hw(Q_fric, D_fric, L_fric, C_hw_fric)
        st.metric("Pérdidas por fricción", f"{Hf_fric:.2f} mca")
        
        # Pérdidas por metro
        Hf_por_metro = Hf_fric / L_fric
        st.metric("Pérdidas por metro", f"{Hf_por_metro:.4f} mca/m")
        
    with col_fric2:
        st.subheader("Análisis de Eficiencia")
        area_fric = np.pi * (D_fric / 2)**2
        velocity_fric = Q_fric / area_fric if area_fric > 0 else 0
        st.metric("Velocidad del flujo", f"{velocity_fric:.2f} m/s")
        
        # Gráfico de pérdidas vs caudal
        Q_range = np.linspace(0.01, Q_fric * 2, 50)
        Hf_range = [calculate_friction_losses_hw(q, D_fric, L_fric, C_hw_fric) for q in Q_range]
        
        fig_fric = go.Figure()
        fig_fric.add_trace(go.Scatter(
            x=Q_range * 1000, y=Hf_range,
            mode='lines', name='Pérdidas por fricción',
            line=dict(color='#1f77b4')
        ))
        
        fig_fric.update_layout(
            title="Pérdidas por Fricción vs Caudal",
            xaxis_title="Caudal (L/s)",
            yaxis_title="Pérdidas (mca)",
            height=400
        )
        
        st.plotly_chart(fig_fric, use_container_width=True)

# --- Información adicional ---
st.sidebar.markdown("### 📚 Información Técnica")
with st.sidebar.expander("Acerca de la cavitación"):
    st.markdown("""
    **Cavitación** es la formación de burbujas de vapor en un líquido cuando la presión local cae por debajo de la presión de vapor del fluido.
    
    **Efectos:**
    - Erosión de materiales
    - Vibraciones y ruido
    - Reducción de eficiencia
    - Daños a equipos
    
    **Prevención:**
    - Mantener presiones adecuadas
    - Diseño correcto de válvulas
    - Selección de materiales resistentes
    """)

with st.sidebar.expander("Referencias técnicas"):
    st.markdown("""
    - **API 610**: Bombas centrífugas para servicios de refinería
    - **ISO 5199**: Bombas centrífugas técnicas
    - **ANSI/HI 9.6.1**: Guía para cavitación en bombas
    - **ASME B73.1**: Bombas horizontales de proceso
    """)

# --- Pie de página ---
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Calculadora de Índice de Cavitación - Herramienta de análisis para sistemas de bombeo</p>
    <p>Desarrollada para análisis técnico de cavitación en válvulas, bombas y sistemas hidráulicos</p>
</div>
""", unsafe_allow_html=True)
