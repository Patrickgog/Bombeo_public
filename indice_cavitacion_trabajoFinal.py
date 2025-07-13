import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Tabla de densidad y presión de vapor ---
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

# --- Funciones Auxiliares ---
def get_vapor_pressure_mca(temperature_celsius):
    A = 8.07131
    B = 1730.63
    C = 233.426
    T = temperature_celsius
    P_mmHg = 10 ** (A - (B / (C + T)))
    P_mca = P_mmHg * 0.0136
    return P_mca

class CavitationCalculator:
    def __init__(self, fluid_temperature_celsius):
        self.fluid_temperature_celsius = fluid_temperature_celsius
        self.Pv = get_vapor_pressure_mca(fluid_temperature_celsius)

    def calculate_cavitation_index(self, P_upstream_mca, P_downstream_mca):
        if P_upstream_mca <= P_downstream_mca:
            return None
        delta_P = P_upstream_mca - P_downstream_mca
        if delta_P == 0:
            return float('inf')
        numerator = P_downstream_mca - self.Pv
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

st.set_page_config(layout="wide", page_title="Simulación de Índice de Cavitación - Trabajo Final")
st.title("Simulación de Índice de Cavitación: Trabajo Final")

# --- Parámetros Generales ---
st.sidebar.header("Parámetros Generales")
T_fluid = st.sidebar.number_input("Temperatura del fluido (°C)", min_value=0, max_value=100, value=20, step=1)
densidad_agua = interpola_tabla(T_fluid, tabla_temp, tabla_dens)
st.sidebar.info(f"Densidad del agua a {T_fluid}°C: **{densidad_agua:.1f} kg/m³**")

calc = CavitationCalculator(T_fluid)

# --- Pestañas ---
tab_diam, tab_caudal = st.tabs([
    "Índice de Cavitación vs Diámetro",
    "Índice de Cavitación vs Caudal"
])

# Pestaña de diámetro
with tab_diam:
    st.header("Índice de Cavitación variando el Diámetro de la Válvula")
    col1, col2, col3, col4 = st.columns([0.22, 0.38, 0.28, 0.12])
    with col1:
        st.subheader("Datos")
        Q = st.number_input("Caudal fijo (L/s)", min_value=0.01, value=10.0, step=0.1, key="Q_diam")
        P1_calc = st.number_input("Presión de entrada P1 (mca)", min_value=0.0, value=10.0, step=0.1, key="P1_diam_calc")
        D_calc_puntual = st.number_input("Diámetro de cálculo (mm)", min_value=1.0, value=100.0, step=1.0, key="D_calc_puntual")
        K_coef = st.number_input("Coeficiente de pérdidas K", min_value=0.1, value=10.0, step=0.1, key="K_diam")
        # Cálculos puntuales
        D_calc_m = D_calc_puntual / 1000
        A = np.pi * (D_calc_m/2)**2
        v = (Q/1000) / A if A > 0 else 0
        K = K_coef
        g = 9.81
        hL = K * v**2 / (2*g)
        P2 = P1_calc - hL
        Pv = get_vapor_pressure_mca(T_fluid)
        theta = calc.calculate_cavitation_index(P1_calc, P2)
        # Mostrar resultados
        st.markdown("## Resultados")
        st.metric(label="Índice de Cavitación (θ)", value=f"{theta:.3f}")
        # Mensaje visual según el riesgo
        if theta is not None:
            if theta < 0.5:
                st.error("🚨 Riesgo CRÍTICO de daño por cavitación (θ < 0.5)")
            elif theta < 0.8:
                st.warning("⚠️ Riesgo ALTO de ruido por cavitación (0.5 ≤ θ < 0.8)")
            else:
                st.success("✅ Riesgo de cavitación BAJO o NULO.")
        st.markdown(f"**Área (A):** {A:.5f} m²")
        st.markdown(f"**Velocidad (v):** {v:.3f} m/s")
        st.markdown(f"**Coeficiente K:** {K:.1f}")
        st.markdown(f"**hL:** {hL:.3f} mca")
        st.markdown(f"**P2:** {P2:.3f} mca")
        st.markdown(f"**Pv (presión de vapor):** {Pv:.3f} mca")
        st.markdown("---")
        st.subheader("Datos gráfico")
        D_min = st.number_input("Diámetro mínimo (mm)", min_value=1.0, value=20.0, step=1.0, key="D_min")
        D_max = st.number_input("Diámetro máximo (mm)", min_value=1.0, value=200.0, step=1.0, key="D_max")
        D_step = st.number_input("Paso de diámetro (mm)", min_value=0.1, value=5.0, step=0.1, key="D_step")
        ordenada_max = st.number_input("Ordenada máxima", min_value=0.1, value=30.0, step=0.1, key="ordenada_max_diam")
        D_range = np.arange(D_min, D_max + D_step, D_step) / 1000  # m
        st.markdown("---")
        st.markdown("**Resultados principales:**")
        st.markdown("- θ < 0.5: Riesgo crítico de daño por cavitación")
        st.markdown("- 0.5 ≤ θ < 0.8: Riesgo alto de ruido por cavitación")
        st.markdown("- θ ≥ 0.8: Riesgo bajo o nulo de cavitación")
        
        # Panel informativo sobre coeficiente K
        with st.expander("Valores típicos del coeficiente K"):
            st.markdown("""
            **Coeficiente de pérdidas K - Valores típicos:**
            
            | Tipo de Válvula | K (típico) | Rango |
            |:---------------:|:----------:|:-----:|
            | Válvula de globo (totalmente abierta) | 6-10 | 5-15 |
            | Válvula de globo (parcialmente abierta) | 15-30 | 10-50 |
            | Válvula de compuerta (totalmente abierta) | 0.15-0.3 | 0.1-0.5 |
            | Válvula de compuerta (parcialmente abierta) | 2-5 | 1-10 |
            | Válvula de mariposa (totalmente abierta) | 0.3-0.5 | 0.2-1.0 |
            | Válvula de mariposa (parcialmente abierta) | 5-15 | 3-25 |
            | Válvula de bola (totalmente abierta) | 0.05-0.1 | 0.03-0.2 |
            | Válvula de bola (parcialmente abierta) | 2-8 | 1-15 |
            | Codo de 90° | 0.3-0.5 | 0.2-0.8 |
            | Tee en línea | 0.2-0.4 | 0.1-0.6 |
            | Entrada de depósito | 0.5-1.0 | 0.3-1.5 |
            | Salida a depósito | 1.0 | 0.8-1.2 |
            
            **Nota:** Los valores pueden variar según el fabricante, tamaño y condiciones de operación.
            """)
    with col2:
        results = []
        for D in D_range:
            area = np.pi * (D/2)**2
            v = (Q/1000) / area if area > 0 else 0
            K = K_coef # Use the K_coef input value
            g = 9.81
            hL = K * v**2 / (2*g)
            P2 = P1_calc - hL
            theta = calc.calculate_cavitation_index(P1_calc, P2)
            results.append({
                "Diámetro (mm)": D*1000,
                "Índice θ": theta,
                "P2 (mca)": P2,
                "Riesgo": calc.get_cavitation_risk_description(theta)
            })
        df = pd.DataFrame(results)
        st.subheader("Gráfico θ vs Diámetro")
        
        # Selector de escala del eje Y
        escala_y = st.radio("Escala del eje Y", [f"Detalle (0-1)", f"Completa (0-{ordenada_max})"], key="escala_diam")
        y_max = ordenada_max if escala_y == f"Completa (0-{ordenada_max})" else 1
        
        # Crear máscaras para los rangos de riesgo
        theta_array = np.array(df["Índice θ"])
        mask_rojo = theta_array < 0.5
        mask_amarillo = (theta_array >= 0.5) & (theta_array < 0.8)
        mask_verde = theta_array >= 0.8
        
        # Crear arrays para cada rango de riesgo
        y_rojo = np.where(mask_rojo, theta_array, np.nan)
        y_amarillo = np.where(mask_amarillo, theta_array, np.nan)
        y_verde = np.where(mask_verde, theta_array, np.nan)
        
        fig = go.Figure()
        
        # Agregar áreas pintadas para cada rango de riesgo
        # Área roja (riesgo crítico)
        fig.add_trace(go.Scatter(
            x=df["Diámetro (mm)"], y=y_rojo, mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            name='Riesgo Crítico (θ < 0.5)',
            showlegend=False,
            hoverinfo='skip'
        ))
        # Área amarilla (ahora morada, riesgo alto)
        fig.add_trace(go.Scatter(
            x=df["Diámetro (mm)"], y=y_amarillo, mode='lines',
            line=dict(color='rgba(128,0,128,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.2)',
            name='Riesgo Alto (0.5 ≤ θ < 0.8)',
            showlegend=False,
            hoverinfo='skip'
        ))
        # Área verde (riesgo bajo)
        fig.add_trace(go.Scatter(
            x=df["Diámetro (mm)"], y=y_verde, mode='lines',
            line=dict(color='rgba(0,200,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(0,200,0,0.2)',
            name='Riesgo Bajo (θ ≥ 0.8)',
            showlegend=False,
            hoverinfo='skip'
        ))
        # Agregar la curva principal encima de las áreas
        fig.add_trace(go.Scatter(
            x=df["Diámetro (mm)"], y=df["Índice θ"], mode='lines',
            line=dict(color='#1f77b4', width=3),
            name='Índice de Cavitación',
            hovertemplate='θ = %{y:.3f}<extra></extra>',
            hoverlabel=dict(bgcolor='#1f77b4', font=dict(color='white', family='Arial Black'))
        ))
        # Calcular el valor puntual exactamente igual que la curva
        area_puntual = np.pi * (D_calc_puntual/1000/2)**2
        v_puntual = (Q/1000) / area_puntual if area_puntual > 0 else 0
        hL_puntual = K_coef * v_puntual**2 / (2*g)
        P2_puntual = P1_calc - hL_puntual
        theta_puntual = calc.calculate_cavitation_index(P1_calc, P2_puntual)
        # Agregar punto puntual de cálculo
        fig.add_trace(go.Scatter(
            x=[D_calc_puntual], y=[theta_puntual], mode='markers',
            marker=dict(color='red', size=8, symbol='diamond'),
            name='Punto calculado',
            hovertemplate='Diámetro = %{x:.2f} mm<br>θ = %{y:.3f}<extra></extra>',
            hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
        ))
        
        # Agregar líneas horizontales de límites
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=2, 
                     annotation_text="Límite Daño (0.5)", secondary_y=False)
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange", line_width=2, 
                     annotation_text="Límite Ruido (0.8)", secondary_y=False)
        
        fig.update_layout(
            xaxis_title="Diámetro de válvula (mm)",
            yaxis_title="Índice de Cavitación θ",
            height=400,
            showlegend=False,
            hovermode="x",
            yaxis_range=[0,y_max]
        )
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="#888",
            showline=True,
            showticklabels=True,
            tickformat=".2f",
            hoverformat=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Simbología debajo del gráfico
        st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <b>Simbología:</b><br><br>
            <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin: 10px 0; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 30px; height: 3px; background-color: #1f77b4; border-radius: 2px;"></div>
                    <span style="font-size: 0.9em;">Índice de cavitación</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(255,0,0,0.2); border: 1px solid rgba(255,0,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo crítico (θ < 0.5)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(128,0,128,0.2); border: 1px solid rgba(128,0,128,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo alto (0.5 ≤ θ < 0.8)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(0,200,0,0.2); border: 1px solid rgba(0,200,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo bajo (θ ≥ 0.8)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.subheader("Tabla de resultados")
        st.dataframe(df, use_container_width=True)
        
        # Panel informativo sobre conceptos del índice de cavitación
        with st.expander("📚 Conceptos del Índice de Cavitación"):
            st.markdown("""
            ## **Fórmula del Índice de Cavitación:**
            ```
            θ = (P2 - Pv) / (P1 - P2)
            ```
            
            ## **Elementos de la fórmula:**
            
            ### **1. P1 (Presión de entrada)**
            - **Concepto**: Presión aguas arriba de la válvula, antes de la restricción
            - **Unidad**: mca (metros de columna de agua)
            - **Significado**: Es la presión disponible en el sistema antes de pasar por la válvula
            
            ### **2. P2 (Presión de salida)**
            - **Concepto**: Presión aguas abajo de la válvula, después de la restricción
            - **Cálculo**: `P2 = P1 - hL`
            - **Unidad**: mca
            - **Significado**: Presión que queda después de las pérdidas en la válvula
            
            ### **3. Pv (Presión de vapor)**
            - **Concepto**: Presión a la cual el agua hierve a la temperatura del fluido
            - **Dependencia**: Varía con la temperatura del agua
            - **Unidad**: mca
            - **Significado**: Presión mínima necesaria para evitar que el agua se evapore
            
            ### **4. hL (Pérdida de carga)**
            - **Concepto**: Pérdida de presión debido a la restricción en la válvula
            - **Fórmula**: `hL = K × v² / (2g)`
            - **Unidad**: mca
            - **Significado**: Energía perdida por fricción y turbulencia
            
            ### **5. K (Coeficiente de pérdidas)**
            - **Concepto**: Factor que caracteriza la resistencia de la válvula
            - **Valores típicos**: 
              - Válvula de globo: 6-10
              - Válvula de compuerta: 0.15-0.3
              - Válvula de mariposa: 0.3-0.5
            - **Significado**: Cuanto mayor K, mayor pérdida de presión
            
            ### **6. v (Velocidad del fluido)**
            - **Concepto**: Velocidad del agua al pasar por la válvula
            - **Fórmula**: `v = Q / A`
            - **Unidad**: m/s
            - **Significado**: Velocidad que determina la energía cinética
            
            ### **7. A (Área de paso)**
            - **Concepto**: Área transversal por donde pasa el fluido
            - **Fórmula**: `A = π × (D/2)²`
            - **Unidad**: m²
            - **Significado**: Sección de paso que determina la velocidad
            
            ## **Interpretación del Índice θ:**
            
            ### **θ < 0.5: RIESGO CRÍTICO** 🚨
            - **Significado**: La presión P2 está muy cerca de Pv
            - **Consecuencia**: Alto riesgo de daño por cavitación
            - **Fenómeno**: El agua puede vaporizarse y causar erosión
            
            ### **0.5 ≤ θ < 0.8: RIESGO ALTO** ⚠️
            - **Significado**: P2 está en zona de riesgo
            - **Consecuencia**: Posible ruido y vibraciones por cavitación
            - **Fenómeno**: Inicio de formación de burbujas de vapor
            
            ### **θ ≥ 0.8: RIESGO BAJO** ✅
            - **Significado**: P2 está bien por encima de Pv
            - **Consecuencia**: Operación segura sin cavitación
            - **Fenómeno**: Presión suficiente para evitar vaporización
            
            ## **Relación física:**
            - **θ alto** = Mayor margen de seguridad
            - **θ bajo** = Menor margen de seguridad
            - **θ = 0** = P2 = Pv (inicio de cavitación)
            - **θ < 0** = P2 < Pv (cavitación activa)
            """)
        
        # Panel informativo sobre qué es el índice de cavitación
        with st.expander("🔬 ¿Qué es el Índice de Cavitación?"):
            st.markdown("""
            El **Índice de Cavitación (θ)** es un parámetro fundamental en ingeniería hidráulica que evalúa el riesgo de cavitación en sistemas de flujo de fluidos.
            
            ## **¿Qué es la Cavitación?**
            
            La **cavitación** es un fenómeno físico donde se forman burbujas de vapor dentro de un líquido cuando la presión local cae por debajo de la presión de vapor del fluido. Cuando estas burbujas colapsan, generan ondas de choque que pueden causar:
            
            - **Erosión** en las superficies metálicas
            - **Ruido** y vibraciones
            - **Pérdida de eficiencia** del sistema
            - **Daños** a válvulas, bombas y tuberías
            
            ## **¿Qué es el Índice de Cavitación (θ)?**
            
            Es una **medida adimensional** que cuantifica qué tan cerca está la presión aguas abajo (P2) de la presión de vapor (Pv), comparada con la caída de presión total en el sistema.
            
            ### **Fórmula:**
            ```
            θ = (P2 - Pv) / (P1 - P2)
            ```
            
            ### **Interpretación física:**
            
            - **θ alto** = Mayor margen de seguridad (P2 está muy por encima de Pv)
            - **θ bajo** = Menor margen de seguridad (P2 se acerca a Pv)
            - **θ = 0** = P2 = Pv (inicio de cavitación)
            - **θ < 0** = P2 < Pv (cavitación activa)
            
            ## **Rangos de Riesgo:**
            
            ### **🚨 θ < 0.5: RIESGO CRÍTICO**
            - La presión P2 está muy cerca de Pv
            - Alto riesgo de daño por cavitación
            - El agua puede vaporizarse y causar erosión severa
            
            ### **⚠️ 0.5 ≤ θ < 0.8: RIESGO ALTO**
            - P2 está en zona de riesgo
            - Posible ruido y vibraciones por cavitación
            - Inicio de formación de burbujas de vapor
            
            ### **✅ θ ≥ 0.8: RIESGO BAJO**
            - P2 está bien por encima de Pv
            - Operación segura sin cavitación
            - Presión suficiente para evitar vaporización
            
            ## **Aplicaciones prácticas:**
            
            1. **Diseño de válvulas**: Seleccionar válvulas que mantengan θ ≥ 0.8
            2. **Operación de sistemas**: Monitorear θ para evitar cavitación
            3. **Mantenimiento**: Identificar condiciones de riesgo antes de que ocurran daños
            4. **Optimización**: Ajustar parámetros para mejorar el margen de seguridad
            
            ## **Factores que afectan θ:**
            
            - **Presión de entrada (P1)**: Mayor P1 = mayor θ
            - **Coeficiente K**: Mayor K = menor θ
            - **Caudal (Q)**: Mayor Q = menor θ
            - **Diámetro (D)**: Mayor D = mayor θ
            - **Temperatura**: Mayor temperatura = menor θ (por mayor Pv)
            
            El índice de cavitación es esencial para el diseño y operación segura de sistemas hidráulicos, permitiendo prevenir daños costosos y mantener la eficiencia del sistema.
            """)
        
        # Panel informativo sobre presión de vapor y cavitación
        with st.expander("💧 Presión de Vapor y Cavitación"):
            st.markdown("""
            La **presión de vapor (Pv)** es un concepto fundamental en la cavitación.
            
            ## **¿Qué es la Presión de Vapor (Pv)?**
            
            La **presión de vapor** es la presión a la cual un líquido comienza a hervir (evaporarse) a una temperatura específica. Es la presión mínima necesaria para mantener el líquido en estado líquido.
            
            ### **Características importantes:**
            
            - **Depende de la temperatura**: A mayor temperatura, mayor Pv
            - **Es específica del fluido**: Cada líquido tiene su propia Pv
            - **Se mide en unidades de presión**: mca, Pa, bar, etc.
            
            ## **¿Cómo interviene en la Cavitación?**
            
            ### **1. Mecanismo de la Cavitación:**
            
            ```
            Si P2 < Pv → El agua se evapora → Se forman burbujas de vapor
            ```
            
            - Cuando la **presión local (P2)** cae por debajo de la **presión de vapor (Pv)**
            - El agua cambia de estado líquido a vapor
            - Se forman **burbujas de vapor** dentro del fluido
            
            ### **2. Efecto en el Índice de Cavitación:**
            
            En la fórmula: `θ = (P2 - Pv) / (P1 - P2)`
            
            - **P2 - Pv**: Es el **margen de seguridad**
            - Si P2 se acerca a Pv → θ disminuye → Mayor riesgo
            - Si P2 está muy por encima de Pv → θ aumenta → Menor riesgo
            
            ### **3. Relación con la Temperatura:**
            
            | Temperatura (°C) | Pv (mca) |
            |:----------------:|:--------:|
            | 0 | 0.06 |
            | 20 | 0.25 |
            | 40 | 0.76 |
            | 60 | 2.03 |
            | 80 | 4.85 |
            | 100 | 10.33 |
            
            **A mayor temperatura → Mayor Pv → Menor margen de seguridad**
            
            ## **¿Por qué es Crítica?**
            
            ### **1. Punto de Inicio de Cavitación:**
            - Cuando P2 = Pv → θ = 0 (inicio de cavitación)
            - Cuando P2 < Pv → θ < 0 (cavitación activa)
            
            ### **2. Diseño de Sistemas:**
            - Debe asegurarse que P2 > Pv en todo momento
            - Mayor margen = Mayor seguridad
            
            ### **3. Operación:**
            - Monitorear temperatura del fluido
            - Ajustar presiones según la temperatura
            - Considerar Pv en el diseño de válvulas
            
            ## **Factores que Afectan Pv:**
            
            1. **Temperatura del fluido** (principal factor)
            2. **Composición del fluido** (agua pura vs. soluciones)
            3. **Presión atmosférica** (en sistemas abiertos)
            4. **Altitud** (afecta la presión atmosférica)
            
            La presión de vapor es el **umbral crítico** que determina cuándo comienza la cavitación, por eso es fundamental en el análisis del índice de cavitación.
            """)
    with col4:
        pass

# Pestaña de caudal
with tab_caudal:
    st.header("Índice de Cavitación variando el Caudal en la Válvula")
    col1, col2, col3, col4 = st.columns([0.22, 0.38, 0.28, 0.12])
    with col1:
        st.subheader("Datos")
        D = st.number_input("Diámetro fijo (mm)", min_value=1.0, value=50.0, step=1.0, key="D_caudal")
        P1 = st.number_input("Presión de entrada P1 (mca)", min_value=0.0, value=10.0, step=0.1, key="P1_caudal")
        K_coef = st.number_input("Coeficiente de pérdidas K", min_value=0.1, value=10.0, step=0.1, key="K_caudal")
        Q_puntual = st.number_input("Caudal de cálculo (L/s)", min_value=0.01, value=5.0, step=0.1, key="Q_puntual")
        # Cálculo puntual
        area_puntual = np.pi * (D/1000/2)**2
        v_puntual = (Q_puntual/1000) / area_puntual if area_puntual > 0 else 0
        hL_puntual = K_coef * v_puntual**2 / (2*9.81)
        P2_puntual = P1 - hL_puntual
        Pv_puntual = get_vapor_pressure_mca(T_fluid)
        theta_puntual = calc.calculate_cavitation_index(P1, P2_puntual)
        st.markdown("## Resultados")
        st.metric(label="Índice de Cavitación (θ)", value=f"{theta_puntual:.3f}")
        if theta_puntual is not None:
            if theta_puntual < 0.5:
                st.error("🚨 Riesgo CRÍTICO de daño por cavitación (θ < 0.5)")
            elif theta_puntual < 0.8:
                st.warning("⚠️ Riesgo ALTO de ruido por cavitación (0.5 ≤ θ < 0.8)")
            else:
                st.success("✅ Riesgo de cavitación BAJO o NULO.")
        st.markdown(f"**Área (A):** {area_puntual:.5f} m²")
        st.markdown(f"**Velocidad (v):** {v_puntual:.3f} m/s")
        st.markdown(f"**Coeficiente K:** {K_coef:.1f}")
        st.markdown(f"**hL:** {hL_puntual:.3f} mca")
        st.markdown(f"**P2:** {P2_puntual:.3f} mca")
        st.markdown(f"**Pv (presión de vapor):** {Pv_puntual:.3f} mca")
        st.markdown("---")
        st.subheader("Datos gráfico")
        Q_min = st.number_input("Caudal mínimo (L/s)", min_value=0.01, value=1.0, step=0.1, key="Q_min")
        Q_max = st.number_input("Caudal máximo (L/s)", min_value=0.01, value=20.0, step=0.1, key="Q_max")
        Q_step = st.number_input("Paso de caudal (L/s)", min_value=0.01, value=0.5, step=0.01, key="Q_step")
        ordenada_max = st.number_input("Ordenada máxima", min_value=0.1, value=30.0, step=0.1, key="ordenada_max_caudal")
        Q_range = np.arange(Q_min, Q_max + Q_step, Q_step) / 1000  # m3/s
        # Panel informativo sobre coeficiente K
        with st.expander("Valores típicos del coeficiente K"):
            st.markdown("""
            **Coeficiente de pérdidas K - Valores típicos:**
            
            | Tipo de Válvula | K (típico) | Rango |
            |:---------------:|:----------:|:-----:|
            | Válvula de globo (totalmente abierta) | 6-10 | 5-15 |
            | Válvula de globo (parcialmente abierta) | 15-30 | 10-50 |
            | Válvula de compuerta (totalmente abierta) | 0.15-0.3 | 0.1-0.5 |
            | Válvula de compuerta (parcialmente abierta) | 2-5 | 1-10 |
            | Válvula de mariposa (totalmente abierta) | 0.3-0.5 | 0.2-1.0 |
            | Válvula de mariposa (parcialmente abierta) | 5-15 | 3-25 |
            | Válvula de bola (totalmente abierta) | 0.05-0.1 | 0.03-0.2 |
            | Válvula de bola (parcialmente abierta) | 2-8 | 1-15 |
            | Codo de 90° | 0.3-0.5 | 0.2-0.8 |
            | Tee en línea | 0.2-0.4 | 0.1-0.6 |
            | Entrada de depósito | 0.5-1.0 | 0.3-1.5 |
            | Salida a depósito | 1.0 | 0.8-1.2 |
            
            **Nota:** Los valores pueden variar según el fabricante, tamaño y condiciones de operación.
            """)
    with col2:
        results = []
        for Q in Q_range:
            area = np.pi * (D/1000/2)**2
            v = Q / area if area > 0 else 0
            K = K_coef # Use the K_coef input value
            g = 9.81
            hL = K * v**2 / (2*g)
            P2 = P1 - hL
            theta = calc.calculate_cavitation_index(P1, P2)
            results.append({
                "Caudal (L/s)": Q*1000,
                "Índice θ": theta,
                "P2 (mca)": P2,
                "Riesgo": calc.get_cavitation_risk_description(theta)
            })
        df = pd.DataFrame(results)
        st.subheader("Gráfico θ vs Caudal")
        
        # Selector de escala del eje Y
        escala_y = st.radio("Escala del eje Y", [f"Detalle (0-1)", f"Completa (0-{ordenada_max})"], key="escala_caudal")
        y_max = ordenada_max if escala_y == f"Completa (0-{ordenada_max})" else 1
        
        # Crear máscaras para los rangos de riesgo
        theta_array = np.array(df["Índice θ"])
        mask_rojo = theta_array < 0.5
        mask_amarillo = (theta_array >= 0.5) & (theta_array < 0.8)
        mask_verde = theta_array >= 0.8
        
        # Crear arrays para cada rango de riesgo
        y_rojo = np.where(mask_rojo, theta_array, np.nan)
        y_amarillo = np.where(mask_amarillo, theta_array, np.nan)
        y_verde = np.where(mask_verde, theta_array, np.nan)
        
        fig = go.Figure()
        
        # Agregar áreas pintadas para cada rango de riesgo
        # Área roja (riesgo crítico)
        fig.add_trace(go.Scatter(
            x=df["Caudal (L/s)"], y=y_rojo, mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            name='Riesgo Crítico (θ < 0.5)',
            showlegend=False,
            hoverinfo='skip'
        ))
        # Área amarilla (ahora morada, riesgo alto)
        fig.add_trace(go.Scatter(
            x=df["Caudal (L/s)"], y=y_amarillo, mode='lines',
            line=dict(color='rgba(128,0,128,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.2)',
            name='Riesgo Alto (0.5 ≤ θ < 0.8)',
            showlegend=False,
            hoverinfo='skip'
        ))
        # Área verde (riesgo bajo)
        fig.add_trace(go.Scatter(
            x=df["Caudal (L/s)"], y=y_verde, mode='lines',
            line=dict(color='rgba(0,200,0,0.3)', width=0),
            fill='tozeroy',
            fillcolor='rgba(0,200,0,0.2)',
            name='Riesgo Bajo (θ ≥ 0.8)',
            showlegend=False,
            hoverinfo='skip'
        ))
        # Agregar la curva principal encima de las áreas
        fig.add_trace(go.Scatter(
            x=df["Caudal (L/s)"], y=df["Índice θ"], mode='lines',
            line=dict(color='#d62728', width=3),
            name='Índice de Cavitación',
            hovertemplate='θ = %{y:.3f}<extra></extra>',
            hoverlabel=dict(bgcolor='#d62728', font=dict(color='white', family='Arial Black'))
        ))
        # Agregar punto puntual de cálculo en el gráfico de caudal
        fig.add_trace(go.Scatter(
            x=[Q_puntual], y=[theta_puntual], mode='markers',
            marker=dict(color='red', size=8, symbol='diamond'),
            name='Punto calculado',
            hovertemplate='Caudal = %{x:.2f} L/s<br>θ = %{y:.3f}<extra></extra>',
            hoverlabel=dict(bgcolor='red', font=dict(color='white', family='Arial Black'))
        ))
        
        # Agregar líneas horizontales de límites
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=2, 
                     annotation_text="Límite Daño (0.5)", secondary_y=False)
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange", line_width=2, 
                     annotation_text="Límite Ruido (0.8)", secondary_y=False)
        
        fig.update_layout(
            xaxis_title="Caudal (L/s)",
            yaxis_title="Índice de Cavitación θ",
            height=400,
            showlegend=False,
            hovermode="x",
            yaxis_range=[0,y_max]
        )
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="#888",
            showline=True,
            showticklabels=True,
            tickformat=".2f",
            hoverformat=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Simbología debajo del gráfico
        st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <b>Simbología:</b><br><br>
            <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin: 10px 0; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 30px; height: 3px; background-color: #d62728; border-radius: 2px;"></div>
                    <span style="font-size: 0.9em;">Índice de cavitación</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(255,0,0,0.2); border: 1px solid rgba(255,0,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo crítico (θ < 0.5)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(128,0,128,0.2); border: 1px solid rgba(128,0,128,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo alto (0.5 ≤ θ < 0.8)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: rgba(0,200,0,0.2); border: 1px solid rgba(0,200,0,0.5); border-radius: 3px;"></div>
                    <span style="font-size: 0.9em;">Riesgo bajo (θ ≥ 0.8)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.subheader("Tabla de resultados")
        st.dataframe(df, use_container_width=True)
        
        # Panel informativo sobre conceptos del índice de cavitación
        with st.expander("📚 Conceptos del Índice de Cavitación"):
            st.markdown("""
            ## **Fórmula del Índice de Cavitación:**
            ```
            θ = (P2 - Pv) / (P1 - P2)
            ```
            
            ## **Elementos de la fórmula:**
            
            ### **1. P1 (Presión de entrada)**
            - **Concepto**: Presión aguas arriba de la válvula, antes de la restricción
            - **Unidad**: mca (metros de columna de agua)
            - **Significado**: Es la presión disponible en el sistema antes de pasar por la válvula
            
            ### **2. P2 (Presión de salida)**
            - **Concepto**: Presión aguas abajo de la válvula, después de la restricción
            - **Cálculo**: `P2 = P1 - hL`
            - **Unidad**: mca
            - **Significado**: Presión que queda después de las pérdidas en la válvula
            
            ### **3. Pv (Presión de vapor)**
            - **Concepto**: Presión a la cual el agua hierve a la temperatura del fluido
            - **Dependencia**: Varía con la temperatura del agua
            - **Unidad**: mca
            - **Significado**: Presión mínima necesaria para evitar que el agua se evapore
            
            ### **4. hL (Pérdida de carga)**
            - **Concepto**: Pérdida de presión debido a la restricción en la válvula
            - **Fórmula**: `hL = K × v² / (2g)`
            - **Unidad**: mca
            - **Significado**: Energía perdida por fricción y turbulencia
            
            ### **5. K (Coeficiente de pérdidas)**
            - **Concepto**: Factor que caracteriza la resistencia de la válvula
            - **Valores típicos**: 
              - Válvula de globo: 6-10
              - Válvula de compuerta: 0.15-0.3
              - Válvula de mariposa: 0.3-0.5
            - **Significado**: Cuanto mayor K, mayor pérdida de presión
            
            ### **6. v (Velocidad del fluido)**
            - **Concepto**: Velocidad del agua al pasar por la válvula
            - **Fórmula**: `v = Q / A`
            - **Unidad**: m/s
            - **Significado**: Velocidad que determina la energía cinética
            
            ### **7. A (Área de paso)**
            - **Concepto**: Área transversal por donde pasa el fluido
            - **Fórmula**: `A = π × (D/2)²`
            - **Unidad**: m²
            - **Significado**: Sección de paso que determina la velocidad
            
            ## **Interpretación del Índice θ:**
            
            ### **θ < 0.5: RIESGO CRÍTICO** 🚨
            - **Significado**: La presión P2 está muy cerca de Pv
            - **Consecuencia**: Alto riesgo de daño por cavitación
            - **Fenómeno**: El agua puede vaporizarse y causar erosión
            
            ### **0.5 ≤ θ < 0.8: RIESGO ALTO** ⚠️
            - **Significado**: P2 está en zona de riesgo
            - **Consecuencia**: Posible ruido y vibraciones por cavitación
            - **Fenómeno**: Inicio de formación de burbujas de vapor
            
            ### **θ ≥ 0.8: RIESGO BAJO** ✅
            - **Significado**: P2 está bien por encima de Pv
            - **Consecuencia**: Operación segura sin cavitación
            - **Fenómeno**: Presión suficiente para evitar vaporización
            
            ## **Relación física:**
            - **θ alto** = Mayor margen de seguridad
            - **θ bajo** = Menor margen de seguridad
            - **θ = 0** = P2 = Pv (inicio de cavitación)
            - **θ < 0** = P2 < Pv (cavitación activa)
            """)
        
        # Panel informativo sobre qué es el índice de cavitación
        with st.expander("🔬 ¿Qué es el Índice de Cavitación?"):
            st.markdown("""
            El **Índice de Cavitación (θ)** es un parámetro fundamental en ingeniería hidráulica que evalúa el riesgo de cavitación en sistemas de flujo de fluidos.
            
            ## **¿Qué es la Cavitación?**
            
            La **cavitación** es un fenómeno físico donde se forman burbujas de vapor dentro de un líquido cuando la presión local cae por debajo de la presión de vapor del fluido. Cuando estas burbujas colapsan, generan ondas de choque que pueden causar:
            
            - **Erosión** en las superficies metálicas
            - **Ruido** y vibraciones
            - **Pérdida de eficiencia** del sistema
            - **Daños** a válvulas, bombas y tuberías
            
            ## **¿Qué es el Índice de Cavitación (θ)?**
            
            Es una **medida adimensional** que cuantifica qué tan cerca está la presión aguas abajo (P2) de la presión de vapor (Pv), comparada con la caída de presión total en el sistema.
            
            ### **Fórmula:**
            ```
            θ = (P2 - Pv) / (P1 - P2)
            ```
            
            ### **Interpretación física:**
            
            - **θ alto** = Mayor margen de seguridad (P2 está muy por encima de Pv)
            - **θ bajo** = Menor margen de seguridad (P2 se acerca a Pv)
            - **θ = 0** = P2 = Pv (inicio de cavitación)
            - **θ < 0** = P2 < Pv (cavitación activa)
            
            ## **Rangos de Riesgo:**
            
            ### **🚨 θ < 0.5: RIESGO CRÍTICO**
            - La presión P2 está muy cerca de Pv
            - Alto riesgo de daño por cavitación
            - El agua puede vaporizarse y causar erosión severa
            
            ### **⚠️ 0.5 ≤ θ < 0.8: RIESGO ALTO**
            - P2 está en zona de riesgo
            - Posible ruido y vibraciones por cavitación
            - Inicio de formación de burbujas de vapor
            
            ### **✅ θ ≥ 0.8: RIESGO BAJO**
            - P2 está bien por encima de Pv
            - Operación segura sin cavitación
            - Presión suficiente para evitar vaporización
            
            ## **Aplicaciones prácticas:**
            
            1. **Diseño de válvulas**: Seleccionar válvulas que mantengan θ ≥ 0.8
            2. **Operación de sistemas**: Monitorear θ para evitar cavitación
            3. **Mantenimiento**: Identificar condiciones de riesgo antes de que ocurran daños
            4. **Optimización**: Ajustar parámetros para mejorar el margen de seguridad
            
            ## **Factores que afectan θ:**
            
            - **Presión de entrada (P1)**: Mayor P1 = mayor θ
            - **Coeficiente K**: Mayor K = menor θ
            - **Caudal (Q)**: Mayor Q = menor θ
            - **Diámetro (D)**: Mayor D = mayor θ
            - **Temperatura**: Mayor temperatura = menor θ (por mayor Pv)
            
            El índice de cavitación es esencial para el diseño y operación segura de sistemas hidráulicos, permitiendo prevenir daños costosos y mantener la eficiencia del sistema.
            """)
        
        # Panel informativo sobre presión de vapor y cavitación
        with st.expander("💧 Presión de Vapor y Cavitación"):
            st.markdown("""
            La **presión de vapor (Pv)** es un concepto fundamental en la cavitación.
            
            ## **¿Qué es la Presión de Vapor (Pv)?**
            
            La **presión de vapor** es la presión a la cual un líquido comienza a hervir (evaporarse) a una temperatura específica. Es la presión mínima necesaria para mantener el líquido en estado líquido.
            
            ### **Características importantes:**
            
            - **Depende de la temperatura**: A mayor temperatura, mayor Pv
            - **Es específica del fluido**: Cada líquido tiene su propia Pv
            - **Se mide en unidades de presión**: mca, Pa, bar, etc.
            
            ## **¿Cómo interviene en la Cavitación?**
            
            ### **1. Mecanismo de la Cavitación:**
            
            ```
            Si P2 < Pv → El agua se evapora → Se forman burbujas de vapor
            ```
            
            - Cuando la **presión local (P2)** cae por debajo de la **presión de vapor (Pv)**
            - El agua cambia de estado líquido a vapor
            - Se forman **burbujas de vapor** dentro del fluido
            
            ### **2. Efecto en el Índice de Cavitación:**
            
            En la fórmula: `θ = (P2 - Pv) / (P1 - P2)`
            
            - **P2 - Pv**: Es el **margen de seguridad**
            - Si P2 se acerca a Pv → θ disminuye → Mayor riesgo
            - Si P2 está muy por encima de Pv → θ aumenta → Menor riesgo
            
            ### **3. Relación con la Temperatura:**
            
            | Temperatura (°C) | Pv (mca) |
            |:----------------:|:--------:|
            | 0 | 0.06 |
            | 20 | 0.25 |
            | 40 | 0.76 |
            | 60 | 2.03 |
            | 80 | 4.85 |
            | 100 | 10.33 |
            
            **A mayor temperatura → Mayor Pv → Menor margen de seguridad**
            
            ## **¿Por qué es Crítica?**
            
            ### **1. Punto de Inicio de Cavitación:**
            - Cuando P2 = Pv → θ = 0 (inicio de cavitación)
            - Cuando P2 < Pv → θ < 0 (cavitación activa)
            
            ### **2. Diseño de Sistemas:**
            - Debe asegurarse que P2 > Pv en todo momento
            - Mayor margen = Mayor seguridad
            
            ### **3. Operación:**
            - Monitorear temperatura del fluido
            - Ajustar presiones según la temperatura
            - Considerar Pv en el diseño de válvulas
            
            ## **Factores que Afectan Pv:**
            
            1. **Temperatura del fluido** (principal factor)
            2. **Composición del fluido** (agua pura vs. soluciones)
            3. **Presión atmosférica** (en sistemas abiertos)
            4. **Altitud** (afecta la presión atmosférica)
            
            La presión de vapor es el **umbral crítico** que determina cuándo comienza la cavitación, por eso es fundamental en el análisis del índice de cavitación.
            """)
    with col4:
        pass 

# --- Pie de página personalizado ---
st.markdown('---')
st.markdown('<div style="text-align:center; font-size:16px; margin-top:20px; color:#555; font-family:Segoe UI;">Desarrollador: Patricio Sarmiento Reinoso<br>Maestría HS- UDA-2025</div>', unsafe_allow_html=True) 