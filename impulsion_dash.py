import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
from impulsion_design import ImpulsionPotable

# --- Función para obtener texto de criterios ---
def get_criterio_texto():
    return """
# Criterios de Diseño - Impulsión

## **Velocidad de Diseño**
- **Rango recomendado:** 0.5 - 2.0 m/s
- **Velocidad óptima:** 1.0 - 1.5 m/s
- **Límite máximo:** 2.5 m/s (para evitar erosión)

## **Presión de Operación**
- **Presión mínima:** 10 m H₂O (1 bar)
- **Presión máxima:** Según material de tubería
- **Factor de seguridad:** 1.2 - 1.5

## **Pérdidas de Carga**
- **Máximo 20%** de la altura total
- **Optimización:** Reducir accesorios
- **Verificación:** NPSH disponible

## **Materiales y PN**
- **Hierro Dúctil:** PN según ISO 2531
- **Acero:** PN según especificaciones
- **PEAD:** PN según ISO 4427
- **PVC:** PN según ISO 4422
"""

# --- Panel de datos de entrada ---
def get_input_panel():
    materiales = [
        ("Hierro Dúctil", "Hierro Dúctil"),
        ("Acero", "Acero"),
        ("PEAD", "PEAD"),
        ("PVC", "PVC")
    ]
    campos = [
        ("Material:", dcc.Dropdown(
            id='input-material',
            options=[{'label': m, 'value': v} for m, v in materiales],
            value='Hierro Dúctil',
            clearable=False,
            style={'width': '180px', 'marginBottom': '6px'}
        )),
        ("Caudal Q (L/s):", dcc.Input(id='input-q', type='number', value=11, step=0.1, style={'width': '120px'})),
        ("Longitud L (m):", dcc.Input(id='input-l', type='number', value=2000, step=1, style={'width': '120px'})),
        ("Altura h₀ (m):", dcc.Input(id='input-h0', type='number', value=190, step=1, style={'width': '120px'})),
        ("PN (bar):", dcc.Input(id='input-pn', type='number', value=40, step=1, style={'width': '120px'})),
    ]
    
    # Configuración de rango de diámetros
    config_rango = html.Div([
        html.Label("Rango de diámetros:", style={'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginBottom': '10px', 'display': 'block', 'textAlign': 'center', 'fontWeight': 'bold'}),
        html.Div([
            html.Label("Mín (mm):", style={'fontFamily': 'Segoe UI', 'fontSize': '14px', 'color': '#333', 'marginRight': '10px'}),
            dcc.Input(id='input-dmin', type='number', value=0, step=10, min=0, style={'width': '80px', 'marginRight': '20px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '8px'}),
        html.Div([
            html.Label("Máx (mm):", style={'fontFamily': 'Segoe UI', 'fontSize': '14px', 'color': '#333', 'marginRight': '10px'}),
            dcc.Input(id='input-dmax', type='number', value=500, step=10, min=0, style={'width': '80px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'})
    filas = [
        html.Div([
            html.Label(label, style={'flex': '1', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginRight': '10px', 'textAlign': 'right'}),
            html.Div(comp, style={'flex': '1', 'textAlign': 'left'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'marginBottom': '12px'})
        for label, comp in campos
    ]
    
    # Botones principales
    botones_principales = html.Div([
        html.Button('Calcular', id='btn-calc', n_clicks=0, style={'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '120px', 'background': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '6px'}),
        html.Button('Cerrar', id='btn-close-panel', n_clicks=0, style={'marginTop': '12px', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'width': '120px', 'background': '#eee', 'color': '#333', 'border': 'none', 'borderRadius': '6px'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'})
    
    # Diámetro a analizar
    diametro_analizar = html.Div([
        html.Label("Diámetro a analizar (mm):", style={'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginBottom': '8px', 'display': 'block'}),
        dcc.Input(id='input-danalizar', type='number', value='', step=1, style={'width': '120px', 'marginBottom': '15px'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'})
    
    # Botones de punto (uno junto al otro)
    botones_punto = html.Div([
        html.Button('Calcular punto', id='btn-calc-punto', n_clicks=0, style={'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#28a745', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'marginRight': '10px'}),
        html.Button('Limpiar punto', id='btn-clear-punto', n_clicks=0, style={'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#dc3545', 'color': 'white', 'border': 'none', 'borderRadius': '6px'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'})
    
    # Botón de análisis
    boton_analisis = html.Div([
        html.Button('Análisis', id='btn-analisis', n_clicks=0, style={'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#6f42c1', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'opacity': '0.5'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'})
    
    # Configuración de gráficos
    config_graficos = html.Div([
        html.Label("Configuración de gráficos:", style={'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginBottom': '10px', 'display': 'block', 'textAlign': 'center', 'fontWeight': 'bold'}),
        html.Div([
            html.Label("Ancho (%):", style={'fontFamily': 'Segoe UI', 'fontSize': '14px', 'color': '#333', 'marginRight': '10px'}),
            dcc.Input(id='input-ancho', type='number', value=60, step=5, min=30, max=90, style={'width': '80px', 'marginRight': '20px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '8px'}),
        html.Div([
            html.Label("Alto (px):", style={'fontFamily': 'Segoe UI', 'fontSize': '14px', 'color': '#333', 'marginRight': '10px'}),
            dcc.Input(id='input-alto', type='number', value=700, step=50, min=400, max=1000, style={'width': '80px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ], style={'textAlign': 'center'})
    
    return html.Div([
        html.H3('Datos de Impulsión', style={'fontFamily': 'Segoe UI', 'fontWeight': 'bold', 'fontSize': '22px', 'color': '#007bff', 'marginBottom': '18px', 'textAlign': 'center', 'letterSpacing': '1px'}),
        *filas,
        config_rango,
        botones_principales,
        diametro_analizar,
        botones_punto,
        boton_analisis,
        config_graficos,
    ], id='input-panel-content', style={'padding': '24px', 'fontFamily': 'Segoe UI'})

# --- Layout principal ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('IMPULSION PIPE DESIGN', style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'fontWeight': 'bold', 'fontSize': '2.5rem', 'color': '#007bff', 'marginBottom': '10px', 'marginTop': '10px', 'letterSpacing': '2px'}),
    dcc.Store(id='panel-open', data=False),
    dcc.Store(id='criterio-open', data=False),
    dcc.Store(id='punto-analizado', data=None),
    dcc.Store(id='analisis-data', data=None),
    html.Div(id='graph-script-container'),
    html.Div(id='copy-notification'),
    # Script simple para el panel
    html.Script('''
        // Script básico para el panel
        console.log('Panel script loaded');
    '''),
    html.Div([
        html.Div('DATOS', id='tab-datos', n_clicks=0, style={
            'position': 'absolute', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
            'background': 'rgba(0,123,255,0.25)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }),
        html.Div(get_input_panel(), id='input-panel', style={
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.6)', 'backdropFilter': 'blur(20px)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(100%)',
            'borderLeft': '2px solid #007bff',
        })
    ], style={'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'zIndex': 20}),
    html.Div([
        html.Div('CRITERIOS', id='tab-criterio', n_clicks=0, style={
            'position': 'absolute', 'bottom': '40px', 'right': 0, 'width': '36px', 'height': '100px',
            'background': 'rgba(0,123,255,0.18)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }),
        html.Div(dcc.Markdown(get_criterio_texto(), style={'margin': 0, 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333'}), id='criterio-box', style={
            'position': 'fixed', 'bottom': '30px', 'right': '50px', 'background': 'rgba(255,255,255,0.97)',
            'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'zIndex': 30,
            'transition': 'transform 0.4s', 'transform': 'translateX(120%)',
            'borderLeft': '2px solid #007bff',
        })
    ], style={'position': 'fixed', 'bottom': 0, 'right': 0, 'height': '100vh', 'zIndex': 20}),
    html.Div([
        dcc.Loading(
            dcc.Tabs(id='tabs-graficos', value='tab-1', children=[
                dcc.Tab(label='Análisis Hidráulico', value='tab-1', children=html.Div(id='tab-content-1')),
                dcc.Tab(label='Presión y Límites', value='tab-2', children=html.Div(id='tab-content-2')),
            ]),
            type='circle',
            color='#007bff',
            style={'minHeight': '900px'}
        ),
    ], style={'padding': '30px', 'fontFamily': 'Segoe UI', 'position': 'relative', 'minHeight': '100vh', 'background': '#f8f9fa'})
], style={'width': '100vw', 'height': '100vh', 'overflow': 'hidden', 'fontFamily': 'Segoe UI', 'background': '#f4f7fb'})

# --- Callbacks para paneles laterales ---
@app.callback(
    [Output('input-panel', 'style'),
     Output('panel-open', 'data')],
    [Input('tab-datos', 'n_clicks'),
     Input('btn-close-panel', 'n_clicks')],
    [State('panel-open', 'data')]
)
def toggle_input_panel(tab_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'tab-datos':
        new_style = {
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.6)', 'backdropFilter': 'blur(20px)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(0%)' if not is_open else 'translateX(100%)',
            'borderLeft': '2px solid #007bff',
        }
        return new_style, not is_open
    elif button_id == 'btn-close-panel':
        new_style = {
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.6)', 'backdropFilter': 'blur(20px)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(100%)',
            'borderLeft': '2px solid #007bff',
        }
        return new_style, False
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('criterio-box', 'style'),
     Output('criterio-open', 'data')],
    [Input('tab-criterio', 'n_clicks')],
    [State('criterio-open', 'data')]
)
def toggle_criterio_panel(tab_clicks, is_open):
    if not tab_clicks:
        return dash.no_update, dash.no_update
    
    new_style = {
        'position': 'fixed', 'bottom': '30px', 'right': '50px', 'background': 'rgba(255,255,255,0.97)',
        'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
        'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'zIndex': 30,
        'transition': 'transform 0.4s', 'transform': 'translateX(0%)' if not is_open else 'translateX(120%)',
        'borderLeft': '2px solid #007bff',
    }
    return new_style, not is_open

# --- Función para generar gráficos de impulsión ---
def generate_impulsion_graphs(material, Q_lps, L, h0, pn_bar, punto_analizado, dmin=None, dmax=None):
    """Genera todos los gráficos de análisis de impulsión"""
    
    if Q_lps is None or L is None or h0 is None or pn_bar is None:
        Q_lps = 11
        L = 2000
        h0 = 190
        pn_bar = 40
        material = 'Hierro Dúctil'
    
    # Valores por defecto para el rango de diámetros
    if dmin is None:
        dmin = 0
    if dmax is None:
        dmax = 500
    
    Q = Q_lps / 1000
    imp = ImpulsionPotable(Q=Q, L=L, h0=h0, pn_bar=pn_bar)
    
    # Función para calcular límites Y dinámicos basados en el rango de diámetros
    def calcular_limites_y(datos_por_material, dmin, dmax):
        """Calcula los límites Y basándose en los datos del rango especificado"""
        valores_en_rango = []
        
        # Filtrar datos que están en el rango especificado
        for diam, valores in zip(imp.D, zip(*datos_por_material)):
            if dmin <= diam <= dmax:
                valores_en_rango.extend(valores)
        
        if valores_en_rango:
            min_val = min(valores_en_rango)
            max_val = max(valores_en_rango)
            # Agregar un margen del 10% para mejor visualización
            margen = (max_val - min_val) * 0.1
            return max(0, min_val - margen), max_val + margen
        else:
            return 0, 100  # Valores por defecto si no hay datos
    
    # Factores de límites por material
    def get_limite_factors(material):
        if material == 'Hierro Dúctil':
            return {'Operacion': 1.0, 'Transitorio': 1.2, 'Prueba': (1.2, 5)}
        elif material == 'Acero':
            return {'Operacion': 1.0, 'Transitorio': 1.2, 'Prueba': (1.5, 0)}
        elif material == 'PEAD':
            return {'Operacion': 1.0, 'Transitorio': 1.25, 'Prueba': (1.5, 0)}
        elif material == 'PVC':
            return {'Operacion': 1.0, 'Transitorio': 1.25, 'Prueba': (1.5, 0)}
        else:
            return {'Operacion': 1.0, 'Transitorio': 1.2, 'Prueba': (1.2, 5)}
    
    def calcular_limites(material, pn_bar):
        bar_to_m = 10.197
        factors = get_limite_factors(material)
        limites = {}
        limites['Operación'] = factors['Operacion'] * pn_bar * bar_to_m
        limites['Transitorio'] = factors['Transitorio'] * pn_bar * bar_to_m
        if isinstance(factors['Prueba'], tuple):
            limites['Prueba'] = factors['Prueba'][0] * pn_bar * bar_to_m + factors['Prueba'][1] * bar_to_m
        else:
            limites['Prueba'] = factors['Prueba'] * pn_bar * bar_to_m
        return limites
    
    # Crear gráficos individuales
    # 1. Gráfico de Pérdida de Carga
    fig1 = go.Figure()
    datos_hl = []
    for m in imp.C:
        hl = [imp._hl(d)[m] for d in imp.D]
        datos_hl.append(hl)
        fig1.add_trace(go.Scatter(x=imp.D, y=hl, mode='lines', name=f'HL {m}', line=dict(width=3)))
    
    # Calcular límites Y dinámicos para pérdida de carga
    ymin_hl, ymax_hl = calcular_limites_y(datos_hl, dmin, dmax)
    
    fig1.update_layout(
        title='Pérdida de Carga (m H₂O)',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Pérdida de Carga (m)',
        height=600, showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(range=[dmin, dmax], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        yaxis=dict(range=[ymin_hl, ymax_hl], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Segoe UI'},
        hovermode='x',
        shapes=[],
    )
    # Marco dinámico siempre detrás
    fig1.add_shape(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1,
                   line=dict(color='#cccccc', width=1), fillcolor='rgba(0,0,0,0)', layer='below')
    
    # 2. Gráfico de Velocidad
    fig2 = go.Figure()
    datos_vel = []
    for m in imp.C:
        vel = [imp._vel(d)[m] for d in imp.D]
        datos_vel.append(vel)
        fig2.add_trace(go.Scatter(x=imp.D, y=vel, mode='lines', name=f'Vel {m}', line=dict(width=3)))
    
    # Calcular límites Y dinámicos para velocidad
    ymin_vel, ymax_vel = calcular_limites_y(datos_vel, dmin, dmax)
    
    fig2.update_layout(
        title='Velocidad (m/s)',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Velocidad (m/s)',
        height=600, showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(range=[dmin, dmax], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        yaxis=dict(range=[ymin_vel, ymax_vel], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Segoe UI'},
        hovermode='x',
        shapes=[],
    )
    fig2.add_shape(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1,
                   line=dict(color='#cccccc', width=1), fillcolor='rgba(0,0,0,0)', layer='below')
    
    # 3. Gráfico de Sobrepresión
    fig3 = go.Figure()
    datos_op = []
    for m in imp.C:
        op = [imp._op(d)[m] for d in imp.D]
        datos_op.append(op)
        fig3.add_trace(go.Scatter(x=imp.D, y=op, mode='lines', name=f'OP {m}', line=dict(width=3)))
    
    # Calcular límites Y dinámicos para sobrepresión
    ymin_op, ymax_op = calcular_limites_y(datos_op, dmin, dmax)
    
    # Límites solo para el material seleccionado
    limites = calcular_limites(material, pn_bar)
    colores_limites = {'Prueba': 'red', 'Transitorio': 'blue', 'Operación': 'green'}
    for label, val in limites.items():
        fig3.add_hline(y=val, line_dash='dash', line_color=colores_limites.get(label, 'gray'),
                      annotation_text=f"{label} {material}: {val:.1f} m H₂O",
                      annotation_position="top left", layer='above')
    
    fig3.update_layout(
        title='Sobrepresión (m H₂O)',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Sobrepresión (m H₂O)',
        height=600, showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(range=[dmin, dmax], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        yaxis=dict(range=[ymin_op, ymax_op], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Segoe UI'},
        hovermode='x',
        shapes=[],
    )
    fig3.add_shape(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1,
                   line=dict(color='#cccccc', width=1), fillcolor='rgba(0,0,0,0)', layer='below')
    
    # 4. Gráfico de Total
    fig4 = go.Figure()
    datos_tt = []
    for m in imp.C:
        tt = [h + imp.h0 for h in [imp._hl(d)[m] for d in imp.D]]
        datos_tt.append(tt)
        fig4.add_trace(go.Scatter(x=imp.D, y=tt, mode='lines', name=f'TT {m}', line=dict(width=3)))
    
    # Calcular límites Y dinámicos para total
    ymin_tt, ymax_tt = calcular_limites_y(datos_tt, dmin, dmax)
    
    # Límites en el gráfico de total también
    for label, val in limites.items():
        fig4.add_hline(y=val, line_dash='dash', line_color=colores_limites.get(label, 'gray'),
                      annotation_text=f"{label} {material}: {val:.1f} m H₂O",
                      annotation_position="top right", layer='above')
    
    fig4.update_layout(
        title='Total = HL + h₀ (m H₂O)',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Total (m H₂O)',
        height=600, showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(range=[dmin, dmax], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        yaxis=dict(range=[ymin_tt, ymax_tt], showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Segoe UI'},
        hovermode='x',
        shapes=[],
    )
    fig4.add_shape(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1,
                   line=dict(color='#cccccc', width=1), fillcolor='rgba(0,0,0,0)', layer='below')
    
    # Si se ha solicitado un punto de análisis, agregar marcadores
    punto_data = None
    if punto_analizado is not None:
        try:
            d_analizar_mm = float(punto_analizado)
            d_analizar_m = d_analizar_mm / 1000
            
            # Interpolar valores para el diámetro analizado SOLO para el material seleccionado
            diam = np.array(imp.D)
            
            # Obtener valores solo para el material seleccionado
            hl_values = [imp._hl(d)[material] for d in imp.D]
            vel_values = [imp._vel(d)[material] for d in imp.D]
            op_values = [imp._op(d)[material] for d in imp.D]
            tt_values = [h + imp.h0 for h in hl_values]
            
            # Interpolar valores para el diámetro analizado
            hl_interp = np.interp(d_analizar_mm, diam, hl_values)
            vel_interp = np.interp(d_analizar_mm, diam, vel_values)
            op_interp = np.interp(d_analizar_mm, diam, op_values)
            tt_interp = np.interp(d_analizar_mm, diam, tt_values)
            
            # Agregar marcadores solo para el material seleccionado
            fig1.add_trace(go.Scatter(x=[d_analizar_mm], y=[hl_interp], mode='markers', 
                                   marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig2.add_trace(go.Scatter(x=[d_analizar_mm], y=[vel_interp], mode='markers', 
                                   marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig3.add_trace(go.Scatter(x=[d_analizar_mm], y=[op_interp], mode='markers', 
                                   marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig4.add_trace(go.Scatter(x=[d_analizar_mm], y=[tt_interp], mode='markers', 
                                   marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            
            # Devolver datos del punto analizado para el material seleccionado
            punto_data = {
                'Diámetro': f'{d_analizar_mm:.0f} mm',
                'Pérdida de Carga': f'{hl_interp:.3f} m',
                'Velocidad': f'{vel_interp:.2f} m/s',
                'Sobrepresión': f'{op_interp:.2f} m',
                'Total': f'{tt_interp:.2f} m'
            }
            
        except Exception as e:
            punto_data = {'Error': f'Error al calcular el punto: {str(e)}'}
    
    return fig1, fig2, fig3, fig4, punto_data

# --- Callbacks para generar gráficos ---
@app.callback(
    [Output('tab-content-1', 'children'),
     Output('tab-content-2', 'children')],
    [Input('btn-calc', 'n_clicks'),
     Input('punto-analizado', 'data'),
     Input('analisis-data', 'data')],
    [Input('input-material', 'value'),
     Input('input-q', 'value'),
     Input('input-l', 'value'),
     Input('input-h0', 'value'),
     Input('input-pn', 'value'),
     Input('input-ancho', 'value'),
     Input('input-alto', 'value'),
     Input('input-dmin', 'value'),
     Input('input-dmax', 'value')]
)
def update_graphs(n_clicks, punto_analizado, analisis_data, material, Q_lps, L, h0, pn_bar, ancho, alto, dmin, dmax):
    # Solo ejecutar si se hizo clic en calcular, si hay un punto analizado, o si hay datos de análisis
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div([
            html.H3('Haz clic en "Calcular" para generar los gráficos', 
                   style={'textAlign': 'center', 'color': '#666', 'marginTop': '100px'})
        ]), html.Div([
            html.H3('Haz clic en "Calcular" para generar los gráficos', 
                   style={'textAlign': 'center', 'color': '#666', 'marginTop': '100px'})
        ])
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id not in ['btn-calc', 'punto-analizado', 'analisis-data']:
        return dash.no_update, dash.no_update
    
    # Valores por defecto para tamaño de gráficos
    ancho = ancho or 60
    alto = alto or 700
    
    try:
        fig1, fig2, fig3, fig4, punto_data = generate_impulsion_graphs(material, Q_lps, L, h0, pn_bar, punto_analizado, dmin, dmax)
        
        def box_punto(label, punto_data, key, unidad):
            if punto_data is not None and key in punto_data:
                return html.Div([
                    html.Span('Punto analizado: ', style={'fontWeight': 'bold', 'color': '#007bff'}),
                    html.Span(f"D = {punto_data['Diámetro']}, {label} = {punto_data[key]}", style={'color': '#222'}),
                ], style={'background': '#f8f9fa', 'border': '1px solid #dee2e6', 'borderRadius': '7px', 'padding': '7px 14px', 'margin': '10px auto 0 auto', 'display': 'block', 'fontSize': '15px', 'textAlign': 'center', 'width': 'fit-content'})
            return None
        
        def crear_analisis(analisis_data, tipo):
            print(f"DEBUG: crear_analisis llamado con tipo={tipo}, analisis_data={analisis_data}")
            if analisis_data is None or 'error' in analisis_data:
                print(f"DEBUG: analisis_data es None o tiene error")
                return None
            
            if tipo == 'velocidad' and 'velocidad' in analisis_data:
                data = analisis_data['velocidad']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            elif tipo == 'perdidas' and 'perdidas' in analisis_data:
                data = analisis_data['perdidas']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            elif tipo == 'presion' and 'presion' in analisis_data:
                data = analisis_data['presion']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            elif tipo == 'total' and 'total' in analisis_data:
                data = analisis_data['total']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            else:
                return None
            
            return html.Div([
                html.H4('📊 Análisis Automático', style={'color': color, 'fontSize': '16px', 'marginBottom': '8px', 'textAlign': 'center'}),
                html.P(data['observacion'], style={'fontSize': '14px', 'marginBottom': '6px', 'textAlign': 'center'}),
                html.P(f"💡 Recomendación: {data['recomendacion']}", style={'fontSize': '14px', 'fontStyle': 'italic', 'textAlign': 'center', 'color': '#666'})
            ], style={'background': '#f8f9fa', 'border': f'2px solid {color}', 'borderRadius': '8px', 'padding': '12px', 'margin': '10px auto 0 auto', 'width': '100%', 'maxWidth': '400px'})
        
        # Si el punto_data contiene un error, mostrarlo
        if punto_data is not None and 'Error' in punto_data:
            error_msg = html.Div([
                html.H3('Error en el cálculo', style={'color': 'red', 'textAlign': 'center'}),
                html.P(punto_data['Error'], style={'textAlign': 'center', 'color': '#666'})
            ])
            return error_msg, error_msg
        
        # Contenido de las pestañas
        # Pestaña 1: Pérdida de Carga y Velocidad
        tab1_content = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig1, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'perdida_carga'}}, style={'width': f'{ancho}%', 'height': f'{alto}px', 'display': 'inline-block'}),
                    box_punto('Pérdida de Carga', punto_data, 'Pérdida de Carga', 'm'),
                    crear_analisis(analisis_data, 'perdidas')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    dcc.Graph(figure=fig2, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'velocidad'}}, style={'width': f'{ancho}%', 'height': f'{alto}px', 'display': 'inline-block'}),
                    box_punto('Velocidad', punto_data, 'Velocidad', 'm/s'),
                    crear_analisis(analisis_data, 'velocidad')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'width': '100%', 'textAlign': 'center'})
        ])
        
        # Pestaña 2: Sobrepresión y Total
        tab2_content = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig3, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'sobrepresion'}}, style={'width': f'{ancho}%', 'height': f'{alto}px', 'display': 'inline-block'}),
                    box_punto('Sobrepresión', punto_data, 'Sobrepresión', 'm'),
                    crear_analisis(analisis_data, 'presion')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    dcc.Graph(figure=fig4, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'total'}}, style={'width': f'{ancho}%', 'height': f'{alto}px', 'display': 'inline-block'}),
                    box_punto('Total', punto_data, 'Total', 'm'),
                    crear_analisis(analisis_data, 'total')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'width': '100%', 'textAlign': 'center'})
        ])
        
        return tab1_content, tab2_content
        
    except Exception as e:
        error_msg = html.Div([
            html.H3('Error en el cálculo', style={'color': 'red', 'textAlign': 'center'}),
            html.P(f'Error: {str(e)}', style={'textAlign': 'center', 'color': '#666'})
        ])
        return error_msg, error_msg

# --- Callbacks adicionales ---
@app.callback(
    [Output('btn-analisis', 'style'),
     Output('btn-analisis', 'disabled')],
    [Input('input-danalizar', 'value')]
)
def toggle_analisis_button(d_analizar):
    if d_analizar is not None and d_analizar != '':
        return {
            'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 
            'width': '140px', 'background': '#6f42c1', 'color': 'white', 
            'border': 'none', 'borderRadius': '6px', 'marginLeft': '10px'
        }, False
    else:
        return {
            'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 
            'width': '140px', 'background': '#6f42c1', 'color': 'white', 
            'border': 'none', 'borderRadius': '6px', 'marginLeft': '10px', 'opacity': '0.5'
        }, True

@app.callback(
    Output('punto-analizado', 'data'),
    [Input('btn-calc-punto', 'n_clicks'), Input('btn-clear-punto', 'n_clicks')],
    [State('input-danalizar', 'value')]
)
def set_punto_analizado(n_calc, n_clear, d_analizar):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-clear-punto':
        return None
    if button_id == 'btn-calc-punto' and d_analizar is not None:
        try:
            d_analizar = float(d_analizar)
            return d_analizar
        except:
            return None
    return None

@app.callback(
    Output('copy-notification', 'children'),
    [Input('copy-notification', 'children')]
)
def update_copy_notification(children):
    return children

# Callback para generar análisis automático
@app.callback(
    Output('analisis-data', 'data'),
    [Input('btn-analisis', 'n_clicks')],
    [State('input-danalizar', 'value'),
     State('input-material', 'value'),
     State('input-q', 'value'),
     State('input-l', 'value'),
     State('input-h0', 'value'),
     State('input-pn', 'value'),
     State('punto-analizado', 'data')]
)
def generate_analysis(n_clicks, d_analizar, material, Q_lps, L, h0, pn_bar, punto_analizado):
    print(f"DEBUG: generate_analysis llamado con n_clicks={n_clicks}, d_analizar={d_analizar}, material={material}, punto_analizado={punto_analizado}")
    
    # Solo ejecutar si realmente se hizo clic en el botón
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id != 'btn-analisis':
        return None
    
    # Verificar que tenemos todos los datos necesarios
    if not n_clicks or d_analizar is None or d_analizar == '' or punto_analizado is None:
        print("DEBUG: Retornando None - no hay clicks, d_analizar es None/vacío, o no hay punto analizado")
        return None
    
    # Verificar que el material no sea None
    if material is None:
        material = 'Hierro Dúctil'  # Valor por defecto
        print("DEBUG: Material era None, usando valor por defecto: Hierro Dúctil")
    
    try:
        # Usar los datos del punto analizado que ya fueron calculados
        d_analizar_mm = float(punto_analizado)
        d_analizar_m = d_analizar_mm / 1000
        Q = Q_lps / 1000
        imp = ImpulsionPotable(Q=Q, L=L, h0=h0, pn_bar=pn_bar)
        
        # Asegurar que el material esté definido
        if material is None or material == '':
            material = 'Hierro Dúctil'
            print("DEBUG: Material no definido, usando Hierro Dúctil")
        
        # Calcular valores usando interpolación (igual que en los gráficos)
        diam = np.array(imp.D)
        
        # Obtener valores para el material seleccionado
        hl_values = [imp._hl(d)[material] for d in imp.D]
        vel_values = [imp._vel(d)[material] for d in imp.D]
        op_values = [imp._op(d)[material] for d in imp.D]
        tt_values = [h + imp.h0 for h in hl_values]
        
        # Interpolar valores para el diámetro analizado
        vel = np.interp(d_analizar_mm, diam, vel_values)
        hl = np.interp(d_analizar_mm, diam, hl_values)
        op = np.interp(d_analizar_mm, diam, op_values)
        tt = np.interp(d_analizar_mm, diam, tt_values)
        
        print(f"DEBUG: Valores interpolados - vel={vel:.3f}, hl={hl:.3f}, op={op:.3f}, tt={tt:.3f}")
        
        # Calcular límites de presión
        def get_limite_factors(material):
            if material == 'Hierro Dúctil':
                return {'Operacion': 1.0, 'Transitorio': 1.2, 'Prueba': (1.2, 5)}
            elif material == 'Acero':
                return {'Operacion': 1.0, 'Transitorio': 1.2, 'Prueba': (1.5, 0)}
            elif material == 'PEAD':
                return {'Operacion': 1.0, 'Transitorio': 1.25, 'Prueba': (1.5, 0)}
            elif material == 'PVC':
                return {'Operacion': 1.0, 'Transitorio': 1.25, 'Prueba': (1.5, 0)}
            else:
                return {'Operacion': 1.0, 'Transitorio': 1.2, 'Prueba': (1.2, 5)}
        
        def calcular_limites(material, pn_bar):
            bar_to_m = 10.197
            factors = get_limite_factors(material)
            limites = {}
            limites['Operación'] = factors['Operacion'] * pn_bar * bar_to_m
            limites['Transitorio'] = factors['Transitorio'] * pn_bar * bar_to_m
            if isinstance(factors['Prueba'], tuple):
                limites['Prueba'] = factors['Prueba'][0] * pn_bar * bar_to_m + factors['Prueba'][1] * bar_to_m
            else:
                limites['Prueba'] = factors['Prueba'] * pn_bar * bar_to_m
            return limites
        
        limites = calcular_limites(material, pn_bar)
        
        # Análisis de cada aspecto usando los valores correctos
        analisis = {
            'velocidad': {
                'valor': vel,
                'limite': 2.0,
                'estado': 'OK' if vel <= 2.0 else 'ALTO',
                'observacion': f'La velocidad de {vel:.2f} m/s está {"dentro" if vel <= 2.0 else "por encima"} del límite recomendado de 2.0 m/s.',
                'recomendacion': 'El diseño es adecuado.' if vel <= 2.0 else f'Velocidad alta ({vel:.2f} m/s). Optimizar: aumentar diámetro a {d_analizar_m*1000*1.3:.0f}-{d_analizar_m*1000*1.6:.0f} mm para reducir a 1.0-1.5 m/s.'
            },
            'perdidas': {
                'valor': hl,
                'limite': h0 * 0.2,  # 20% de la altura total
                'estado': 'OK' if hl <= h0 * 0.2 else 'ALTO',
                'observacion': f'Las pérdidas de {hl:.3f} m representan {(hl/tt)*100:.1f}% de la altura total.',
                'recomendacion': 'Las pérdidas son aceptables.' if hl <= h0 * 0.2 else f'Las pérdidas son altas ({(hl/tt)*100:.1f}%). Optimizar: aumentar diámetro a {d_analizar_m*1000*1.2:.0f}-{d_analizar_m*1000*1.5:.0f} mm, reducir accesorios, o usar material más liso.'
            },
            'presion': {
                'valor': op,
                'limite': limites['Operación'],
                'estado': 'OK' if op <= limites['Operación'] else 'ALTO',
                'posicion_relativa': 'El punto está por debajo del límite de operación.' if op <= limites['Operación'] else 'El punto está por encima del límite de operación.',
                'observacion': f'La sobrepresión es {op:.2f} m vs {limites["Operación"]:.1f} m límite de operación. {("El punto está por debajo del límite de operación." if op <= limites["Operación"] else "El punto está por encima del límite de operación.")}',
                'recomendacion': 'La presión es adecuada.' if op <= limites['Operación'] else f'Presión alta ({op:.2f} m). Optimizar: aumentar diámetro a {d_analizar_m*1000*1.4:.0f}-{d_analizar_m*1000*1.8:.0f} mm, o reducir altura a {h0*0.8:.1f} m.'
            },
            'total': {
                'valor': tt,
                'limite': limites['Operación'],
                'estado': 'OK' if tt <= limites['Operación'] else 'ALTO',
                'posicion_relativa': 'El punto está por debajo del límite de operación.' if tt <= limites['Operación'] else 'El punto está por encima del límite de operación.',
                'observacion': f'La altura total es {tt:.2f} m vs {limites["Operación"]:.1f} m límite de operación. {("El punto está por debajo del límite de operación." if tt <= limites["Operación"] else "El punto está por encima del límite de operación.")}',
                'recomendacion': 'La altura total es adecuada.' if tt <= limites['Operación'] else f'Altura total alta ({tt:.2f} m). Optimizar: aumentar diámetro a {d_analizar_m*1000*1.3:.0f}-{d_analizar_m*1000*1.7:.0f} mm, o reducir altura a {h0*0.8:.1f} m.'
            }
        }
        
        return analisis
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052) 