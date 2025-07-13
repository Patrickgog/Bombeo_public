import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
from suction_design import SuctionPipeDesign

# --- Función de criterios de diseño ---
def get_criterio_texto():
    return (
        "**Criterios de diseño de succión:**\n"
        "- Velocidad máxima recomendada: 1.5 m/s\n"
        "- NPSH disponible ≥ NPSH requerido + margen\n"
        "- Evitar cavitación y presión negativa\n"
        "- Sumergencia mínima según normas\n"
        "- Pérdidas de carga aceptables (<20% de altura de succión)\n"
        "- Régimen de flujo preferido: turbulento\n"
    )

# --- Panel de datos de entrada ---
def get_input_panel():
    materiales = [
        ("PVC (0.0015 mm)", 0.0015),
        ("Cobre (0.0015 mm)", 0.0015),
        ("Acero comercial nuevo (0.15 mm)", 0.15),
        ("Acero galvanizado (0.15 mm)", 0.15),
        ("Hierro fundido (0.26 mm)", 0.26),
        ("Concreto (1.0 mm)", 1.0),
        ("Asbesto-cemento (0.05 mm)", 0.05),
        ("Personalizada", None)
    ]
    campos = [
        ("Caudal de diseño (L/s):", dcc.Input(id='input-q', type='number', value=80, step=0.1, style={'width': '120px'})),
        ("Altura de succión (m):", dcc.Input(id='input-hs', type='number', value=3.0, step=0.1, style={'width': '120px'})),
        ("Longitud de tubería (m):", dcc.Input(id='input-length', type='number', value=15, step=1, style={'width': '120px'})),
        ("Material y rugosidad:",
            html.Div([
                dcc.Dropdown(
                    id='input-rough-material',
                    options=[{'label': m, 'value': str(r) if r is not None else 'custom'} for m, r in materiales],
                    value='0.15',
                    clearable=False,
                    style={'width': '180px', 'marginBottom': '6px'}
                ),
                dcc.Input(id='input-rough', type='number', value=0.15, step=0.001, style={'width': '120px', 'display': 'none', 'marginTop': '4px'})
            ])
        ),
        ("K accesorios:", dcc.Input(id='input-k', type='number', value=3.0, step=0.1, style={'width': '120px'})),
        ("Temperatura (°C):", dcc.Input(id='input-temp', type='number', value=25, step=1, style={'width': '120px'})),
        ("Presión atmosférica (kPa):", dcc.Input(id='input-patm', type='number', value=101.325, step=0.001, style={'width': '120px'})),
        ("NPSH requerido (m):", dcc.Input(id='input-npsh', type='number', value=3.0, step=0.1, style={'width': '120px'})),
        ("Diámetro inicial (mm):", dcc.Input(id='input-dmin', type='number', value=100, step=1, style={'width': '120px'})),
        ("Diámetro final (mm):", dcc.Input(id='input-dmax', type='number', value=500, step=1, style={'width': '120px'})),
        ("Diámetro a analizar (mm):", dcc.Input(id='input-danalizar', type='number', value='', step=1, style={'width': '120px'})),
    ]
    filas = [
        html.Div([
            html.Label(label, style={'flex': '1', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginRight': '10px', 'textAlign': 'right'}),
            html.Div(comp, style={'flex': '1', 'textAlign': 'left'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'marginBottom': '12px'})
        for label, comp in campos
    ]
    return html.Div([
        html.H3('Datos de Succión', style={'fontFamily': 'Segoe UI', 'fontWeight': 'bold', 'fontSize': '22px', 'color': '#007bff', 'marginBottom': '18px', 'textAlign': 'center', 'letterSpacing': '1px'}),
        *filas,
        html.Button('Calcular', id='btn-calc', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '120px', 'background': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '6px'}),
        html.Button('Cerrar', id='btn-close-panel', n_clicks=0, style={'marginTop': '12px', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'width': '120px', 'background': '#eee', 'color': '#333', 'border': 'none', 'borderRadius': '6px'}),
        html.Button('Calcular punto', id='btn-calc-punto', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#28a745', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'marginLeft': '10px'}),
        html.Button('Limpiar punto', id='btn-clear-punto', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#dc3545', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'marginLeft': '10px'}),
        html.Button('Análisis', id='btn-analisis', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#6f42c1', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'marginLeft': '10px', 'opacity': '0.5'}),
    ], id='input-panel-content', style={'padding': '24px', 'fontFamily': 'Segoe UI'})

# --- Layout principal ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('SUCTION PIPE DESIGN', style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'fontWeight': 'bold', 'fontSize': '2.5rem', 'color': '#007bff', 'marginBottom': '10px', 'marginTop': '10px', 'letterSpacing': '2px'}),
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
                dcc.Tab(label='Cavitación y NPSH', value='tab-2', children=html.Div(id='tab-content-2')),
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

# --- Función para generar gráficos de succión ---
def generate_suction_graphs(q, hs, length, rough, k_fittings, temp, patm, npsh_req, dmin, dmax, punto_analizado):
    """Genera todos los gráficos de análisis de succión"""
    
    # Crear instancia de SuctionPipeDesign con rango de diámetros personalizado
    design = SuctionPipeDesign(
        fluid_properties={
            'density': 1000,
            'viscosity': 0.001,
            'vapor_pressure': 17567  # Pa a 20°C
        },
        pipe_properties={
            'length': length,
            'roughness': rough,
            'fittings_k': k_fittings
        },
        operating_conditions={
            'flow_rate': q/1000,  # Convertir L/s a m³/s
            'temperature': temp,
            'suction_height': hs,
            'atmospheric_pressure': patm * 1000  # Convertir kPa a Pa
        },
        diam_range=(dmin/1000, dmax/1000)  # mm a m
    )
    
    # Obtener datos del análisis
    df = design.analyze_all_diameters()
    diameters_mm = df['diameter'] * 1000
    
    # Encontrar punto de operación óptimo
    safe_diameters = df[df['cavitation_risk'] == False]
    if len(safe_diameters) > 0:
        optimal_idx = safe_diameters['npsh_available'].idxmax()
    else:
        optimal_idx = df['npsh_available'].idxmax()
    
    optimal_diameter = df.loc[optimal_idx, 'diameter'] * 1000
    optimal_velocity = df.loc[optimal_idx, 'velocity']
    optimal_npsh = df.loc[optimal_idx, 'npsh_available']
    optimal_loss = df.loc[optimal_idx, 'friction_loss']
    
    # 1. Gráfico de Velocidad vs Diámetro
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=diameters_mm, y=df['velocity'],
        mode='lines', name='Velocidad',
        line=dict(color='blue', width=3),
        hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br><b>Velocidad:</b> %{y:.3f} m/s<extra></extra>'
    ))
    fig1.add_hline(y=1.5, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text="Límite recomendado (1.5 m/s)")
    fig1.update_layout(
        title='Velocidad vs Diámetro',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Velocidad (m/s)',
        height=400, showlegend=False,
        hovermode='x unified'
    )
    
    # 2. Gráfico de Reynolds
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=diameters_mm, y=df['reynolds'],
        mode='lines', name='Reynolds',
        line=dict(color='green', width=3),
        hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br><b>Reynolds:</b> %{y:.0f}<extra></extra>'
    ))
    fig2.add_hline(y=2300, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text="Transición laminar-turbulento")
    fig2.add_hline(y=4000, line_dash="dash", line_color="orange", opacity=0.7,
                   annotation_text="Flujo turbulento establecido")
    fig2.update_layout(
        title='Régimen de Flujo (Reynolds)',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Número de Reynolds',
        yaxis_type="log", height=400, showlegend=False,
        hovermode='x unified'
    )
    
    # 3. Gráfico de Pérdidas de Carga
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=diameters_mm, y=df['friction_loss'],
        mode='lines', name='Pérdidas de Carga',
        line=dict(color='brown', width=3),
        hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br><b>Pérdidas:</b> %{y:.4f} m<extra></extra>'
    ))
    fig3.update_layout(
        title='Pérdidas por Fricción',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Pérdidas de Carga (m)',
        height=400, showlegend=False,
        hovermode='x unified'
    )
    
    # 4. Gráfico de NPSH Disponible
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=diameters_mm, y=df['npsh_available'],
        mode='lines', name='NPSH disponible',
        line=dict(color='purple', width=3),
        hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br><b>NPSH disponible:</b> %{y:.3f} m<extra></extra>'
    ))
    fig4.add_hline(y=npsh_req, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text=f"NPSH requerido ({npsh_req} m)")
    fig4.update_layout(
        title='NPSH Disponible vs Diámetro',
        xaxis_title='Diámetro (mm)',
        yaxis_title='NPSH (m)',
        height=400, showlegend=False,
        hovermode='x unified'
    )
    
    # 5. Gráfico de Presión y Cavitación
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=diameters_mm, y=df['min_pressure']/1000,
        mode='lines', name='Presión mínima',
        line=dict(color='blue', width=3),
        hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br><b>Presión mínima:</b> %{y:.2f} kPa<extra></extra>'
    ))
    fig5.add_hline(y=design.fluid['vapor_pressure']/1000, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text="Presión de vapor")
    fig5.update_layout(
        title='Riesgo de Cavitación',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Presión (kPa)',
        height=400, showlegend=False,
        hovermode='x unified'
    )
    
    # 6. Gráfico de Sumergencia Mínima
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=diameters_mm, y=df['submergence_min'],
        mode='lines', name='Sumergencia Mínima',
        line=dict(color='#006400', width=3),
        hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                     '<b>Sumergencia mínima:</b> %{y:.3f} m<extra></extra>'
    ))
    fig6.update_layout(
        title='Sumergencia para Evitar Vórtices',
        xaxis_title='Diámetro (mm)',
        yaxis_title='Sumergencia Mínima (m)',
        height=400, showlegend=False,
        hovermode='x unified'
    )
    
    # Configurar todos los gráficos
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Segoe UI', size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Ajuste de ejes para fig1 (Velocidad)
    min_x, max_x = diameters_mm.min(), diameters_mm.max()
    # Velocidad: incluir 1.5 m/s
    min_y = min(df['velocity'].min(), 1.5)
    max_y = max(df['velocity'].max(), 1.5)
    fig1.update_yaxes(range=[min_y*0.95, max_y*1.05])
    # Gráfico de Reynolds: escala logarítmica profesional con grilla completa
    min_y2 = 1_000
    max_y2 = 1_000_000
    fig2.update_xaxes(range=[min_x, max_x])
    # Ticks logarítmicos: 2-9 solo número, 10k, 100k, 1M con sufijo
    tickvals = []
    ticktext = []
    for exp in range(3, 7):  # 10^3 a 10^6
        for d in range(1, 10):
            val = d * 10**exp
            if val > max_y2:
                break
            tickvals.append(val)
            if val == 10_000:
                ticktext.append("10k")
            elif val == 100_000:
                ticktext.append("100k")
            elif val == 1_000_000:
                ticktext.append("1M")
            elif d == 1:
                ticktext.append("")  # No mostrar nada en 1k, 10k, 100k, 1M salvo los sufijos
            else:
                ticktext.append(str(d))
    fig2.update_yaxes(
        type="log",
        range=[np.log10(min_y2), np.log10(max_y2)],
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    # Líneas de criterio con anotación en la parte inferior derecha, pegadas a la línea y siempre visibles
    fig2.add_hline(y=2300, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text="Transición laminar-turbulento (2300)",
                   annotation_position="bottom right",
                   annotation_font_size=13,
                   annotation_font_color="red",
                   annotation_bgcolor="white",
                   annotation_bordercolor="red",
                   annotation_borderwidth=1)
    fig2.add_hline(y=4000, line_dash="dash", line_color="orange", opacity=0.7,
                   annotation_text="Flujo turbulento (4000)",
                   annotation_position="bottom right",
                   annotation_font_size=13,
                   annotation_font_color="orange",
                   annotation_bgcolor="white",
                   annotation_bordercolor="orange",
                   annotation_borderwidth=1)
    # Anotación general sobre la escala logarítmica
    fig2.add_annotation(
        text="Líneas horizontales: escala logarítmica base 10",
        xref="paper", yref="paper",
        x=0.99, y=0.99, showarrow=False,
        font=dict(size=13, color="#444"),
        align="right",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="#bbb", borderwidth=1
    )

    # Ajuste de ejes para fig3 (Pérdidas)
    min_y3, max_y3 = df['friction_loss'].min(), df['friction_loss'].max()
    fig3.update_xaxes(range=[min_x, max_x])
    fig3.update_yaxes(range=[min_y3*0.95, max_y3*1.05])
    # Ajuste de ejes para fig4 (NPSH)
    min_y4, max_y4 = df['npsh_available'].min(), df['npsh_available'].max()
    # NPSH: incluir NPSH requerido
    min_y4 = min(df['npsh_available'].min(), npsh_req)
    max_y4 = max(df['npsh_available'].max(), npsh_req)
    fig4.update_xaxes(range=[min_x, max_x])
    fig4.update_yaxes(range=[min_y4*0.95, max_y4*1.05])
    # Ajuste de ejes para fig5 (Presión mínima)
    min_y5, max_y5 = (df['min_pressure']/1000).min(), (df['min_pressure']/1000).max()
    # Presión mínima: incluir presión de vapor
    vapor_kpa = design.fluid['vapor_pressure']/1000
    min_y5 = min((df['min_pressure']/1000).min(), vapor_kpa)
    max_y5 = max((df['min_pressure']/1000).max(), vapor_kpa)
    fig5.update_xaxes(range=[min_x, max_x])
    fig5.update_yaxes(range=[min_y5*0.95, max_y5*1.05])
    fig5.add_hline(y=vapor_kpa, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text=f"Presión de vapor = {vapor_kpa:.1f} kPa", annotation_position="top left")
    # Ajuste de ejes para fig6 (Sumergencia)
    min_y6, max_y6 = df['submergence_min'].min(), df['submergence_min'].max()
    fig6.update_xaxes(range=[min_x, max_x])
    fig6.update_yaxes(range=[min_y6*0.95, max_y6*1.05])
    
    # Cuadro de punto de operación
    punto_operacion = html.Div([
        html.H4('Punto de Operación Óptimo', style={'textAlign': 'center', 'color': '#007bff', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.Span('Diámetro: ', style={'fontWeight': 'bold'}),
                html.Span(f'{optimal_diameter:.0f} mm')
            ], style={'marginBottom': '8px'}),
            html.Div([
                html.Span('Velocidad: ', style={'fontWeight': 'bold'}),
                html.Span(f'{optimal_velocity:.2f} m/s')
            ], style={'marginBottom': '8px'}),
            html.Div([
                html.Span('NPSH disponible: ', style={'fontWeight': 'bold'}),
                html.Span(f'{optimal_npsh:.2f} m')
            ], style={'marginBottom': '8px'}),
            html.Div([
                html.Span('Pérdidas de carga: ', style={'fontWeight': 'bold'}),
                html.Span(f'{optimal_loss:.3f} m')
            ], style={'marginBottom': '8px'}),
        ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #dee2e6'})
    ], style={'marginTop': '20px', 'fontFamily': 'Segoe UI'})

    # Si se ha solicitado un punto de análisis, interpolar los valores
    if punto_analizado is not None:
        try:
            d_analizar_mm = float(punto_analizado)
            d_analizar_m = d_analizar_mm / 1000
            # Interpolación para cada variable
            diam = df['diameter'].values
            vel = np.interp(d_analizar_m, diam, df['velocity'].values)
            reyn = np.interp(d_analizar_m, diam, df['reynolds'].values)
            perd = np.interp(d_analizar_m, diam, df['friction_loss'].values)
            npsh = np.interp(d_analizar_m, diam, df['npsh_available'].values)
            pres = np.interp(d_analizar_m, diam, df['min_pressure'].values) / 1000
            sumerg = np.interp(d_analizar_m, diam, df['submergence_min'].values)
            # Añadir marcadores
            fig1.add_trace(go.Scatter(x=[d_analizar_mm], y=[vel], mode='markers', name='Punto Analizado', marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig2.add_trace(go.Scatter(x=[d_analizar_mm], y=[reyn], mode='markers', name='Punto Analizado', marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig3.add_trace(go.Scatter(x=[d_analizar_mm], y=[perd], mode='markers', name='Punto Analizado', marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig4.add_trace(go.Scatter(x=[d_analizar_mm], y=[npsh], mode='markers', name='Punto Analizado', marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig5.add_trace(go.Scatter(x=[d_analizar_mm], y=[pres], mode='markers', name='Punto Analizado', marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            fig6.add_trace(go.Scatter(x=[d_analizar_mm], y=[sumerg], mode='markers', name='Punto Analizado', marker=dict(size=12, color='red', symbol='star'), showlegend=False))
            
            # Agregar anotación del régimen en el gráfico de Reynolds
            if reyn < 2300:
                regimen = "Laminar"
                color_regimen = "#ff7f0e"
            elif reyn < 4000:
                regimen = "Transición"
                color_regimen = "#ff7f0e"
            else:
                regimen = "Turbulento"
                color_regimen = "#ff7f0e"
            
            print(f"DEBUG: Reynolds = {reyn:.0f}, Régimen = {regimen}")  # Debug
            
            fig2.add_annotation(
                x=d_analizar_mm,
                y=reyn,
                text=f"Régimen: {regimen}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color_regimen,
                ax=20,
                ay=-30,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=color_regimen,
                borderwidth=2,
                font=dict(size=13, color=color_regimen, family="Arial Black"),
                xanchor="left",
                yanchor="top"
            )
            # Devolver los valores interpolados para mostrar en la leyenda
            return fig1, fig2, fig3, fig4, fig5, fig6, {
                'Diámetro': f'{d_analizar_mm:.0f} mm',
                'Velocidad': f'{vel:.2f} m/s',
                'Reynolds': f'{reyn:.0f}',
                'NPSH Disponible': f'{npsh:.2f} m',
                'Pérdidas de Carga': f'{perd:.3f} m',
                'Presión Mínima': f'{pres:.2f} kPa',
                'Sumergencia Mínima': f'{sumerg:.3f} m'
            }
        except Exception as e:
            return fig1, fig2, fig3, fig4, fig5, fig6, {
                'Error': f'Error al calcular el punto: {str(e)}'
            }
    else:
        return fig1, fig2, fig3, fig4, fig5, fig6, None

# --- Callbacks para generar gráficos ---
@app.callback(
    [Output('tab-content-1', 'children'),
     Output('tab-content-2', 'children')],
    [Input('btn-calc', 'n_clicks'),
     Input('punto-analizado', 'data')],
    [Input('input-q', 'value'),
     Input('input-hs', 'value'),
     Input('input-length', 'value'),
     Input('input-rough', 'value'), # This input is now hidden
     Input('input-k', 'value'),
     Input('input-temp', 'value'),
     Input('input-patm', 'value'),
     Input('input-npsh', 'value'),
     Input('input-dmin', 'value'),
     Input('input-dmax', 'value'),
     Input('analisis-data', 'data')]
)
def update_graphs(n_clicks, punto_analizado, q, hs, length, rough, k_fittings, temp, patm, npsh_req, dmin, dmax, analisis_data):
    if not n_clicks:
        q = q or 11
        hs = hs or 3
        length = length or 10
        rough = rough or 0.15
        k_fittings = k_fittings or 2.5
        temp = temp or 20
        patm = patm or 101.3
        npsh_req = npsh_req or 3.0
        dmin = dmin or 20
        dmax = dmax or 1000
    try:
        # Paso el rango de diámetros a la función de gráficos (en mm)
        fig1, fig2, fig3, fig4, fig5, fig6, punto_data = generate_suction_graphs(
            q, hs, length, rough, k_fittings, temp, patm, npsh_req, dmin, dmax, punto_analizado
        )
        # Leyendas/simbología para cada gráfico
        simbologia_vel = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'blue', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Velocidad', style={'color': 'blue', 'fontSize': '14px', 'marginRight': '18px'}),
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'red', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Límite recomendado (1.5 m/s)', style={'color': '#333', 'fontSize': '14px'})
        ], style={'textAlign': 'center', 'marginTop': '8px'})
        simbologia_rey = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'green', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Número de Reynolds', style={'color': 'green', 'fontSize': '14px', 'marginRight': '18px'}),
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'red', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Transición laminar-turbulento (2300)', style={'color': 'red', 'fontSize': '14px', 'marginRight': '18px'}),
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'orange', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Flujo turbulento (4000)', style={'color': 'orange', 'fontSize': '14px'})
        ], style={'textAlign': 'center', 'marginTop': '8px'})
        simbologia_perd = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'brown', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Pérdidas de carga', style={'color': 'brown', 'fontSize': '14px'})
        ], style={'textAlign': 'center', 'marginTop': '8px'})
        simbologia_npsh = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'purple', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('NPSH disponible', style={'color': 'purple', 'fontSize': '14px', 'marginRight': '18px'}),
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'red', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('NPSH requerido', style={'color': 'red', 'fontSize': '14px'})
        ], style={'textAlign': 'center', 'marginTop': '8px'})
        simbologia_pres = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'blue', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Presión mínima', style={'color': 'blue', 'fontSize': '14px', 'marginRight': '18px'}),
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': 'red', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Presión de vapor', style={'color': 'red', 'fontSize': '14px'})
        ], style={'textAlign': 'center', 'marginTop': '8px'})
        simbologia_sum = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '30px', 'height': '4px', 'background': '#006400', 'verticalAlign': 'middle', 'marginRight': '8px'}),
            html.Span('Sumergencia mínima', style={'color': '#006400', 'fontSize': '14px'})
        ], style={'textAlign': 'center', 'marginTop': '8px'})

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
            elif tipo == 'reynolds' and 'reynolds' in analisis_data:
                data = analisis_data['reynolds']
                color = '#28a745' if data['estado'] == 'TURBULENTO' else '#ffc107'
            elif tipo == 'perdidas' and 'perdidas' in analisis_data:
                data = analisis_data['perdidas']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            elif tipo == 'npsh' and 'npsh' in analisis_data:
                data = analisis_data['npsh']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            elif tipo == 'presion' and 'presion' in analisis_data:
                data = analisis_data['presion']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            elif tipo == 'sumergencia' and 'sumergencia' in analisis_data:
                data = analisis_data['sumergencia']
                color = '#28a745' if data['estado'] == 'OK' else '#dc3545'
            else:
                return None
            
            return html.Div([
                html.H4('📊 Análisis Automático', style={'color': color, 'fontSize': '16px', 'marginBottom': '8px', 'textAlign': 'center'}),
                html.P(data['observacion'], style={'fontSize': '14px', 'marginBottom': '6px', 'textAlign': 'center'}),
                html.P(f"💡 Recomendación: {data['recomendacion']}", style={'fontSize': '14px', 'fontStyle': 'italic', 'textAlign': 'center', 'color': '#666'})
            ], style={'background': '#f8f9fa', 'border': f'2px solid {color}', 'borderRadius': '8px', 'padding': '12px', 'margin': '10px auto 0 auto', 'width': '100%', 'maxWidth': '400px'})

        # Si el punto_data contiene un error, mostrarlo en lugar de los gráficos
        if punto_data is not None and 'Error' in punto_data:
            error_msg = html.Div([
                html.H3('Error en el cálculo', style={'color': 'red', 'textAlign': 'center'}),
                html.P(punto_data['Error'], style={'textAlign': 'center', 'color': '#666'})
            ])
            return error_msg, error_msg

        tab1_content = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig1, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'velocidad_succion'}}, style={'width': '33.33vw', 'height': '33.33vw', 'minWidth': '300px', 'minHeight': '300px'}),
                    simbologia_vel,
                    box_punto('Velocidad', punto_data, 'Velocidad', 'm/s'),
                    crear_analisis(analisis_data, 'velocidad')
                ], style={'flex': 1}),
                html.Div([
                    dcc.Graph(figure=fig2, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'reynolds_succion'}}, style={'width': '33.33vw', 'height': '33.33vw', 'minWidth': '300px', 'minHeight': '300px'}),
                    simbologia_rey,
                    box_punto('Reynolds', punto_data, 'Reynolds', ''),
                    crear_analisis(analisis_data, 'reynolds')
                ], style={'flex': 1}),
                html.Div([
                    dcc.Graph(figure=fig3, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'perdidas_succion'}}, style={'width': '33.33vw', 'height': '33.33vw', 'minWidth': '300px', 'minHeight': '300px'}),
                    simbologia_perd,
                    box_punto('Pérdidas', punto_data, 'Pérdidas de Carga', 'm'),
                    crear_analisis(analisis_data, 'perdidas')
                ], style={'flex': 1}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '30px', 'width': '100%', 'justifyContent': 'center'})
        ])
        tab2_content = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig4, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'npsh_succion'}}, style={'width': '33.33vw', 'height': '33.33vw', 'minWidth': '300px', 'minHeight': '300px'}),
                    simbologia_npsh,
                    box_punto('NPSH Disponible', punto_data, 'NPSH Disponible', 'm'),
                    crear_analisis(analisis_data, 'npsh')
                ], style={'flex': 1}),
                html.Div([
                    dcc.Graph(figure=fig5, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'presion_succion'}}, style={'width': '33.33vw', 'height': '33.33vw', 'minWidth': '300px', 'minHeight': '300px'}),
                    simbologia_pres,
                    box_punto('Presión Mínima', punto_data, 'Presión Mínima', 'kPa'),
                    crear_analisis(analisis_data, 'presion')
                ], style={'flex': 1}),
                html.Div([
                    dcc.Graph(figure=fig6, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toImage'], 'toImageButtonOptions': {'format': 'png', 'filename': 'sumergencia_succion'}}, style={'width': '33.33vw', 'height': '33.33vw', 'minWidth': '300px', 'minHeight': '300px'}),
                    simbologia_sum,
                    box_punto('Sumergencia Mínima', punto_data, 'Sumergencia Mínima', 'm'),
                    crear_analisis(analisis_data, 'sumergencia')
                ], style={'flex': 1}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '30px', 'width': '100%', 'justifyContent': 'center'})
        ])
        return tab1_content, tab2_content
    except Exception as e:
        error_msg = html.Div([
            html.H3('Error en el cálculo', style={'color': 'red', 'textAlign': 'center'}),
            html.P(f'Error: {str(e)}', style={'textAlign': 'center', 'color': '#666'})
        ])
        return error_msg, error_msg

# Callback para mostrar/ocultar campo de rugosidad personalizada
@app.callback(
    [Output('input-rough', 'style'),
     Output('input-rough', 'value')],
    [Input('input-rough-material', 'value')]
)
def toggle_custom_roughness(selected_value):
    if selected_value == 'custom':
        return {'width': '120px', 'display': 'block', 'marginTop': '4px'}, 0.15
    else:
        return {'width': '120px', 'display': 'none', 'marginTop': '4px'}, float(selected_value)

# Callback para activar/desactivar botón de análisis
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

# Callback para generar análisis automático
@app.callback(
    Output('analisis-data', 'data'),
    [Input('btn-analisis', 'n_clicks')],
    [State('input-danalizar', 'value'),
     State('input-q', 'value'),
     State('input-hs', 'value'),
     State('input-length', 'value'),
     State('input-rough', 'value'),
     State('input-k', 'value'),
     State('input-temp', 'value'),
     State('input-patm', 'value'),
     State('input-npsh', 'value'),
     State('input-dmin', 'value'),
     State('input-dmax', 'value')]
)
def generate_analysis(n_clicks, d_analizar, q, hs, length, rough, k_fittings, temp, patm, npsh_req, dmin, dmax):
    print(f"DEBUG: generate_analysis llamado con n_clicks={n_clicks}, d_analizar={d_analizar}")
    
    # Solo ejecutar si realmente se hizo clic en el botón
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id != 'btn-analisis':
        return None
    
    if not n_clicks or d_analizar is None or d_analizar == '':
        print("DEBUG: Retornando None - no hay clicks o d_analizar es None/vacío")
        return None
    
    try:
        # Calcular valores para el diámetro analizado
        d_analizar_m = float(d_analizar) / 1000
        design = SuctionPipeDesign(
            fluid_properties={'density': 1000, 'viscosity': 0.001, 'vapor_pressure': 2337},
            pipe_properties={'length': length, 'roughness': rough, 'fittings_k': k_fittings},
            operating_conditions={'flow_rate': q/1000, 'temperature': temp, 'suction_height': hs, 'atmospheric_pressure': patm*1000},
            diam_range=(dmin/1000, dmax/1000)
        )
        
        # Calcular valores para el diámetro analizado
        vel = design.calculate_velocity(d_analizar_m, q/1000)
        reyn = design.calculate_reynolds(vel, d_analizar_m)
        perd = design.calculate_friction_loss(vel, d_analizar_m)
        
        # Calcular NPSH disponible
        npsh_available = hs - perd - design.fluid['vapor_pressure']/(1000*9.81)
        
        # Calcular presión mínima
        min_pressure = (patm*1000 - 1000*9.81*(hs + perd))/1000  # kPa
        
        # Calcular sumergencia mínima (aproximación)
        submergence_min = d_analizar_m * 1.5  # 1.5 veces el diámetro
        
        # Determinar estado del régimen de Reynolds
        reyn_estado = 'LAMINAR' if reyn < 2300 else 'TRANSICION' if reyn < 4000 else 'TURBULENTO'
        
        # Análisis de cada aspecto
        analisis = {
            'velocidad': {
                'valor': vel,
                'limite': 1.5,
                'estado': 'OK' if vel <= 1.5 else 'ALTO',
                'observacion': f'La velocidad de {vel:.2f} m/s está {"dentro" if vel <= 1.5 else "por encima"} del límite recomendado de 1.5 m/s.',
                'recomendacion': 'El diseño es adecuado.' if vel <= 1.5 else f'Velocidad alta ({vel:.2f} m/s). Optimizar: aumentar diámetro a {d_analizar_m*1000*1.3:.0f}-{d_analizar_m*1000*1.6:.0f} mm para reducir a 1.0-1.3 m/s.'
            },
            'reynolds': {
                'valor': reyn,
                'estado': reyn_estado,
                'observacion': f'El flujo es {reyn_estado.lower()} con Re = {reyn:.0f}.',
                'recomendacion': 'El régimen turbulento es preferible para sistemas de bombeo.' if reyn >= 4000 else 'Considerar optimizar para flujo turbulento.'
            },
            'perdidas': {
                'valor': perd,
                'limite': hs * 0.2,  # 20% de la altura de succión
                'estado': 'OK' if perd <= hs * 0.2 else 'ALTO',
                'observacion': f'Las pérdidas de {perd:.3f} m representan {(perd/hs)*100:.1f}% de la altura de succión.',
                'recomendacion': 'Las pérdidas son aceptables.' if perd <= hs * 0.2 else f'Las pérdidas son altas ({(perd/hs)*100:.1f}%). Optimizar: aumentar diámetro a {d_analizar_m*1000*1.2:.0f}-{d_analizar_m*1000*1.5:.0f} mm, reducir accesorios, o usar material más liso.'
            },
            'npsh': {
                'valor': npsh_available,
                'limite': npsh_req,
                'estado': 'OK' if npsh_available >= npsh_req else 'BAJO',
                'observacion': f'El NPSH disponible es {npsh_available:.2f} m vs {npsh_req} m requerido.',
                'recomendacion': 'El NPSH es adecuado.' if npsh_available >= npsh_req else f'NPSH insuficiente ({npsh_available:.2f} m). Optimizar: reducir pérdidas aumentando diámetro a {d_analizar_m*1000*1.4:.0f}-{d_analizar_m*1000*1.8:.0f} mm, o aumentar altura de succión a {hs*1.2:.1f} m.'
            },
            'presion': {
                'valor': min_pressure,
                'limite': design.fluid['vapor_pressure']/1000,  # Presión de vapor en kPa
                'estado': 'OK' if min_pressure > design.fluid['vapor_pressure']/1000 else 'BAJA',
                'observacion': f'La presión mínima es {min_pressure:.1f} kPa.',
                'recomendacion': 'La presión es adecuada.' if min_pressure > design.fluid['vapor_pressure']/1000 else f'Presión baja ({min_pressure:.1f} kPa). Optimizar: reducir pérdidas aumentando diámetro a {d_analizar_m*1000*1.3:.0f}-{d_analizar_m*1000*1.7:.0f} mm, o reducir altura de succión a {hs*0.8:.1f} m.'
            },
            'sumergencia': {
                'valor': submergence_min,
                'estado': 'OK',
                'observacion': f'La sumergencia mínima requerida es {submergence_min:.3f} m.',
                'recomendacion': 'Verificar que la sumergencia real sea mayor a este valor para evitar vórtices.'
            }
        }
        
        return analisis
        
    except Exception as e:
        return {'error': str(e)}

# Callback para guardar/eliminar el punto analizado
@app.callback(
    Output('punto-analizado', 'data'),
    [Input('btn-calc-punto', 'n_clicks'), Input('btn-clear-punto', 'n_clicks')],
    [State('input-danalizar', 'value'),
     State('input-dmin', 'value'), State('input-dmax', 'value')]
)
def set_punto_analizado(n_calc, n_clear, d_analizar, dmin, dmax):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-clear-punto':
        return None
    if button_id == 'btn-calc-punto' and d_analizar is not None:
        try:
            d_analizar = float(d_analizar)
            dmin = float(dmin)
            dmax = float(dmax)
            if d_analizar < dmin or d_analizar > dmax:
                return None
            return d_analizar
        except:
            return None
    return None

# Callback para notificaciones de copiado (usando JavaScript)
@app.callback(
    Output('copy-notification', 'children'),
    [Input('copy-notification', 'children')]
)
def update_copy_notification(children):
    return children

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051) 