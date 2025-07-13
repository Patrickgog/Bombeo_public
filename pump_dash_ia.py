import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import numpy as np
from pump_design import CentrifugalPumpDesigner
import requests
import logging

# Configuración del logger
logging.basicConfig(
    filename='app_dash.log',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

def default_hourly_factors():
    return "0.6,0.6,0.6,0.6,0.7,0.85,1.1,1.4,1.7,2.1,1.95,1.8,1.7,1.6,1.5,1.55,1.7,1.9,2.0,1.6,1.2,1.0,0.8,0.7"

def filtrar_figura(fig, title, idx):
    if not fig or not hasattr(fig, 'data') or len(fig.data) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"{title} (sin datos válidos)")
        return empty_fig

    nuevas_trazas = []
    datos_validos = False
    y_vals_all = []
    for trace in fig.data:
        if hasattr(trace, 'y') and hasattr(trace, 'x'):
            y = np.array(trace.y)
            x = np.array(trace.x)
            mask = np.isfinite(y) & np.isfinite(x)
            # Protección básica contra valores absurdos
            if 'eficiencia' in title.lower():
                mask = mask & (y >= 0)
            if np.sum(mask) > 0:
                trace.y = y[mask]
                trace.x = x[mask]
                nuevas_trazas.append(trace)
                y_vals_all.extend(y[mask].tolist())
                datos_validos = True
        else:
            nuevas_trazas.append(trace)
    fig.data = tuple(nuevas_trazas)
    if not datos_validos or not y_vals_all:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"{title} (sin datos válidos)")
        return empty_fig
    # Ajuste dinámico del eje Y
    y_min = float(np.min(y_vals_all))
    y_max = float(np.max(y_vals_all))
    # NO modificar la configuración de ejes - dejar que Plotly maneje automáticamente
    # Solo forzar formato entero para todos los gráficos
    fig.update_yaxes(tickformat="d")
    # Incluir la línea de capacidad de tanque en 'Volumen en tanque'
    if 'volumen en tanque' in title.lower():
        # Buscar la línea horizontal de capacidad de tanque
        for shape in getattr(fig.layout, 'shapes', []):
            if hasattr(shape, 'y0') and hasattr(shape, 'y1'):
                y_max = max(y_max, float(shape.y0), float(shape.y1))
        # O buscar en add_hline si está en layout['shapes']
        if hasattr(fig.layout, 'shapes'):
            for s in fig.layout.shapes:
                if hasattr(s, 'y0') and hasattr(s, 'y1'):
                    y_max = max(y_max, float(s.y0), float(s.y1))
        # O buscar en layout['annotations'] si hay texto de capacidad
        if hasattr(fig.layout, 'annotations'):
            for ann in fig.layout.annotations:
                if hasattr(ann, 'y'):
                    y_max = max(y_max, float(ann.y))
    # Incluir la línea de caudal de bombeo en 'Demanda vs bombeo'
    if 'demanda vs bombeo' in title.lower():
        for trace in fig.data:
            if hasattr(trace, 'name') and 'bombeo' in str(trace.name).lower():
                y_bombeo = np.array(trace.y)
                if y_bombeo.size > 0 and np.isfinite(y_bombeo).any():
                    y_max = max(y_max, float(np.max(y_bombeo)))
    if y_max == y_min:
        y_min = y_min - 0.1 * abs(y_min) if y_min != 0 else -1
        y_max = y_max + 0.1 * abs(y_max) if y_max != 0 else 1
    else:
        margen = 0.1 * (y_max - y_min)
        y_min -= margen
        y_max += margen
    # Protección contra rangos absurdos
    if 'eficiencia' in title.lower():
        y_min = max(0, y_min)
        y_max = min(100, y_max)
    if 'potencia' in title.lower() or 'altura' in title.lower() or 'carga' in title.lower():
        y_min = max(0, y_min)
    if 'costo' in title.lower():
        y_min = max(0, y_min)
    fig.update_yaxes(range=[y_min, y_max], fixedrange=False)
    # Leyenda debajo
    fig.update_layout(
        legend_orientation='h', legend_yanchor='top', legend_y=-0.25, legend_xanchor='center', legend_x=0.5,
        hovermode='x',
    )
    # --- CONFIGURACIÓN DE CUADRÍCULA CON EJES EN (0,0) ---
    # Configurar ejes para que incluyan 0 en los ticks
    if fig.layout.xaxis.range:
        x_min, x_max = fig.layout.xaxis.range
        # Asegurar que 0 esté incluido si está en el rango
        if x_min <= 0 <= x_max:
            # Calcular ticks que incluyan 0
            tick_vals = []
            step = (x_max - x_min) / 10  # Aproximadamente 10 divisiones
            for i in range(int(x_min/step), int(x_max/step) + 1):
                tick_val = i * step
                if x_min <= tick_val <= x_max:
                    tick_vals.append(tick_val)
            # Asegurar que 0 esté incluido
            if 0 not in tick_vals and x_min <= 0 <= x_max:
                tick_vals.append(0)
                tick_vals.sort()
            fig.update_xaxes(tickvals=tick_vals, showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
    
    if fig.layout.yaxis.range:
        y_min, y_max = fig.layout.yaxis.range
        # Asegurar que 0 esté incluido si está en el rango
        if y_min <= 0 <= y_max:
            # Calcular ticks que incluyan 0
            tick_vals = []
            step = (y_max - y_min) / 10  # Aproximadamente 10 divisiones
            for i in range(int(y_min/step), int(y_max/step) + 1):
                tick_val = i * step
                if y_min <= tick_val <= y_max:
                    tick_vals.append(tick_val)
            # Asegurar que 0 esté incluido
            if 0 not in tick_vals and y_min <= 0 <= y_max:
                tick_vals.append(0)
                tick_vals.sort()
            fig.update_yaxes(tickvals=tick_vals, showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
    # --- MARCO DINÁMICO ---
    # Elimina marcos previos si existen
    if 'shapes' in fig.layout:
        fig.layout.shapes = [s for s in fig.layout.shapes if not (getattr(s, 'name', None) == 'marco-dinamico')]
    # Añade el marco siempre detrás
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1,
                  line=dict(color='#cccccc', width=1), fillcolor='rgba(0,0,0,0)', layer='below', name='marco-dinamico')
    # --- EJES EN (0,0) ---
    # Elimina ejes previos si existen
    if 'shapes' in fig.layout:
        fig.layout.shapes = [s for s in fig.layout.shapes if not (getattr(s, 'name', None) in ['eje-x', 'eje-y'])]
    # Añade eje horizontal en y=0 (solo si el rango incluye y=0)
    if fig.layout.yaxis.range and fig.layout.yaxis.range[0] <= 0 <= fig.layout.yaxis.range[1]:
        fig.add_shape(type='line', xref='x', yref='y', x0=fig.layout.xaxis.range[0] if fig.layout.xaxis.range else None, 
                      x1=fig.layout.xaxis.range[1] if fig.layout.xaxis.range else None, y0=0, y1=0,
                      line=dict(color='#cccccc', width=0.7, dash='solid'), name='eje-x', layer='below')
    # Añade eje vertical en x=0 (solo si el rango incluye x=0)
    if fig.layout.xaxis.range and fig.layout.xaxis.range[0] <= 0 <= fig.layout.xaxis.range[1]:
        fig.add_shape(type='line', xref='x', yref='y', x0=0, x1=0, y0=fig.layout.yaxis.range[0] if fig.layout.yaxis.range else None,
                      y1=fig.layout.yaxis.range[1] if fig.layout.yaxis.range else None,
                      line=dict(color='#cccccc', width=0.7, dash='solid'), name='eje-y', layer='below')
    return fig

def get_criterio_texto():
    return (
        "**Criterios de diseño:**\n"
        "- Curva del sistema: Hazen-Williams\n"
        "- Curva de bomba: Parabólica (estimada si no se provee)\n"
        "- Punto de operación: Intersección sistema-bomba\n"
        "- Balance de masa: Método de Rippl\n"
        "- Costos energéticos: Potencia eje y costo unitario\n"
        "- Leyes de afinidad: Variación de RPM\n"
        "- Simulación: Balance horario y volumen de tanque\n"
    )

app = dash.Dash(__name__)

# Diccionario de tooltips para cada campo
TOOLTIPS = {
    'input-q': 'Caudal de diseño (Q): Es el caudal objetivo que debe entregar la bomba bajo condiciones normales. Un valor subestimado puede dejar sin servicio a los usuarios; uno sobrestimado incrementa el costo de inversión y operación. Se recomienda calcularlo considerando la demanda máxima esperada y posibles expansiones futuras. Unidades: litros por segundo (L/s).',
    'input-hstatic': 'Altura estática: Es la diferencia de altura entre el nivel de agua en el tanque de succión y el punto de descarga. No incluye pérdidas por fricción. Un valor incorrecto puede causar cavitación o sobredimensionamiento de la bomba. Unidades: metros (m).',
    'input-rpm': 'RPM: Velocidad de rotación del eje de la bomba. Afecta directamente la curva de rendimiento, la potencia y la vida útil del equipo. Usualmente 2900-3500 rpm para bombas estándar. Consultar la placa del fabricante.',
    'input-length': 'Longitud de tubería: Longitud total de la tubería de impulsión desde la bomba hasta el punto de entrega. Incluye tramos rectos y accesorios. Un valor mayor aumenta las pérdidas por fricción y el consumo energético. Unidades: metros (m).',
    'input-diam': 'Diámetro de tubería: Diámetro interior de la tubería de impulsión. Un diámetro pequeño reduce el costo inicial pero aumenta las pérdidas y el gasto energético. Un diámetro grande reduce pérdidas pero incrementa el costo de inversión. Unidades: milímetros (mm).',
    'input-nparallel': 'Bombas en paralelo: Número de bombas idénticas que operan simultáneamente. Usar varias bombas permite flexibilidad, respaldo y eficiencia en variaciones de demanda. Demasiadas bombas pueden complicar la operación y el mantenimiento.',
    'input-hw': 'C Hazen-Williams: Coeficiente de rugosidad hidráulica de la tubería. Depende del material y la edad. Un valor bajo indica mayor rugosidad y mayores pérdidas. Seleccione el valor adecuado según el material y el estado de la tubería. Consulte normas o tablas especializadas.',
    'input-eff': 'Eficiencia pico (%): Es la eficiencia máxima de la bomba en su punto de mejor rendimiento (BEP). Una eficiencia baja implica mayor consumo eléctrico y costos operativos. Use el valor real del fabricante si está disponible. Valores típicos: 60-85%.',
    'input-cost': 'Costo electricidad (USD/kWh): Precio unitario de la energía eléctrica. Un valor alto impacta fuertemente el costo de operación del sistema. Verifique el contrato con la empresa suministradora.',
    'input-tankinit': 'Nivel inicial tanque (%): Porcentaje de llenado del tanque al inicio de la simulación. Un valor bajo puede provocar arranques frecuentes de la bomba; uno alto reduce la capacidad de reserva.',
    'input-tankmin': 'Nivel mínimo tanque (%): Porcentaje mínimo permitido de llenado del tanque. Si el nivel cae por debajo, la bomba debe detenerse para evitar daños o entrada de aire.',
    'input-days': 'Días de simulación: Número de días consecutivos para simular la operación del sistema. Útil para analizar el comportamiento en periodos prolongados y detectar problemas de capacidad o ciclos de bombeo.',
    'input-hourly': 'Factores horarios: Distribución relativa de la demanda a lo largo de 24 horas. Ingrese 24 valores separados por coma, donde 1.0 representa la demanda promedio. Permite simular variaciones diarias y ajustar la operación de la bomba.',
}

# Tooltip flotante reutilizable
TOOLTIP_STYLE = {
    'position': 'fixed', 'zIndex': 9999, 'background': '#fff', 'color': '#222', 'border': '1px solid #007bff',
    'borderRadius': '7px', 'padding': '10px 16px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.12)', 'fontSize': '15px',
    'maxWidth': '320px', 'display': 'none', 'pointerEvents': 'none'
}

# Panel de datos de entrada (contenido)
def get_input_panel():
    campos = [
        ("Caudal de diseño Q (L/s):", dcc.Input(id='input-q', type='number', value=11, step=0.1, min=0.1, max=300, style={'width': '120px'}), 'input-q'),
        ("Altura estática (m):", dcc.Input(id='input-hstatic', type='number', value=190, step=1, style={'width': '120px'}), 'input-hstatic'),
        ("RPM:", dcc.Input(id='input-rpm', type='number', value=3500, step=10, style={'width': '120px'}), 'input-rpm'),
        ("Longitud tubería (m):", dcc.Input(id='input-length', type='number', value=2000, step=1, style={'width': '120px'}), 'input-length'),
        ("Diámetro tubería (mm):", dcc.Input(id='input-diam', type='number', value=150, step=1, style={'width': '120px'}), 'input-diam'),
        ("Bombas en paralelo:", dcc.Input(id='input-nparallel', type='number', value=3, step=1, style={'width': '120px'}), 'input-nparallel'),
        ("C Hazen-Williams:", dcc.Dropdown(
            id='input-hw',
            options=[
                {'label': 'PVC (C=150)', 'value': 150},
                {'label': 'PEAD (C=150)', 'value': 150},
                {'label': 'Asbesto-cemento (nuevo) (C=140)', 'value': 140},
                {'label': 'Cobre (C=135)', 'value': 135},
                {'label': 'Hierro fundido (nuevo) (C=130)', 'value': 130},
                {'label': 'Hierro fundido (10 años) (C=110)', 'value': 110},
                {'label': 'Acero galvanizado (nuevo) (C=120)', 'value': 120},
                {'label': 'Acero soldado (nuevo) (C=120)', 'value': 120},
                {'label': 'Acero soldado (usado) (C=90)', 'value': 90},
                {'label': 'Concreto (acabado liso) (C=125)', 'value': 125},
                {'label': 'Concreto (acabado común) (C=110)', 'value': 110},
                {'label': 'Hierro dúctil (C=135)', 'value': 135},
                {'label': 'Latón (C=130)', 'value': 130},
                {'label': 'Tuberías rectas muy lisas (C=140)', 'value': 140},
                {'label': 'Tuberías de alcantarillado vitrificadas (C=110)', 'value': 110},
                {'label': 'Tuberías en mal estado (C=70)', 'value': 70},
            ],
            value=130,
            clearable=False,
            style={'width': '270px', 'marginBottom': '6px', 'fontSize': '15px'}
        ), 'input-hw'),
        ("Eficiencia pico (%):", dcc.Input(id='input-eff', type='number', value=78, step=1, min=50, max=95, style={'width': '120px'}), 'input-eff'),
        ("Costo electricidad (USD/kWh):", dcc.Input(id='input-cost', type='number', value=0.12, step=0.01, style={'width': '120px'}), 'input-cost'),
        ("Nivel inicial tanque (%):", dcc.Input(id='input-tankinit', type='number', value=80, step=1, min=0, max=100, style={'width': '120px'}), 'input-tankinit'),
        ("Nivel mínimo tanque (%):", dcc.Input(id='input-tankmin', type='number', value=30, step=1, min=0, max=99, style={'width': '120px'}), 'input-tankmin'),
        ("Días de simulación:", dcc.Input(id='input-days', type='number', value=1, step=1, min=1, style={'width': '120px'}), 'input-days'),
        ("Factores horarios (24, separados por coma):", dcc.Textarea(id='input-hourly', value=default_hourly_factors(), style={'width': '180px', 'height': '60px', 'fontSize': '15px'}), 'input-hourly'),
    ]
    filas = [
        html.Div([
            html.Label([
                label,
                html.Span(' ⓘ', id=f'tip-{idcampo}', n_clicks=0, style={'color': '#007bff', 'cursor': 'pointer', 'fontSize': '16px', 'marginLeft': '6px', 'verticalAlign': 'middle'})
            ], style={'flex': '1', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginRight': '10px', 'textAlign': 'right'}),
            html.Div(comp, style={'flex': '1', 'textAlign': 'left'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'marginBottom': '12px'})
        for label, comp, idcampo in campos
    ]
    return html.Div([
        html.H3('Panel de Datos', style={'fontFamily': 'Segoe UI', 'fontWeight': 'bold', 'fontSize': '22px', 'color': '#007bff', 'marginBottom': '18px', 'textAlign': 'center', 'letterSpacing': '1px'}),
        *filas,
        html.Button('Calcular', id='btn-calc', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '120px', 'background': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '6px'}),
        html.Button('Cerrar', id='btn-close-panel', n_clicks=0, style={'marginTop': '12px', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'width': '120px', 'background': '#eee', 'color': '#333', 'border': 'none', 'borderRadius': '6px'}),
        html.Div([
            html.Button('Análisis', id='btn-analisis', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#6f42c1', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto'})
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),
        html.Div(id='tooltip-panel', style=TOOLTIP_STYLE)
    ], id='input-panel-content', style={'padding': '24px', 'fontFamily': 'Segoe UI'})

# --- Callback unificado para mostrar/ocultar el tooltip flotante y cerrar al hacer clic fuera ---
@app.callback(
    Output('tooltip-active', 'data'),
    [Input(f'tip-{idcampo}', 'n_clicks') for idcampo in TOOLTIPS.keys()] + [Input('tooltip-backdrop', 'n_clicks')],
    [State('tooltip-active', 'data')]
)
def set_tooltip_active_unificado(*args):
    ctx = callback_context
    prev = args[-1]
    n = len(TOOLTIPS)
    # Si se hace clic fuera (fondo)
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith('tooltip-backdrop'):
        return ''
    # Si se hace clic en un ícono
    for i, idcampo in enumerate(TOOLTIPS.keys()):
        if ctx.triggered and ctx.triggered[0]['prop_id'].startswith(f'tip-{idcampo}'):
            return '' if prev == idcampo else idcampo
    return prev

@app.callback(
    Output('tooltip-panel', 'children'),
    Output('tooltip-panel', 'style'),
    Input('tooltip-active', 'data'),
    [State('tooltip-panel', 'style')]
)
def show_tooltip_panel(active_id, style):
    style = style.copy() if style else TOOLTIP_STYLE.copy()
    if active_id and active_id in TOOLTIPS:
        idx = list(TOOLTIPS.keys()).index(active_id)
        style['display'] = 'block'
        style['pointerEvents'] = 'auto'
        style['top'] = f'calc(80px + {idx*44}px)'
        style['left'] = '420px'
        return TOOLTIPS[active_id], style
    style['display'] = 'none'
    return '', style

# Layout principal con panel colapsable y criterios colapsables
app.layout = html.Div([
    html.Div(id='material-subtitle', style={
        'position': 'absolute', 'top': '18px', 'left': '32px', 'fontSize': '14px', 'color': '#555',
        'fontFamily': 'Segoe UI', 'fontWeight': 'normal', 'zIndex': 10, 'letterSpacing': '0.5px',
        'background': 'rgba(255,255,255,0.85)', 'padding': '2px 10px', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),
    html.H1('PUMP DESIGN', style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'fontWeight': 'bold', 'fontSize': '2.5rem', 'color': '#007bff', 'marginBottom': '10px', 'marginTop': '10px', 'letterSpacing': '2px'}),
    dcc.Store(id='panel-open', data=False),
    dcc.Store(id='criterio-open', data=False),
    # Panel lateral colapsable de datos
    html.Div([
        html.Div('DATOS', id='tab-datos', n_clicks=0, style={
            'position': 'absolute', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
            'background': 'rgba(0,123,255,0.25)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }),
        html.Div(get_input_panel(), id='input-panel', style={
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(100%)',
            'borderLeft': '2px solid #007bff',
        })
    ], style={'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'zIndex': 20}),
    # Panel colapsable de criterios
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
    # El resto de la app
    html.Div([
        dcc.Loading(
            dcc.Tabs(id='tabs-graficos', value='tab-1', children=[
                dcc.Tab(label='Curvas y Operación', value='tab-1', children=html.Div(id='tab-content-1')),
                dcc.Tab(label='Eficiencia-Potencia-Costo', value='tab-2', children=html.Div(id='tab-content-2')),
                dcc.Tab(label='Leyes de Afinidad', value='tab-3', children=html.Div(id='tab-content-3')),
                dcc.Tab(label='Tanque y Demanda', value='tab-4', children=html.Div(id='tab-content-4')),
            ]),
            type='circle',
            color='#007bff',
            style={'minHeight': '900px'}
        ),
    ], style={'padding': '30px', 'fontFamily': 'Segoe UI', 'position': 'relative', 'minHeight': '100vh', 'background': '#f8f9fa'})
], style={'width': '100vw', 'height': '100vh', 'overflow': 'hidden', 'fontFamily': 'Segoe UI', 'background': '#f4f7fb'})

# --- Store para mostrar análisis ---
app.layout.children.insert(2, dcc.Store(id='show-analysis', data=False))

# --- Store para el campo activo del tooltip ---
app.layout.children.insert(2, dcc.Store(id='tooltip-active', data=''))

# --- Callback para mostrar análisis solo al hacer clic en el botón ---
@app.callback(
    Output('show-analysis', 'data'),
    [Input('btn-analisis', 'n_clicks')]
)
def show_analysis_callback(n_clicks):
    return bool(n_clicks and n_clicks > 0)

# Callbacks para mostrar/ocultar el panel de datos
@app.callback(
    Output('input-panel', 'style'),
    Output('tab-datos', 'style'),
    Output('panel-open', 'data'),
    [Input('tab-datos', 'n_clicks'), Input('btn-close-panel', 'n_clicks')],
    State('panel-open', 'data')
)
def toggle_panel(tab_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, False
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'tab-datos' and not is_open:
        panel_style = {'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
                       'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
                       'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s', 'transform': 'translateX(0)',
                       'borderLeft': '2px solid #007bff'}
        tab_style = {'position': 'absolute', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
                     'background': 'rgba(0,123,255,0.35)', 'color': '#222', 'writingMode': 'vertical-rl',
                     'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
                     'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'opacity': 0.5, 'fontFamily': 'Segoe UI'}
        return panel_style, tab_style, True
    else:
        panel_style = {'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
                       'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
                       'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s', 'transform': 'translateX(100%)',
                       'borderLeft': '2px solid #007bff'}
        tab_style = {'position': 'absolute', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
                     'background': 'rgba(0,123,255,0.25)', 'color': '#222', 'writingMode': 'vertical-rl',
                     'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
                     'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI'}
        return panel_style, tab_style, False

# Callback para mostrar/ocultar criterios
@app.callback(
    Output('criterio-box', 'style'),
    Output('tab-criterio', 'style'),
    Output('criterio-open', 'data'),
    [Input('tab-criterio', 'n_clicks')],
    State('criterio-open', 'data')
)
def toggle_criterio(tab_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, False
    if not is_open:
        box_style = {'position': 'fixed', 'bottom': '30px', 'right': '50px', 'background': 'rgba(255,255,255,0.97)',
                     'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
                     'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'zIndex': 30,
                     'transition': 'transform 0.4s', 'transform': 'translateX(0)', 'borderLeft': '2px solid #007bff'}
        tab_style = {'position': 'absolute', 'bottom': '40px', 'right': 0, 'width': '36px', 'height': '100px',
                     'background': 'rgba(0,123,255,0.28)', 'color': '#222', 'writingMode': 'vertical-rl',
                     'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'borderRadius': '8px 0 0 8px',
                     'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI'}
        return box_style, tab_style, True
    else:
        box_style = {'position': 'fixed', 'bottom': '30px', 'right': '50px', 'background': 'rgba(255,255,255,0.97)',
                     'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
                     'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'zIndex': 30,
                     'transition': 'transform 0.4s', 'transform': 'translateX(120%)', 'borderLeft': '2px solid #007bff'}
        tab_style = {'position': 'absolute', 'bottom': '40px', 'right': 0, 'width': '36px', 'height': '100px',
                     'background': 'rgba(0,123,255,0.18)', 'color': '#222', 'writingMode': 'vertical-rl',
                     'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'borderRadius': '8px 0 0 8px',
                     'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI'}
        return box_style, tab_style, False

# --- FUNCIÓN DE ANÁLISIS EXPERTO (IA DE CRITERIOS) ---
def analisis_criterios_ia(q_op, h_op, power_op, eff_op, velocidad, hw_c, diam_mm, n_parallel):
    soluciones = []
    advertencias = []
    explicaciones = []
    if eff_op < 60:
        advertencias.append("La eficiencia de la bomba es muy baja (<60%). Esto implica alto consumo energético y desgaste. Considere seleccionar una bomba con mejor curva de eficiencia o ajustar el punto de operación.")
        soluciones.append("• Cambie a una bomba con mayor eficiencia en el rango de operación.")
        soluciones.append("• Ajuste el diámetro de la tubería para acercar el punto de operación al BEP (Best Efficiency Point).")
    if velocidad > 2.5:
        advertencias.append("La velocidad en la tubería es excesiva (>2.5 m/s). Esto puede causar erosión y ruidos.")
        soluciones.append("• Aumente el diámetro de la tubería para reducir la velocidad.")
    if power_op > 50:
        advertencias.append("La potencia requerida es muy alta (>50 kW).")
        soluciones.append("• Revise si la altura estática o las pérdidas pueden optimizarse.")
        soluciones.append("• Considere dividir el bombeo en varias etapas o usar bombas en paralelo.")
    if hw_c < 110:
        advertencias.append("El coeficiente Hazen-Williams es bajo (<110), lo que indica tubería rugosa o envejecida.")
        soluciones.append("• Considere reemplazar la tubería por un material más liso o nuevo.")
    explicaciones.append(f"El punto de operación calculado es {q_op:.2f} L/s a {h_op:.2f} m de carga, con una potencia de {power_op:.2f} kW y eficiencia de {eff_op:.2f}%.")
    if not advertencias:
        explicaciones.append("No se detectaron problemas críticos. El diseño es adecuado según los criterios técnicos habituales.")
    return advertencias, soluciones, explicaciones

# FUNCIÓN DE ANÁLISIS CON OLLAMA/LLAMA2

def analisis_ollama_llama2(datos):
    try:
        prompt = (
            f"Eres un ingeniero hidráulico experto. Analiza este sistema de bombeo y entrega un análisis profesional, advertencias, criterios técnicos y sugerencias de mejora en español:\n"
            f"Caudal: {datos.get('q_op', 0):.2f} L/s\n"
            f"Altura: {datos.get('h_op', 0):.2f} m\n"
            f"Potencia: {datos.get('power_op', 0):.2f} kW\n"
            f"Eficiencia: {datos.get('eff_op', 0):.2f}%\n"
            f"Velocidad: {datos.get('velocidad', 0):.2f} m/s\n"
            f"C Hazen-Williams: {datos.get('hw_c', 0)}\n"
            f"Diámetro: {datos.get('diam_mm', 0)} mm\n"
            f"Bombas en paralelo: {datos.get('n_parallel', 0)}\n"
            f"Da un análisis técnico, advertencias y sugerencias en español."
        )
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.ok:
            return response.json().get("response", "(La IA no devolvió respuesta)")
        else:
            print(f"[Ollama/Llama2] Error HTTP: {response.status_code} - {response.text}")
            return "No se pudo obtener el análisis de la IA (Llama2). Verifique que Ollama esté corriendo y el modelo cargado."
    except Exception as e:
        print(f"[Ollama/Llama2] Excepción: {e}")
        return "No se pudo conectar con la IA local (Llama2). Revise que Ollama esté activo y el modelo cargado. Si el problema persiste, consulte la consola para más detalles."

# Callback para actualizar los gráficos en cada pestaña
@app.callback(
    [Output('tab-content-1', 'children'),
     Output('tab-content-2', 'children'),
     Output('tab-content-3', 'children'),
     Output('tab-content-4', 'children'),
     Output('criterio-box', 'children')],
    [Input('btn-calc', 'n_clicks'),
     Input('input-q', 'value'),
     Input('input-hstatic', 'value'),
     Input('input-rpm', 'value'),
     Input('input-length', 'value'),
     Input('input-diam', 'value'),
     Input('input-nparallel', 'value'),
     Input('input-hw', 'value'),
     Input('input-eff', 'value'),
     Input('input-cost', 'value'),
     Input('input-tankinit', 'value'),
     Input('input-tankmin', 'value'),
     Input('input-days', 'value'),
     Input('input-hourly', 'value'),
     Input('show-analysis', 'data')],
)
def update_dashboard(n_clicks, Q_lps, h_static, rpm, length, diam_mm, n_parallel, hw_c, eff_peak, elec_cost, tank_init, tank_min, days, hourly_str, show_analysis):
    logging.info("--- DEBUG: Entradas del callback update_dashboard ---")
    logging.info(f"Q_lps={Q_lps}, h_static={h_static}, rpm={rpm}, length={length}, diam_mm={diam_mm}, n_parallel={n_parallel}, hw_c={hw_c}, eff_peak={eff_peak}, elec_cost={elec_cost}, tank_init={tank_init}, tank_min={tank_min}, days={days}, hourly_str={hourly_str}, show_analysis={show_analysis}")
    # Validación reforzada
    if None in [Q_lps, h_static, rpm, length, diam_mm, n_parallel, hw_c, eff_peak, elec_cost, tank_init, tank_min, days, hourly_str]:
        logging.warning("[ADVERTENCIA] Algún campo de entrada es None")
        advert = dcc.Markdown('**Advertencia:** Complete todos los campos del panel de datos antes de calcular.')
        return [advert], [advert], [advert], [advert], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
    try:
        if not (0.1 <= Q_lps <= 300):
            logging.warning("[ADVERTENCIA] Caudal fuera de rango")
            advert = dcc.Markdown('**Advertencia:** El caudal de diseño debe estar entre 0.1 y 300 L/s.')
            return [advert], [advert], [advert], [advert], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
        if not (0 <= h_static <= 500):
            raise ValueError("Altura estática debe estar entre 0 y 500 m")
        if not (1000 <= rpm <= 5000):
            raise ValueError("RPM debe estar entre 1000 y 5000")
        if not (0 < length <= 10000):
            raise ValueError("Longitud de tubería debe estar entre 0 y 10000 m")
        if not (50 <= diam_mm <= 1000):
            raise ValueError("Diámetro de tubería debe estar entre 50 y 1000 mm")
        if not (1 <= n_parallel <= 10):
            raise ValueError("Número de bombas en paralelo debe estar entre 1 y 10")
        if not (100 <= hw_c <= 150):
            raise ValueError("Coeficiente Hazen-Williams debe estar entre 100 y 150")
        if not (0.5 <= eff_peak <= 0.95):
            raise ValueError("Eficiencia pico debe estar entre 0.5 y 0.95")
        if not (0 <= elec_cost <= 1):
            raise ValueError("Costo de electricidad debe estar entre 0 y 1 USD/kWh")
        if not (0 <= tank_init <= 100):
            raise ValueError("Nivel inicial de tanque debe estar entre 0 y 100%")
        if not (0 <= tank_min < tank_init):
            raise ValueError("Nivel mínimo de tanque debe ser menor que el nivel inicial")
        if not (1 <= days <= 30):
            raise ValueError("Días de simulación deben estar entre 1 y 30")
        hourly_factors = [float(x.strip()) for x in hourly_str.split(',')]
        if len(hourly_factors) != 24 or not all(0 <= f <= 5 for f in hourly_factors):
            raise ValueError("Ingrese 24 factores horarios entre 0 y 5, separados por coma")
    except ValueError as e:
        logging.error(f"[ERROR] {str(e)}")
        advert = dcc.Markdown(f'**Error:** {str(e)}')
        return [advert], [advert], [advert], [advert], dcc.Markdown(get_criterio_texto(), style={'margin': 0})

    # --- BLOQUE DE DEPURACIÓN PROFUNDA ---
    import traceback
    try:
        logging.info("[DEBUG] Creando objeto CentrifugalPumpDesigner...")
        q_design = Q_lps / 1000
        pipe_diameter_m = diam_mm / 1000
        designer = CentrifugalPumpDesigner(
            q_design=q_design,
            h_static=h_static,
            rpm=rpm,
            hourly_factors=hourly_factors,
            pipe_length_m=length,
            pipe_diameter_m=pipe_diameter_m,
            n_parallel=n_parallel,
            hw_c=hw_c,
            eff_peak=eff_peak,
            electricity_cost=elec_cost,
            min_tank_level_perc=tank_min/100,
            initial_tank_level_perc=tank_init/100,
            simulation_days=days
        )
        logging.info("[DEBUG] Objeto designer creado correctamente.")
        figs = [
            designer.plot_system_vs_pump(),
            designer.plot_vfd_comparison(),
            designer.plot_pump_operating_range(),
            designer.plot_efficiency_vs_flow(),
            designer.plot_total_power_analysis(),
            designer.plot_cost_per_m3_vs_flow(),
            designer.plot_affinity(),
            designer.plot_power_affinity(),
            designer.plot_efficiency_affinity(),
            designer.plot_tank_volume(),
            designer.plot_demand_vs_inflow(),
        ]
        titles = [
            "Curva del sistema vs bomba",
            "Comparación VFD",
            "Rango operacional bomba",
            "Eficiencia vs caudal",
            "Potencia total",
            "Costo unitario vs caudal",
            "Leyes de afinidad",
            "Potencia vs caudal (afinidad)",
            "Eficiencia vs caudal (afinidad)",
            "Volumen en tanque",
            "Demanda vs bombeo"
        ]
        # Validar si alguna figura es None o vacía
        for idx, fig in enumerate(figs):
            if fig is None or not hasattr(fig, 'data') or len(fig.data) == 0:
                logging.error(f"[ERROR] La figura '{titles[idx]}' es None o está vacía.")
                advert = dcc.Markdown(f'**Error:** No se pudo generar el gráfico "{titles[idx]}". Revise los datos de entrada.')
                return [advert], [advert], [advert], [advert], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
        logging.info("[DEBUG] Todas las figuras generadas correctamente.")
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.critical(f"[EXCEPCIÓN CRÍTICA] {e}\n{tb_str}")
        # Escribir el error en un archivo de log (además del logger)
        with open('error_dash.log', 'w', encoding='utf-8') as f:
            f.write(tb_str)
        # Mostrar el error y el traceback en la interfaz
        advert = html.Div([
            html.H4('Error crítico en la app', style={'color': 'red'}),
            html.Pre(tb_str, style={'fontSize': '13px', 'color': '#b30000', 'background': '#fff0f0', 'padding': '10px', 'borderRadius': '7px', 'overflowX': 'auto'})
        ], style={'background': '#ffeaea', 'border': '2px solid #b30000', 'borderRadius': '8px', 'padding': '12px', 'margin': '20px auto', 'maxWidth': '900px'})
        return [advert], [advert], [advert], [advert], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
    # Distribución de gráficos por pestaña
    tabs_graficos = [
        [0, 1, 2],   # Tab 1
        [3, 4, 5],   # Tab 2
        [6, 7, 8],   # Tab 3
        [9, 10]      # Tab 4
    ]
    tab_contents = []
    graph_ids = []
    for tab_idx, tab in enumerate(tabs_graficos):
        children = []
        for i in tab:
            graph_id = f'graph-{i}'
            graph_ids.append(graph_id)
            fig_filtrada = filtrar_figura(figs[i], titles[i], i)
            # Si es pestaña 3 (índices 6, 7, 8), modificar hovertemplate de cada traza
            if i in [6, 7, 8]:
                for trace in fig_filtrada.data:
                    if hasattr(trace, 'hovertemplate'):
                        trace.hovertemplate = '%{y}<extra></extra>'
            # Cuadro de punto de operación para los gráficos relevantes
            op_box = None
            analisis = []  # <-- INICIALIZACIÓN AQUÍ
            if i == 0:  # Curva del sistema vs bomba
                q_op = designer.q_op * designer.n_parallel * 1000  # L/s total
                h_op = designer.h_op  # m
                power_op = designer.power_at_op_kW() * designer.n_parallel
                power_hp = power_op / 0.7457
                op_box = html.Div(f"Punto de operación: {q_op:.2f} L/s, {h_op:.2f} m de carga, {power_op:.2f} kW ({power_hp:.2f} HP)", style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'} )
            elif i == 1:  # Comparación VFD
                vfd = designer.vfd_results
                if vfd:
                    q_op = vfd.get('q_op_total_lps', 0)
                    h_op = vfd.get('h_op', 0)
                    rpm_op = designer.rpm * vfd.get('speed_ratio', 1)
                    power_op = vfd.get('total_power_kw', 0)
                    power_hp = power_op / 0.7457
                    op_box = html.Div(
                        f"Punto de operación: {q_op:.2f} L/s, {h_op:.2f} m de carga, {rpm_op:.1f} rpm, {power_op:.2f} kW ({power_hp:.2f} HP)",
                        style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'}
                    )
            elif i == 3:  # Eficiencia vs caudal
                q_op = designer.q_op * 1000  # L/s por bomba
                eff_op = designer.efficiency(designer.q_op) * 100
                power_op = designer.power_at_op_kW()
                power_hp = power_op / 0.7457
                op_box = html.Div(f"Punto de operación: {q_op:.2f} L/s, {eff_op:.2f}% eficiencia, {power_op:.2f} kW ({power_hp:.2f} HP)", style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'} )
            elif i == 4:  # Potencia total
                if hasattr(designer, 'vfd_results') and designer.vfd_results:
                    q_op = designer.vfd_results.get('q_op_total_lps', 0)
                    power_op = designer.vfd_results.get('total_power_kw', 0)
                    power_hp = power_op / 0.7457
                    h_op = designer.vfd_results.get('h_op', 0)
                    eff_op = designer.efficiency(designer.q_op) * 100
                    velocidad = designer.velocity(designer.q_op)
                    op_box = html.Div(
                        f"Punto de operación: {q_op:.2f} L/s, {power_op:.2f} kW ({power_hp:.2f} HP)",
                        style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px',
                               'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center',
                               'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'}
                    )
                    # --- ANÁLISIS IA OLLAMA ---
                    # datos_ia = dict(q_op=q_op, h_op=h_op, power_op=power_op, eff_op=eff_op, velocidad=velocidad, hw_c=hw_c, diam_mm=diam_mm, n_parallel=n_parallel)
                    # analisis_ia = analisis_ollama_llama2(datos_ia)
                    analisis.append(html.Div([
                        html.H4('Análisis IA (Llama2)', style={'color': '#6f42c1', 'fontSize': '16px', 'marginBottom': '8px', 'textAlign': 'center'}),
                        html.P("Esto es un análisis de prueba.", style={'fontSize': '15px', 'textAlign': 'left'})
                    ], style={'background': '#f8f9fa', 'border': '2px solid #6f42c1', 'borderRadius': '8px', 'padding': '12px', 'margin': '10px auto 0 auto', 'width': '100%', 'maxWidth': '500px'}))
                    # --- FIN ANÁLISIS IA ---
                    # Calcular potencia de referencia a ese caudal
                    q_ref = q_op / (designer.n_parallel if designer.n_parallel else 1) / 1000
                    power_ref = designer.power_kW(q_ref) * designer.n_parallel
                    ahorro = power_ref - power_op
                    ahorro_pct = 100 * (ahorro / power_ref) if power_ref > 0 else 0
                    advertencia = ""
                    if power_op < power_ref * 0.95:
                        advertencia = (f"\n\n<b>Advertencia:</b> El punto de operación con VFD está por debajo de la curva de potencia de referencia. "
                            f"Esto es normal y significa que el variador de frecuencia está ahorrando energía. "
                            f"El ahorro estimado es de <b>{ahorro:.2f} kW</b> ({ahorro_pct:.1f}%). "
                            f"Sin embargo, si la bomba opera mucho tiempo a baja velocidad, verifique que no esté fuera del rango óptimo de eficiencia (BEP) y que no haya riesgos de inestabilidad hidráulica o cavitación.")
                    analisis.append(f"<b>Análisis:</b> La potencia total requerida es <b>{power_op:.2f} kW</b>. Si la potencia es muy alta, revise el dimensionamiento del sistema y la selección de la bomba." + advertencia)
                else:
                    q_op = designer.q_op * designer.n_parallel * 1000  # L/s total
                    power_op = designer.power_kW(designer.q_op) * designer.n_parallel
                    power_hp = power_op / 0.7457
                    h_op = designer.h_op
                    eff_op = designer.efficiency(designer.q_op) * 100
                    velocidad = designer.velocity(designer.q_op)
                    op_box = html.Div(
                        f"Punto de operación: {q_op:.2f} L/s, {power_op:.2f} kW ({power_hp:.2f} HP)",
                        style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px',
                               'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center',
                               'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'}
                    )
                    # --- ANÁLISIS IA OLLAMA ---
                    # datos_ia = dict(q_op=q_op, h_op=h_op, power_op=power_op, eff_op=eff_op, velocidad=velocidad, hw_c=hw_c, diam_mm=diam_mm, n_parallel=n_parallel)
                    # analisis_ia = analisis_ollama_llama2(datos_ia)
                    analisis.append(html.Div([
                        html.H4('Análisis IA (Llama2)', style={'color': '#6f42c1', 'fontSize': '16px', 'marginBottom': '8px', 'textAlign': 'center'}),
                        html.P("Esto es un análisis de prueba.", style={'fontSize': '15px', 'textAlign': 'left'})
                    ], style={'background': '#f8f9fa', 'border': '2px solid #6f42c1', 'borderRadius': '8px', 'padding': '12px', 'margin': '10px auto 0 auto', 'width': '100%', 'maxWidth': '500px'}))
                    # --- FIN ANÁLISIS IA ---
                    analisis.append(f"<b>Análisis:</b> La potencia total requerida es <b>{power_op:.2f} kW</b> (<b>{power_hp:.2f} HP</b>). Si la potencia es muy alta, revise el dimensionamiento del sistema y la selección de la bomba.")
            elif i == 5:  # Costo unitario vs caudal
                q_op = designer.q_op * 1000
                cost_op = (designer.power_kW(designer.q_op) * designer.electricity_cost) / (designer.q_op * 3600)
                power_op = designer.power_kW(designer.q_op)
                power_hp = power_op / 0.7457
                op_box = html.Div(f"Punto de operación: {q_op:.2f} L/s, {cost_op:.4f} USD/m³, {power_op:.2f} kW ({power_hp:.2f} HP)", style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'} )
            elif i == 2 and designer.vfd_results:  # Rango operacional bomba
                q_op = designer.vfd_results.get('q_op_per_pump_lps', 0)
                h_op = designer.vfd_results.get('h_op', 0)
                op_box = html.Div(f"Punto de operación: {q_op:.2f} L/s, {h_op:.2f} m de carga", style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'} )
            elif i == 7:  # Potencia vs caudal (afinidad)
                q_op = designer.q_op * 1000
                power_op = designer.power_at_op_kW()
                op_box = html.Div(f"Punto de operación: {q_op:.2f} L/s, {power_op:.2f} kW", style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'} )
            elif i == 8:  # Eficiencia vs caudal (afinidad)
                q_op = designer.q_op * 1000
                eff_op = designer.efficiency(designer.q_op) * 100
                op_box = html.Div(f"Punto de operación: {q_op:.2f} L/s, {eff_op:.2f}% eficiencia", style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px', 'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center', 'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'} )
            # --- ANÁLISIS Y RECOMENDACIONES ESPECÍFICAS ---
            analysis_box = None
            if op_box and show_analysis:
                analisis = []
                if i == 0:  # Curva del sistema vs bomba
                    analisis.append(f"<b>Análisis:</b> La bomba seleccionada entrega <b>{q_op:.2f} L/s</b> a <b>{h_op:.2f} m</b> de carga. Si el punto de operación está muy lejos del óptimo, la eficiencia y la vida útil pueden verse afectadas. Si la potencia es alta (<b>{power_op:.2f} kW</b>), considere aumentar el diámetro de la tubería o reducir la altura estática. Si el caudal es insuficiente, revise la selección de la bomba.")
                elif i == 1:  # Comparación VFD
                    analisis.append(f"<b>Análisis:</b> Con variador de frecuencia, la bomba opera a <b>{rpm_op:.1f} rpm</b> para entregar <b>{q_op:.2f} L/s</b> a <b>{h_op:.2f} m</b>. Esto permite ajustar el punto de operación y ahorrar energía (<b>{power_op:.2f} kW</b>, <b>{power_hp:.2f} HP</b>). Si la velocidad es muy baja, verifique que la bomba mantenga buena eficiencia y evite zonas de inestabilidad hidráulica.")
                elif i == 2 and designer.vfd_results:  # Rango operacional bomba
                    q_min = designer.vfd_results.get('q_min_lps', None)
                    q_max = designer.vfd_results.get('q_max_lps', None)
                    analisis.append(f"<b>Análisis:</b> El rango operacional muestra los caudales mínimos y máximos recomendados para la bomba seleccionada. Operar dentro de este rango garantiza flexibilidad y evita daños por cavitación o sobrecarga. Si el punto de operación está fuera de este rango, ajuste la selección de bomba o la configuración del sistema.")
                elif i == 3:  # Eficiencia vs caudal
                    analisis.append(f"<b>Análisis:</b> La eficiencia de la bomba en el punto de operación es <b>{eff_op:.2f}%</b>. Procure que el punto de operación esté cerca del máximo de la curva para minimizar el consumo energético y el desgaste. Si la eficiencia es baja, considere otra bomba o ajuste el sistema.")
                elif i == 4:  # Potencia total
                    # La advertencia ya se incluyó en el análisis anterior cuando se definió
                    analisis.append(f"<b>Análisis:</b> La potencia total requerida es <b>{power_op:.2f} kW</b>. Si la potencia es muy alta, revise el dimensionamiento del sistema y la selección de la bomba.")
                elif i == 5:  # Costo unitario vs caudal
                    analisis.append(f"<b>Análisis:</b> El costo unitario de bombeo es <b>{cost_op:.4f} USD/m³</b>. Para reducir costos, optimice la eficiencia y evite operar fuera del rango recomendado.")
                analysis_box = html.Div([
                    html.Div(dcc.Markdown("\n".join(analisis), dangerously_allow_html=True),
                             style={'background': '#f8f9fa', 'border': '1px solid #dee2e6', 'borderRadius': '7px', 'padding': '7px 14px', 'margin': '10px auto 0 auto', 'display': 'block', 'fontSize': '15px', 'textAlign': 'left', 'width': 'fit-content'})
                ])
            # --- FIN ANÁLISIS ---
            if op_box or analysis_box:
                children.append(html.Div([
                    dcc.Graph(id=graph_id, figure=fig_filtrada, style={'width': '32vw', 'height': '60vh', 'marginBottom': '0'}, clear_on_unhover=False),
                    op_box if op_box else None,
                    analysis_box if analysis_box else None
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'flex': '1 1 0', 'minWidth': '320px'}))
            else:
                children.append(html.Div([
                    dcc.Graph(id=graph_id, figure=fig_filtrada, style={'width': '32vw', 'height': '60vh', 'marginBottom': '0'}, clear_on_unhover=False)
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'flex': '1 1 0', 'minWidth': '320px'}))
        tab_contents.append(html.Div(children, style={'display': 'flex', 'flexDirection': 'row', 'gap': '30px', 'marginRight': '400px', 'marginTop': '20px', 'justifyContent': 'flex-start', 'alignItems': 'stretch', 'width': '100%'}))

    # --- Eliminar callbacks dinámicos de hover ---
    # (Elimina el bloque que genera y ejecuta graph_callbacks_code con exec)
    # --- FIN ---
    return (*tab_contents, dcc.Markdown(get_criterio_texto(), style={'margin': 0}))

# --- Subtítulo dinámico del material ---
@app.callback(
    Output('material-subtitle', 'children'),
    [Input('input-hw', 'value')]
)
def update_material_subtitle(hw_c):
    materiales = {
        150: 'PVC o PEAD',
        140: 'Asbesto-cemento (nuevo), Tuberías rectas muy lisas',
        135: 'Cobre, Hierro dúctil',
        130: 'Hierro fundido (nuevo), Latón',
        125: 'Concreto (acabado liso)',
        120: 'Acero galvanizado (nuevo), Acero soldado (nuevo)',
        110: 'Hierro fundido (10 años), Concreto (acabado común), Tuberías de alcantarillado vitrificadas',
        100: 'No especificado',
        90: 'Acero soldado (usado)',
        70: 'Tuberías en mal estado',
    }
    texto = materiales.get(hw_c, f'Material con C={hw_c}')
    return f"Material empleado en el análisis: {texto} (C={hw_c})"

# Agregar un fondo transparente para cerrar el tooltip al hacer clic fuera
app.layout.children.append(html.Div(id='tooltip-backdrop', n_clicks=0, style={
    'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
    'zIndex': 9998, 'background': 'rgba(0,0,0,0)', 'display': 'none'
}))

# Callback para mostrar/ocultar el fondo del tooltip
@app.callback(
    Output('tooltip-backdrop', 'style'),
    Input('tooltip-active', 'data')
)
def show_tooltip_backdrop(active_id):
    if active_id:
        return {'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh', 'zIndex': 9998, 'background': 'rgba(0,0,0,0)', 'display': 'block'}
    return {'display': 'none'}

if __name__ == '__main__':
    app.run(debug=True)