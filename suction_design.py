import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

class SuctionPipeDesign:
    """
    Clase para el diseño y análisis de tuberías de succión en estaciones de bombeo centrífugo.
    Incluye análisis hidráulico completo, NPSH, cavitación y criterios de sumergencia.
    """
    
    def __init__(self, fluid_properties=None, pipe_properties=None, operating_conditions=None):
        """
        Inicializa la clase con propiedades del fluido, tubería y condiciones operativas.
        
        Parameters:
        -----------
        fluid_properties : dict
            {'density': kg/m³, 'viscosity': Pa·s, 'vapor_pressure': Pa}
        pipe_properties : dict
            {'length': m, 'roughness': mm, 'fittings_k': adimensional}
        operating_conditions : dict
            {'flow_rate': m³/s, 'temperature': °C, 'suction_height': m, 'atmospheric_pressure': Pa}
        """
        # Propiedades por defecto del agua a 20°C
        self.fluid = fluid_properties or {
            'density': 1000,  # kg/m³
            'viscosity': 0.001,  # Pa·s
            'vapor_pressure': 2337  # Pa
        }
        
        self.pipe = pipe_properties or {
            'length': 10,  # m
            'roughness': 0.15,  # mm
            'fittings_k': 2.5  # Factor K total de accesorios
        }
        
        self.operating = operating_conditions or {
            'flow_rate': 0.05,  # m³/s
            'temperature': 20,  # °C
            'suction_height': 3,  # m
            'atmospheric_pressure': 101325  # Pa
        }
        
        # Rango de diámetros para análisis
        self.diameters = np.linspace(0.1, 0.5, 50)  # metros
        
    def calculate_velocity(self, diameter, flow_rate):
        """Calcula la velocidad del flujo en la tubería"""
        area = np.pi * (diameter/2)**2
        return flow_rate / area
    
    def calculate_reynolds(self, velocity, diameter):
        """Calcula el número de Reynolds"""
        return (self.fluid['density'] * velocity * diameter) / self.fluid['viscosity']
    
    def calculate_friction_factor(self, reynolds, diameter):
        """Calcula el factor de fricción usando la ecuación de Colebrook-White"""
        if reynolds < 2300:
            return 64 / reynolds
        else:
            # Aproximación de Swamee-Jain para flujo turbulento
            relative_roughness = (self.pipe['roughness']/1000) / diameter
            return 0.25 / (np.log10(relative_roughness/3.7 + 5.74/reynolds**0.9))**2
    
    def calculate_friction_loss(self, velocity, diameter):
        """Calcula las pérdidas por fricción usando Darcy-Weisbach"""
        reynolds = self.calculate_reynolds(velocity, diameter)
        f = self.calculate_friction_factor(reynolds, diameter)
        
        # Pérdidas por fricción
        h_friction = f * (self.pipe['length'] / diameter) * (velocity**2) / (2 * 9.81)
        
        # Pérdidas menores
        h_minor = self.pipe['fittings_k'] * (velocity**2) / (2 * 9.81)
        
        return h_friction + h_minor
    
    def calculate_npsh_available(self, velocity, diameter):
        """Calcula el NPSH disponible"""
        h_friction = self.calculate_friction_loss(velocity, diameter)
        
        npsh_a = (self.operating['atmospheric_pressure'] / (self.fluid['density'] * 9.81) -
                 self.operating['suction_height'] -
                 h_friction -
                 self.fluid['vapor_pressure'] / (self.fluid['density'] * 9.81))
        
        return npsh_a
    
    def calculate_minimum_pressure(self, velocity, diameter):
        """Calcula la presión mínima en la tubería"""
        h_friction = self.calculate_friction_loss(velocity, diameter)
        
        # Presión en el punto más crítico (entrada de la bomba)
        p_min = (self.operating['atmospheric_pressure'] -
                self.fluid['density'] * 9.81 * self.operating['suction_height'] -
                self.fluid['density'] * 9.81 * h_friction)
        
        return p_min
    
    def calculate_minimum_submergence(self, diameter):
        """
        Calcula la sumergencia mínima usando criterios normativos para bombas centrífugas
        
        Criterios aplicados:
        - Ec. 3.18: S = 2.5*D + 0.1 (Impedir ingreso de aire)
        - Ec. 3.19: S = 2.5*(v²/2g) + 0.2 (Condición hidráulica)
        - Mínimo 0.7 m para v=0.6-0.9 m/s y D<250mm
        """
        velocity = self.calculate_velocity(diameter, self.operating['flow_rate'])
        
        # Criterio 1: Impedir ingreso de aire (Ec. 3.18)
        # S = 2.5*D + 0.1
        s_air = 2.5 * diameter + 0.1
        
        # Criterio 2: Condición hidráulica (Ec. 3.19)
        # S = 2.5*(v²/2g) + 0.2
        s_hydraulic = 2.5 * (velocity**2 / (2 * 9.81)) + 0.2
        
        # Criterio 3: Condición especial para velocidades bajas y diámetros pequeños
        # Mínimo 0.7 m para v=0.6-0.9 m/s y D<250mm
        s_special = 0.3  # Valor por defecto
        if 0.6 <= velocity <= 0.9 and diameter < 0.25:
            s_special = 0.7
        
        # Tomar el mayor de los tres criterios (más conservador)
        s_min = max(s_air, s_hydraulic, s_special)
        
        # Límite mínimo absoluto de seguridad
        s_min = max(s_min, 0.3)
        
        return s_min
    
    def analyze_all_diameters(self):
        """Realiza el análisis completo para todos los diámetros"""
        results = {
            'diameter': [],
            'velocity': [],
            'reynolds': [],
            'friction_loss': [],
            'npsh_available': [],
            'min_pressure': [],
            'cavitation_risk': [],
            'submergence_min': []
        }
        
        for d in self.diameters:
            v = self.calculate_velocity(d, self.operating['flow_rate'])
            re = self.calculate_reynolds(v, d)
            h_loss = self.calculate_friction_loss(v, d)
            npsh_a = self.calculate_npsh_available(v, d)
            p_min = self.calculate_minimum_pressure(v, d)
            cavitation_risk = p_min < self.fluid['vapor_pressure']
            s_min = self.calculate_minimum_submergence(d)
            
            results['diameter'].append(d)
            results['velocity'].append(v)
            results['reynolds'].append(re)
            results['friction_loss'].append(h_loss)
            results['npsh_available'].append(npsh_a)
            results['min_pressure'].append(p_min)
            results['cavitation_risk'].append(cavitation_risk)
            results['submergence_min'].append(s_min)
        
        return pd.DataFrame(results)
    
    
    def plot_comprehensive_analysis_interactive(self):
        """Genera gráficos interactivos completos del análisis usando Plotly"""
        df = self.analyze_all_diameters()
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Velocidad vs Diámetro',
                'Régimen de Flujo (Reynolds)',
                'NPSH Disponible vs Diámetro',
                'Pérdidas por Fricción',
                'Riesgo de Cavitación',
                'Sumergencia para Evitar Vórtices'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Datos comunes
        diameters_mm = df['diameter'] * 1000
        
        # 1. Velocidad vs Diámetro
        fig.add_trace(
            go.Scatter(
                x=diameters_mm,
                y=df['velocity'],
                mode='lines',
                name='Velocidad',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                             '<b>Velocidad:</b> %{y:.3f} m/s<br>' +
                             '<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Líneas de referencia para velocidad
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", opacity=0.7,
                      annotation_text="Límite recomendado (1.5 m/s)", row=1, col=1)

        
        # 2. Número de Reynolds (escala logarítmica)
        fig.add_trace(
            go.Scatter(
                x=diameters_mm,
                y=df['reynolds'],
                mode='lines',
                name='Reynolds',
                line=dict(color='green', width=3),
                hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                             '<b>Reynolds:</b> %{y:.0f}<br>' +
                             '<b>Régimen:</b> %{customdata}<br>' +
                             '<extra></extra>',
                customdata=['Laminar' if re < 2300 else 'Transición' if re < 4000 else 'Turbulento'
                           for re in df['reynolds']],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Actualizar eje Y a escala logarítmica para Reynolds
        fig.update_yaxes(type="log", row=1, col=2)
        
        # Líneas de referencia para Reynolds
        fig.add_hline(y=2300, line_dash="dash", line_color="red", opacity=0.7,
                      annotation_text="Transición laminar-turbulento", row=1, col=2)
        fig.add_hline(y=4000, line_dash="dash", line_color="orange", opacity=0.7,
                      annotation_text="Flujo turbulento establecido", row=1, col=2)
        
        # 3. NPSH Disponible
        fig.add_trace(
            go.Scatter(
                x=diameters_mm,
                y=df['npsh_available'],
                mode='lines',
                name='NPSH disponible',
                line=dict(color='purple', width=3),
                hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                             '<b>NPSH disponible:</b> %{y:.3f} m<br>' +
                             '<b>Estado:</b> %{customdata}<br>' +
                             '<extra></extra>',
                customdata=['Suficiente' if npsh >= 3.0 else 'Insuficiente'
                           for npsh in df['npsh_available']],
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Línea de referencia para NPSH
        fig.add_hline(y=self.operating['NPSH_required'], line_dash="dash", line_color="red", opacity=0.7,
                      annotation_text="NPSH mínimo requerido", row=1, col=3)
        
        # 4. Pérdidas de Carga
        fig.add_trace(
            go.Scatter(
                x=diameters_mm,
                y=df['friction_loss'],
                mode='lines',
                name='Pérdidas de Carga',
                line=dict(color='brown', width=3),
                hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                             '<b>Pérdidas:</b> %{y:.4f} m<br>' +
                             '<b>% de altura succión:</b> %{customdata:.1f}%<br>' +
                             '<extra></extra>',
                customdata=[(loss/self.operating['suction_height'])*100
                           for loss in df['friction_loss']],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Análisis de Presión y Cavitación
        fig.add_trace(
            go.Scatter(
                x=diameters_mm,
                y=df['min_pressure']/1000,
                mode='lines',
                name='Presión mínima',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                             '<b>Presión mínima:</b> %{y:.2f} kPa<br>' +
                             '<b>Riesgo cavitación:</b> %{customdata}<br>' +
                             '<extra></extra>',
                customdata=['SÍ' if risk else 'NO' for risk in df['cavitation_risk']],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Línea de referencia para presión de vapor
        fig.add_hline(y=self.fluid['vapor_pressure']/1000, line_dash="dash", line_color="red", opacity=0.7,
                      annotation_text="Presión de vapor", row=2, col=2)
        
        # 6. Sumergencia Mínima
        fig.add_trace(
            go.Scatter(
                x=diameters_mm,
                y=df['submergence_min'],
                mode='lines',
                name='Sumergencia Mínima',
                line=dict(color='cyan', width=3),
                hovertemplate='<b>Diámetro:</b> %{x:.0f} mm<br>' +
                             '<b>Sumergencia mínima:</b> %{y:.3f} m<br>' +
                             '<b>Velocidad:</b> %{customdata:.2f} m/s<br>' +
                             '<extra></extra>',
                customdata=df['velocity'],
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Configurar títulos de ejes
        fig.update_xaxes(title_text="Diámetro (mm)", row=1, col=1)
        fig.update_yaxes(title_text="Velocidad (m/s)", row=1, col=1)
        
        fig.update_xaxes(title_text="Diámetro (mm)", row=1, col=2)
        fig.update_yaxes(title_text="Número de Reynolds", row=1, col=2)
        
        fig.update_xaxes(title_text="Diámetro (mm)", row=1, col=3)
        fig.update_yaxes(title_text="NPSH (m)", row=1, col=3)
        
        fig.update_xaxes(title_text="Diámetro (mm)", row=2, col=1)
        fig.update_yaxes(title_text="Pérdidas de Carga (m)", row=2, col=1)
        
        fig.update_xaxes(title_text="Diámetro (mm)", row=2, col=2)
        fig.update_yaxes(title_text="Presión (kPa)", row=2, col=2)
        
        fig.update_xaxes(title_text="Diámetro (mm)", row=2, col=3)
        fig.update_yaxes(title_text="Sumergencia Mínima (m)", row=2, col=3)
        

    
    
    
        fig.update_layout(
            title={
                'text': f'Tubería de Succión - q: {self.operating["flow_rate"]*1000:.1f} L/s, succion negativa: {self.operating["suction_height"]:.1f} m',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            height=1600,
            width=2500,
            showlegend=True,
            hovermode='closest',
            # Configuración de grillas
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Agregar grillas a todos los subplots
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Mostrar el gráfico
        fig.show()
        
        return df


    def generate_design_report(self):
        """Genera un reporte técnico del análisis"""
        df = self.analyze_all_diameters()
        
        # Encuentra el diámetro óptimo (mínimo riesgo, máximo NPSH, velocidad aceptable)
        safe_diameters = df[df['cavitation_risk'] == False]
        if len(safe_diameters) > 0:
            optimal_idx = safe_diameters['npsh_available'].idxmax()
            optimal_diameter = safe_diameters.loc[optimal_idx, 'diameter']
        else:
            optimal_idx = df['npsh_available'].idxmax()
            optimal_diameter = df.loc[optimal_idx, 'diameter']
        
        print("="*80)
        print("REPORTE TÉCNICO - DISEÑO DE TUBERÍA DE SUCCIÓN")
        print("="*80)
        print(f"\nCONDICIONES DE DISEÑO:")
        print(f"• Caudal: {self.operating['flow_rate']:.3f} m³/s")
        print(f"• Altura de succión: {self.operating['suction_height']:.1f} m")
        print(f"• Longitud de tubería: {self.pipe['length']:.1f} m")
        print(f"• Temperatura: {self.operating['temperature']:.1f} °C")
        
        print(f"\nDIÁMETRO ÓPTIMO RECOMENDADO:")
        print(f"• Diámetro: {optimal_diameter*1000:.0f} mm")
        print(f"• Velocidad: {df.loc[optimal_idx, 'velocity']:.2f} m/s")
        print(f"• Reynolds: {df.loc[optimal_idx, 'reynolds']:.0f}")
        print(f"• NPSH disponible: {df.loc[optimal_idx, 'npsh_available']:.2f} m")
        print(f"• Pérdidas de carga: {df.loc[optimal_idx, 'friction_loss']:.3f} m")
        print(f"• Sumergencia mínima: {df.loc[optimal_idx, 'submergence_min']:.2f} m")
        print(f"• Riesgo de cavitación: {'SÍ' if df.loc[optimal_idx, 'cavitation_risk'] else 'NO'}")
        
        print(f"\nCRITERIOS DE EVALUACIÓN:")
        velocities_ok = df[(df['velocity'] >= 1.0) & (df['velocity'] <= 3.0)]
        npsh_ok = df[df['npsh_available'] >= 3.0]
        no_cavitation = df[df['cavitation_risk'] == False]
        
        print(f"• Diámetros con velocidad aceptable (1-3 m/s): {len(velocities_ok)} de {len(df)}")
        print(f"• Diámetros con NPSH suficiente (≥3m): {len(npsh_ok)} de {len(df)}")
        print(f"• Diámetros sin riesgo de cavitación: {len(no_cavitation)} de {len(df)}")
        
        print(f"\nRECOMENDACIONES:")
        if df.loc[optimal_idx, 'velocity'] > 3.0:
            print("⚠️  ADVERTENCIA: Velocidad alta - considerar aumentar diámetro")
        if df.loc[optimal_idx, 'npsh_available'] < 3.0:
            print("⚠️  ADVERTENCIA: NPSH insuficiente - revisar diseño")
        if df.loc[optimal_idx, 'cavitation_risk']:
            print("🚨 CRÍTICO: Riesgo de cavitación - rediseño necesario")
        
        print("="*80)
        
        return df


if __name__ == "__main__":

    

    
    # Crear instancia con parámetros personalizados para análisis completo
    design = SuctionPipeDesign(
        fluid_properties={
            'density': 1000,
            'viscosity': 0.001,
            'vapor_pressure': 17567
        },
        pipe_properties={
            'length': 15,
            'roughness': 0.15,
            'fittings_k': 3.0
        },
        operating_conditions={
            'flow_rate': 80/1000,  # 11 L/s = 0.011 m³/s
            'temperature': 25,
            'suction_height': 3.0,  # Altura de succión
            'NPSH_required': 3.0,
            'atmospheric_pressure': 101325
        }
    )


    results = design.plot_comprehensive_analysis_interactive()

    # Generar reporte técnico
    design.generate_design_report()


    # Mostrar primeros resultados
    print("\nPrimeros 10 resultados del análisis:")
    print(results.head(100).round(3))