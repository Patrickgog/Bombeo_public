# -*- coding: utf-8 -*-
"""
centrifugal_pump_design.py
==========================

----------------------------------
* Curva del sistema con Hazen-Williams o polinómica.
* Curva de bomba (parabólica) y punto de operación.
* Eficiencia, potencia, leyes de afinidad (100–10 %).
* Balance de masa diario y costos energéticos.

Requisitos:
    pip install numpy plotly pandas
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constantes físicas
G = 9.80665            # m/s^2
RHO_WATER = 1000.0     # kg/m^3 (agua a ~20 °C)


class CentrifugalPumpDesigner:
    """Herramienta para dimensionar y analizar bombas centrífugas."""

    def __init__(
        self,
        q_design: float,
        h_static: float,
        rpm: float,
        hourly_factors: Sequence[float],
        pipe_length_m: float,
        pipe_diameter_m: float,
        *,
        n_parallel: int = 1,
        hw_c: float = 130.0,
        pump_coeffs: tuple[float, float] | None = None,
        eff_peak: float = 0.78,
        electricity_cost: float = 0.12,
        tank_capacity_m3: float | None = None,
        min_tank_level_perc: float = 0.3,
        initial_tank_level_perc: float = 1.0,
        simulation_days: int = 1,
    ) -> None:
        # Validaciones
        if len(hourly_factors) != 24:
            raise ValueError("hourly_factors debe contener 24 valores: uno por cada hora del día")
        if not 0.0 <= min_tank_level_perc < 1.0:
            raise ValueError("min_tank_level_perc debe estar entre 0.0 y 1.0")
        if not 0.0 <= initial_tank_level_perc <= 1.0:
            raise ValueError("initial_tank_level_perc debe estar entre 0.0 y 1.0")
        if not simulation_days > 0:
            raise ValueError("simulation_days debe ser mayor que 0")

        # Parámetros básicos
        self.q_design = float(q_design)
        self.h_static = float(h_static)
        self.rpm = float(rpm)
        self.hourly_factors = np.asarray(hourly_factors, dtype=float)
        self.n_parallel = int(n_parallel)
        self.initial_tank_level_perc = float(initial_tank_level_perc)
        self.simulation_days = int(simulation_days)

        # Sistema hidráulico (Hazen-Williams)
        self.pipe_length_m = float(pipe_length_m)
        self.pipe_diameter_m = float(pipe_diameter_m)
        self.hw_c = float(hw_c)

        # Curva de la bomba y eficiencia
        self.eff_peak = eff_peak
        self.electricity_cost = electricity_cost

        if pump_coeffs is None:
            # Estimar curva de la bomba si no se provee
            # 1. Carga de diseño es la del sistema para el caudal de diseño
            h_design = self.system_head(self.q_design)
            # 2. Carga de cierre (shutoff) ~130% de la de diseño (típico)
            h_shutoff = 1.3 * h_design
            # 3. Coeficiente 'a' para la parábola H = H_cierre - a * Q^2
            #    que pasa por (q_design, h_design)
            a_p = (h_shutoff - h_design) / self.q_design**2 if self.q_design > 1e-9 else 0.0
            self.pump_coeffs = (a_p, h_shutoff)
        else:
            self.pump_coeffs = pump_coeffs

        # Rango de caudales para gráficas (0 - 150% Q_design)
        self.q_range = np.linspace(1e-6, 1.5 * self.q_design, 400)

        # Calcular punto de operación
        self.q_op, self.h_op = self._solve_operating_point()

        # Parámetros del tanque y balance hídrico
        self.min_tank_level_perc = min_tank_level_perc
        self.tank_capacity_m3_user = tank_capacity_m3
        if self.tank_capacity_m3_user is None:
            # Modo dimensionamiento: calcular el tamaño del tanque
            self.tank_capacity_m3 = self._size_reservoir()
            self.initial_volume_m3 = self.tank_capacity_m3 * self.initial_tank_level_perc
        else:
            # Modo evaluación: usar un tamaño de tanque existente
            self.tank_capacity_m3 = float(self.tank_capacity_m3_user)
            # Iniciar con el porcentaje especificado por el usuario.
            self.initial_volume_m3 = self.tank_capacity_m3 * self.initial_tank_level_perc

        # Simular el balance diario una vez con la configuración final del tanque
        self.vol_hourly, self.demand_hourly, self.pump_on_hourly = self.daily_balance()

        # --- Análisis VFD ---
        self.vfd_results = self._analyze_vfd_operation_by_flow(self.q_design * 1000)

    def system_head(self, q: np.ndarray | float) -> np.ndarray | float:
        """Carga del sistema H (m) para caudal q (m3/s) usando Hazen-Williams."""
        q = np.asarray(q, dtype=float)
        # Ecuación de Hazen-Williams para pérdidas por fricción
        L, D, C = self.pipe_length_m, self.pipe_diameter_m, self.hw_c
        hf = 10.67 * L * q**1.852 / (C**1.852 * D**4.871)
        return self.h_static + hf

    def pump_head(self, q: np.ndarray | float) -> np.ndarray | float:
        """Curva H-Q de la bomba (m), modelada como H = H_shutoff - a_p * Q^2."""
        a_p, h_shutoff = self.pump_coeffs
        return h_shutoff - a_p * np.asarray(q, dtype=float)**2

    def _analyze_vfd_operation_by_flow(self, target_flow_lps: float) -> dict | None:
        """
        Analiza la operación con VFD para alcanzar un caudal total objetivo.
        """
        q_target_total_m3s = target_flow_lps / 1000.0
        
        # 1. Calcular la altura requerida por el sistema para este caudal
        target_head = self.system_head(q_target_total_m3s)
        
        if target_head > self.pump_coeffs[1]: # H_shutoff @ 100% RPM
            print(f"Advertencia: La altura requerida ({target_head:.2f}m) para el caudal objetivo es mayor que la altura de cierre de la bomba.")
            return None

        q_target_per_pump_m3s = q_target_total_m3s / self.n_parallel

        # 2. Calcular la velocidad de la bomba (ratio) necesaria usando leyes de afinidad
        a_p, h_shutoff = self.pump_coeffs
        
        # H_op = H_shutoff * r^2 - a_p * (Q_op/r)^2 * r^2  -> H_op = H_shutoff * r^2 - a_p * Q_op^2
        # No, la ecuación de la bomba a velocidad variable es H = H_shutoff_100 * r^2 - a_p_100 * Q^2
        # Esto es incorrecto. La curva de la bomba se desplaza.
        # H_new = H_old * r^2; Q_new = Q_old * r
        # H_new(Q_new) = H_shutoff*r^2 - a_p * (Q_new/r)^2 = H_shutoff*r^2 - a_p/r^2 * Q_new^2
        # Necesitamos encontrar 'r' tal que H_new(q_target_per_pump) = target_head
        # target_head = h_shutoff * r**2 - a_p * (q_target_per_pump)**2
        # No, la curva de la bomba a velocidad r es H(Q) = H0*r^2 - a*(Q/r)^2
        # No, es H(Q) = H_shutoff*r^2 - a_p*Q^2. Esto es un error común.
        # La curva H-Q de una bomba a velocidad variable (r) es: H = (H_shutoff_100) * r^2 - a_p_100 * (Q / r)^2
        # No, la forma correcta es H = A*r^2 y Q = B*r. La curva H(Q) se desplaza.
        # H = H_shutoff*r^2 - a_p * Q^2 no es correcto.
        # La curva homóloga es H/r^2 = f(Q/r).
        # H = r^2 * f(Q/r) = r^2 * (H_shutoff - a_p * (Q/r)^2) = H_shutoff*r^2 - a_p*Q^2
        # Sí, parece que esta es la aproximación que se usa.
        
        numerator = target_head
        denominator = self.pump_head(q_target_per_pump_m3s)
        if denominator <= 0: return None
        
        # H_sys(Q_total) = H_pump(Q_total / n_parallel / r) * r^2
        # target_head = self.pump_head(q_target_per_pump / r) * r^2
        # target_head = (H_shutoff - a_p * (q_target_per_pump/r)^2) * r^2
        # target_head = H_shutoff * r^2 - a_p * q_target_per_pump^2
        
        h_shutoff_100, a_p_100 = self.pump_coeffs[1], self.pump_coeffs[0]
        
        # Resolver para r^2
        r_squared = (target_head + a_p_100 * q_target_per_pump_m3s**2) / h_shutoff_100
        if r_squared < 0: return None
        speed_ratio = np.sqrt(r_squared)

        # 3. Calcular rendimiento en el nuevo punto de operación
        # La eficiencia se mantiene aproximadamente constante para puntos homólogos
        q_homologous_100_rpm = q_target_per_pump_m3s / speed_ratio if speed_ratio > 1e-6 else 0
        efficiency = self.efficiency(q_homologous_100_rpm)
        
        power_kw = (RHO_WATER * G * q_target_per_pump_m3s * target_head / efficiency) / 1000 if efficiency > 1e-6 else 0.0

        return {
            "target_flow_lps": target_flow_lps,
            "speed_ratio": speed_ratio,
            "q_op_per_pump_lps": q_target_per_pump_m3s * 1000,
            "q_op_total_lps": q_target_total_m3s * 1000,
            "h_op": target_head,
            "efficiency": efficiency * 100,
            "power_per_pump_kw": power_kw,
            "total_power_kw": power_kw * self.n_parallel,
        }

    def _solve_operating_point(self) -> tuple[float, float]:
        """
        Encuentra Q y H donde la curva combinada de las bombas se intersecta con la del sistema.
        """
        # self.q_range representa el caudal POR BOMBA.
        # El caudal total del sistema es q_total = self.q_range * self.n_parallel
        q_total = self.q_range * self.n_parallel
        
        # La curva de la bomba es para una sola bomba, así que usamos self.q_range
        pump_head_curve = self.pump_head(self.q_range)
        
        # La curva del sistema depende del caudal total
        system_head_curve = self.system_head(q_total)
        
        # La diferencia es entre la curva de una bomba y la del sistema
        diff = np.abs(pump_head_curve - system_head_curve)
        idx = diff.argmin()
        
        # q_op es el caudal por bomba
        q_op_per_pump = self.q_range[idx]
        
        # h_op es la carga en el punto de operación
        h_op = pump_head_curve[idx]
        
        return float(q_op_per_pump), float(h_op)

    def efficiency(self, q: np.ndarray | float) -> np.ndarray | float:
        """Eficiencia (%) en función de q, aproximada como parábola."""
        q = np.asarray(q, dtype=float)
        eta = self.eff_peak * (1 - ((q - self.q_design) / self.q_design)**2)
        return np.clip(eta, 0, None)

    def power_kW(self, q: np.ndarray | float) -> np.ndarray | float:
        """Potencia eje (kW) para caudal q."""
        head = self.pump_head(q)
        eta = self.efficiency(q)
        with np.errstate(divide='ignore', invalid='ignore'):
            power = (RHO_WATER * G * q * head / eta) / 1e3
        return np.nan_to_num(power)

    def power_at_op_kW(self) -> float:
        """Potencia en el eje (kW) en el punto de operación para una sola bomba."""
        q_op = self.q_op
        h_op = self.h_op
        eta_op = self.efficiency(q_op)
        if eta_op < 1e-6:
            return 0.0
        # Potencia en el eje en kW
        return (RHO_WATER * G * q_op * h_op / eta_op) / 1000

    def _size_reservoir(self) -> float:
        """
        Dimensiona el tanque por Rippl “curva-masa”:
        1) Calcula el balance horario (entrada – demanda) [m³]
        2) Acumula para generar la curva masa
        3) Activo = max(curva) – min(curva)
        4) Capacidad total = Activo / (1 – reserva_min)
        5) Redondea al múltiplo de 50 m³ y garantiza al menos 50 m³
        """
        # (A) Horas a simular
        H = self.simulation_days * 24
    
        # (B) Serie de aporte [m³/s] constante = punto de operación total
        q_in = np.full(H, self.q_op * self.n_parallel)
    
        # (C) Serie de demanda [m³/s]
        q_out = np.tile(self.q_design * self.hourly_factors, self.simulation_days)
    
        # (D) Balance horario [m³]
        hourly_balance = (q_in - q_out) * 3600.0
    
        # (E) Curva-masa acumulada [m³]
        cum_mass = np.cumsum(hourly_balance)
    
        # (F) Almacenamiento activo requerido [m³]
        active_storage = cum_mass.max() - cum_mass.min()
    
        # (G) Capacidad total considerando reserva mínima
        total_storage = active_storage / (1.0 - self.min_tank_level_perc)
    
        # (H) Redondeo práctico y piso
        cap = np.ceil(total_storage / 50.0) * 50.0
        return float(max(cap, 50.0))
    

    def affinity_curves(self, rpm_list: Iterable[float]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Genera curvas de afinidad para velocidades en rpm_list."""
        curves = {}
        for n in rpm_list:
            ratio = n / self.rpm
            q_new = self.q_range * ratio
            h_new = self.pump_head(self.q_range) * ratio**2
            curves[n] = (q_new, h_new)
        return curves

    def daily_balance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simula el balance de agua y la operación de la bomba durante el período definido."""
        # Usar el caudal de operación real para la simulación, no el de diseño.
        q_supply = self.q_op * self.n_parallel
        q_demand = np.tile(self.q_design * self.hourly_factors, self.simulation_days)
        
        num_hours = 24 * self.simulation_days
        vol = np.zeros(num_hours + 1)
        pump_on = np.zeros(num_hours)
        vol[0] = self.initial_volume_m3

        for hour in range(num_hours):
            # La bomba funciona si el tanque no está lleno al inicio de la hora
            if vol[hour] < self.tank_capacity_m3:
                pump_on[hour] = 1.0
                current_supply = q_supply
            else:
                # Si el tanque está lleno, la bomba se apaga para evitar rebalses
                pump_on[hour] = 0.0
                current_supply = 0.0
            
            delta_vol = (current_supply - q_demand[hour]) * 3600
            vol[hour+1] = np.clip(vol[hour] + delta_vol, 0, self.tank_capacity_m3)
            
        return vol, q_demand, pump_on




    def energy_costs(self) -> tuple[np.ndarray, float, float]:
        """Calcula costos energéticos horarios, costo total del período y costo por m3."""
        power_per_pump = self.power_at_op_kW()
        total_power = power_per_pump * self.n_parallel
        
        hourly_cost = total_power * self.pump_on_hourly * self.electricity_cost
        total_cost = np.sum(hourly_cost)
        
        total_volume_pumped = self.q_design * self.n_parallel * np.sum(self.pump_on_hourly) * 3600
        cost_per_m3 = total_cost / total_volume_pumped if total_volume_pumped > 0 else 0
        
        return hourly_cost, total_cost, cost_per_m3

    def _apply_plotly_layout(self, fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str) -> go.Figure:
        """Aplica un layout estándar de Plotly a una figura."""
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'family': 'Arial Black'}},
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            hovermode='x unified',
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        return fig

    def plot_system_vs_pump(self) -> go.Figure:
        """Genera un gráfico interactivo de las curvas H-Q del sistema y las bombas."""
        fig = go.Figure()

        # Rango de caudal total para el eje X
        q_total_range_m3s = self.q_range * self.n_parallel
        q_total_range_lps = q_total_range_m3s * 1000

        # 1. Curva del sistema (H vs Q_total)
        system_head_curve = self.system_head(q_total_range_m3s)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=system_head_curve, mode='lines', name='Curva del Sistema', line=dict(color='blue', width=3)))

        # 2. Curva de una bomba (H vs Q_individual)
        q_single_range_lps = self.q_range * 1000
        fig.add_trace(go.Scatter(x=q_single_range_lps, y=self.pump_head(self.q_range), mode='lines', name='Curva Bomba Individual', line=dict(color='grey', width=2, dash='dash')))

        # 3. Curva combinada de bombas en paralelo (H vs Q_total)
        combined_pump_head = self.pump_head(self.q_range)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=combined_pump_head, mode='lines', name=f'Curva Combinada ({self.n_parallel} bombas)', line=dict(color='green', width=3)))

        # 4. Punto de operación
        q_op_total_lps = self.q_op * self.n_parallel * 1000
        fig.add_trace(go.Scatter(x=[q_op_total_lps], y=[self.h_op], mode='markers', name='Punto de Operación', marker=dict(color='red', size=12, symbol='x')))
        
        return self._apply_plotly_layout(fig, 'Curvas H-Q: Sistema vs. Bombas', 'Caudal Total (L/s)', 'Carga (m)')

    def plot_efficiency_vs_flow(self) -> go.Figure:
        """Genera un gráfico interactivo de la eficiencia de la bomba vs. el caudal por bomba."""
        fig = go.Figure()
        q_range_lps = self.q_range * 1000
        q_op_lps = self.q_op * 1000
        eff_op = self.efficiency(self.q_op) * 100
        
        fig.add_trace(go.Scatter(x=q_range_lps, y=self.efficiency(self.q_range) * 100, mode='lines', name='Eficiencia', line=dict(color='purple', width=3)))
        
        fig.add_trace(go.Scatter(x=[q_op_lps], y=[eff_op], mode='markers+text', name='Punto de Operación',
                               marker=dict(color='red', size=12, symbol='x'),
                               text=[f" {eff_op:.1f}% @ {q_op_lps:.1f} L/s"], textposition="top right"))
                               
        return self._apply_plotly_layout(fig, 'Eficiencia por Bomba vs. Caudal por Bomba', 'Caudal por Bomba (L/s)', 'Eficiencia (%)')

    def plot_total_power_analysis(self) -> go.Figure:
        """Muestra la curva de potencia de referencia y el punto de operación VFD."""
        fig = go.Figure()

        # Curva de Potencia Total de referencia a 100% RPM
        q_total_range_lps = self.q_range * self.n_parallel * 1000
        total_power_range_kw = self.power_kW(self.q_range) * self.n_parallel
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=total_power_range_kw, mode='lines', name='Potencia de Referencia @ 100% RPM', line=dict(color='lightgrey', width=2, dash='dash')))

        # Punto de operación con VFD (el único que se marca)
        if self.vfd_results:
            q_op_total_vfd_lps = self.vfd_results['q_op_total_lps']
            power_op_total_vfd_kw = self.vfd_results['total_power_kw']
            fig.add_trace(go.Scatter(x=[q_op_total_vfd_lps], y=[power_op_total_vfd_kw], mode='markers+text', name='OP con VFD',
                                   marker=dict(color='magenta', size=12, symbol='star'),
                                   text=[f" {power_op_total_vfd_kw:.1f} kW"], textposition="middle right"))

        return self._apply_plotly_layout(fig, 'Análisis de Potencia Total', 'Caudal Total (L/s)', 'Potencia Total (kW)')

    def plot_affinity(self) -> go.Figure:
        """Genera un gráfico interactivo de las leyes de afinidad de la bomba."""
        fig = go.Figure()
        rpm_list = [self.rpm * r for r in np.linspace(1, 0.1, 10)]
        for n in rpm_list:
            q, h = self.affinity_curves([n])[n]
            fig.add_trace(go.Scatter(x=q * 1000, y=h, mode='lines', name=f"{n/self.rpm*100:.0f}% RPM"))
        return self._apply_plotly_layout(fig, 'Leyes de Afinidad de la Bomba', 'Caudal (L/s)', 'Carga (m)')

    def plot_tank_volume(self) -> go.Figure:
        """Genera un gráfico interactivo del volumen del tanque."""
        fig = go.Figure()
        num_hours = 24 * self.simulation_days
        hrs_vol = np.arange(num_hours + 1)
        
        fig.add_trace(go.Scatter(x=hrs_vol, y=self.vol_hourly, name='Volumen Tanque (m³)',
                                 mode='lines', line=dict(color='blue', width=3)))

        fig.add_hline(y=self.tank_capacity_m3, line_dash="dash", line_color="black",
                      annotation_text=f'Capacidad Tanque ({self.tank_capacity_m3:.0f} m³)')
        fig.add_hline(y=self.tank_capacity_m3 * self.min_tank_level_perc, line_dash="dash", line_color="red",
                      annotation_text=f'Reserva Mínima ({self.min_tank_level_perc*100:.0f}%)')

        fig.update_yaxes(range=[0, self.tank_capacity_m3 * 1.1])
        return self._apply_plotly_layout(fig, f'Simulación Operacional ({self.simulation_days} días): Volumen en Tanque', 'Hora', 'Volumen (m³)')

    def plot_demand_vs_inflow(self) -> go.Figure:
        """Compara el caudal de demanda con el caudal de bombeo."""
        fig = go.Figure()
        num_hours = 24 * self.simulation_days
        hrs_op = np.arange(num_hours)

        # Caudal de bombeo constante para visualización
        constant_inflow_lps = np.full(num_hours, self.q_design * self.n_parallel * 1000)
        demand_lps = self.demand_hourly * 1000

        fig.add_trace(go.Bar(x=hrs_op, y=demand_lps, name='Demanda Horaria (L/s)',
                             marker_color='red', opacity=0.6))

        # Mostrar el caudal de bombeo como una línea constante
        fig.add_trace(go.Scatter(x=hrs_op, y=constant_inflow_lps, name='Caudal de Bombeo (L/s)',
                                 mode='lines', line=dict(color='green', width=3)))

        return self._apply_plotly_layout(fig, f'Simulación Operacional ({self.simulation_days} días): Demanda vs. Bombeo', 'Hora', 'Caudal (L/s)')

    def plot_power_affinity(self) -> go.Figure:
        """Genera un gráfico interactivo de la potencia vs. caudal para diferentes velocidades."""
        fig = go.Figure()
        rpm_list = [self.rpm * r for r in np.linspace(1, 0.5, 6)]
        for n in rpm_list:
            ratio = n / self.rpm
            q_new = self.q_range * ratio
            power_new = self.power_kW(self.q_range) * ratio**3
            fig.add_trace(go.Scatter(x=q_new * 1000, y=power_new, mode='lines', name=f"{n/self.rpm*100:.0f}% RPM"))
        
        # Añadir punto de operación
        q_op_lps = self.q_op * 1000
        power_op_kw = self.power_at_op_kW()
        fig.add_trace(go.Scatter(x=[q_op_lps], y=[power_op_kw], mode='markers', name='Punto de Operación',
                               marker=dict(color='red', size=12, symbol='x')))
        
        return self._apply_plotly_layout(fig, 'Potencia vs. Caudal (Afinidad)', 'Caudal (L/s)', 'Potencia (kW)')

    def plot_efficiency_affinity(self) -> go.Figure:
        """Genera un gráfico interactivo de la eficiencia vs. caudal para diferentes velocidades."""
        fig = go.Figure()
        rpm_list = [self.rpm * r for r in np.linspace(1, 0.5, 6)]
        for n in rpm_list:
            ratio = n / self.rpm
            q_new = self.q_range * ratio
            efficiency_vals = self.efficiency(self.q_range) * 100
            fig.add_trace(go.Scatter(x=q_new * 1000, y=efficiency_vals, mode='lines', name=f"{n/self.rpm*100:.0f}% RPM"))
        return self._apply_plotly_layout(fig, 'Eficiencia vs. Caudal (Afinidad)', 'Caudal (L/s)', 'Eficiencia (%)')

    def plot_cost_per_m3_vs_flow(self) -> go.Figure:
        """Genera un gráfico del costo unitario de bombeo vs. el caudal."""
        fig = go.Figure()
        
        # Evitar división por cero en el caudal
        q_valid_m3s = self.q_range[self.q_range > 1e-6]
        q_valid_lps = q_valid_m3s * 1000
        
        # Costo (USD/kWh) * Potencia (kW) / Caudal (m³/h)
        # Caudal en m³/h = q_m3_s * 3600
        cost_per_m3 = (self.power_kW(q_valid_m3s) * self.electricity_cost) / (q_valid_m3s * 3600)
        
        fig.add_trace(go.Scatter(x=q_valid_lps, y=cost_per_m3, mode='lines', name='Costo Unitario', line=dict(color='teal', width=3)))
        
        # Punto de operación
        if self.q_op > 1e-6:
            q_op_lps = self.q_op * 1000
            cost_op = (self.power_kW(self.q_op) * self.electricity_cost) / (self.q_op * 3600)
            fig.add_trace(go.Scatter(x=[q_op_lps], y=[cost_op], mode='markers', name='Punto de Operación', marker=dict(color='red', size=12, symbol='x')))

        return self._apply_plotly_layout(fig, 'Costo Unitario vs. Caudal', 'Caudal (L/s)', 'Costo (USD/m³)')

    def plot_pump_operating_range(self) -> go.Figure | None:
        """Muestra el rango operacional de una bomba individual y su punto de operación VFD."""
        if not self.vfd_results:
            return None

        fig = go.Figure()

        # Curva de la bomba individual
        q_range_lps = self.q_range * 1000
        h_range = self.pump_head(self.q_range)
        fig.add_trace(go.Scatter(x=q_range_lps, y=h_range, mode='lines', name='Curva Bomba Individual', line=dict(color='grey')))

        # BEP (Punto de Mejor Eficiencia)
        q_bep_lps = self.q_design * 1000
        h_bep = self.pump_head(self.q_design)
        fig.add_trace(go.Scatter(x=[q_bep_lps], y=[h_bep], mode='markers', name='BEP',
                               marker=dict(color='blue', symbol='diamond', size=12)))

        # Rango Operacional Preferido (70% - 120% del caudal del BEP)
        q_min_pref = q_bep_lps * 0.7
        q_max_pref = q_bep_lps * 1.2
        fig.add_vrect(x0=q_min_pref, x1=q_max_pref,
                      annotation_text="Rango Preferido", annotation_position="top left",
                      fillcolor="green", opacity=0.15, line_width=0)

        # Punto de operación real por bomba con VFD
        q_op_per_pump_lps = self.vfd_results['q_op_per_pump_lps']
        h_op_vfd = self.vfd_results['h_op']
        fig.add_trace(go.Scatter(x=[q_op_per_pump_lps], y=[h_op_vfd], mode='markers+text', name='OP Real por Bomba (VFD)',
                               marker=dict(color='magenta', symbol='star', size=12),
                               text=[f"  {q_op_per_pump_lps:.1f} L/s"], textposition="bottom right"))

        return self._apply_plotly_layout(fig, 'Rango Operacional de la Bomba', 'Caudal por Bomba (L/s)', 'Carga (m)')

    def plot_vfd_comparison(self) -> go.Figure | None:
        """
        Genera un gráfico comparando la operación a velocidad nominal vs. con VFD.
        """
        if not self.vfd_results:
            return None

        fig = go.Figure()
        r = self.vfd_results['speed_ratio']
        
        # Curva de la bomba combinada a 100% RPM
        q_total_range_lps = self.q_range * self.n_parallel * 1000
        combined_pump_head_100 = self.pump_head(self.q_range)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=combined_pump_head_100, mode='lines', name=f'Curva Combinada @ 100% RPM', line=dict(color='green', width=3)))\

        # Curva de la bomba combinada a velocidad reducida (VFD)
        # H_vfd(Q_total) = H_100(Q_total / n_parallel / r) * r^2
        q_per_pump_vfd = (q_total_range_lps / 1000) / self.n_parallel
        q_homologous_100 = q_per_pump_vfd / r
        combined_pump_head_vfd = self.pump_head(q_homologous_100) * r**2
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=combined_pump_head_vfd, mode='lines', name=f'Curva Combinada @ {r*100:.1f}% RPM', line=dict(color='orange', width=3, dash='dash')))\

        # Curva del sistema
        system_head_curve = self.system_head(self.q_range * self.n_parallel)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=system_head_curve, mode='lines', name='Curva del Sistema', line=dict(color='blue', width=3)))

        # Punto de operación con VFD
        q_op_total_vfd_lps = self.vfd_results['q_op_total_lps']
        h_op_vfd = self.vfd_results['h_op']
        power_vfd = self.vfd_results['total_power_kw']
        fig.add_trace(go.Scatter(x=[q_op_total_vfd_lps], y=[h_op_vfd], mode='markers+text', name=f'OP @ {r*100:.1f}% RPM',
                               marker=dict(color='magenta', size=12, symbol='star'),
                               text=[f"  {power_vfd:.1f} kW"], textposition="bottom right"))

        return self._apply_plotly_layout(fig, 'Análisis de Operación con VFD', 'Caudal Total (L/s)', 'Carga (m)')

    def plot_dashboard(self):
        """Genera un dashboard interactivo con todos los gráficos de análisis."""
        
        figs = [
            self.plot_system_vs_pump(),
            self.plot_vfd_comparison(),
            self.plot_pump_operating_range(),
            self.plot_efficiency_vs_flow(),
            self.plot_total_power_analysis(),
            self.plot_cost_per_m3_vs_flow(),
            self.plot_affinity(),
            self.plot_power_affinity(),
            self.plot_efficiency_affinity(),
            self.plot_tank_volume(),
            self.plot_demand_vs_inflow(),
        ]
        
        figs = [fig for fig in figs if fig is not None]
        subplot_titles = [fig.layout.title.text for fig in figs]

        n_plots = len(figs)
        cols = 3
        rows = (n_plots + cols - 1) // cols

        dashboard_fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.06
        )

        fig_iter = iter(figs)
        for i in range(rows * cols):
            row = i // cols + 1
            col = i % cols + 1
            
            try:
                fig = next(fig_iter)
                for trace in fig.data:
                    dashboard_fig.add_trace(trace, row=row, col=col)
                
                dashboard_fig.update_xaxes(title_text=fig.layout.xaxis.title.text, row=row, col=col)
                dashboard_fig.update_yaxes(title_text=fig.layout.yaxis.title.text, row=row, col=col)
                
                if fig.layout.shapes:
                    for shape in fig.layout.shapes:
                        dashboard_fig.add_shape(shape, row=row, col=col)

                if fig.layout.annotations:
                    for annotation in fig.layout.annotations:
                        
                        dashboard_fig.add_annotation(annotation, row=row, col=col)
            except StopIteration:
                break


        dashboard_fig.update_layout(
            title_text=f"Dashboard de Diseño de Bombeo ({self.n_parallel} bombas) - Caudal de Diseño: {self.q_design * 1000:.1f} L/s",
            height=750* rows,
            width=2100,
            showlegend=True,
            hovermode='closest'
        )
        dashboard_fig.write_html("pump_design.html")
        dashboard_fig.show()
        return dashboard_fig

    def summary(self) -> str:
        """Genera un resumen técnico del diseño."""
        hourly_cost, total_cost, cost_unit = self.energy_costs()
        total_pump_hours = np.sum(self.pump_on_hourly)
        tank_vol_str = (f"Volumen del tanque={self.tank_capacity_m3:.1f} m³ (calculado)"
                        if self.tank_capacity_m3_user is None
                        else f"Volumen del tanque={self.tank_capacity_m3:.1f} m³ (provisto)")
        
        power_per_pump = self.power_at_op_kW()
        total_power = power_per_pump * self.n_parallel
        q_op_total_lps = self.q_op * self.n_parallel * 1000
        q_op_per_pump_lps = self.q_op * 1000

        summary_str = ""

        if self.vfd_results:
            power_100_total = self.power_at_op_kW() * self.n_parallel
            power_vfd_total = self.vfd_results['total_power_kw']
            power_saving_kw = power_100_total - power_vfd_total
            power_saving_perc = (power_saving_kw / power_100_total) * 100 if power_100_total > 0 else 0

            vfd_summary = f"""
--- RESUMEN DEL DISEÑO ---

[Análisis de Operación Optimizada con VFD]
  - Objetivo de Caudal: {self.vfd_results['target_flow_lps']:.2f} L/s
  - Carga Requerida para Objetivo: {self.vfd_results['h_op']:.2f} m
  - Velocidad Requerida: {self.vfd_results['speed_ratio']*100:.1f}% RPM
  - Potencia Total con VFD: {power_vfd_total:.1f} kW
  - AHORRO DE POTENCIA vs 100% RPM: {power_saving_kw:.1f} kW ({power_saving_perc:.1f}%)
"""
            summary_str += vfd_summary

        summary_str += f"""
[Operación y Costos ({self.simulation_days} días)]
  - {tank_vol_str}
  - Horas de bombeo totales: {total_pump_hours:.1f} h
  - Costo total del período: {total_cost:.2f} USD
  - Costo por m³ bombeado: {cost_unit:.4f} USD/m³

[Análisis de Balance de Masa]
  - ADVERTENCIA: El caudal de bombeo ({self.q_design * 1000:.1f} L/s) es menor que la demanda diaria promedio ({np.mean(self.demand_hourly) * 1000:.1f} L/s).
    Esto significa que el sistema tiene un déficit de agua y el tanque se vaciará con el tiempo, como se ve en la simulación.
  - Dimensionamiento del Tanque: El volumen se calculó con el método de Rippl, que asume un bombeo capaz de satisfacer la demanda promedio.
    El resultado ({self.tank_capacity_m3:.1f} m³) es el volumen necesario para un sistema balanceado.
  - Conclusión: Para un funcionamiento sostenible, el caudal de la bomba debe ser igual o mayor a la demanda promedio.
--------------------------------"""
        return summary_str.strip()


if __name__ == "__main__":
    # Ejemplo de uso ( poner las curvas aprendidas en otros cursos)
    hourly = [0.6,0.6,0.6,0.6,0.7,0.85,1.1,1.4,1.7,2.1,1.95,1.8,
              1.7,1.6,1.5,1.55,1.7,1.9,2.0,1.6,1.2,1.0,0.8,0.7]

    # --- Modo de Dimensionamiento (comportamiento por defecto) ---
    print("--- Modo de Dimensionamiento ---")
    pump_designer = CentrifugalPumpDesigner(
        q_design=11/1000, h_static=190, rpm=3500, hourly_factors=hourly,
        pipe_length_m=2000,  pipe_diameter_m=0.15,
        n_parallel=3, hw_c=130,
        min_tank_level_perc=0.3,
        initial_tank_level_perc=0.8,
    )
    print(pump_designer.summary())

    # Generar y mostrar el dashboard con todos los gráficos
    pump_designer.plot_dashboard()
