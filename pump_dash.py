import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import numpy as np

# --- CLASE CENTRIFUGAL PUMP DESIGNER INDEPENDIENTE ---
class CentrifugalPumpDesigner:
    RHO_WATER = 1000.0
    G = 9.80665
    def __init__(self, q_design, h_static, rpm, hourly_factors, pipe_length_m, pipe_diameter_m, 
                 n_parallel, hw_c, eff_peak, electricity_cost, min_tank_level_perc, 
                 initial_tank_level_perc, simulation_days=1, tank_capacity_m3=None, tank_round_m3=50):
        # Validaciones y asignación robusta
        if len(hourly_factors) != 24:
            raise ValueError("hourly_factors debe contener 24 valores: uno por cada hora del día")
        if not 0.0 <= min_tank_level_perc < 1.0:
            raise ValueError("min_tank_level_perc debe estar entre 0.0 y 1.0")
        if not 0.0 <= initial_tank_level_perc <= 1.0:
            raise ValueError("initial_tank_level_perc debe estar entre 0.0 y 1.0")
        if not simulation_days > 0:
            raise ValueError("simulation_days debe ser mayor que 0")
        self.q_design = float(q_design)
        self.h_static = float(h_static)
        self.rpm = float(rpm)
        self.hourly_factors = np.asarray(hourly_factors, dtype=float)
        self.n_parallel = int(n_parallel)
        self.initial_tank_level_perc = float(initial_tank_level_perc)
        self.simulation_days = int(simulation_days)
        self.pipe_length_m = float(pipe_length_m)
        self.pipe_diameter_m = float(pipe_diameter_m)
        self.hw_c = float(hw_c)
        self.eff_peak = eff_peak
        self.electricity_cost = electricity_cost
        # Curva de la bomba (parábola): H = H_shutoff - a_p * Q^2
        h_design = self.system_head(self.q_design)
        h_shutoff = 1.3 * h_design
        a_p = (h_shutoff - h_design) / self.q_design**2 if self.q_design > 1e-9 else 0.0
        self.pump_coeffs = (a_p, h_shutoff)
        self.q_range = np.linspace(1e-6, 1.5 * self.q_design, 400)
        self.q_op, self.h_op = self._solve_operating_point()
        self.min_tank_level_perc = min_tank_level_perc
        self.tank_capacity_m3_user = tank_capacity_m3
        self.tank_round_m3 = tank_round_m3 if tank_round_m3 is not None and tank_round_m3 > 0 else 50.0
        if self.tank_capacity_m3_user is None:
            self.tank_capacity_m3 = self._size_reservoir()
            self.initial_volume_m3 = self.tank_capacity_m3 * self.initial_tank_level_perc
        else:
            self.tank_capacity_m3 = float(self.tank_capacity_m3_user)
            self.initial_volume_m3 = self.tank_capacity_m3 * self.initial_tank_level_perc
        self.vol_hourly, self.demand_hourly, self.pump_on_hourly = self.daily_balance()
        self.vfd_results = self._analyze_vfd_operation_by_flow(self.q_design * 1000)

    def system_head(self, q):
        q = np.asarray(q, dtype=float)
        L, D, C = self.pipe_length_m, self.pipe_diameter_m, self.hw_c
        hf = 10.67 * L * q**1.852 / (C**1.852 * D**4.871)
        return self.h_static + hf

    def pump_head(self, q):
        a_p, h_shutoff = self.pump_coeffs
        return h_shutoff - a_p * np.asarray(q, dtype=float)**2

    def _solve_operating_point(self):
        q_total = self.q_range * self.n_parallel
        pump_head_curve = self.pump_head(self.q_range)
        system_head_curve = self.system_head(q_total)
        diff = np.abs(pump_head_curve - system_head_curve)
        idx = diff.argmin()
        q_op_per_pump = self.q_range[idx]
        h_op = pump_head_curve[idx]
        return float(q_op_per_pump), float(h_op)

    def efficiency(self, q):
        q = np.asarray(q, dtype=float)
        eta = self.eff_peak * (1 - ((q - self.q_design) / self.q_design)**2)
        return np.clip(eta, 0, None)

    def power_kW(self, q):
        head = self.pump_head(q)
        eta = self.efficiency(q)
        with np.errstate(divide='ignore', invalid='ignore'):
            power = (self.RHO_WATER * self.G * q * head / eta) / 1e3
        return np.nan_to_num(power)

    def power_at_op_kW(self):
        q_op = self.q_op
        h_op = self.h_op
        eta_op = self.efficiency(q_op)
        if eta_op < 1e-6:
            return 0.0
        return (self.RHO_WATER * self.G * q_op * h_op / eta_op) / 1000

    def _size_reservoir(self):
        H = self.simulation_days * 24
        q_in = np.full(H, self.q_op * self.n_parallel)
        q_out = np.tile(self.q_design * self.hourly_factors, self.simulation_days)
        hourly_balance = (q_in - q_out) * 3600.0
        cum_mass = np.cumsum(hourly_balance)
        active_storage = cum_mass.max() - cum_mass.min()
        total_storage = active_storage / (1.0 - self.min_tank_level_perc)
        # --- USAR EL REDONDEO ELEGIDO POR EL USUARIO ---
        round_m3 = self.tank_round_m3 if hasattr(self, 'tank_round_m3') and self.tank_round_m3 > 0 else 50.0
        cap = np.ceil(total_storage / round_m3) * round_m3
        return float(max(cap, round_m3))

    def affinity_curves(self, rpm_list):
        curves = {}
        for n in rpm_list:
            ratio = n / self.rpm
            q_new = self.q_range * ratio
            h_new = self.pump_head(self.q_range) * ratio**2
            curves[n] = (q_new, h_new)
        return curves

    def daily_balance(self):
        q_supply = self.q_op * self.n_parallel
        q_demand = np.tile(self.q_design * self.hourly_factors, self.simulation_days)
        num_hours = 24 * self.simulation_days
        vol = np.zeros(num_hours + 1)
        pump_on = np.zeros(num_hours)
        vol[0] = self.initial_volume_m3
        for hour in range(num_hours):
            if vol[hour] < self.tank_capacity_m3:
                pump_on[hour] = 1.0
                current_supply = q_supply
            else:
                pump_on[hour] = 0.0
                current_supply = 0.0
            delta_vol = (current_supply - q_demand[hour]) * 3600
            vol[hour+1] = np.clip(vol[hour] + delta_vol, 0, self.tank_capacity_m3)
        return vol, q_demand, pump_on

    def energy_costs(self):
        power_per_pump = self.power_at_op_kW()
        total_power = power_per_pump * self.n_parallel
        hourly_cost = total_power * self.pump_on_hourly * self.electricity_cost
        total_cost = np.sum(hourly_cost)
        total_volume_pumped = self.q_design * self.n_parallel * np.sum(self.pump_on_hourly) * 3600
        cost_per_m3 = total_cost / total_volume_pumped if total_volume_pumped > 0 else 0
        return hourly_cost, total_cost, cost_per_m3

    def _analyze_vfd_operation_by_flow(self, target_flow_lps):
        q_target_total_m3s = target_flow_lps / 1000.0
        target_head = self.system_head(q_target_total_m3s)
        if target_head > self.pump_coeffs[1]:
            return None
        q_target_per_pump_m3s = q_target_total_m3s / self.n_parallel
        a_p, h_shutoff = self.pump_coeffs
        h_shutoff_100, a_p_100 = h_shutoff, a_p
        r_squared = (target_head + a_p_100 * q_target_per_pump_m3s**2) / h_shutoff_100
        if r_squared < 0:
            return None
        speed_ratio = np.sqrt(r_squared)
        q_homologous_100_rpm = q_target_per_pump_m3s / speed_ratio if speed_ratio > 1e-6 else 0
        efficiency = self.efficiency(q_homologous_100_rpm)
        power_kw = (self.RHO_WATER * self.G * q_target_per_pump_m3s * target_head / efficiency) / 1000 if efficiency > 1e-6 else 0.0
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

    def plot_system_vs_pump(self):
        fig = go.Figure()
        q_total_range_m3s = self.q_range * self.n_parallel
        q_total_range_lps = q_total_range_m3s * 1000
        system_head_curve = self.system_head(q_total_range_m3s)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=system_head_curve, mode='lines', name='Curva del Sistema', line=dict(color='blue', width=3)))
        q_single_range_lps = self.q_range * 1000
        fig.add_trace(go.Scatter(x=q_single_range_lps, y=self.pump_head(self.q_range), mode='lines', name='Curva Bomba Individual', line=dict(color='grey', width=2, dash='dash')))
        combined_pump_head = self.pump_head(self.q_range)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=combined_pump_head, mode='lines', name=f'Curva Combinada ({self.n_parallel} bombas)', line=dict(color='green', width=3)))
        q_op_total_lps = self.q_op * self.n_parallel * 1000
        fig.add_trace(go.Scatter(x=[q_op_total_lps], y=[self.h_op], mode='markers', name='Punto de Operación', marker=dict(color='red', size=12, symbol='x')))
        fig.update_layout(title='Curvas H-Q: Sistema vs. Bombas', xaxis_title='Caudal Total (L/s)', yaxis_title='Carga (m)')
        return fig

    def plot_vfd_comparison(self):
        if not self.vfd_results:
            return None
        fig = go.Figure()
        r = self.vfd_results['speed_ratio']
        q_total_range_lps = self.q_range * self.n_parallel * 1000
        combined_pump_head_100 = self.pump_head(self.q_range)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=combined_pump_head_100, mode='lines', name=f'Curva Combinada @ 100% RPM', line=dict(color='green', width=3)))
        q_per_pump_vfd = (q_total_range_lps / 1000) / self.n_parallel
        q_homologous_100 = q_per_pump_vfd / r
        combined_pump_head_vfd = self.pump_head(q_homologous_100) * r**2
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=combined_pump_head_vfd, mode='lines', name=f'Curva Combinada @ {r*100:.1f}% RPM', line=dict(color='orange', width=3, dash='dash')))
        system_head_curve = self.system_head(self.q_range * self.n_parallel)
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=system_head_curve, mode='lines', name='Curva del Sistema', line=dict(color='blue', width=3)))
        q_op_total_vfd_lps = self.vfd_results['q_op_total_lps']
        h_op_vfd = self.vfd_results['h_op']
        power_vfd = self.vfd_results['total_power_kw']
        fig.add_trace(go.Scatter(x=[q_op_total_vfd_lps], y=[h_op_vfd], mode='markers+text', name=f'OP @ {r*100:.1f}% RPM', marker=dict(color='magenta', size=12, symbol='star'), text=[f"  {power_vfd:.1f} kW"], textposition="bottom right"))
        fig.update_layout(title='Análisis de Operación con VFD', xaxis_title='Caudal Total (L/s)', yaxis_title='Carga (m)')
        return fig

    def plot_pump_operating_range(self):
        if not self.vfd_results:
            return None
        fig = go.Figure()
        q_range_lps = self.q_range * 1000
        h_range = self.pump_head(self.q_range)
        fig.add_trace(go.Scatter(x=q_range_lps, y=h_range, mode='lines', name='Curva Bomba Individual', line=dict(color='grey')))
        q_bep_lps = self.q_design * 1000
        h_bep = self.pump_head(self.q_design)
        fig.add_trace(go.Scatter(x=[q_bep_lps], y=[h_bep], mode='markers', name='BEP', marker=dict(color='blue', symbol='diamond', size=12)))
        q_min_pref = q_bep_lps * 0.7
        q_max_pref = q_bep_lps * 1.2
        fig.add_vrect(x0=q_min_pref, x1=q_max_pref, annotation_text="Rango Preferido", annotation_position="top left", fillcolor="green", opacity=0.15, line_width=0)
        q_op_per_pump_lps = self.vfd_results['q_op_per_pump_lps']
        h_op_vfd = self.vfd_results['h_op']
        fig.add_trace(go.Scatter(x=[q_op_per_pump_lps], y=[h_op_vfd], mode='markers+text', name='OP Real por Bomba (VFD)', marker=dict(color='magenta', symbol='star', size=12), text=[f"  {q_op_per_pump_lps:.1f} L/s"], textposition="bottom right"))
        fig.update_layout(title='Rango Operacional de la Bomba', xaxis_title='Caudal por Bomba (L/s)', yaxis_title='Carga (m)')
        return fig

    def plot_efficiency_vs_flow(self):
        fig = go.Figure()
        q_range_lps = self.q_range * 1000
        q_op_lps = self.q_op * 1000
        eff_op = self.efficiency(self.q_op) * 100
        fig.add_trace(go.Scatter(x=q_range_lps, y=self.efficiency(self.q_range) * 100, mode='lines', name='Eficiencia', line=dict(color='purple', width=3)))
        fig.add_trace(go.Scatter(x=[q_op_lps], y=[eff_op], mode='markers+text', name='Punto de Operación', marker=dict(color='red', size=12, symbol='x'), text=[f" {eff_op:.1f}% @ {q_op_lps:.1f} L/s"], textposition="top right"))
        fig.update_layout(title='Eficiencia por Bomba vs. Caudal por Bomba', xaxis_title='Caudal por Bomba (L/s)', yaxis_title='Eficiencia (%)')
        return fig

    def plot_total_power_analysis(self):
        fig = go.Figure()
        q_total_range_lps = self.q_range * self.n_parallel * 1000
        total_power_range_kw = self.power_kW(self.q_range) * self.n_parallel
        fig.add_trace(go.Scatter(x=q_total_range_lps, y=total_power_range_kw, mode='lines', name='Potencia de Referencia @ 100% RPM', line=dict(color='lightgrey', width=2, dash='dash')))
        if self.vfd_results:
            q_op_total_vfd_lps = self.vfd_results['q_op_total_lps']
            power_op_total_vfd_kw = self.vfd_results['total_power_kw']
            fig.add_trace(go.Scatter(x=[q_op_total_vfd_lps], y=[power_op_total_vfd_kw], mode='markers+text', name='OP con VFD', marker=dict(color='magenta', size=12, symbol='star'), text=[f" {power_op_total_vfd_kw:.1f} kW"], textposition="middle right"))
        fig.update_layout(title='Análisis de Potencia Total', xaxis_title='Caudal Total (L/s)', yaxis_title='Potencia Total (kW)')
        return fig

    def plot_cost_per_m3_vs_flow(self):
        fig = go.Figure()
        q_valid_m3s = self.q_range[self.q_range > 1e-6]
        q_valid_lps = q_valid_m3s * 1000
        cost_per_m3 = (self.power_kW(q_valid_m3s) * self.electricity_cost) / (q_valid_m3s * 3600)
        fig.add_trace(go.Scatter(x=q_valid_lps, y=cost_per_m3, mode='lines', name='Costo Unitario', line=dict(color='teal', width=3)))
        if self.q_op > 1e-6:
            q_op_lps = self.q_op * 1000
            cost_op = (self.power_kW(self.q_op) * self.electricity_cost) / (self.q_op * 3600)
            fig.add_trace(go.Scatter(x=[q_op_lps], y=[cost_op], mode='markers', name='Punto de Operación', marker=dict(color='red', size=12, symbol='x')))
        fig.update_layout(title='Costo Unitario vs. Caudal', xaxis_title='Caudal (L/s)', yaxis_title='Costo (USD/m³)')
        return fig

    def plot_affinity(self):
        fig = go.Figure()
        rpm_list = [self.rpm * r for r in np.linspace(1, 0.1, 10)]
        for n in rpm_list:
            q, h = self.affinity_curves([n])[n]
            fig.add_trace(go.Scatter(x=q * 1000, y=h, mode='lines', name=f"{n/self.rpm*100:.0f}% RPM"))
        fig.update_layout(title='Leyes de Afinidad de la Bomba', xaxis_title='Caudal (L/s)', yaxis_title='Carga (m)')
        return fig

    def plot_power_affinity(self):
        fig = go.Figure()
        rpm_list = [self.rpm * r for r in np.linspace(1, 0.5, 6)]
        for n in rpm_list:
            ratio = n / self.rpm
            q_new = self.q_range * ratio
            power_new = self.power_kW(self.q_range) * ratio**3
            fig.add_trace(go.Scatter(x=q_new * 1000, y=power_new, mode='lines', name=f"{n/self.rpm*100:.0f}% RPM"))
        q_op_lps = self.q_op * 1000
        power_op_kw = self.power_at_op_kW()
        fig.add_trace(go.Scatter(x=[q_op_lps], y=[power_op_kw], mode='markers', name='Punto de Operación', marker=dict(color='red', size=12, symbol='x')))
        fig.update_layout(title='Potencia vs. Caudal (Afinidad)', xaxis_title='Caudal (L/s)', yaxis_title='Potencia (kW)')
        return fig

    def plot_efficiency_affinity(self):
        fig = go.Figure()
        rpm_list = [self.rpm * r for r in np.linspace(1, 0.5, 6)]
        for n in rpm_list:
            ratio = n / self.rpm
            q_new = self.q_range * ratio
            efficiency_vals = self.efficiency(self.q_range) * 100
            fig.add_trace(go.Scatter(x=q_new * 1000, y=efficiency_vals, mode='lines', name=f"{n/self.rpm*100:.0f}% RPM"))
        fig.update_layout(title='Eficiencia vs. Caudal (Afinidad)', xaxis_title='Caudal (L/s)', yaxis_title='Eficiencia (%)')
        return fig

    def plot_tank_volume(self):
        fig = go.Figure()
        num_hours = 24 * self.simulation_days
        hrs_vol = np.arange(num_hours + 1)
        fig.add_trace(go.Scatter(x=hrs_vol, y=self.vol_hourly, name='Volumen Tanque (m³)', mode='lines', line=dict(color='blue', width=3)))
        fig.add_hline(y=self.tank_capacity_m3, line_dash="dash", line_color="black", annotation_text=f'Capacidad Tanque ({self.tank_capacity_m3:.0f} m³)')
        fig.add_hline(y=self.tank_capacity_m3 * self.min_tank_level_perc, line_dash="dash", line_color="red", annotation_text=f'Reserva Mínima ({self.min_tank_level_perc*100:.0f}%)')
        fig.update_yaxes(range=[0, self.tank_capacity_m3 * 1.1])
        fig.update_layout(title=f'Simulación Operacional ({self.simulation_days} días): Volumen en Tanque', xaxis_title='Hora', yaxis_title='Volumen (m³)')
        return fig

    def plot_demand_vs_inflow(self):
        fig = go.Figure()
        num_hours = 24 * self.simulation_days
        hrs_op = np.arange(num_hours)
        constant_inflow_lps = np.full(num_hours, self.q_design * self.n_parallel * 1000)
        demand_lps = self.demand_hourly * 1000
        fig.add_trace(go.Bar(x=hrs_op, y=demand_lps, name='Demanda Horaria (L/s)', marker_color='red', opacity=0.6))
        fig.add_trace(go.Scatter(x=hrs_op, y=constant_inflow_lps, name='Caudal de Bombeo (L/s)', mode='lines', line=dict(color='green', width=3)))
        fig.update_layout(title=f'Simulación Operacional ({self.simulation_days} días): Demanda vs. Bombeo', xaxis_title='Hora', yaxis_title='Caudal (L/s)')
        return fig

def default_hourly_factors():
    return "0.6,0.6,0.6,0.6,0.7,0.85,1.1,1.4,1.7,2.1,1.95,1.8,1.7,1.6,1.5,1.55,1.7,1.9,2.0,1.6,1.2,1.0,0.8,0.7"

def filtrar_figura(fig, title, idx):
    if fig is None or not hasattr(fig, 'data') or len(fig.data) == 0:
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
            # Asegurar que x e y tengan la misma longitud
            min_len = min(len(y), len(x))
            if min_len > 0:
                y = y[:min_len]
                x = x[:min_len]
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
    if not datos_validos or len(y_vals_all) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"{title} (sin datos válidos)")
        return empty_fig
    # Ajuste dinámico del eje Y
    y_min = float(np.min(y_vals_all))
    y_max = float(np.max(y_vals_all))
    # Protección contra NaN o infinitos
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"{title} (sin datos válidos)")
        return empty_fig
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
    # Protección contra NaN o infinitos en el rango final
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"{title} (sin datos válidos)")
        return empty_fig
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
            # Protección contra step NaN o 0
            if not np.isfinite(step) or step == 0:
                fig.update_xaxes(showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
            else:
                try:
                    for i in range(int(x_min/step), int(x_max/step) + 1):
                        tick_val = i * step
                        if x_min <= tick_val <= x_max:
                            tick_vals.append(tick_val)
                    # Asegurar que 0 esté incluido
                    if 0 not in tick_vals and x_min <= 0 <= x_max:
                        tick_vals.append(0)
                        tick_vals.sort()
                    fig.update_xaxes(tickvals=tick_vals, showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
                except (ValueError, OverflowError):
                    # Si hay problemas con los cálculos, usar configuración básica
                    fig.update_xaxes(showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
    if fig.layout.yaxis.range:
        y_min, y_max = fig.layout.yaxis.range
        # Asegurar que 0 esté incluido si está en el rango
        if y_min <= 0 <= y_max:
            # Calcular ticks que incluyan 0
            tick_vals = []
            step = (y_max - y_min) / 10  # Aproximadamente 10 divisiones
            # Protección contra step NaN o 0
            if not np.isfinite(step) or step == 0:
                fig.update_yaxes(showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
            else:
                try:
                    for i in range(int(y_min/step), int(y_max/step) + 1):
                        tick_val = i * step
                        if y_min <= tick_val <= y_max:
                            tick_vals.append(tick_val)
                    # Asegurar que 0 esté incluido
                    if 0 not in tick_vals and y_min <= 0 <= y_max:
                        tick_vals.append(0)
                        tick_vals.sort()
                    fig.update_yaxes(tickvals=tick_vals, showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
                except (ValueError, OverflowError):
                    # Si hay problemas con los cálculos, usar configuración básica
                    fig.update_yaxes(showgrid=True, gridcolor='#cccccc', gridwidth=0.7, zeroline=False)
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
def get_input_panel(dias=None):
    dias = dias if dias is not None else 1
    campos = [
        ("Caudal de diseño Q (L/s):", dcc.Input(id='input-q', type='number', value=11, step=0.01, min=0.1, max=300, style={'width': '120px'}), 'input-q'),
        ("Altura estática (m):", dcc.Input(id='input-hstatic', type='number', value=190, step=0.01, style={'width': '120px'}), 'input-hstatic'),
        ("RPM:", dcc.Input(id='input-rpm', type='number', value=3500, step=0.01, style={'width': '120px'}), 'input-rpm'),
        ("Longitud tubería (m):", dcc.Input(id='input-length', type='number', value=2000, step=0.01, style={'width': '120px'}), 'input-length'),
        ("Diámetro tubería (mm):", dcc.Input(id='input-diam', type='number', value=150, step=0.01, style={'width': '120px'}), 'input-diam'),
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
        ("Eficiencia pico (%):", dcc.Input(id='input-eff', type='number', value=78, step=0.01, min=50, max=95, style={'width': '120px'}), 'input-eff'),
        ("Costo electricidad (USD/kWh):", dcc.Input(id='input-cost', type='number', value=0.12, step=0.01, style={'width': '120px'}), 'input-cost'),
        ("Nivel inicial tanque (%):", dcc.Input(id='input-tankinit', type='number', value=80, step=0.01, min=0, max=100, style={'width': '120px'}), 'input-tankinit'),
        ("Nivel mínimo tanque (%):", dcc.Input(id='input-tankmin', type='number', value=30, step=0.01, min=0, max=99, style={'width': '120px'}), 'input-tankmin'),
        ("Días de simulación:", dcc.Input(id='input-days', type='number', value=dias, step=1, min=1, style={'width': '120px'}), 'input-days'),
        ("Factores horarios (24, separados por coma):", dcc.Textarea(id='input-hourly', value=default_hourly_factors(), style={'width': '180px', 'height': '60px', 'fontSize': '15px'}), 'input-hourly'),
        ("% RPM curva combinada:", dcc.Input(id='input-rpm-combinada', type='number', value=100, step=0.01, min=50, max=100, style={'width': '120px'}), 'input-rpm-combinada'),
        # ("Capacidad tanque (m³):", dcc.Input(
        #     id='input-tank-cap',
        #     type='number',
        #     value=None,
        #     step=0.01,
        #     min=1,
        #     style={'width': '120px'},
        #     disabled=(dias == 1)
        # ), 'input-tank-cap'),
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
        html.Div([
            html.Button('Calcular', id='btn-calc', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '120px', 'background': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'marginRight': '10px', 'display': 'inline-block'}),
            html.Button('Análisis', id='btn-analisis', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '140px', 'background': '#6f42c1', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'display': 'inline-block', 'marginRight': '10px'}),
            html.Button('Borrar', id='btn-borrar-analisis', n_clicks=0, style={'marginTop': '18px', 'fontFamily': 'Segoe UI', 'fontSize': '16px', 'width': '100px', 'background': '#dc3545', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'display': 'inline-block'})
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

    # Panel lateral colapsable de datos (estructura igual a CRITERIOS)
    html.Div([
        html.Div(get_input_panel(), id='input-panel', style={
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(100%)', 'borderLeft': '2px solid #007bff',
        }),
        html.Div('DATOS', id='tab-datos', n_clicks=0, style={
            'position': 'fixed', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
            'background': 'rgba(0,123,255,0.25)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 31, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        })
    ], style={'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'zIndex': 20}),
    # Panel colapsable de criterios (igual que antes)
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
    # Panel colapsable TANQUES (igual estructura, pero a la izquierda y sin contenido)
    html.Div([
        html.Div('TANQUES', id='tab-tanques', n_clicks=0, style={
            'position': 'absolute', 'bottom': '40px', 'left': 0, 'width': '36px', 'height': '100px',
            'background': 'rgba(0,123,255,0.18)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'borderRadius': '0 8px 8px 0',
            'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }),
        html.Div([
            html.Label("Método de cálculo de capacidad de tanque:", style={'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginBottom': '4px', 'textAlign': 'left', 'width': '100%'}),
            dcc.RadioItems(
                id='input-tank-method',
                options=[
                    {'label': 'Cálculo acumulativo (simulación)', 'value': 'simulacion'},
                    {'label': 'Norma CO 10.7-602', 'value': 'norma'},
                    {'label': 'Manual', 'value': 'manual'}
                ],
                value='simulacion',
                labelStyle={'display': 'block', 'marginBottom': '4px'},
                style={'marginBottom': '12px'}
            ),
            html.Div([
                html.Label([
                    "Capacidad tanque (m³):",
                    html.Span(' ⓘ', id='tip-input-tank-cap', n_clicks=0, style={'color': '#007bff', 'cursor': 'pointer', 'fontSize': '16px', 'marginLeft': '6px', 'verticalAlign': 'middle'})
                ], style={'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginBottom': '4px', 'textAlign': 'left', 'width': '100%'}),
                dcc.Input(
                    id='input-tank-cap',
                    type='number',
                    value=None,
                    step=0.01,
                    min=1,
                    style={'width': '100%', 'marginBottom': '18px'},
                    disabled=True  # Solo editable si es manual
                )
            ], id='div-tank-cap', style={'width': '100%', 'marginBottom': '12px', 'display': 'flex', 'flexDirection': 'column'}),
            html.Div([
                html.Label([
                    "Redondeo (múltiplo de):",
                    html.Span(' ⓘ', id='tip-input-tank-round', n_clicks=0, style={'color': '#007bff', 'cursor': 'pointer', 'fontSize': '16px', 'marginLeft': '6px', 'verticalAlign': 'middle'})
                ], style={'fontFamily': 'Segoe UI', 'fontSize': '15px', 'color': '#333', 'marginBottom': '4px', 'textAlign': 'left', 'width': '100%'}),
                dcc.Input(
                    id='input-tank-round',
                    type='number',
                    value=50,
                    step=1,
                    min=1,
                    style={'width': '100%'},
                    disabled=False
                )
            ], style={'width': '100%', 'marginBottom': '0', 'display': 'flex', 'flexDirection': 'column'})
        ], id='tanques-box', style={
            'position': 'fixed', 'bottom': '30px', 'left': '50px', 'background': 'rgba(255,255,255,0.97)',
            'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'maxHeight': '420px', 'overflowY': 'auto', 'zIndex': 30,
            'transition': 'transform 0.4s', 'transform': 'translateX(0)',
            'borderRight': '2px solid #007bff',
            'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch', 'gap': '0px'
        })
    ], style={'position': 'fixed', 'bottom': 0, 'left': 0, 'height': '100vh', 'zIndex': 20}),
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

# --- Store para el estado del panel TANQUES ---
app.layout.children.insert(2, dcc.Store(id='tanques-open', data=False))

# --- Callback para mostrar análisis solo al hacer clic en el botón ---
@app.callback(
    Output('show-analysis', 'data'),
    [Input('btn-analisis', 'n_clicks'), Input('btn-borrar-analisis', 'n_clicks')]
)
def show_analysis_callback(n_analisis, n_borrar):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'btn-analisis' and n_analisis:
        return True
    elif trigger == 'btn-borrar-analisis' and n_borrar:
        return False
    return False

# --- Callback para mostrar/ocultar el panel de datos igual que el de criterios ---
@app.callback(
    Output('input-panel', 'style'),
    Output('tab-datos', 'style'),
    Output('panel-open', 'data'),
    [Input('tab-datos', 'n_clicks')],
    [State('panel-open', 'data')]
)
def toggle_panel(tab_clicks, is_open):
    if tab_clicks is None:
        # Estado inicial: cerrado
        panel_style = {
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(100%)', 'borderLeft': '2px solid #007bff'
        }
        tab_style = {
            'position': 'fixed', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
            'background': 'rgba(0,123,255,0.25)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 31, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }
        return panel_style, tab_style, False
    open_now = not is_open if is_open is not None else True
    if open_now:
        panel_style = {
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(0)', 'borderLeft': '2px solid #007bff'
        }
        tab_style = {
            'position': 'fixed', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
            'background': 'rgba(0,123,255,0.28)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 31, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }
        return panel_style, tab_style, True
    else:
        panel_style = {
            'position': 'fixed', 'top': 0, 'right': 0, 'height': '100vh', 'width': '340px',
            'background': 'rgba(255,255,255,0.98)', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
            'zIndex': 30, 'overflowY': 'auto', 'transition': 'transform 0.4s',
            'transform': 'translateX(100%)', 'borderLeft': '2px solid #007bff'
        }
        tab_style = {
            'position': 'fixed', 'top': '40%', 'right': 0, 'width': '36px', 'height': '120px',
            'background': 'rgba(0,123,255,0.25)', 'color': '#222', 'writingMode': 'vertical-rl',
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'borderRadius': '8px 0 0 8px',
            'cursor': 'pointer', 'zIndex': 31, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI',
        }
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

# Callback para mostrar/ocultar TANQUES
@app.callback(
    Output('tanques-box', 'style'),
    Output('tab-tanques', 'style'),
    Output('tanques-open', 'data'),
    [Input('tab-tanques', 'n_clicks')],
    [State('tanques-open', 'data')]
)
def toggle_tanques(tab_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, False
    if not is_open:
        box_style = {'position': 'fixed', 'bottom': '30px', 'left': '50px', 'background': 'rgba(255,255,255,0.97)',
                     'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
                     'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'maxHeight': '420px', 'overflowY': 'auto', 'zIndex': 30,
                     'transition': 'transform 0.4s', 'transform': 'translateX(0)',
                     'borderRight': '2px solid #007bff',
                     'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch', 'gap': '0px'
                     }
        tab_style = {'position': 'absolute', 'bottom': '40px', 'left': 0, 'width': '36px', 'height': '100px',
                     'background': 'rgba(0,123,255,0.28)', 'color': '#222', 'writingMode': 'vertical-rl',
                     'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'borderRadius': '0 8px 8px 0',
                     'cursor': 'pointer', 'zIndex': 20, 'transition': 'background 0.3s', 'fontFamily': 'Segoe UI'}
        return box_style, tab_style, True
    else:
        box_style = {'position': 'fixed', 'bottom': '30px', 'left': '50px', 'background': 'rgba(255,255,255,0.97)',
                     'padding': '18px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.10)',
                     'fontFamily': 'Segoe UI', 'fontSize': '15px', 'maxWidth': '340px', 'maxHeight': '420px', 'overflowY': 'auto', 'zIndex': 30,
                     'transition': 'transform 0.4s', 'transform': 'translateX(-120%)',
                     'borderRight': '2px solid #007bff',
                     'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch', 'gap': '0px'
                     }
        tab_style = {'position': 'absolute', 'bottom': '40px', 'left': 0, 'width': '36px', 'height': '100px',
                     'background': 'rgba(0,123,255,0.18)', 'color': '#222', 'writingMode': 'vertical-rl',
                     'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'borderRadius': '0 8px 8px 0',
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

# --- Callback combinado para habilitar/deshabilitar y actualizar el valor de capacidad de tanque ---
@app.callback(
    [Output('input-tank-cap', 'disabled'),
     Output('input-tank-cap', 'value')],
    [Input('input-tank-method', 'value'),
     Input('input-q', 'value'),
     Input('input-tank-cap', 'value'),
     Input('input-tank-round', 'value')],
)
def update_tank_cap_field(method, Q_lps, tank_cap, tank_round):
    # Lógica para disabled
    disabled = method != 'manual'
    # Lógica para value
    if method == 'manual':
        value = tank_cap
    elif method == 'norma':
        # Norma CO 10.7-602: Va = 0.5 * Qm * 86400 (Qm en L/s)
        if Q_lps is not None and Q_lps > 0:
            round_val = tank_round if tank_round and float(tank_round) > 0 else 50.0
            value = 0.5 * Q_lps * 86400 / 1000.0
            value = float(int(-(-value//round_val))*round_val)
        else:
            value = 50.0
    else:
        value = None  # El valor lo pone el backend en simulación
    return disabled, value

# --- Callback principal: ahora recibe el método de cálculo ---
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
     Input('input-rpm-combinada', 'value'),
     Input('input-tank-cap', 'value'),
     Input('input-tank-round', 'value'),
     Input('input-tank-method', 'value'),
     Input('show-analysis', 'data')],
)
def update_dashboard(n_clicks, Q_lps, h_static, rpm, length, diam_mm, n_parallel, hw_c, eff_peak, elec_cost, tank_init, tank_min, days, hourly_str, rpm_combinada, tank_cap, tank_round, tank_method, show_analysis):
    import traceback
    try:
        # --- ASIGNAR VALORES POR DEFECTO A TODOS LOS CAMPOS ---
        # Valores por defecto del panel de entrada
        Q_lps = 11 if Q_lps is None else Q_lps
        h_static = 190 if h_static is None else h_static
        rpm = 3500 if rpm is None else rpm
        length = 2000 if length is None else length
        diam_mm = 150 if diam_mm is None else diam_mm
        n_parallel = 3 if n_parallel is None else n_parallel
        hw_c = 130 if hw_c is None else hw_c
        elec_cost = 0.12 if elec_cost is None else elec_cost
        tank_init = 80 if tank_init is None else tank_init
        tank_min = 30 if tank_min is None else tank_min
        days = 1 if days is None else days
        hourly_str = default_hourly_factors() if hourly_str is None or hourly_str.strip() == "" else hourly_str
        rpm_combinada = 100 if rpm_combinada is None else rpm_combinada
        
        # --- CONVERSIÓN ROBUSTA DE EFICIENCIA PICO ---
        valor_recibido = eff_peak
        if eff_peak is None or (isinstance(eff_peak, str) and eff_peak.strip() == ""):
            eff_peak_val = 0.78  # 78% por defecto
        else:
            try:
                eff_peak_val = float(eff_peak)
            except Exception:
                eff_peak_val = 0.78  # 78% por defecto si no es convertible
        # Convertir a decimal si es porcentaje
        if eff_peak_val > 1.0:
            eff_peak_val = eff_peak_val / 100.0
        # Validar rango después de conversión
        if not (0.5 <= eff_peak_val <= 0.95):
            return [dcc.Markdown(f'**Error:** Eficiencia pico debe estar entre 0.5 y 0.95. Valor recibido: {valor_recibido} (convertido: {eff_peak_val:.3f})')], [], [], [], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
        eff_peak = eff_peak_val
        
        # --- VALIDACIONES DE RANGO ---
        if not (0.1 <= Q_lps <= 300):
            return [dcc.Markdown('**Advertencia:** El caudal de diseño debe estar entre 0.1 y 300 L/s.')], [], [], [], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
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
        if not (0 <= elec_cost <= 1):
            raise ValueError("Costo de electricidad debe estar entre 0 y 1 USD/kWh")
        if not (0 <= tank_init <= 100):
            raise ValueError("Nivel inicial de tanque debe estar entre 0 y 100%")
        if not (0 <= tank_min < tank_init):
            raise ValueError("Nivel mínimo de tanque debe ser menor que el nivel inicial")
        if not (1 <= days <= 30):
            raise ValueError("Días de simulación deben estar entre 1 y 30")
        
        # --- VALIDACIÓN DE FACTORES HORARIOS ---
        try:
            # Limpiar y validar el string de factores horarios
            if not hourly_str or hourly_str.strip() == "":
                hourly_str = default_hourly_factors()
            
            # Convertir a lista de floats de forma robusta
            hourly_factors = []
            for x in hourly_str.split(','):
                try:
                    val = float(x.strip())
                    if 0 <= val <= 5:
                        hourly_factors.append(val)
                    else:
                        raise ValueError(f"Factor horario {val} fuera del rango [0, 5]")
                except ValueError as ve:
                    raise ValueError(f"Factor horario inválido: '{x.strip()}'. Debe ser un número entre 0 y 5")
            
            if len(hourly_factors) != 24:
                raise ValueError(f"Ingrese exactamente 24 factores horarios, separados por coma. Se recibieron {len(hourly_factors)} valores")
                
        except Exception as e:
            return [dcc.Markdown(f'**Error en factores horarios:** {str(e)}')], [], [], [], dcc.Markdown(get_criterio_texto(), style={'margin': 0})
        
        if not (50 <= rpm_combinada <= 100):
            raise ValueError("El porcentaje de RPM para la curva combinada debe estar entre 50 y 100%")

        # --- CONVERSIONES Y CÁLCULOS ---
        q_design = Q_lps / 1000
        pipe_diameter_m = diam_mm / 1000
        
        # Validar input de capacidad de tanque (robusto para string vacío)
        if isinstance(tank_cap, str) and tank_cap.strip() == '':
            tank_cap = None
        # --- NUEVO: obtener múltiplo de redondeo ---
        if tank_round is None or (isinstance(tank_round, str) and (tank_round.strip() == '' or float(tank_round) <= 0)):
            tank_round_val = 50.0
        else:
            try:
                tank_round_val = float(tank_round)
                if tank_round_val <= 0:
                    tank_round_val = 50.0
            except Exception:
                tank_round_val = 50.0
        # Si el usuario ingresa capacidad de tanque, usarla; si no, calcular y redondear con el múltiplo elegido
        if days > 1 and tank_cap is not None and float(tank_cap) > 0:
            capacidad_tanque = float(tank_cap)
        else:
            capacidad_tanque = None  # Se calculará automáticamente en el backend

        # --- NUEVO: lógica de método de cálculo ---
        capacidad_tanque_norma = None
        capacidad_tanque_sim = None
        if tank_method == 'manual':
            if tank_cap is not None and float(tank_cap) > 0:
                capacidad_tanque = float(tank_cap)
            else:
                capacidad_tanque = None
        elif tank_method == 'norma':
            # Norma CO 10.7-602: Va = 0.5 * Qm * 86400 (Qm en L/s)
            if Q_lps is not None and Q_lps > 0:
                capacidad_tanque = 0.5 * Q_lps * 86400 / 1000.0  # en m³
                # Aplicar redondeo
                round_val = tank_round if tank_round and float(tank_round) > 0 else 50.0
                capacidad_tanque = float(int(-(-capacidad_tanque//round_val))*round_val)
            else:
                capacidad_tanque = 50.0
        else:  # simulacion
            capacidad_tanque = None  # Se calculará en el backend como antes

        # --- CREAR INSTANCIA DEL DISEÑADOR ---
        designer = CentrifugalPumpDesigner(
            q_design=q_design,
            h_static=h_static,
            rpm=rpm,
            hourly_factors=hourly_factors,  # <-- lista de floats
            pipe_length_m=length,
            pipe_diameter_m=pipe_diameter_m,
            n_parallel=n_parallel,
            hw_c=hw_c,
            eff_peak=eff_peak,
            electricity_cost=elec_cost,
            min_tank_level_perc=tank_min/100,
            initial_tank_level_perc=tank_init/100,
            simulation_days=days,
            tank_capacity_m3=capacidad_tanque,
            tank_round_m3=tank_round_val  # <-- nuevo argumento
        )
        
        # --- GENERAR GRÁFICOS ---
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
        
        # --- DISTRIBUCIÓN DE GRÁFICOS POR PESTAÑA ---
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
                
                # Ajustar hovertemplate y formato de ejes para todos los gráficos
                for trace in fig_filtrada.data:
                    # Determinar la unidad y formato según el gráfico
                    if i == 0:  # Curva del sistema vs bomba
                        trace.hovertemplate = '%{y:.2f} m<extra></extra>'
                    elif i == 1:  # Comparación VFD
                        trace.hovertemplate = '%{y:.2f} m<extra></extra>'
                    elif i == 2:  # Rango operacional bomba
                        trace.hovertemplate = '%{y:.2f} m<extra></extra>'
                    elif i == 3:  # Eficiencia vs caudal
                        trace.hovertemplate = '%{y:.2f} %<extra></extra>'
                    elif i == 4:  # Potencia total
                        trace.hovertemplate = '%{y:.2f} kW<extra></extra>'
                    elif i == 5:  # Costo unitario vs caudal
                        trace.hovertemplate = '%{y:.4f} USD/m³<extra></extra>'
                    elif i == 6:  # Leyes de afinidad
                        trace.hovertemplate = '%{y:.2f} m<extra></extra>'
                    elif i == 7:  # Potencia vs caudal (afinidad)
                        trace.hovertemplate = '%{y:.2f} kW<extra></extra>'
                    elif i == 8:  # Eficiencia vs caudal (afinidad)
                        trace.hovertemplate = '%{y:.2f} %<extra></extra>'
                    elif i == 9:  # Volumen en tanque
                        trace.hovertemplate = '%{y:.2f} m³<extra></extra>'
                    elif i == 10:  # Demanda vs bombeo
                        trace.hovertemplate = '%{y:.2f} L/s<extra></extra>'
                    else:
                        trace.hovertemplate = '%{y:.2f}<extra></extra>'
                
                # Formato del eje X (abscisas)
                if i in [9, 10]:  # Tanque y Demanda: eje X son horas (entero)
                    fig_filtrada.update_xaxes(tickformat='.0f')
                else:
                    fig_filtrada.update_xaxes(tickformat='.2f')
                
                # Modificaciones específicas para el gráfico de volumen en tanque (i==9)
                if i == 9:  # Volumen en tanque
                    # Calcular el volumen mínimo
                    volumen_minimo = designer.tank_capacity_m3 * designer.min_tank_level_perc
                    porcentaje_minimo = designer.min_tank_level_perc * 100
                    # Modificar la anotación existente de la línea roja (reserva mínima)
                    if hasattr(fig_filtrada.layout, 'annotations'):
                        for ann in fig_filtrada.layout.annotations:
                            if hasattr(ann, 'text') and 'reserva mínima' in str(ann.text).lower():
                                ann.text = f"Volumen mínimo {volumen_minimo:.0f} m³ ({porcentaje_minimo:.0f}%)"
                                ann.font = dict(size=12, color='red')
                                try:
                                    ann.bgcolor = None
                                    ann.bordercolor = None
                                    ann.borderwidth = None
                                except Exception:
                                    pass
                                break
                    # Asegurar que las líneas horizontales aparezcan en la leyenda
                    tiene_capacidad = False
                    tiene_minimo = False
                    for trace in fig_filtrada.data:
                        if hasattr(trace, 'name'):
                            if 'capacidad' in str(trace.name).lower():
                                tiene_capacidad = True
                            if 'mínima' in str(trace.name).lower() or 'reserva' in str(trace.name).lower():
                                tiene_minimo = True
                    # Si no están en la leyenda, agregar trazas invisibles solo para la leyenda
                    if not tiene_capacidad:
                        fig_filtrada.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='lines',
                            name=f'Capacidad Tanque ({designer.tank_capacity_m3:.0f} m³)',
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=True
                        ))
                    if not tiene_minimo:
                        fig_filtrada.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='lines',
                            name=f'Reserva Mínima ({porcentaje_minimo:.0f}%)',
                            line=dict(color='red', width=2, dash='dash'),
                            showlegend=True
                        ))
                
                # Si es pestaña 3, Potencia vs Caudal (afinidad, i==7), mostrar potencia con 2 decimales y solo el valor Y
                if i == 7:
                    for trace in fig_filtrada.data:
                        trace.hovertemplate = '%{y:.2f} kW<extra></extra>'
                
                # Cambiar el color del texto del hover a blanco para la curva bomba individual en gráfico 1 y 3
                if i == 0:
                    for trace in fig_filtrada.data:
                        if hasattr(trace, 'name') and 'bomba individual' in str(trace.name).lower():
                            trace.hoverlabel = dict(font_color='white')
                if i == 2:
                    for trace in fig_filtrada.data:
                        if hasattr(trace, 'name') and 'bomba individual' in str(trace.name).lower():
                            trace.hoverlabel = dict(font_color='white')
                
                # Si es Comparación VFD (i==1), mostrar la curva original y la curva combinada editable al % de RPM seleccionado
                if i == 1:
                    q_total_range_lps = designer.q_range * designer.n_parallel * 1000
                    r = rpm_combinada / 100 if rpm_combinada else 1.0
                    q_per_pump_vfd = (q_total_range_lps / 1000) / designer.n_parallel
                    q_homologous_100 = q_per_pump_vfd / r
                    h_range_edit = designer.pump_head(q_homologous_100) * r**2
                    # Calcular Hf y V para cada punto de caudal
                    L = designer.pipe_length_m
                    D = designer.pipe_diameter_m
                    C = designer.hw_c
                    Q_m3s = q_total_range_lps / 1000  # Caudal total en m3/s
                    A = 3.14159265359 * (D/2)**2  # Área de la tubería en m2
                    V = Q_m3s / A  # Velocidad en m/s
                    # Hazen-Williams: hf = 10.67 * L * Q^1.852 / (C^1.852 * D^4.871)
                    Hf = 10.67 * L * (Q_m3s**1.852) / (C**1.852 * D**4.871)
                    customdata = np.stack([Hf, V], axis=-1)
                    fig_filtrada.add_trace(go.Scatter(
                        x=q_total_range_lps,
                        y=h_range_edit,
                        mode='lines',
                        name=f'Curva Combinada @ {rpm_combinada:.1f}% RPM',
                        line=dict(color='blue', width=2, dash='dot'),
                        customdata=customdata,
                        hovertemplate='%{y:.2f} m<br>Hf: %{customdata[0]:.2f} m<br>V: %{customdata[1]:.2f} m/s<extra></extra>'
                    ))
                
                # En el gráfico 3 (i==2), agregar la curva azul dinámica (bomba individual a %RPM) y la curva naranja (bomba individual a velocidad VFD)
                if i == 2:
                    # Curva azul dinámica (editable por %RPM)
                    r = rpm_combinada / 100 if rpm_combinada else 1.0
                    q_range_lps = designer.q_range * 1000
                    q_homologous_100 = designer.q_range / r
                    h_range_azul = designer.pump_head(q_homologous_100) * r**2
                    fig_filtrada.add_trace(go.Scatter(
                        x=q_range_lps * r,  # Caudal ajustado a %RPM
                        y=h_range_azul,
                        mode='lines',
                        name=f'Bomba Individual @ {rpm_combinada:.1f}% RPM',
                        line=dict(color='blue', width=2, dash='dot')
                    ))
                    # Curva naranja (bomba individual a velocidad VFD real)
                    if designer.vfd_results:
                        r_vfd = designer.vfd_results.get('speed_ratio', 1)
                        q_homologous_100_vfd = designer.q_range / r_vfd
                        h_range_naranja = designer.pump_head(q_homologous_100_vfd) * r_vfd**2
                        fig_filtrada.add_trace(go.Scatter(
                            x=q_range_lps * r_vfd,
                            y=h_range_naranja,
                            mode='lines',
                            name=f'Bomba Individual @ {r_vfd*100:.1f}% RPM',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                
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
                        eff_op = designer.efficiency(designer.q_op) * 100
                        velocidad = designer.q_op / (3.1416 * (designer.pipe_diameter_m / 2) ** 2)
                        op_box = html.Div(
                            f"Punto de operación: {q_op:.2f} L/s, {power_op:.2f} kW ({power_hp:.2f} HP)",
                            style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px',
                                   'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center',
                                   'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'}
                        )
                        # --- ANÁLISIS EXPERTO ---
                        advertencias, soluciones, explicaciones = analisis_criterios_ia(q_op, h_op, power_op, eff_op, velocidad, hw_c, diam_mm, n_parallel)
                        if advertencias:
                            analisis.append('<b>Advertencias técnicas:</b>\n' + '<br>'.join(advertencias))
                        if soluciones:
                            analisis.append('<b>Soluciones sugeridas:</b>\n' + '<br>'.join(soluciones))
                        if explicaciones:
                            analisis.append('<b>Explicación técnica:</b>\n' + '<br>'.join(explicaciones))
                        # --- FIN ANÁLISIS EXPERTO ---
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
                        eff_op = designer.efficiency(designer.q_op) * 100
                        velocidad = designer.q_op / (3.1416 * (designer.pipe_diameter_m / 2) ** 2)
                        op_box = html.Div(
                            f"Punto de operación: {q_op:.2f} L/s, {power_op:.2f} kW ({power_hp:.2f} HP)",
                            style={'background': '#eaf4ff', 'color': '#007bff', 'fontFamily': 'Segoe UI', 'fontSize': '15px',
                                   'padding': '7px 14px', 'borderRadius': '7px', 'margin': '10px 0 0 0', 'textAlign': 'center',
                                   'boxShadow': '0 1px 4px rgba(0,123,255,0.08)'}
                        )
                        # --- ANÁLISIS EXPERTO ---
                        advertencias, soluciones, explicaciones = analisis_criterios_ia(q_op, h_op, power_op, eff_op, velocidad, hw_c, diam_mm, n_parallel)
                        if advertencias:
                            analisis.append('<b>Advertencias técnicas:</b>\n' + '<br>'.join(advertencias))
                        if soluciones:
                            analisis.append('<b>Soluciones sugeridas:</b>\n' + '<br>'.join(soluciones))
                        if explicaciones:
                            analisis.append('<b>Explicación técnica:</b>\n' + '<br>'.join(explicaciones))
                        # --- FIN ANÁLISIS EXPERTO ---
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
        return (*tab_contents, dcc.Markdown(get_criterio_texto(), style={'margin': 0}))
    except Exception as e:
        import sys
        print("\nERROR EN CALLBACK PRINCIPAL DE DASH:")
        traceback.print_exc(file=sys.stdout)
        error_msg = f"**Error crítico en Dash:** {str(e)}"
        return [dcc.Markdown(error_msg)], [], [], [], dcc.Markdown(get_criterio_texto(), style={'margin': 0})

# --- Subtítulo dinámico del material ---

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

# --- 2. Callback para actualizar el panel de datos cuando cambian los días ---

def toggle_tank_cap_input(dias):
    return dias == 1



if __name__ == '__main__':
    app.run(debug=True)