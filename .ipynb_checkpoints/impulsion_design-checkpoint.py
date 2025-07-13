import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ImpulsionPotable:
    """
    Visualización de impulsión de agua potable para cuatro materiales (PVC, PEAD, Acero, HierroDúctil).

    Gráficos (cuadrícula 2×2):
      • **Headloss** – pérdida de carga Hazen‑Williams   [mH₂O]  (eje 0‑50m)
      • **Velocidad** – velocidad de flujo              [m/s]
      • **Sobrepresión** – presión transitoria Joukowsky [mH₂O]  (límites sólo para HD)
      • **Total** – Headloss + **altura estática**      [mH₂O]

    ### Lógica de límites HierroDúctil (ISO2531 – PN 10 como ejemplo)
      * **PN** – Presión nominal de la tubería (PFA).  Se ingresa en **bar**.
      * **PFA** = PN  →  Operación
      * **PMA** = 1.2×PN →  Transitorio
      * **PEA** = 1.2×PN + 5 bar →  Prueba

    Los tres valores se convierten a metros de columna de agua (1bar ≈10.197mH₂O) y se dibujan sólo
    en el gráfico de Sobrepresión.

    La **altura estática** (**h₀**, en mH₂O) se usa únicamente para sumar a la pérdida por fricción en la
    gráfica "Total".
    """

    def __init__(self, Q: float, L: float, h0: float, pn_bar: float = 10):
        """Parámetros
        ----------
        Q       : Caudal [m³/s]
        L       : Longitud de la tubería [m]
        h0      : Altura estática [mH₂O]
        pn_bar  : Presión nominal de HierroDúctil en **bar** (por defecto PN10).
        """
        self.Q, self.L, self.h0 = Q, L, h0
        self.rho, self.g, self.K = 1000, 9.81, 2.2e9

        # Conversión bar → mH₂O
        self.bar_to_m = 10.197
        pn_m = pn_bar * self.bar_to_m  # PN en metros de columna de agua

        # Límites HD según ISO2531
        self.lim = {
            'Operación':   pn_m,                      # PFA
            'Transitorio': 1.2 * pn_m,               # PMA = 1.2·PN
            'Prueba':      1.2 * pn_m + 5 * self.bar_to_m  # PEA = 1.2·PN + 5 bar
        }

        # Hazen–Williams y propiedades elásticas
        self.C = {
            'PVC': 150,
            'PEAD': 140,
            'Acero': 120,
            'Hierro Dúctil': 130
        }
        self.prop = {
            'PVC': (4e9, 0.005),
            'PEAD': (1e9, 0.006),
            'Acero': (2e11, 0.01),
            'Hierro Dúctil': (1e11, 0.012)
        }

        # Diámetros (mm) evaluados
        self.D = np.arange(50, 600, 10)
        
        # Valores inventados
        self.precios = {
            'PVC': {
                50: 3.50, 63: 4.20, 75: 5.10, 90: 6.80, 110: 8.50, 125: 10.20,
                140: 12.50, 160: 15.80, 200: 22.40, 250: 32.60, 315: 48.20,
                400: 72.50, 500: 125.80
            },
            'PEAD': {
                50: 4.80, 63: 6.20, 75: 7.90, 90: 10.50, 110: 14.20, 125: 17.80,
                140: 22.40, 160: 28.60, 200: 42.80, 250: 65.40, 315: 98.20,
                400: 152.80, 500: 245.60
            },
            'Acero': {
                50: 12.50, 63: 15.80, 75: 19.60, 90: 24.80, 110: 32.40, 125: 38.90,
                140: 46.80, 160: 58.20, 200: 82.60, 250: 125.40, 315: 186.80,
                400: 285.60, 500: 425.80
            },
            'Hierro Dúctil': {
                50: 18.60, 63: 22.80, 75: 28.40, 90: 36.20, 110: 48.60, 125: 58.40,
                140: 69.80, 160: 86.40, 200: 125.80, 250: 186.40, 315: 278.60,
                400: 425.80, 500: 685.40
            }
        }

    # ----------------  ----------------
    def _hl(self, d_mm):
        d_m = d_mm / 1000
        return {m: 10.67 * self.L * self.Q ** 1.852 / (C ** 1.852 * d_m ** 4.87)
                for m, C in self.C.items()}

    def _vel(self, d_mm):
        area = np.pi * (d_mm / 1000 / 2) ** 2
        return {m: self.Q / area for m in self.C}

    def _op(self, d_mm):
        area = np.pi * (d_mm / 1000 / 2) ** 2
        u = self.Q / area
        res = {}
        for m, (E, delta) in self.prop.items():
            a = np.sqrt((self.K / self.rho) / (1 + (self.K * (d_mm / 1000)) / (E * delta)))
            res[m] = (self.rho * a * u) / (self.rho * self.g)
        return res

    # ---------------- GRÁFICOS ----------------
    def plot(self):
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Pérdida de Carga (m H₂O)",
                "Velocidad (m/s)",
                "Sobrepresión (m H₂O)",
                "Total = HL + h₀ (m H₂O)"
            )
        )

        for m in self.C:
            hl  = [self._hl(d)[m]  for d in self.D]
            vel = [self._vel(d)[m] for d in self.D]
            op  = [self._op(d)[m]  for d in self.D]
            tt  = [h + self.h0 for h in hl]

            fig.add_trace(go.Scatter(x=self.D, y=hl,  mode='lines', name=f'HL {m}'),  row=1, col=1)
            fig.add_trace(go.Scatter(x=self.D, y=vel, mode='lines', name=f'Vel {m}'), row=1, col=2)
            fig.add_trace(go.Scatter(x=self.D, y=op,  mode='lines', name=f'OP {m}'),  row=2, col=1)
            fig.add_trace(go.Scatter(x=self.D, y=tt,  mode='lines', name=f'TT {m}'),  row=2, col=2)

        # Dibujar límites sólo para HD en sobrepresión y gráfico Total
        for label, val in self.lim.items():
            # Sobrepresión (fila 2, col 1)
            fig.add_hline(y=val, row=2, col=1, line_dash='dash',
                          annotation_text=f"{label}: {val:.1f} m H₂O",
                          annotation_position="top left")
            # Total (fila 2, col 2) - ESTA ES LA ÚNICA LÍNEA AGREGADA
            fig.add_hline(y=val, row=2, col=2, line_dash='dash',
                          annotation_text=f"{label}: {val:.1f} m H₂O",
                          annotation_position="top right")

        # Configuración de ejes
        for r in range(1, 3):
            for c in range(1, 3):
                fig.update_xaxes(title_text='Diámetro (mm)', row=r, col=c)
        fig.update_yaxes(title_text='m H₂O', row=1, col=1, range=[0, 50])
        fig.update_yaxes(title_text='m/s',   row=1, col=2)
        fig.update_yaxes(title_text='m H₂O', row=2, col=1, range=[0, max(self.lim.values()) * 1.1])
        fig.update_yaxes(title_text='m H₂O', row=2, col=2, range=[0, self.h0 * 2])

        fig.update_layout(
            showlegend=False,
            height=1500,
            width=2100,
            hovermode='x',
            title=f"Impulsión- q:{self.Q * 1000}L/s,  L:{self.L}m, est:{self.h0}m"
        )
        fig.show()

# Ejemplo de uso
if __name__ == '__main__':
    ImpulsionPotable(Q=11/1000, L=2000, h0=190, pn_bar=40).plot()