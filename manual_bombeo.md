# Manual Avanzado de Análisis y Diseño de Sistemas de Bombeo

## 1. Explicación Profunda de los Datos de Entrada

### Caudal de diseño Q (L/s)
**Concepto:**  
El caudal de diseño es el flujo máximo de agua que el sistema debe ser capaz de transportar de manera continua. Se expresa en litros por segundo (L/s) y se determina a partir de la demanda máxima esperada, considerando consumos simultáneos, crecimiento futuro y factores de seguridad.

**Importancia en el diseño:**  
Un caudal subestimado puede provocar desabastecimiento, mientras que uno sobredimensionado incrementa costos de inversión y operación. Es fundamental analizar la demanda real, los patrones horarios y estacionales, y prever expansiones.

**Recomendación:**  
Utilizar estudios de demanda, registros históricos y aplicar factores de simultaneidad y crecimiento.

---

### Altura estática (m)
**Concepto:**  
Es la diferencia de elevación entre el nivel de agua en el punto de succión (típicamente un pozo o tanque inferior) y el punto de descarga (tanque elevado, red, etc.). No depende del caudal.

**Importancia en el diseño:**  
Determina la carga mínima que la bomba debe vencer, independientemente de las pérdidas por fricción.

**Recomendación:**  
Medir con precisión y considerar variaciones de nivel en el punto de succión y descarga.

---

### RPM (Revoluciones por minuto)
**Concepto:**  
Es la velocidad de rotación del eje de la bomba. Afecta directamente la curva característica de la bomba (caudal, carga y potencia).

**Importancia en el diseño:**  
La selección de RPM influye en la eficiencia, el rango de operación y la posibilidad de usar variadores de frecuencia (VFD).

**Recomendación:**  
Usar la velocidad recomendada por el fabricante y considerar el uso de VFD para flexibilidad operativa.

---

### Longitud de tubería (m)
**Concepto:**  
Es la distancia total que recorre el agua desde la bomba hasta el punto de entrega.

**Importancia en el diseño:**  
Afecta las pérdidas por fricción, que aumentan con la longitud y el caudal.

**Recomendación:**  
Minimizar la longitud cuando sea posible y considerar trayectorias alternativas para reducir pérdidas.

---

### Diámetro de tubería (mm)
**Concepto:**  
Diámetro interno de la tubería de impulsión.

**Importancia en el diseño:**  
Un diámetro mayor reduce las pérdidas por fricción, pero incrementa el costo de materiales y la velocidad mínima de autolimpieza.

**Recomendación:**  
Buscar un equilibrio entre costo, velocidad mínima y pérdidas aceptables. Usar fórmulas de Hazen-Williams o Darcy-Weisbach para el cálculo.

---

### Bombas en paralelo
**Concepto:**  
Número de bombas instaladas en paralelo.

**Importancia en el diseño:**  
Permite modular la capacidad, mejorar la redundancia y facilitar el mantenimiento.

**Recomendación:**  
Diseñar para que una bomba pueda cubrir la demanda básica y las otras entren en operación en picos o ante fallas.

---

### C Hazen-Williams
**Concepto:**  
Coeficiente de rugosidad de la tubería según la fórmula de Hazen-Williams. Depende del material, la edad y el estado interno de la tubería.

**Importancia en el diseño:**  
Afecta el cálculo de pérdidas por fricción. Valores típicos: PVC (140-150), hierro dúctil (120-140), acero (100-120).

**Recomendación:**  
Usar valores conservadores para tuberías viejas o con incrustaciones.

---

### Eficiencia pico
**Concepto:**  
Eficiencia máxima de la bomba, normalmente en el punto de mejor eficiencia (BEP).

**Importancia en el diseño:**  
Afecta el consumo energético y el costo operativo.

**Recomendación:**  
Seleccionar bombas con alta eficiencia en el rango de operación esperado.

---

### Costo electricidad (USD/kWh)
**Concepto:**  
Costo unitario de la energía eléctrica.

**Importancia en el diseño:**  
Permite calcular el costo operativo y comparar alternativas de diseño.

**Recomendación:**  
Considerar tarifas horarias, penalizaciones por demanda máxima y posibles incrementos futuros.

---

### Nivel inicial tanque (%)
**Concepto:**  
Porcentaje de llenado del tanque al inicio de la simulación.

**Importancia en el diseño:**  
Afecta la dinámica inicial de la simulación y la capacidad de respuesta ante demandas imprevistas.

**Recomendación:**  
Usar valores realistas según la operación habitual.

---

### Nivel mínimo tanque (%)
**Concepto:**  
Porcentaje mínimo permitido de llenado del tanque.

**Importancia en el diseño:**  
Evita la succión de aire, daños a la bomba y asegura reserva ante emergencias.

**Recomendación:**  
Definir según normativas y experiencia operativa.

---

### Días de simulación
**Concepto:**  
Duración de la simulación operacional.

**Importancia en el diseño:**  
Permite analizar el comportamiento del sistema ante variaciones diarias y eventos críticos.

**Recomendación:**  
Simular al menos varios días para capturar ciclos completos de demanda.

---

### Factores horarios (24, separados por coma)
**Concepto:**  
Perfil horario de la demanda, expresado como factores multiplicativos para cada hora del día.

**Importancia en el diseño:**  
Permite simular la variación real de la demanda y dimensionar adecuadamente el sistema.

**Recomendación:**  
Basar los factores en registros históricos o estudios de consumo.

---

## 2. Análisis Profundo de los 11 Gráficos Generados

### 1. Curva del sistema vs bomba
**Descripción:**  
Superpone la curva de carga del sistema (incluyendo pérdidas por fricción y altura estática) con la curva característica de la bomba y la curva combinada de bombas en paralelo. El punto de intersección es el punto de operación.

**Función:**  
Permite verificar si la bomba seleccionada es adecuada para el sistema y si el punto de operación está dentro del rango recomendado.

**Estrategias de diseño:**  
- El punto de operación debe estar cerca del BEP.
- La curva de la bomba debe superar la curva del sistema en todo el rango de operación.
- Evitar operar en los extremos de la curva de la bomba.

---

### 2. Comparación VFD
**Descripción:**  
Compara la operación de la bomba a velocidad nominal y con variador de frecuencia (VFD), mostrando el desplazamiento del punto de operación y el ahorro energético potencial.

**Función:**  
Evalúa la flexibilidad y eficiencia que aporta el uso de VFD.

**Estrategias de diseño:**  
- Usar VFD para adaptar la bomba a variaciones de demanda.
- Analizar el ahorro energético y el retorno de inversión del VFD.

---

### 3. Rango operacional bomba
**Descripción:**  
Muestra el rango preferido de operación de la bomba (típicamente 70-120% del BEP), el BEP y el punto real de operación.

**Función:**  
Verifica que la bomba opere dentro de los límites recomendados por el fabricante.

**Estrategias de diseño:**  
- Operar dentro del rango preferido para maximizar eficiencia y vida útil.
- Evitar cavitación y sobrecarga.

---

### 4. Eficiencia vs caudal
**Descripción:**  
Curva de eficiencia de la bomba en función del caudal, destacando el punto de operación.

**Función:**  
Permite identificar el caudal óptimo para máxima eficiencia.

**Estrategias de diseño:**  
- Seleccionar la bomba para que el punto de operación esté cerca del máximo de la curva.
- Evitar operar en zonas de baja eficiencia.

---

### 5. Potencia total
**Descripción:**  
Curva de potencia requerida por el sistema en función del caudal, mostrando el punto de operación.

**Función:**  
Permite dimensionar el motor y estimar el consumo energético.

**Estrategias de diseño:**  
- Considerar márgenes de seguridad.
- Seleccionar motores con eficiencia adecuada y capacidad suficiente.

---

### 6. Costo unitario vs caudal
**Descripción:**  
Costo energético por metro cúbico bombeado en función del caudal, con el punto de operación resaltado.

**Función:**  
Evalúa la eficiencia económica del sistema.

**Estrategias de diseño:**  
- Optimizar el diseño para minimizar el costo unitario.
- Considerar tarifas eléctricas y horarios de menor costo.

---

### 7. Leyes de afinidad
**Descripción:**  
Muestra cómo varían caudal, carga y potencia con la velocidad de la bomba, según las leyes de afinidad.

**Función:**  
Permite analizar el efecto de cambios de velocidad (VFD) sobre el comportamiento hidráulico.

**Estrategias de diseño:**  
- Usar VFD para ajustar la operación a la demanda real.
- Evaluar el impacto en eficiencia y consumo.

---

### 8. Potencia vs caudal (afinidad)
**Descripción:**  
Curvas de potencia para diferentes velocidades de la bomba, mostrando el punto de operación.

**Función:**  
Evalúa el impacto del control de velocidad en el consumo energético.

**Estrategias de diseño:**  
- Ajustar la velocidad para reducir el consumo en baja demanda.
- Verificar que el motor soporte la gama de potencias requeridas.

---

### 9. Eficiencia vs caudal (afinidad)
**Descripción:**  
Curvas de eficiencia para diferentes velocidades de la bomba, con el punto de operación.

**Función:**  
Identifica el rango óptimo de operación bajo control de velocidad.

**Estrategias de diseño:**  
- Operar cerca del máximo de eficiencia en cada velocidad.
- Evitar operar en zonas de baja eficiencia.

---

### 10. Volumen en tanque
**Descripción:**  
Simulación del volumen de agua en el tanque a lo largo del tiempo, mostrando la capacidad máxima y el nivel mínimo permitido.

**Función:**  
Verifica que el tanque nunca baje del nivel mínimo ni sobrepase la capacidad.

**Estrategias de diseño:**  
- Ajustar el tamaño del tanque y la operación de las bombas para mantener niveles seguros.
- Analizar escenarios de alta y baja demanda.

---

### 11. Demanda vs bombeo
**Descripción:**  
Comparación entre la demanda horaria y el caudal bombeado, mostrando la capacidad de respuesta del sistema.

**Función:**  
Evalúa si el sistema puede satisfacer la demanda en todo momento y si hay excedentes o déficits.

**Estrategias de diseño:**  
- Ajustar la programación de bombeo y el tamaño del tanque para evitar déficits.
- Sincronizar la operación de las bombas con los picos de demanda.

---

## 3. Criterios de Diseño Utilizados

- **Curva del sistema:** Hazen-Williams para pérdidas por fricción.
- **Curva de bomba:** Parabólica, estimada si no se provee.
- **Punto de operación:** Intersección sistema-bomba.
- **Balance de masa:** Método de Rippl para simulación de tanque.
- **Costos energéticos:** Basados en potencia eje y costo unitario.
- **Leyes de afinidad:** Para analizar variación de RPM.
- **Simulación:** Balance horario y volumen de tanque.

---

## 4. Estrategia de Uso de la App

1. **Ingrese los datos de entrada** en el panel lateral “DATOS”, asegurándose de comprender el significado y la importancia de cada parámetro.
2. **Presione “Calcular”** para actualizar los gráficos y resultados.
3. **Analice cada pestaña**:
   - En “Curvas y Operación”, verifique que el punto de operación sea adecuado y que la bomba seleccionada cumpla con los requisitos del sistema.
   - En “Eficiencia-Potencia-Costo”, busque la máxima eficiencia y el menor costo unitario, y dimensione el motor correctamente.
   - En “Leyes de Afinidad”, evalúe el impacto de variar la velocidad de la bomba y la conveniencia de usar VFD.
   - En “Tanque y Demanda”, asegúrese de que el sistema cubre la demanda y el tanque opera en niveles seguros.
4. **Ajuste los parámetros** según los resultados para optimizar el diseño, minimizando costos y maximizando la confiabilidad.
5. **Consulte los criterios de diseño** en la pestaña “CRITERIOS” para recordar las bases técnicas del análisis y tomar decisiones fundamentadas. 