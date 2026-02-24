# AAPL Time Series Statistical Analysis
> *"In God we trust. All others must bring data."* — W. Edwards Deming

Análisis estadístico cuantitativo completo de la serie temporal de precios de AAPL (Apple Inc.) para el período 2019–2024, implementado **desde primeros principios**: todos los tests estadísticos se construyen sobre `numpy` y `scipy` sin depender de `statsmodels`, garantizando máxima portabilidad y comprensión metodológica.

---

## Tabla de contenido

- [Descripción general](#descripción-general)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Módulos](#módulos)
  - [data\_loader.py](#data_loaderpy)
  - [transformations.py](#transformationspy)
  - [statistical\_tests.py](#statistical_testspy)
  - [decomposition.py](#decompositionpy)
  - [visualization.py](#visualizationpy)
- [Análisis del notebook](#análisis-del-notebook)
- [Resultados clave](#resultados-clave)
- [Próximos pasos](#próximos-pasos)
- [Stack tecnológico](#stack-tecnológico)

---

## Descripción general

Este proyecto realiza un análisis exploratorio estadístico profundo de una serie temporal financiera de alta frecuencia (datos diarios OHLCV). El objetivo es caracterizar completamente la distribución de los retornos, identificar estructura temporal (tendencia, estacionalidad, autocorrelación) y cuantificar el riesgo antes de proceder a etapas de modelado predictivo.

**Datos:** Serie temporal sintética de AAPL generada mediante Movimiento Browniano Geométrico (GBM) con volatilidad estocástica al estilo Heston, colas gruesas (distribución *t* de Student con ν = 5) y cambios de régimen realistas que replican el *crash* del COVID-19 (feb–mar 2020), el rally post-pandemia y el mercado bajista de 2022.

**Período:** 1 enero 2019 → 1 enero 2024 | **1 305 días hábiles de negociación**

---

## Estructura del proyecto

```
aapl-timeseries-analysis/
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb   # Análisis exploratorio completo
│
├── src/
│   ├── data_loader.py                  # Carga, validación y persistencia de datos
│   ├── transformations.py              # Feature engineering (retornos, volatilidad, BB)
│   ├── statistical_tests.py            # ADF, KPSS, ACF, PACF desde cero
│   ├── decomposition.py                # STL-lite: tendencia + estacionalidad + residuo
│   └── visualization.py               # 10 gráficos de calidad publicación
│
├── data/
│   ├── raw/
│   │   └── aapl_ohlcv.csv             # Datos crudos (generados automáticamente)
│   └── processed/
│       └── aapl_ohlcv.csv             # Datos validados (caché)
│
├── reports/
│   └── figures/                        # Gráficos exportados (PNG, 150 dpi)
│       ├── 01_price_overview.png
│       ├── 02_return_distribution.png
│       ├── 03_rolling_volatility.png
│       ├── 04_stationarity_tests.png
│       ├── 05_acf_pacf_correlogram.png
│       ├── 06_decomposition.png
│       ├── 07_volatility_clustering.png
│       ├── 08_monthly_return_heatmap.png
│       ├── 09_drawdown_analysis.png
│       └── 10_statistical_summary.png
│
└── requirements.txt
```

---

## Instalación

**Requisitos:** Python 3.10+

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/aapl-timeseries-analysis.git
cd aapl-timeseries-analysis

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy>=1.26
pandas>=2.2
scipy>=1.13
matplotlib>=3.9
seaborn>=0.13
jupyter>=1.0
```

> **Nota:** el proyecto no requiere `statsmodels`. Todos los tests estadísticos (ADF, KPSS, ACF, PACF) están implementados internamente.

---

## Uso rápido

```python
# Cargar datos
from src.data_loader import load_data
df = load_data()
close = df["Close"]

# Calcular retornos logarítmicos
from src.transformations import log_returns, descriptive_stats
lr = log_returns(close)
stats = descriptive_stats(lr)
print(stats)

# Test de estacionariedad
from src.statistical_tests import stationarity_diagnosis
diag = stationarity_diagnosis(lr, label="Log Returns")
print(diag["verdict"])

# Descomposición
from src.decomposition import classical_decompose
dec = classical_decompose(close, period=252, model="additive")
print(dec.summary())

# Generar todos los gráficos
from src.visualization import plot_price_overview
plot_price_overview(df, ticker="AAPL")
```

Para ejecutar el análisis completo, abrir el notebook:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

## Módulos

### `data_loader.py`

Responsable de la carga, validación y persistencia de datos OHLCV. Implementa una estrategia de caché por prioridades:

1. Caché procesada (CSV) → más rápido
2. CSV crudo → intermedio
3. Generación sintética → siempre disponible (sin red)

**Función principal:**

```python
load_data(force_regenerate: bool = False) -> pd.DataFrame
```

**Generación sintética (`generate_synthetic_aapl`):**

Los datos se generan con un modelo GBM de volatilidad estocástica inspirado en Heston con los siguientes parámetros:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `mu` | 0.0003 | Drift diario (~7.5% anualizado) |
| `kappa` | 0.05 | Velocidad de reversión a la media |
| `theta` | 0.0002 | Varianza de largo plazo |
| `xi` | 0.01 | Vol-of-vol |
| `rho` | −0.7 | Efecto leverage (correlación negativa) |
| `nu` | 5 | Grados de libertad (colas gruesas) |

Los regímenes de mercado simulados incluyen el *crash* del COVID-19 (−2.5% de drift diario entre feb–mar 2020), el rally post-pandemia (+0.3% diario hasta ago 2020) y el mercado bajista por alzas de tasas en 2022 (−0.1% diario).

**Validación automática:** el método `_validate()` verifica columnas requeridas, detecta y corrige valores nulos mediante *forward-fill*, y sanea las filas donde `High < Low` intercambiando los valores.

---

### `transformations.py`

Feature engineering de series temporales financieras. Todas las funciones son **puras** (sin efectos secundarios), diseñadas para facilitar testing unitario.

#### `log_returns(prices, col)`
Retornos continuamente compuestos: $r_t = \ln(P_t / P_{t-1})$. Preferibles a retornos simples para análisis estadístico por su aditividad temporal.

#### `simple_returns(prices)`
Retornos aritméticos: $(P_t - P_{t-1}) / P_{t-1}$.

#### `rolling_statistics(series, windows)`
Calcula media, desviación estándar, volatilidad anualizada y z-score móvil para múltiples ventanas temporales. Por defecto: 5d (semana), 21d (mes), 63d (trimestre), 252d (año).

#### `realised_volatility(log_rets, window, annualise)`
Volatilidad histórica realizada: desviación estándar móvil de retornos logarítmicos, anualizada por defecto (×√252).

#### `bollinger_bands(prices, window, n_std)`
Bandas de Bollinger: banda media ± n desviaciones estándar. Incluye el ancho normalizado de banda como indicador de régimen de volatilidad.

#### `ewma_volatility(log_rets, span)`
Volatilidad EWMA al estilo RiskMetrics: pondera más las observaciones recientes. Útil para capturar cambios rápidos de régimen.

#### `descriptive_stats(series)`
Resumen distribucional completo: media, desviación estándar, asimetría, curtosis en exceso, mínimo, máximo, VaR 5%, CVaR 5% y test de Jarque-Bera.

---

### `statistical_tests.py`

Tests estadísticos profesionales implementados desde cero con `numpy`/`scipy`.

#### Test ADF — `adf_test(series, max_lags)`

Test de Dickey-Fuller Aumentado siguiendo Said & Dickey (1984).

- **H₀:** la serie tiene raíz unitaria (no estacionaria)
- **H₁:** la serie es estacionaria
- Selección de rezagos por AIC (regla de Schwert para `max_lags` por defecto)
- p-valor aproximado mediante superficie de respuesta de MacKinnon (1994)

```python
result = adf_test(log_returns_series)
print(result.summary())
# → ADF Test Result: ✅ STATIONARY
#     Statistic  : -28.1234
#     p-value    : 0.0100
#     ...
```

#### Test KPSS — `kpss_test(series, regression)`

Test de Kwiatkowski-Phillips-Schmidt-Shin.

- **H₀:** la serie es estacionaria (o estacionaria en tendencia)
- **H₁:** la serie tiene raíz unitaria
- Estimación de varianza de largo plazo con kernel de Bartlett (Newey-West)
- Parámetro `regression='c'` para estacionariedad en nivel; `'ct'` para estacionariedad en tendencia

#### Diagnóstico combinado — `stationarity_diagnosis(series, label)`

Ejecuta ADF + KPSS y produce un veredicto conjunto siguiendo la matriz de decisión estándar:

| ADF | KPSS | Veredicto |
|-----|------|-----------|
| Rechaza H₀ | No rechaza H₀ | ✅ Estacionaria — usar directamente |
| No rechaza H₀ | Rechaza H₀ | ❌ Raíz unitaria — diferenciar |
| Rechaza H₀ | Rechaza H₀ | ⚠️ Estacionaria en tendencia — de-trending |
| No rechaza H₀ | No rechaza H₀ | ⚠️ Inconcluso — posible ruptura estructural |

#### ACF — `acf(series, n_lags, alpha)`

Función de Autocorrelación Muestral: $\rho(k) = \text{Cov}(y_t, y_{t-k}) / \text{Var}(y_t)$. Banda de confianza de Bartlett: $\pm z_{\alpha/2} / \sqrt{n}$.

#### PACF — `pacf(series, n_lags, alpha)`

Función de Autocorrelación Parcial mediante la recursión de Levinson-Durbin sobre las ecuaciones de Yule-Walker.

---

### `decomposition.py`

Descomposición clásica de series temporales en Tendencia + Estacionalidad + Residuo (STL simplificado).

#### `classical_decompose(series, period, model)`

Implementa descomposición aditiva ($y = T + S + R$) o multiplicativa ($y = T \times S \times R$) mediante:

1. **Tendencia:** media móvil centrada de orden `period` (252 para datos diarios anuales)
2. **Componente estacional:** promedio por posición dentro del período, normalizado (media 0 en aditivo, media 1 en multiplicativo)
3. **Residuo:** diferencia (o cociente) entre serie original, tendencia y estacionalidad

El objeto `DecompositionResult` expone:

- `trend_strength`: métrica de Wang et al. (2006), rango [0, 1]
- `seasonal_strength`: análoga a trend strength
- `summary()`: resumen textual de los componentes

#### `hodrick_prescott(series, lamb)`

Filtro Hodrick-Prescott para extracción de tendencia y ciclo económico, resuelto eficientemente mediante álgebra matricular dispersa (`scipy.sparse`). Para datos diarios financieros se recomiendan valores de λ entre 129 600 (convención diaria estándar) y 1 600 (convención trimestral).

---

### `visualization.py`

10 gráficos de calidad publicación, exportados automáticamente a `reports/figures/`.

| # | Función | Descripción |
|---|---------|-------------|
| 01 | `plot_price_overview` | Precio de cierre con Bandas de Bollinger, volumen y retornos diarios |
| 02 | `plot_return_distribution` | Histograma + KDE vs Normal, Q-Q plot y momentos móviles |
| 03 | `plot_rolling_volatility` | Volatilidad anualizada en ventanas de 21d, 63d, 252d y EWMA |
| 04 | `plot_stationarity` | Serie temporal, estadísticos rodantes y tarjetas de resultado ADF/KPSS |
| 05 | `plot_correlogram` | ACF y PACF con rezagos significativos resaltados |
| 06 | `plot_decomposition` | Original, tendencia, estacionalidad y residuo apilados |
| 07 | `plot_volatility_clustering` | Retornos², ACF de retornos², scatter $r_t$ vs $r_{t-1}$ |
| 08 | `plot_monthly_returns_heatmap` | Calendario de retornos mensuales (mapa de calor RdYlGn) |
| 09 | `plot_drawdown` | Precio vs máximo histórico y curva de drawdown |
| 10 | `plot_summary_dashboard` | Dashboard de estadísticos clave para reportes |

Todos los gráficos usan una paleta consistente de azul marino / ámbar / rojo sobre fondo `#f8f9fa`, sin bordes superiores ni derechos, para un aspecto limpio y profesional.

---

## Análisis del notebook

El notebook `01_exploratory_analysis.ipynb` estructura el análisis en 10 secciones:

**Sección 1 — Carga y validación de datos**
Carga 1 305 días hábiles de datos OHLCV. Verifica cero valores nulos en las cinco columnas. Precio medio de cierre de ~$197.69 USD; rango completo de $61.82 a $315.91.

**Sección 2 — Descripción y estadísticos**
Calcula el conjunto completo de estadísticos descriptivos sobre los retornos logarítmicos usando `descriptive_stats()`.

**Sección 3 — Visualización OHLCV**
Gráfico de panorama de precio, volumen y retornos diarios con Bandas de Bollinger.

**Sección 4 — Distribución de retornos**
Histograma contra distribución normal teórica, Q-Q plot y momentos estadísticos rodantes en ventana de 63 días.

**Sección 5 — Volatilidad rodante**
Evolución de la volatilidad en cuatro escalas temporales más estimación EWMA. Identifica tres regímenes claramente diferenciados.

**Sección 6 — Estacionariedad**
Diagnóstico conjunto ADF + KPSS sobre precios y retornos logarítmicos, con visualización del resultado en tarjetas de color.

**Sección 7 — Correlogramas ACF / PACF**
Análisis de autocorrelación en los retornos logarítmicos para identificar estructura ARMA candidata.

**Sección 8 — Descomposición**
Descomposición aditiva con período anual (252 días) extrayendo tendencia, estacionalidad semanal/anual y residuo.

**Sección 9 — Análisis de drawdown**
Evolución del precio frente al máximo histórico y curva de caída acumulada, con líneas de referencia en −10% y −20%.

**Sección 10 — Resumen y conclusiones**
Tabla de hallazgos con implicaciones para el modelado.

---

## Resultados clave

| Hallazgo | Valor | Implicación para el modelado |
|----------|-------|------------------------------|
| Estacionariedad de retornos | ✅ Confirmada (ADF + KPSS) | Usar retornos, no precios, como variable de entrada |
| Curtosis en exceso | 7.63 | Colas gruesas — usar *t*-Student o EVT; descartar Normal |
| Asimetría | −0.25 | Leve cola izquierda — asimetría en el riesgo de caída |
| VaR diario (5%) | −3.26% | Umbral de pérdida 1 de cada 20 días |
| CVaR diario (5%) | −5.19% | Pérdida promedio en el peor 5% de los días |
| Rezagos ACF significativos | 1, 2, 4, 6, 8 | Predictibilidad débil → considerar ARMA(4, q) |
| Efectos ARCH | Confirmados | Necesario modelo GARCH para predicción de volatilidad |
| Fuerza de tendencia | 0.87 | Serie fuertemente dominada por tendencia |
| Drawdown máximo | ~−35% | *Crash* del COVID-19 (feb–mar 2020) |

---

## Próximos pasos

1. **GARCH(1,1) / GJR-GARCH** — modelado y predicción de volatilidad condicional
2. **ARMA-GARCH conjunto** — estimación simultánea de media y varianza condicionales
3. **Hidden Markov Model (HMM)** — formalización de los tres regímenes de volatilidad identificados visualmente
4. **Simulación de Monte Carlo** — VaR de cartera bajo supuestos de colas gruesas

---

## Stack tecnológico

| Librería | Versión | Uso |
|----------|---------|-----|
| `Python` | 3.12.7 | Lenguaje base |
| `numpy` | 1.26.4 | Álgebra matricular, generación de números aleatorios |
| `pandas` | 2.2.3 | Manipulación de series temporales indexadas |
| `scipy` | ≥1.13 | Distribuciones, álgebra dispersa, regresión |
| `matplotlib` | ≥3.9 | Generación de gráficos |
| `seaborn` | ≥0.13 | Mapa de calor de retornos mensuales |
| `jupyter` | ≥1.0 | Entorno de análisis interactivo |

> **Sin dependencias de `statsmodels`** — todos los tests estadísticos (ADF, KPSS, ACF, PACF, Hodrick-Prescott) están implementados internamente usando únicamente `numpy` y `scipy`.

---

## Licencia

MIT License — libre para uso académico y comercial con atribución.