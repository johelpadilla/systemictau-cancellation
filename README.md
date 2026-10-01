# systemictau-cancellation 0.1.0

Canal de cancelación al lado de la media con signo de Tau Sistémico.

La nota [10.5281/zenodo.23073069](https://doi.org/10.5281/zenodo.23073069) parte el cambio absoluto de las concordancias por pares:

```text
D = |Δ̄| + C,    κ = C / D
```

`Δ̄` es el cambio de la media con signo: el τ_s que ya devuelve `systemictau` 4.6.1. `C` es la parte que se cancela cuando unos pares ganan concordancia y otros la pierden. Este paquete calcula las dos piezas. No importa `systemictau` y no modifica la compuerta, el reloj `T_n`, `Protocol.frozen_v1` ni `nested-recd`.

Instalación: `pip install systemictau-cancellation`.

## Uso

`X` tiene la misma forma que en `systemictau`: `(tiempo, módulos)`.

```python
from systemictau_cancellation import cancellation_channel, format_report

out = cancellation_channel(X, window=13, n_cal=19, alpha=0.05)
print(format_report(out))
print(out.mode)      # calibration, quiet, coherent, cancellation, mixed
print(out.tau_s)     # media con signo, tau-b
print(out.D, out.C, out.kappa)
```

Las primeras `n_cal` ventanas son la línea base. El paso por defecto es igual a la ventana, así que los bloques no se solapan: esa es la disposición en la que la nota calibra el valor p. Con `alpha=0.05` hacen falta al menos 19 ventanas de calibración; si no, el valor p mínimo no alcanza 0.05.

| Modo | Qué pasó en esa ventana |
|---|---|
| `coherent` | Se movió la media con signo |
| `cancellation` | Se movió el cambio absoluto y la media con signo se quedó |
| `mixed` | Las dos alarmas se encienden |
| `quiet` | Ninguna alarma |
| `calibration` | Ventana usada como línea base |

Un desplazamiento grande y puro enciende las dos alarmas, porque ahí `D = |Δ̄|`. El modo queda `mixed`. La cancelación es el caso en el que `D` se enciende y la media no.

## Cómo leer el modo

`cancellation` dice que `D` salió de la calibración y la media con signo no. En la nota, ese patrón es el recableado balanceado. En un ciclo corto cuyos estados vecinos están más juntos que el ruido, el mismo patrón puede ser el ruido volteando rangos mientras el orden medio se sostiene. El modo no nombra la causa.

El valor p es la calibración de la nota cuando los bloques son intercambiables y no se solapan. En una órbita continua las ventanas siguen siendo dependientes aunque el paso sea igual a la ventana, y el valor p queda descriptivo. Si el paso es menor que la ventana, `windows_overlap` es verdadero y el informe lo dice.

## Qué τ se usa

Cada ventana usa tau-b, el coseno de los vectores de signos, como la nota. El camino Numba de `systemictau` 4.6.1 usa tau-a. Sin empates los dos coinciden. `signed_tau_series(X, window=13)` devuelve la media con signo en el mismo índice temporal que `compute_taus`.

## Reproducir las ilustraciones

```bash
python -m pytest -q
python examples/demo_modes.py
```

La figura queda en `examples/modes_demo.png`. Esas series son un ejemplo corto. La Tabla 1 de la nota (semilla 20260930, 1000 ventanas) sigue viviendo en `cancellation-aware-tau` y no se regenera aquí.

## Licencia

MIT. Johel Padilla-Villanueva, ORCID 0000-0002-5797-6931.
