# Pokemon Prism Memory Notes

Estado actual del trabajo de memoria para `Prism`.

## Confirmado

- La ROM `PM_PRISM` arranca en `PyBoy`.
- El entorno `PrismGymEnv` puede hacer `reset()` y `step()`.
- Hay actividad estable en la zona `0xD35D-0xD362`, que hoy estamos usando como candidatos de mapa/coordenadas.

## Candidatos observados

- `0xD35E`: candidato fuerte para `map id`
- `0xD361`: candidato fuerte para `y`
- `0xD362`: candidato fuerte para `x`
- `0xD356`: byte interesante cerca del bloque de coordenadas; pendiente de validar como badge/event/status
- `0xDCB5-0xDCB8`: candidatos fuertes estilo Gen2 para `mapGroup`, `mapNumber`, `y`, `x`

## Layout confirmado desde el código fuente de Prism

- `0xDCD7`: número de Pokémon en el equipo
- `0xDCDF`: inicio de la primera estructura de Pokémon; cada estructura ocupa
  `0x30` bytes
- niveles: `0xDCFE`, `0xDD2E`, `0xDD5E`, `0xDD8E`, `0xDDBE`, `0xDDEE`
- HP actual: `0xDD01`, `0xDD31`, `0xDD61`, `0xDD91`, `0xDDC1`, `0xDDF1`
- HP máximo: `0xDD03`, `0xDD33`, `0xDD63`, `0xDD93`, `0xDDC3`, `0xDDF3`
- `0xDE99-0xDEB8`: Pokédex capturados
- `0xDEB9-0xDED8`: Pokédex vistos
- `0xDED9`: medallas de Naljo
- `0xDEDA`: medallas de Rijon
- `0xDEDB`: otras medallas

Estas direcciones están centralizadas en `prism_memory.py`. El entorno expone
los contadores de Pokédex en la observación y en TensorBoard, y los incorpora a
la recompensa de progreso.

Nuevo checkpoint útil:

- `map_ready_adam.state`: estado ya en overworld
- desde ese state se observa `0xDCB5=1`, `0xDCB6=1`, `0xDCB7=6`, `0xDCB8=10`
- esto refuerza que Prism usa layout de memoria tipo Crystal/PolishedCrystal para coordenadas

Primer diff útil:

- snapshot base desde `prism_init.state`: `0xD35D`, `0xD35E`, `0xD361`, `0xD362`, `0xD356` en `0`
- tras `a,wait:120`: esos bytes pasan a valores no nulos (`219`, `39`, `217`, `39`, `39`)

Eso sugiere que el `init state` todavía está muy cerca del arranque, pero también que un único input ya nos mete en una fase donde el bloque de coordenadas empieza a poblarse.

## Limitación actual

El `init state` generado automáticamente todavía cae muy temprano en el arranque del juego, así que aún no hay validación fuerte de:

- Pokédex seen/caught
- número de Pokémon en party
- badges reales
- event flags de progresión

## Scripts útiles

Capturar snapshot de memoria:

```bash
cd v2
python generate_prism_init_state.py --rom ../PokemonPrism.gbc
python prism_memory_scan.py --rom ../PokemonPrism.gbc --state ../prism_init.state --label base
python prism_memory_scan.py --rom ../PokemonPrism.gbc --state ../prism_init.state --label after_a --script a,wait:120
```

Comparar snapshots:

```bash
cd v2
python prism_memory_diff.py ../memory/base.npz ../memory/after_a.npz --output ../memory/base_vs_after_a.json
```

Generar estados intermedios reproducibles del onboarding:

```bash
cd v2
python prism_bootstrap.py --rom ../PokemonPrism.gbc --preset title
python prism_bootstrap.py --rom ../PokemonPrism.gbc --preset new_game_menu
python prism_bootstrap.py --rom ../PokemonPrism.gbc --preset calendar
python prism_bootstrap.py --rom ../PokemonPrism.gbc --preset intro_text
```

Esto crea `.state`, `.png` y `.json` para cada checkpoint bajo `bootstrap_states/`.

Checkpoints útiles ya verificados:

- `title`: pantalla de título de Prism
- `new_game_menu`: menú `Nueva Partida / Ajustes`
- `calendar`: selección de fecha/hora
- `intro_text`: confirmación `¿Es correcto?`
- `name_selection`: lista de nombres prefijados
- `name_adam`: intro ya continuada con el nombre `Adam`
- `map_ready_adam`: primer estado confirmado en overworld, sin diálogo abierto,
  tras elegir `Adam`

El entrenamiento usa `bootstrap_states/map_ready_adam.state` por defecto. Esto
evita gastar episodios en el título y el onboarding. Se puede seleccionar otro
checkpoint mediante `PRISM_INIT_STATE`; la ROM también se puede seleccionar con
`PRISM_ROM`.

## Siguiente objetivo

Crear o capturar un `state` ya dentro de gameplay o menú para validar bytes de:

- badges
- party count
- Pokédex seen/caught
- event flags de progreso
