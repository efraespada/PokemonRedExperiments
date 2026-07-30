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
- `map_ready_adam`: primer estado confirmado en overworld tras elegir `Adam`

## Siguiente objetivo

Crear o capturar un `state` ya dentro de gameplay o menú para validar bytes de:

- badges
- party count
- Pokédex seen/caught
- event flags de progreso
