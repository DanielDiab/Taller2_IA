# Taller 2 — CSP y Búsqueda Adversaria

En este archivo puede encontrar comandos de ejecución de todos los layouts con los distintos algoritmos que usted debe implementar. No es necesario que pruebe todas las combinaciones: eliga aquellas que le parezcan interesantes y que le permitan realizar un buen análisis de sus algoritmos. Recuerde que también puede proponer nuevos layouts o editar los existentes. Para los problemas probabilísticos, puede escoger el valor de `p` que le parezca más relevante; se muestran ejemplos con cazadores greedy (p=0), estocásticos (p=1) y mixtos (p=0.3).

El desempeño en el modo adversarial está fuertemente influenciado por la definición de la función de evaluación. Algunas funciones tienen mejor desempeño en ciertos layouts que en otros, por lo que es importante probar varias combinaciones para obtener una visión completa de las fortalezas y debilidades de su función de evaluación. No se preocupe si su función de evaluación no obtiene buenos resultados en todos los layouts: lo importante es que pueda analizar y explicar por qué se desempeña mejor o peor en cada caso, y que pueda identificar oportunidades de mejora.

## CSP — Todas las combinaciones

### Layouts CSP disponibles (22 layouts)

| Layout | Descripción |
|--------|-------------|
| `twin_bases` | Dos bases separadas por muros, una entrega en cada extremo |
| `triple_fleet` | Tres drones en paralelo, corredores separados por muros |
| `jungle_outpost` | Selva con niebla, múltiples bases y entregas |
| `tight_battery` | Batería justa con zonas de niebla que aumentan el costo |
| `heavy_cargo` | Cargas pesadas, dos drones con capacidades distintas |
| `storm_corridor` | Corredor de tormenta eléctrica, ruta de desvío obligatoria |
| `round_trip` | Una entrega con ventana temprana y otra tardía (espera forzada) |
| `tight_windows` | Ventanas de tiempo muy ajustadas que fuerzan asignaciones específicas |
| `big_fleet` | Cuatro drones, cinco entregas, mapa grande con terreno variado |
| `battery_edge` | Batería = costo exacto del viaje de ida y vuelta (límite exacto) |
| `detour_required` | Tormenta bloquea ruta directa; Dijkstra obliga un desvío más barato |
| `hero_drone` | Un dron con batería mínima, otro asume la mayoría de entregas |
| `weight_distribution` | Distribución de pesos fuerza asignación específica por capacidad |
| `fog_valley` | Valle de niebla: rutas caras que limitan la batería útil |
| `mountain_bypass` | Montañas bloquean ruta directa; desvío más largo pero necesario |
| `staggered_time` | Ventanas no superpuestas requieren drones distintos para entregas simultáneas |
| `maze_csp` | Laberinto complejo con muros; rutas largas y costosas |
| `cross_mission` | Entregas en esquinas opuestas; drones se cruzan en el mapa |
| `tight_overload` | Capacidad exacta: ningún dron puede cargar más de lo asignado |
| `single_path` | Pasillo único; entregas distribuidas en el corredor |
| `corner_challenge` | Zona de niebla en el centro; drones deben rodear el mapa |
| `storm_shortcut` | Dron de batería baja sólo alcanza la entrega cercana; otro cubre la lejana |

### CSP — `backtracking`

```bash
python main.py -m csp -a backtracking -l twin_bases
python main.py -m csp -a backtracking -l triple_fleet
python main.py -m csp -a backtracking -l jungle_outpost
python main.py -m csp -a backtracking -l tight_battery
python main.py -m csp -a backtracking -l heavy_cargo
python main.py -m csp -a backtracking -l storm_corridor
python main.py -m csp -a backtracking -l round_trip
python main.py -m csp -a backtracking -l tight_windows
python main.py -m csp -a backtracking -l big_fleet
python main.py -m csp -a backtracking -l battery_edge
python main.py -m csp -a backtracking -l detour_required
python main.py -m csp -a backtracking -l hero_drone
python main.py -m csp -a backtracking -l weight_distribution
python main.py -m csp -a backtracking -l fog_valley
python main.py -m csp -a backtracking -l mountain_bypass
python main.py -m csp -a backtracking -l staggered_time
python main.py -m csp -a backtracking -l maze_csp
python main.py -m csp -a backtracking -l cross_mission
python main.py -m csp -a backtracking -l tight_overload
python main.py -m csp -a backtracking -l single_path
python main.py -m csp -a backtracking -l corner_challenge
python main.py -m csp -a backtracking -l storm_shortcut
```

### CSP — `backtracking_fc`

```bash
python main.py -m csp -a backtracking_fc -l twin_bases
python main.py -m csp -a backtracking_fc -l triple_fleet
python main.py -m csp -a backtracking_fc -l jungle_outpost
python main.py -m csp -a backtracking_fc -l tight_battery
python main.py -m csp -a backtracking_fc -l heavy_cargo
python main.py -m csp -a backtracking_fc -l storm_corridor
python main.py -m csp -a backtracking_fc -l round_trip
python main.py -m csp -a backtracking_fc -l tight_windows
python main.py -m csp -a backtracking_fc -l big_fleet
python main.py -m csp -a backtracking_fc -l battery_edge
python main.py -m csp -a backtracking_fc -l detour_required
python main.py -m csp -a backtracking_fc -l hero_drone
python main.py -m csp -a backtracking_fc -l weight_distribution
python main.py -m csp -a backtracking_fc -l fog_valley
python main.py -m csp -a backtracking_fc -l mountain_bypass
python main.py -m csp -a backtracking_fc -l staggered_time
python main.py -m csp -a backtracking_fc -l maze_csp
python main.py -m csp -a backtracking_fc -l cross_mission
python main.py -m csp -a backtracking_fc -l tight_overload
python main.py -m csp -a backtracking_fc -l single_path
python main.py -m csp -a backtracking_fc -l corner_challenge
python main.py -m csp -a backtracking_fc -l storm_shortcut
```

### CSP — `backtracking_ac3`

```bash
python main.py -m csp -a backtracking_ac3 -l twin_bases
python main.py -m csp -a backtracking_ac3 -l triple_fleet
python main.py -m csp -a backtracking_ac3 -l jungle_outpost
python main.py -m csp -a backtracking_ac3 -l tight_battery
python main.py -m csp -a backtracking_ac3 -l heavy_cargo
python main.py -m csp -a backtracking_ac3 -l storm_corridor
python main.py -m csp -a backtracking_ac3 -l round_trip
python main.py -m csp -a backtracking_ac3 -l tight_windows
python main.py -m csp -a backtracking_ac3 -l big_fleet
python main.py -m csp -a backtracking_ac3 -l battery_edge
python main.py -m csp -a backtracking_ac3 -l detour_required
python main.py -m csp -a backtracking_ac3 -l hero_drone
python main.py -m csp -a backtracking_ac3 -l weight_distribution
python main.py -m csp -a backtracking_ac3 -l fog_valley
python main.py -m csp -a backtracking_ac3 -l mountain_bypass
python main.py -m csp -a backtracking_ac3 -l staggered_time
python main.py -m csp -a backtracking_ac3 -l maze_csp
python main.py -m csp -a backtracking_ac3 -l cross_mission
python main.py -m csp -a backtracking_ac3 -l tight_overload
python main.py -m csp -a backtracking_ac3 -l single_path
python main.py -m csp -a backtracking_ac3 -l corner_challenge
python main.py -m csp -a backtracking_ac3 -l storm_shortcut
```

### CSP — `backtracking_mrv_lcv` (Bono)

```bash
python main.py -m csp -a backtracking_mrv_lcv -l twin_bases
python main.py -m csp -a backtracking_mrv_lcv -l triple_fleet
python main.py -m csp -a backtracking_mrv_lcv -l jungle_outpost
python main.py -m csp -a backtracking_mrv_lcv -l tight_battery
python main.py -m csp -a backtracking_mrv_lcv -l heavy_cargo
python main.py -m csp -a backtracking_mrv_lcv -l storm_corridor
python main.py -m csp -a backtracking_mrv_lcv -l round_trip
python main.py -m csp -a backtracking_mrv_lcv -l tight_windows
python main.py -m csp -a backtracking_mrv_lcv -l big_fleet
python main.py -m csp -a backtracking_mrv_lcv -l battery_edge
python main.py -m csp -a backtracking_mrv_lcv -l detour_required
python main.py -m csp -a backtracking_mrv_lcv -l hero_drone
python main.py -m csp -a backtracking_mrv_lcv -l weight_distribution
python main.py -m csp -a backtracking_mrv_lcv -l fog_valley
python main.py -m csp -a backtracking_mrv_lcv -l mountain_bypass
python main.py -m csp -a backtracking_mrv_lcv -l staggered_time
python main.py -m csp -a backtracking_mrv_lcv -l maze_csp
python main.py -m csp -a backtracking_mrv_lcv -l cross_mission
python main.py -m csp -a backtracking_mrv_lcv -l tight_overload
python main.py -m csp -a backtracking_mrv_lcv -l single_path
python main.py -m csp -a backtracking_mrv_lcv -l corner_challenge
python main.py -m csp -a backtracking_mrv_lcv -l storm_shortcut
```

## Adversarial — Todas las combinaciones

### Layouts adversariales disponibles (24 layouts)

| Layout | Descripción |
|--------|-------------|
| `tiny_hunt` | Mapa mínimo 5×5, un cazador, una entrega |
| `small_hunt` | Mapa 7×7, un cazador, dos entregas |
| `open_field` | Campo abierto con muros simétricos |
| `drone_escape` | Corredor con muros, dron debe escapar hacia las entregas |
| `supply_run` | Terreno mixto (niebla, montaña), un cazador |
| `swamp_run` | Pantano denso de niebla, dificultad para los cazadores |
| `jungle_chase` | Selva con pasillos, dos cazadores en posiciones opuestas |
| `contested_territory` | Territorio disputado, dos cazadores cercanos al inicio |
| `maze_hunt` | Laberinto con muros, dos cazadores |
| `mountain_pass` | Paso montañoso, dos cazadores; montañas restringen movilidad |
| `wide_open` | Gran espacio abierto, cadena montañosa separa al cazador del dron |
| `bottleneck` | Cuello de botella con muros densos |
| `gauntlet` | Corredor estrecho con dos cazadores en línea |
| `triple_threat` | Dos cazadores dispersos, laberinto de muros |
| `scattered_hunters` | Dos cazadores dispersos, terreno variado (niebla, montaña) |
| `terrain_trap` | Trampa de terreno: montaña y niebla separando zonas |
| `storm_tunnel` | Túnel de tormenta eléctrica que sólo el dron puede usar |
| `arena` | Arena con muro de montañas central, dos cazadores |
| `hunter_cage` | Cazadores en zona de niebla (ventaja para el dron) |
| `tunnel_escape` | Corredor de montañas inatravesable por cazadores; dron sobrevuela libremente |
| `mountain_overfly` | Cadena montañosa divide el mapa; dron sobrevuela en línea recta |
| `hunter_swarm` | Tres cazadores agrupados; montañas separan al dron de los cazadores |
| `pacman_maze` | Laberinto denso estilo Pac-Man; pasillos estrechos |
| `desperate_run` | Montañas separan las zonas; dos cazadores en lados opuestos |

### MinimaxAgent — p=0 (cazador greedy (determinístico))

```bash
python main.py -m adversarial -a MinimaxAgent -l tiny_hunt
python main.py -m adversarial -a MinimaxAgent -l small_hunt
python main.py -m adversarial -a MinimaxAgent -l open_field
python main.py -m adversarial -a MinimaxAgent -l drone_escape
python main.py -m adversarial -a MinimaxAgent -l supply_run
python main.py -m adversarial -a MinimaxAgent -l swamp_run
python main.py -m adversarial -a MinimaxAgent -l jungle_chase
python main.py -m adversarial -a MinimaxAgent -l contested_territory
python main.py -m adversarial -a MinimaxAgent -l maze_hunt
python main.py -m adversarial -a MinimaxAgent -l mountain_pass
python main.py -m adversarial -a MinimaxAgent -l wide_open
python main.py -m adversarial -a MinimaxAgent -l bottleneck
python main.py -m adversarial -a MinimaxAgent -l gauntlet
python main.py -m adversarial -a MinimaxAgent -l triple_threat
python main.py -m adversarial -a MinimaxAgent -l scattered_hunters
python main.py -m adversarial -a MinimaxAgent -l terrain_trap
python main.py -m adversarial -a MinimaxAgent -l storm_tunnel
python main.py -m adversarial -a MinimaxAgent -l arena
python main.py -m adversarial -a MinimaxAgent -l hunter_cage
python main.py -m adversarial -a MinimaxAgent -l tunnel_escape
python main.py -m adversarial -a MinimaxAgent -l mountain_overfly
python main.py -m adversarial -a MinimaxAgent -l hunter_swarm
python main.py -m adversarial -a MinimaxAgent -l pacman_maze
python main.py -m adversarial -a MinimaxAgent -l desperate_run
```

### MinimaxAgent — p=0.3 (cazador mixto 30% aleatorio)

```bash
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l tiny_hunt
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l small_hunt
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l open_field
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l drone_escape
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l supply_run
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l swamp_run
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l jungle_chase
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l contested_territory
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l maze_hunt
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l mountain_pass
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l wide_open
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l bottleneck
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l gauntlet
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l triple_threat
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l scattered_hunters
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l terrain_trap
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l storm_tunnel
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l arena
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l hunter_cage
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l tunnel_escape
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l mountain_overfly
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l hunter_swarm
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l pacman_maze
python main.py -m adversarial -a MinimaxAgent -p 0.3 -l desperate_run
```

### MinimaxAgent — p=1.0 (cazador completamente aleatorio)

```bash
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l tiny_hunt
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l small_hunt
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l open_field
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l drone_escape
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l supply_run
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l swamp_run
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l jungle_chase
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l contested_territory
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l maze_hunt
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l mountain_pass
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l wide_open
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l bottleneck
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l gauntlet
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l triple_threat
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l scattered_hunters
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l terrain_trap
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l storm_tunnel
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l arena
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l hunter_cage
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l tunnel_escape
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l mountain_overfly
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l hunter_swarm
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l pacman_maze
python main.py -m adversarial -a MinimaxAgent -p 1.0 -l desperate_run
```

### AlphaBetaAgent — p=0 (cazador greedy (determinístico))

```bash
python main.py -m adversarial -a AlphaBetaAgent -l tiny_hunt
python main.py -m adversarial -a AlphaBetaAgent -l small_hunt
python main.py -m adversarial -a AlphaBetaAgent -l open_field
python main.py -m adversarial -a AlphaBetaAgent -l drone_escape
python main.py -m adversarial -a AlphaBetaAgent -l supply_run
python main.py -m adversarial -a AlphaBetaAgent -l swamp_run
python main.py -m adversarial -a AlphaBetaAgent -l jungle_chase
python main.py -m adversarial -a AlphaBetaAgent -l contested_territory
python main.py -m adversarial -a AlphaBetaAgent -l maze_hunt
python main.py -m adversarial -a AlphaBetaAgent -l mountain_pass
python main.py -m adversarial -a AlphaBetaAgent -l wide_open
python main.py -m adversarial -a AlphaBetaAgent -l bottleneck
python main.py -m adversarial -a AlphaBetaAgent -l gauntlet
python main.py -m adversarial -a AlphaBetaAgent -l triple_threat
python main.py -m adversarial -a AlphaBetaAgent -l scattered_hunters
python main.py -m adversarial -a AlphaBetaAgent -l terrain_trap
python main.py -m adversarial -a AlphaBetaAgent -l storm_tunnel
python main.py -m adversarial -a AlphaBetaAgent -l arena
python main.py -m adversarial -a AlphaBetaAgent -l hunter_cage
python main.py -m adversarial -a AlphaBetaAgent -l tunnel_escape
python main.py -m adversarial -a AlphaBetaAgent -l mountain_overfly
python main.py -m adversarial -a AlphaBetaAgent -l hunter_swarm
python main.py -m adversarial -a AlphaBetaAgent -l pacman_maze
python main.py -m adversarial -a AlphaBetaAgent -l desperate_run
```

### AlphaBetaAgent — p=0.3 (cazador mixto 30% aleatorio)

```bash
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l tiny_hunt
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l small_hunt
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l open_field
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l drone_escape
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l supply_run
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l swamp_run
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l jungle_chase
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l contested_territory
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l maze_hunt
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l mountain_pass
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l wide_open
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l bottleneck
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l gauntlet
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l triple_threat
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l scattered_hunters
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l terrain_trap
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l storm_tunnel
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l arena
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l hunter_cage
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l tunnel_escape
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l mountain_overfly
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l hunter_swarm
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l pacman_maze
python main.py -m adversarial -a AlphaBetaAgent -p 0.3 -l desperate_run
```

### AlphaBetaAgent — p=1.0 (cazador completamente aleatorio)

```bash
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l tiny_hunt
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l small_hunt
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l open_field
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l drone_escape
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l supply_run
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l swamp_run
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l jungle_chase
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l contested_territory
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l maze_hunt
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l mountain_pass
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l wide_open
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l bottleneck
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l gauntlet
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l triple_threat
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l scattered_hunters
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l terrain_trap
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l storm_tunnel
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l arena
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l hunter_cage
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l tunnel_escape
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l mountain_overfly
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l hunter_swarm
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l pacman_maze
python main.py -m adversarial -a AlphaBetaAgent -p 1.0 -l desperate_run
```

### ExpectimaxAgent — p=0 (cazador greedy)

```bash
python main.py -m adversarial -a ExpectimaxAgent -l tiny_hunt
python main.py -m adversarial -a ExpectimaxAgent -l small_hunt
python main.py -m adversarial -a ExpectimaxAgent -l open_field
python main.py -m adversarial -a ExpectimaxAgent -l drone_escape
python main.py -m adversarial -a ExpectimaxAgent -l supply_run
python main.py -m adversarial -a ExpectimaxAgent -l swamp_run
python main.py -m adversarial -a ExpectimaxAgent -l jungle_chase
python main.py -m adversarial -a ExpectimaxAgent -l contested_territory
python main.py -m adversarial -a ExpectimaxAgent -l maze_hunt
python main.py -m adversarial -a ExpectimaxAgent -l mountain_pass
python main.py -m adversarial -a ExpectimaxAgent -l wide_open
python main.py -m adversarial -a ExpectimaxAgent -l bottleneck
python main.py -m adversarial -a ExpectimaxAgent -l gauntlet
python main.py -m adversarial -a ExpectimaxAgent -l triple_threat
python main.py -m adversarial -a ExpectimaxAgent -l scattered_hunters
python main.py -m adversarial -a ExpectimaxAgent -l terrain_trap
python main.py -m adversarial -a ExpectimaxAgent -l storm_tunnel
python main.py -m adversarial -a ExpectimaxAgent -l arena
python main.py -m adversarial -a ExpectimaxAgent -l hunter_cage
python main.py -m adversarial -a ExpectimaxAgent -l tunnel_escape
python main.py -m adversarial -a ExpectimaxAgent -l mountain_overfly
python main.py -m adversarial -a ExpectimaxAgent -l hunter_swarm
python main.py -m adversarial -a ExpectimaxAgent -l pacman_maze
python main.py -m adversarial -a ExpectimaxAgent -l desperate_run
```

### ExpectimaxAgent — p=0.3 (cazador mixto 30% aleatorio)

```bash
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l tiny_hunt
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l small_hunt
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l open_field
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l drone_escape
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l supply_run
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l swamp_run
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l jungle_chase
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l contested_territory
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l maze_hunt
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l mountain_pass
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l wide_open
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l bottleneck
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l gauntlet
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l triple_threat
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l scattered_hunters
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l terrain_trap
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l storm_tunnel
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l arena
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l hunter_cage
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l tunnel_escape
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l mountain_overfly
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l hunter_swarm
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l pacman_maze
python main.py -m adversarial -a ExpectimaxAgent -p 0.3 -l desperate_run
```

### ExpectimaxAgent — p=1.0 (cazador completamente aleatorio)

```bash
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l tiny_hunt
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l small_hunt
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l open_field
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l drone_escape
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l supply_run
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l swamp_run
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l jungle_chase
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l contested_territory
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l maze_hunt
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l mountain_pass
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l wide_open
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l bottleneck
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l gauntlet
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l triple_threat
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l scattered_hunters
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l terrain_trap
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l storm_tunnel
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l arena
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l hunter_cage
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l tunnel_escape
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l mountain_overfly
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l hunter_swarm
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l pacman_maze
python main.py -m adversarial -a ExpectimaxAgent -p 1.0 -l desperate_run
```
