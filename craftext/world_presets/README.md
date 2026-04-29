# World Presets

Каждый YAML в этой папке описывает world preset для `craftext_play.py`.

Минимальный пример:

```yaml
preset: box
env_name: Craftax-Classic-Pixels-v1
map_size: 64
seed: 123
box_inner_size: 3
perimeter_tree_prob: 0.8
blocked_block: out_of_bounds
floor_block: grass
perimeter_block: tree
starting_inventory:
  wood:
    value: 3
    probability: 1.0
  sapling:
    value: 1
    probability: 0.5
starting_intrinsics:
  player_health:
    value: 9
    probability: 1.0
  player_energy:
    value: 6
    probability: 1.0
intrinsic_rates:
  player_hunger_rate:
    value: 0.0
    probability: 1.0
  player_thirst_rate:
    value: 0.0
    probability: 1.0
intrinsic_thresholds:
  player_hunger_threshold:
    value: 5.0
    probability: 1.0
  player_thirst_threshold:
    value: 5.0
    probability: 1.0
recovery_rules:
  instant_sleep_recovery:
    value: true
    probability: 1.0
  sleep_energy_value:
    value: 9
    probability: 1.0
  sleep_fatigue_value:
    value: 0.0
    probability: 1.0
  wake_after_sleep_recovery:
    value: true
    probability: 1.0
map_rules:
  solid_out_of_bounds:
    value: true
    probability: 1.0
  solid_blocks:
    value: [tree]
    probability: 1.0
```

Поддерживаемые поля:

- `preset`: `default`, `random`, `fixed`, `ring_random`, `ring_fixed`, `ring`, `box`, `box3_random_trees`
- `env_name` или `base_environment`
- `seed`
- `map_size`
- `blocked_block`
- `floor_block`
- `perimeter_block`
- `ring_inner_radius`
- `ring_outer_radius`
- `box_inner_size`
- `perimeter_tree_prob`
- `starting_inventory`
- `starting_intrinsics`
- `intrinsic_rates`
- `intrinsic_thresholds`
- `recovery_rules`
- `map_rules`
- `extends`: имя другого YAML preset без `.yaml`

Замечания:

- `env_name` в YAML не переключает активный env, а только валидирует совместимость. Источник истины для env остаётся `--env` или `base_environment` из scenario config.
- preset применяется как overlay к `state` внутри `reset`, после чего обычный `Craftax.step(...)` продолжает игру уже из модифицированного состояния.
- если `map_size` не указан, используется обычный `make_craftax_env_from_name(...)`, как в `verl-agent-craftext`.

Пример наследования:

```yaml
extends: box3_random_trees
seed: 7
perimeter_tree_prob: 1.0
```

Пример кольца:

```yaml
preset: ring
ring_inner_radius: 2
ring_outer_radius: 8
blocked_block: water
floor_block: grass
```

Пример стартового инвентаря:

```yaml
starting_inventory:
  wood:
    value: 5
    probability: 1.0
  stone:
    value: 2
    probability: 0.25
  armour:
    value: [1, 0, 0, 0]
    probability: 1.0
```

Замечания по `starting_inventory`:

- для обычных scalar-полей (`wood`, `stone`, `sapling`, `sword`, `torches`) используй число
- для array-полей (`armour`, `potions`) используй список нужной формы
- `probability` применяется отдельно к каждому полю на каждом `reset`

Замечания по `starting_intrinsics`, `intrinsic_rates` и `intrinsic_thresholds`:

- `starting_intrinsics` меняет стартовые значения полей состояния игрока, например `player_health`, `player_food`, `player_drink`, `player_energy`, `player_mana`
- `intrinsic_rates` задаёт нашу дополнительную post-step динамику, независимую от версии `craftax`
- основные имена:
  - `player_hunger_rate`
  - `player_thirst_rate`
  - `player_fatigue_rate`
  - `player_fatigue_sleep_rate`
  - `player_recover_rate`
  - `player_recover_penalty_rate`
- `intrinsic_thresholds` задаёт пороги срабатывания этой динамики:
  - `player_hunger_threshold`
  - `player_thirst_threshold`
  - `player_fatigue_threshold`
  - `player_fatigue_recover_threshold`
  - `player_recover_positive_threshold`
  - `player_recover_negative_threshold`
- оба блока работают через тот же формат `{ value, probability }`

Замечания по `recovery_rules`:

- это отдельный post-step adapter, не зависящий от внутренней логики `craftax`
- полезные ключи:
  - `instant_sleep_recovery`
  - `sleep_energy_value`
  - `sleep_fatigue_value`
  - `sleep_mana_value`
  - `sleep_health_value`
  - `wake_after_sleep_recovery`
  - `instant_rest_recovery`
  - `rest_energy_value`
  - `rest_fatigue_value`
  - `stop_rest_after_recovery`
- если `instant_sleep_recovery: true`, то при сне можно сразу восстановить энергию до нужного значения за один шаг

Замечания по `map_rules`:

- это отдельный adapter для коллизий по карте, без патча `craftax`
- полезные ключи:
  - `solid_out_of_bounds`
  - `solid_blocks`
- `solid_blocks` можно задавать строкой или списком имён блоков, например `tree`, `water`, `wall`

Запуск:

```bash
python craftext/environment/craftext_play.py --world-preset tiny_box_clear
```
