# API Endpoint Specification

## Base URL

```
https://api.ainm.no/astar-island
```

All endpoints require authentication. The API accepts either:

- **Cookie:** `access_token` JWT cookie (set automatically when you log in at app.ainm.no)
- **Bearer token:** `Authorization: Bearer <token>` header

Both methods use the same JWT token. Use whichever is more convenient for your setup.

## Endpoints Overview

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/astar-island/rounds` | Public | List all rounds |
| `GET` | `/astar-island/rounds/{round_id}` | Public | Round details + initial states |
| `GET` | `/astar-island/budget` | Team | Query budget for active round |
| `POST` | `/astar-island/simulate` | Team | Observe one simulation through viewport |
| `POST` | `/astar-island/submit` | Team | Submit prediction tensor |
| `GET` | `/astar-island/my-rounds` | Team | Rounds with your scores, rank, budget |
| `GET` | `/astar-island/my-predictions/{round_id}` | Team | Your predictions with argmax/confidence |
| `GET` | `/astar-island/analysis/{round_id}/{seed_index}` | Team | Post-round ground truth comparison |
| `GET` | `/astar-island/leaderboard` | Public | Astar Island leaderboard |

## GET /astar-island/rounds

List all rounds with status and timing.

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "event_date": "2026-03-19",
    "status": "active",
    "map_width": 40,
    "map_height": 40,
    "prediction_window_minutes": 165,
    "started_at": "2026-03-19T10:00:00Z",
    "closes_at": "2026-03-19T10:45:00Z",
    "round_weight": 1,
    "created_at": "2026-03-19T09:00:00Z"
  }
]
```

### Round Status

| Status | Meaning |
|--------|---------|
| `pending` | Round created but not yet started |
| `active` | Queries and submissions open |
| `scoring` | Submissions closed, scoring in progress |
| `completed` | Scores finalized |

## GET /astar-island/rounds/{round_id}

Returns round details including **initial map states** for all seeds. Use this to reconstruct the starting terrain locally.

**Note:** Settlement data in initial states shows only position and port status. Internal stats (population, food, wealth, defense) are not exposed.

```json
{
  "id": "uuid",
  "round_number": 1,
  "status": "active",
  "map_width": 40,
  "map_height": 40,
  "seeds_count": 5,
  "initial_states": [
    {
      "grid": [[10, 10, 10, ...], ...],
      "settlements": [
        {
          "x": 5, "y": 12,
          "has_port": true,
          "alive": true
        }
      ]
    }
  ]
}
```

### Grid Cell Values

| Value | Terrain |
|-------|---------|
| 0 | Empty |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |
| 10 | Ocean |
| 11 | Plains |

## GET /astar-island/budget

Check your team's remaining query budget for the active round.

```json
{
  "round_id": "uuid",
  "queries_used": 23,
  "queries_max": 50,
  "active": true
}
```

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /simulate` | 5 requests/second per team |
| `POST /submit` | 2 requests/second per team |

Exceeding these limits returns `429 Too Many Requests`.

## POST /astar-island/simulate

**This is the core observation endpoint.** Each call runs one stochastic simulation and reveals a viewport window of the result. Costs one query from your budget (50 per round).

![Viewport: full map with highlighted window vs. what gets returned](/docs/astar-island/viewport-demo.png)

### Request

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

| Field | Type | Description |
|-------|------|-------------|
| `round_id` | string | UUID of the active round |
| `seed_index` | int (0–4) | Which of the 5 seeds to simulate |
| `viewport_x` | int (>=0) | Left edge of viewport (default 0) |
| `viewport_y` | int (>=0) | Top edge of viewport (default 0) |
| `viewport_w` | int (5–15) | Viewport width (default 15) |
| `viewport_h` | int (5–15) | Viewport height (default 15) |

### Response

```json
{
  "grid": [[4, 11, 1, ...], ...],
  "settlements": [
    {
      "x": 12, "y": 7,
      "population": 2.8,
      "food": 0.4,
      "wealth": 0.7,
      "defense": 0.6,
      "has_port": true,
      "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

The `grid` contains only the viewport region (viewport_h × viewport_w), not the full map. The `settlements` list includes only settlements within the viewport. The `viewport` object confirms the actual viewport bounds (clamped to map edges). `width` and `height` give the full map dimensions.

Each call uses a different random sim_seed, so you get a different stochastic outcome.

### Error Codes

| Status | Meaning |
|--------|---------|
| 400 | Round not active, or invalid seed_index |
| 403 | Not on a team |
| 404 | Round not found |
| 429 | Query budget exhausted (50/50) or rate limit exceeded (max 5 req/sec) |

## POST /astar-island/submit

Submit your prediction for one seed. You must submit all 5 seeds for a complete score.

### Request

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "prediction": [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      [0.10, 0.40, 0.30, 0.10, 0.05, 0.05],
      ...
    ],
    ...
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `round_id` | string | UUID of the active round |
| `seed_index` | int (0–4) | Which seed this prediction is for |
| `prediction` | float[][][] | H×W×6 tensor — probability per cell per class |

### Prediction Format

The `prediction` is a 3D array: `prediction[y][x][class]`

- Outer dimension: **H** rows (height)
- Middle dimension: **W** columns (width)
- Inner dimension: **6** probabilities (one per class)
- Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance)
- All probabilities must be non-negative

### Class Indices

| Index | Class |
|-------|-------|
| 0 | Empty (Ocean, Plains, Empty) |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |

### Response

```json
{
  "status": "accepted",
  "round_id": "uuid",
  "seed_index": 3
}
```

Resubmitting for the same seed overwrites your previous prediction. Only the last submission counts.

### Validation Errors

| Error | Cause |
|-------|-------|
| `Expected H rows, got N` | Wrong number of rows |
| `Row Y: expected W cols, got N` | Wrong number of columns |
| `Cell (Y,X): expected 6 probs, got N` | Wrong probability vector length |
| `Cell (Y,X): probs sum to S, expected 1.0` | Probabilities don't sum to 1.0 |
| `Cell (Y,X): negative probability` | Negative value in probability vector |

## GET /astar-island/my-rounds

Returns all rounds enriched with your team's scores, submission counts, rank, and query budget. This is the team-specific version of `/rounds`.

**Auth required.**

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "event_date": "2026-03-19",
    "status": "completed",
    "map_width": 40,
    "map_height": 40,
    "seeds_count": 5,
    "round_weight": 1,
    "started_at": "2026-03-19T10:00:00+00:00",
    "closes_at": "2026-03-19T10:45:00+00:00",
    "prediction_window_minutes": 165,
    "round_score": 72.5,
    "seed_scores": [80.1, 65.3, 71.9, ...],
    "seeds_submitted": 5,
    "rank": 3,
    "total_teams": 12,
    "queries_used": 48,
    "queries_max": 50,
    "initial_grid": [[10, 10, 10, ...], ...]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `round_score` | float \| null | Your team's average score across all seeds (null if not scored) |
| `seed_scores` | float[] \| null | Per-seed scores (null if not scored) |
| `seeds_submitted` | int | Number of seeds your team has submitted predictions for |
| `rank` | int \| null | Your team's rank for this round (null if not scored) |
| `total_teams` | int \| null | Total teams scored in this round |
| `queries_used` | int | Simulation queries used by your team |
| `queries_max` | int | Maximum queries allowed (default 50) |
| `initial_grid` | int[][] | Initial terrain grid for the first seed |

### Error Codes

| Status | Meaning |
|--------|---------|
| 403 | Not on a team |

## GET /astar-island/my-predictions/{round_id}

Returns your team's submitted predictions for a given round, with derived argmax and confidence grids for easy visualization.

**Auth required.**

```json
[
  {
    "seed_index": 0,
    "argmax_grid": [[0, 4, 5, ...], ...],
    "confidence_grid": [[0.85, 0.72, 0.93, ...], ...],
    "score": 78.2,
    "submitted_at": "2026-03-19T10:30:00+00:00"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `seed_index` | int | Which seed this prediction is for (0–4) |
| `argmax_grid` | int[][] | H×W grid of predicted class indices (argmax of probability vector) |
| `confidence_grid` | float[][] | H×W grid of confidence values (max probability per cell, rounded to 3 decimals) |
| `score` | float \| null | Score for this seed (null if not yet scored) |
| `submitted_at` | string \| null | ISO 8601 timestamp of submission |

The `argmax_grid` uses the same class indices as the prediction format (0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain).

### Error Codes

| Status | Meaning |
|--------|---------|
| 403 | Not on a team |

## GET /astar-island/analysis/{round_id}/{seed_index}

Post-round analysis endpoint. Returns your prediction alongside the ground truth for a specific seed, enabling detailed comparison. Only available after a round is completed (or during scoring).

**Auth required.**

```json
{
  "prediction": [[[0.85, 0.05, 0.02, 0.03, 0.03, 0.02], ...], ...],
  "ground_truth": [[[0.90, 0.03, 0.01, 0.02, 0.02, 0.02], ...], ...],
  "score": 78.2,
  "width": 40,
  "height": 40,
  "initial_grid": [[10, 10, 10, ...], ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | float[][][] | Your submitted H×W×6 probability tensor |
| `ground_truth` | float[][][] | The actual H×W×6 probability distribution (computed from Monte Carlo simulations) |
| `score` | float \| null | Your score for this seed |
| `width` | int | Map width |
| `height` | int | Map height |
| `initial_grid` | int[][] \| null | Initial terrain grid for this seed |

### Error Codes

| Status | Meaning |
|--------|---------|
| 400 | Round not completed/scoring yet, or invalid seed_index |
| 403 | Not on a team |
| 404 | Round not found, or no prediction submitted for this seed |

## GET /astar-island/leaderboard

Public leaderboard. Each team's score is their **best round score of all time** (weighted by round weight).

```json
[
  {
    "team_id": "uuid",
    "team_name": "Vikings ML",
    "team_slug": "vikings-ml",
    "weighted_score": 72.5,
    "rounds_participated": 3,
    "hot_streak_score": 78.1,
    "rank": 1,
    "is_verified": true
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `weighted_score` | float | Best `round_score × round_weight` across all rounds |
| `rounds_participated` | int | Total rounds this team has submitted predictions |
| `hot_streak_score` | float | Average score of last 3 rounds |
| `is_verified` | bool | Whether all team members are Vipps-verified |
| `rank` | int | Current leaderboard rank |
# Astar Island Simulation Mechanics

## The World

The world is a rectangular grid (default 40×40) with 8 terrain types that map to **6 prediction classes**:

![Terrain Types](/docs/astar-island/terrain-types.png)

| Internal Code | Terrain | Class Index | Description |
|--------------|---------|-------------|-------------|
| 10 | Ocean | 0 (Empty) | Impassable water, borders the map |
| 11 | Plains | 0 (Empty) | Flat land, buildable |
| 0 | Empty | 0 | Generic empty cell |
| 1 | Settlement | 1 | Active Norse settlement |
| 2 | Port | 2 | Coastal settlement with harbour |
| 3 | Ruin | 3 | Collapsed settlement |
| 4 | Forest | 4 | Provides food to adjacent settlements |
| 5 | Mountain | 5 | Impassable terrain |

Ocean, Plains, and Empty all map to **class 0** in predictions. Mountains are static (never change). Forests are mostly static but can reclaim ruined land. The interesting cells are those that can become Settlements, Ports, or Ruins.

## Map Generation

Each map is procedurally generated from a **map seed**:

- **Ocean borders** surround the map
- **Fjords** cut inland from random edges
- **Mountain chains** form via random walks
- **Forest patches** cover land with clustered groves
- **Initial settlements** placed on land cells, spaced apart

The map seed is visible to you — you can reconstruct the initial terrain layout locally.

## Simulation Lifecycle

Each of the 50 years cycles through multiple phases. The world goes through **growth, conflict, trade, harsh winters, and environmental change** — in that order.

![Simulation Phases](/docs/astar-island/simulation-phases.png)

### Growth

Settlements produce food based on adjacent terrain. When conditions are right, settlements grow in population, develop ports along coastlines, and build longships for naval operations. Prosperous settlements expand by founding new settlements on nearby land.

### Conflict

Settlements raid each other. Longships extend raiding range significantly. Desperate settlements (low food) raid more aggressively. Successful raids loot resources and damage the defender. Sometimes, conquered settlements change allegiance to the raiding faction.

![Faction dynamics — settlements change color as allegiances shift](/docs/astar-island/faction-dynamics.gif)

### Trade

Ports within range of each other can trade if not at war. Trade generates wealth and food for both parties, and technology diffuses between trading partners.

### Winter

Each year ends with a winter of varying severity. All settlements lose food. Settlements can collapse from starvation, sustained raids, or harsh winters — becoming Ruins and dispersing population to nearby friendly settlements.

### Environment

The natural world slowly reclaims abandoned land. Nearby thriving settlements may reclaim and rebuild ruined sites, establishing new outposts that inherit a portion of their patron's resources and knowledge. Coastal ruins can even be restored as ports. If no settlement steps in, ruins are eventually overtaken by forest growth or fade back into open plains.

## Settlement Properties

Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, and faction allegiance (owner_id).

Initial states expose settlement positions and port status. Internal stats (population, food, wealth, defense) are only visible through simulation queries.

## The World in Motion

Watch a full 50-year simulation unfold — settlements grow, expand, get raided, and some collapse:

![50-year simulation](/docs/astar-island/simulation.gif)

Initial state vs. after 50 years of simulation:

![Before and after](/docs/astar-island/simulation-before-after.png)

With high expansion, settlements rapidly colonise available land:

![High expansion simulation](/docs/astar-island/expansion.gif)
# Astar Island — Viking Civilisation Prediction

## What is this?

Astar Island is a machine learning challenge where you observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state. The simulator runs a procedurally generated Norse world for 50 years — settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

Your goal: **observe, learn the world's hidden rules, and predict the probability distribution of terrain types across the entire map.**

- **Task type**: Observation + probabilistic prediction
- **Platform**: [app.ainm.no](https://app.ainm.no)
- **API**: REST endpoints at `api.ainm.no/astar-island/`

![Initial Astar Island Map](/docs/astar-island/map-initial.png)

## How It Works

1. **A round starts** — the admin creates a round with a fixed map, many hidden parameters, and 5 random seeds
2. **Observe through a viewport** — call `POST /astar-island/simulate` with viewport coordinates to observe one stochastic run through a window (max 15×15 cells). You have 50 queries total per round, shared across all 5 seeds.
3. **Learn the hidden rules** — analyze viewport observations to understand the forces that govern the world
4. **Generate predictions** — use your understanding to build probability distributions for the full map
5. **Submit predictions** — for each of the 5 seeds, submit a W×H×6 probability tensor predicting terrain type probabilities per cell
6. **Scoring** — your prediction is compared against the ground truth using entropy-weighted KL divergence

## The Core Challenge

The simulation is **stochastic** — the same map and parameters produce different outcomes every run. With only **50 queries** shared across **5 seeds**, and each query only revealing a **15×15 viewport** of the 40×40 map, you must be strategic about what you observe and how you use that information.

![Same map, different stochastic outcomes](/docs/astar-island/stochastic-outcomes.png)

![Viewport: you only see a small window of the full map](/docs/astar-island/viewport-demo.png)

The world is governed by many hidden forces that interact in complex ways. Teams that understand these interactions can build accurate models and generate predictions far beyond what raw observation provides.

## Quick Start

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team
3. Go to the Astar Island page
4. When a round is active, use the API to observe the simulator
5. Analyze results, build your model, submit predictions for all 5 seeds

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Map seed** | Determines terrain layout (fixed per seed, visible to you) |
| **Sim seed** | Random seed for each simulation run (different every query) |
| **Hidden parameters** | Values controlling the world's behavior (same for all seeds in a round) |
| **50 queries** | Your budget per round, shared across all 5 seeds |
| **Viewport** | Each query reveals a max 15×15 window of the map |
| **W×H×6 tensor** | Your prediction — probability of each of 6 terrain classes per cell |
| **50 years** | Each simulation runs for 50 time steps |
# Astar Island Quickstart

## Authentication

All endpoints require authentication. Log in at app.ainm.no, then inspect cookies in your browser to grab your `access_token` JWT.

You can authenticate using either a cookie or a Bearer token header:

```python
import requests

BASE = "https://api.ainm.no"

# Option 1: Cookie-based auth
session = requests.Session()
session.cookies.set("access_token", "YOUR_JWT_TOKEN")

# Option 2: Bearer token auth
session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

## Step 1: Get the Active Round

```python
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)

if active:
    round_id = active["id"]
    print(f"Active round: {active['round_number']}")
```

## Step 2: Get Round Details

Fetch the detail endpoint to get full round info including `seeds_count` and initial states:

```python
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

width = detail["map_width"]      # 40
height = detail["map_height"]    # 40
seeds = detail["seeds_count"]    # 5
print(f"Round: {width}x{height}, {seeds} seeds")

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]           # height x width terrain codes
    settlements = state["settlements"]  # [{x, y, has_port, alive}, ...]
    print(f"Seed {i}: {len(settlements)} settlements")
```

## Step 3: Query the Simulator

You have 50 queries per round, shared across all seeds. Each query reveals a 5-15 cell wide viewport:

```python
result = session.post(f"{BASE}/astar-island/simulate", json={
    "round_id": round_id,
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15,
}).json()

grid = result["grid"]                # 15x15 terrain after simulation
settlements = result["settlements"]  # settlements in viewport with full stats
viewport = result["viewport"]        # {x, y, w, h}
```

## Step 4: Build and Submit Predictions

For each seed, submit a `height x width x 6` probability tensor. Each cell has 6 values representing the probability of each terrain class (Empty, Settlement, Port, Ruin, Forest, Mountain). They must sum to 1.0:

```python
import numpy as np

for seed_idx in range(seeds):
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline

    # TODO: replace with your model's predictions
    # prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code}")
```

A uniform prediction scores ~1-5. Use your queries to build better predictions.

> **Warning:** Never assign probability 0.0 to any class. If the ground truth has any non-zero probability for a class you marked as zero, KL divergence becomes infinite and your score for that cell is destroyed. Always enforce a minimum floor (e.g., 0.01) and renormalize. See the [scoring docs](/docs/astar-island/scoring.md#common-pitfalls) for details.

## Using the MCP Server

Add the documentation server to Claude Code for AI-assisted development:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
# Astar Island Scoring

## Score Formula

Your score is based on **entropy-weighted KL divergence** between your prediction and the ground truth.

### Ground Truth

For each seed, the organizers pre-compute ground truth by running the simulation **hundreds of times** with the true hidden parameters. This produces a probability distribution for each cell.

For example, a cell might have ground truth `[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]` — meaning 60% chance of Settlement, 25% Port, 15% Ruin, after 50 years.

![Probability tensor — per-cell class probabilities](/docs/astar-island/probability-tensor.png)

### KL Divergence

For each cell, the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measures how different your prediction is from the ground truth:

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)
```

Where `p` = ground truth, `q` = your prediction. Lower KL = better match.

### Entropy Weighting

Not all cells are equally important. Static cells (ocean stays ocean, mountain stays mountain) have near-zero entropy and are excluded from scoring.

**Only dynamic cells** (those that change between simulation runs) contribute to your score, weighted by their entropy:

```
entropy(cell) = -Σ pᵢ × log(pᵢ)
```

Cells with higher entropy (more uncertain outcomes) count more toward your score. This focuses scoring on the interesting parts of the map.

### Final Score

```
weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ─────────────────────────────────────────────────────────
                            Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- **100** = perfect prediction (your distribution matches ground truth exactly)
- **0** = terrible prediction (high KL divergence)
- The exponential decay means small improvements in prediction accuracy yield diminishing score gains

## Common Pitfalls

**Never assign probability 0.0 to any class.** KL divergence includes the term `pᵢ × log(pᵢ / qᵢ)`. If the ground truth has `pᵢ > 0` but your prediction has `qᵢ = 0`, the divergence goes to **infinity** — destroying your entire score for that cell.

Even if you're confident a cell is Forest, the ground truth may assign a small probability to Settlement or Ruin across thousands of simulations. A single zero in your prediction can tank your score.

**Recommendation:** Always enforce a minimum probability floor of **0.01** per class, then renormalize so the values still sum to 1.0:

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

This small safety margin costs almost nothing in score but protects against catastrophic KL blowups.

## Per-Round Score

Each round has **5 seeds**. Your round score is the **average** of your per-seed scores:

```
round_score = (score_seed_0 + score_seed_1 + ... + score_seed_4) / 5
```

If you don't submit a prediction for a seed, that seed scores **0**. Always submit something for every seed — even a uniform prediction beats 0.

## Leaderboard

Your leaderboard score is your **best round score of all time**:

```
leaderboard_score = max(round_score) across all rounds
```

Later rounds may have higher weights, meaning a good score on a later round counts for more. Only your single best result matters — keep improving your model across rounds.

A **hot streak score** (average of your last 3 rounds) is also tracked.

## Game End

Each round has a **prediction window** (typically 2 hours 45 minutes). After the window closes:

1. Round status changes to `scoring`
2. All predictions are scored against ground truth
3. Per-seed scores are averaged to compute round score
4. Leaderboard updates with weighted averages
5. Round status changes to `completed`
# WebSocket Protocol Specification

## Connection

Connect via WebSocket to the URL provided when you request a game token:

```
wss://game.ainm.no/ws?token=<jwt_token>
```

Get a token by clicking "Play" on a map at [app.ainm.no/challenge](https://app.ainm.no/challenge), or by calling the `request_game(map_id)` MCP tool.

## Message Flow

```
Server → Client: {"type": "game_state", ...}     (round 0)
Client → Server: {"actions": [...]}
Server → Client: {"type": "game_state", ...}     (round 1)
Client → Server: {"actions": [...]}
...
Server → Client: {"type": "game_over", ...}       (final)
```

## Game State Message

```json
{
  "type": "game_state",
  "round": 42,
  "max_rounds": 300,
  "action_status": "ok",
  "grid": {
    "width": 14,
    "height": 10,
    "walls": [[1,1], [1,2], [3,1]]
  },
  "bots": [
    {"id": 0, "position": [3, 7], "inventory": ["milk"]},
    {"id": 1, "position": [5, 3], "inventory": []},
    {"id": 2, "position": [10, 7], "inventory": ["bread", "eggs"]}
  ],
  "items": [
    {"id": "item_0", "type": "milk", "position": [2, 1]},
    {"id": "item_1", "type": "bread", "position": [4, 1]}
  ],
  "orders": [
    {
      "id": "order_0",
      "items_required": ["milk", "bread", "eggs"],
      "items_delivered": ["milk"],
      "complete": false,
      "status": "active"
    },
    {
      "id": "order_1",
      "items_required": ["cheese", "butter", "pasta"],
      "items_delivered": [],
      "complete": false,
      "status": "preview"
    }
  ],
  "drop_off": [6, 9],
  "score": 12,
  "active_order_index": 0,
  "total_orders": 8
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Current round number (0-indexed) |
| `max_rounds` | int | Maximum rounds (300, or 500 for Nightmare) |
| `action_status` | string | Result of your last action: `"ok"`, `"timeout"`, or `"error"` |
| `grid.width` | int | Grid width in cells |
| `grid.height` | int | Grid height in cells |
| `grid.walls` | int[][] | List of [x, y] wall positions |
| `bots` | object[] | All bots (1-10 depending on difficulty) with id, position [x,y], and inventory |
| `items` | object[] | All items on shelves with id, type, and position [x,y] |
| `orders` | object[] | Only active + preview orders (max 2). Each has `status`: `"active"` or `"preview"` |
| `drop_off` | int[] | [x, y] position of the drop-off zone (Easy–Expert) |
| `drop_off_zones` | int[][] | Array of [x, y] positions (Nightmare only, 3 zones) |
| `score` | int | Current score |
| `active_order_index` | int | Index of the current active order |
| `total_orders` | int | Total number of orders in the game |

### `action_status`

Every `game_state` message includes `action_status`, which tells you whether the server received and processed your last response:

| Value | Meaning |
|-------|---------|
| `"ok"` | Your actions were received and applied normally |
| `"timeout"` | Your bot didn't respond within the 2-second window — all actions were set to `null` (bots waited) |
| `"error"` | Your message was received but couldn't be parsed (invalid JSON, wrong format, etc.) — all actions were set to `null` |

On round 0, `action_status` is always `"ok"` since there was no previous action.

Use this field to detect and debug connectivity or parsing issues. If you see repeated `"timeout"` values, your bot is too slow. If you see `"error"`, check your JSON format.

## Bot Response

Send within **2 seconds** of receiving the game state:

```json
{
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "drop_off"}
  ]
}
```

### Optional `round` Field

You can include an optional `round` field in your action message to guard against desync:

```json
{
  "round": 42,
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "drop_off"}
  ]
}
```

If `round` is included and doesn't match the server's current round number, your actions are rejected (treated as if you sent nothing — all bots wait). This is useful for detecting when your bot has fallen out of sync with the server, for example due to network latency causing you to respond to a stale game state.

If you omit the `round` field, the server accepts your actions unconditionally. Including it is recommended but not required.

### Actions

| Action | Extra Fields | Description |
|--------|-------------|-------------|
| `move_up` | — | Move one cell up (y-1) |
| `move_down` | — | Move one cell down (y+1) |
| `move_left` | — | Move one cell left (x-1) |
| `move_right` | — | Move one cell right (x+1) |
| `pick_up` | `item_id` | Pick up item from adjacent shelf |
| `drop_off` | — | Deliver matching items to active order at drop-off zone |
| `wait` | — | Do nothing |

### Move Rules

- Moves to walls, shelves, or out-of-bounds cells fail silently (treated as `wait`)
- Moves to a cell occupied by another bot fail silently (`blocked_by_bot`)
- Actions resolve in **bot ID order** — bot 0 moves first, then bot 1, etc.
- The spawn tile (bottom-right) is exempt from collision — bots can share it at game start

### Pickup Rules

- Bot must be **adjacent** (Manhattan distance 1) to the shelf containing the item
- Bot inventory must not be full (max 3 items)
- `item_id` must match an item on the map

### Dropoff Rules

- Bot must be standing **on** the drop-off cell
- Bot must have items in inventory
- Only items matching the **active order** are delivered — non-matching items **stay in inventory**
- Each delivered item = **+1 point**
- Completed order = **+5 bonus points**
- When the active order completes, the next order activates immediately and remaining items are re-checked

## Game Over Message

When the game ends, the server sends a `game_over` message instead of another `game_state`. This is the final message — the WebSocket closes after this.

```json
{
  "type": "game_over",
  "score": 47,
  "rounds_used": 200,
  "items_delivered": 22,
  "orders_completed": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `score` | int | Final score (`items_delivered + orders_completed * 5`) |
| `rounds_used` | int | Number of rounds played before the game ended |
| `items_delivered` | int | Total items delivered across all orders |
| `orders_completed` | int | Number of fully completed orders |

The game ends when any of these conditions is met:
- **Max rounds reached** — 300 rounds (500 for nightmare difficulty)
- **Wall-clock time limit** — 120 seconds (300 for nightmare)
- **Client disconnect** — your WebSocket connection drops

Your bot **must** handle `game_over` messages. Check `data["type"]` before processing — if it's `"game_over"`, print the results and exit cleanly. Failing to handle this will cause your bot to error when it tries to parse the message as a game state.

## Timeouts & Errors

- **2 second** timeout per round for your response
- Timeout → all bots wait (no action), next `action_status` will be `"timeout"`
- Unparseable message → all bots wait, next `action_status` will be `"error"`
- Invalid individual actions → treated as `wait` (but `action_status` is still `"ok"`)
- Disconnect → game ends immediately, score is saved
- **120 second** wall-clock limit per game (300 seconds for nightmare difficulty)

### Coordinate System

- Origin `(0, 0)` is the **top-left** corner
- X increases to the right
- Y increases downward
# Submission Guide

## How to Play

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team
3. Go to the [Challenge page](https://app.ainm.no/challenge)
4. Pick a map and click "Play" to get a WebSocket URL with token
5. Connect your bot to the WebSocket URL
6. Your bot receives game state each round and responds with actions
7. Best score per map is saved automatically

## Rate Limits

- 60 second cooldown between games
- Max 40 games per hour per team
- Max 300 games per day per team

## Example Bot (Python + websockets)

A minimal bot that connects via WebSocket:

```python
import asyncio
import json
import websockets

WS_URL = "wss://game.ainm.no/ws?token=YOUR_TOKEN_HERE"


async def play():
    async with websockets.connect(WS_URL) as ws:
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "game_over":
                print(f"Game over! Score: {data['score']}, Rounds: {data['rounds_used']}")
                break

            if data["type"] == "game_state":
                actions = decide_actions(data)
                await ws.send(json.dumps({"actions": actions}))


def decide_actions(state):
    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off = state["drop_off"]

    actions = []
    for bot in bots:
        action = decide_bot_action(bot, items, orders, drop_off)
        actions.append(action)
    return actions


def decide_bot_action(bot, items, orders, drop_off):
    bx, by = bot["position"]
    inventory = bot["inventory"]

    # Find the active order (status == "active")
    active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
    if not active:
        return {"bot": bot["id"], "action": "wait"}

    # What does the active order still need?
    needed = {}
    for item in active["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in active["items_delivered"]:
        needed[item] = needed.get(item, 0) - 1
    needed = {k: v for k, v in needed.items() if v > 0}

    # If we have useful items and we're at dropoff, deliver
    has_useful = any(needed.get(item, 0) > 0 for item in inventory)
    if has_useful and bx == drop_off[0] and by == drop_off[1]:
        return {"bot": bot["id"], "action": "drop_off"}

    # If inventory full or has useful items, go deliver
    if len(inventory) >= 3 or (has_useful and not needed):
        return navigate_to(bot["id"], bx, by, drop_off[0], drop_off[1])

    # Find nearest needed item
    best_item = None
    best_dist = float("inf")
    for item in items:
        if needed.get(item["type"], 0) > 0:
            ix, iy = item["position"]
            dist = abs(bx - ix) + abs(by - iy)
            if dist < best_dist:
                best_dist = dist
                best_item = item

    if best_item:
        ix, iy = best_item["position"]
        if abs(bx - ix) + abs(by - iy) == 1:
            return {"bot": bot["id"], "action": "pick_up", "item_id": best_item["id"]}
        return navigate_to(bot["id"], bx, by, ix, iy)

    if has_useful:
        return navigate_to(bot["id"], bx, by, drop_off[0], drop_off[1])

    return {"bot": bot["id"], "action": "wait"}


def navigate_to(bot_id, x, y, tx, ty):
    dx = tx - x
    dy = ty - y
    if abs(dx) > abs(dy):
        return {"bot": bot_id, "action": "move_right" if dx > 0 else "move_left"}
    if dy != 0:
        return {"bot": bot_id, "action": "move_down" if dy > 0 else "move_up"}
    if dx != 0:
        return {"bot": bot_id, "action": "move_right" if dx > 0 else "move_left"}
    return {"bot": bot_id, "action": "wait"}


asyncio.run(play())
```

This simple bot treats each bot identically — they all greedily go for the nearest needed item. To improve:

- **Assign roles** — use `bot["id"]` to split bots into different map regions
- **Add pathfinding** — BFS/A* around walls and shelves
- **Coordinate pickups** — track what each bot is targeting to avoid duplication
- **Order prioritization** — focus on nearly-complete orders first

## Deploying Your Bot

Your bot runs **locally** — it connects out to the game server via WebSocket. No hosting required!

If you want to run it on a server:
- Any machine with Python and internet access works
- No HTTPS or public endpoint needed
- The bot is the WebSocket **client**, not server

## Using This MCP Server with Claude

Add this MCP server to Claude Code:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

Then Claude can read the challenge docs and help you build your bot.

## Debugging Tips

- Print the full `state` on the first round to understand the structure
- Track `score` changes between rounds to verify deliveries
- Check if bots are moving (compare positions between rounds)
- Common issues:
  - Moving into walls (check `grid.walls` before moving)
  - Trying to pick up from non-adjacent shelves (must be Manhattan distance 1)
  - Trying to drop off when not on the drop-off cell
  - Not handling `game_over` messages (causes connection errors)
# Grocery Bot Game Mechanics

## Concept

You control bots navigating a grocery store to fulfill orders sequentially. Pick up items from shelves, deliver them to the drop-off zone, complete orders one at a time for bonus points. Bot count scales by difficulty.

## Store Layout

The store is a rectangular grid with border walls:

- **Floor** (`.`) — walkable cells
- **Walls** (`#`) — impassable barriers (borders + aisle walls)
- **Shelves** — contain items, not walkable. Pick up by standing adjacent.
- **Drop-off** (`D`) — where you deliver items, also walkable

Stores have parallel vertical aisles (shelf-walkway-shelf, 3 cells wide), connected by horizontal corridors at top, bottom, and mid-height.

## 5 Difficulty Levels

| Level | Grid | Bots | Aisles | Item Types | Maps | Rounds | Time Limit |
|-------|------|------|--------|------------|------|--------|------------|
| Easy | 12×10 | 1 | 2 | 4 | 5 | 300 | 2 min |
| Medium | 16×12 | 3 | 3 | 8 | 5 | 300 | 2 min |
| Hard | 22×14 | 5 | 4 | 12 | 5 | 300 | 2 min |
| Expert | 28×18 | 10 | 5 | 16 | 5 | 300 | 2 min |
| Nightmare | 30×18 | 20 | 6 | 21 | 1 | 500 | 5 min |

21 maps total. Nightmare features 3 drop-off zones instead of 1. Grid structure is fixed per map. **Item placement and orders change daily** (seeded from map_seed + day_of_competition). Same day = same game (deterministic).

## Game Flow

1. All bots start at bottom-right of the store (inside border)
2. Each round, your bot receives the full game state via WebSocket
3. You respond with actions for each bot
4. The game runs for **300 rounds** maximum
5. Wall-clock limit: **120 seconds** per game

## Bots

- **Bot count varies** by difficulty (1, 3, 5, 10, or 20)
- **Inventory capacity**: 3 items per bot
- **Collision**: bots block each other — no two bots can occupy the same tile. Actions resolve in bot ID order (lower IDs move first). Spawn tile is exempt so bots can start stacked.
- **Full visibility**: all items on all shelves are always visible

## Sequential Orders (Infinite)

Orders are revealed **one at a time** and keep generating indefinitely:

- **Active order**: the current order you must complete. Full details visible. You can deliver items for this order.
- **Preview order**: the next order. Full details visible. You CANNOT deliver items for it yet, but you can pre-pick items.
- **Hidden orders**: all remaining orders are not shown.
- **Infinite**: when you complete an order, a new one appears. Orders never run out. Rounds are the only limit.

When the active order is completed:
- The preview order becomes active
- A new order becomes the preview
- Any items in bot inventories that match the new active order are auto-delivered

### Pickup rules
- Bots can pick up **any item** from any shelf, regardless of which order needs it
- Bad picks waste inventory slots — choose wisely

### Dropoff rules
- Only the **active order** can be delivered to
- Items matching the active order are consumed; non-matching items **stay in inventory**
- When the active order completes, the next order activates immediately and remaining items are re-checked

### Order sizes

| Level | Items per Order |
|-------|----------------|
| Easy | 3-4 |
| Medium | 3-5 |
| Hard | 3-5 |
| Expert | 4-6 |
| Nightmare | 4-7 |

## Actions

Each bot can perform one action per round:

| Action | Description |
|--------|-------------|
| `move_up` | Move one cell up |
| `move_down` | Move one cell down |
| `move_left` | Move one cell left |
| `move_right` | Move one cell right |
| `pick_up` | Pick up an item from adjacent shelf (requires `item_id`) |
| `drop_off` | Deliver matching inventory at the drop-off zone |
| `wait` | Do nothing |

Invalid actions are treated as `wait` — no penalty, no error.

## Key Constraints

- **300 rounds** — plan carefully, every round counts
- **3 items per bot** inventory capacity
- **Sequential orders** — complete one before the next activates
- **Infinite orders** — rounds are the only limit
- **No fog of war** — full map visible every round
- **Deterministic per day** — same game every run within a day
- **60s cooldown** between games, max **40/hour** and **300/day** per team
- **Disconnect = game over** — score what you have, no reconnect
# Grocery Bot Challenge

## What is this?

The Grocery Bot was the pre-competition warm-up challenge. It is not part of the main competition scoring.

- **Task type**: Real-time game (WebSocket)
- **Platform**: [app.ainm.no](https://app.ainm.no)

## How It Works

1. **Pick a map** from the 21 available maps on the Challenge page
2. **Get a WebSocket URL** — click Play to get a game token
3. **Connect your bot** to the WebSocket URL
4. **Receive game state** each round as JSON
5. **Respond with actions** — one per bot (move, pickup, dropoff, or wait)
6. **Best score per map** is saved automatically. Leaderboard = sum of all 21 best scores.

## Difficulty Levels

Bot count and grid size increase with difficulty:

| Level | Bots | Grid | Maps | Rounds | Time Limit |
|-------|------|------|------|--------|------------|
| Easy | 1 | 12×10 | 5 | 300 | 2 min |
| Medium | 3 | 16×12 | 5 | 300 | 2 min |
| Hard | 5 | 22×14 | 5 | 300 | 2 min |
| Expert | 10 | 28×18 | 5 | 300 | 2 min |
| Nightmare | 20 | 30×18 | 1 | 500 | 5 min |

Nightmare features 3 drop-off zones instead of 1.

## Quick Start

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team
3. Go to the Challenge page, pick a difficulty, click Play
4. Copy the WebSocket URL and connect your bot
5. Play all 21 maps to maximize your leaderboard score

## Key Features

- **WebSocket** — you connect to the game server, not the other way around
- **No fog of war** — full map visible from round 1
- **Bot collision** — bots block each other (no two on same tile, except spawn)
- **Infinite orders** — orders keep generating, rounds are the only limit
- **Daily rotation** — item placement and orders change daily to prevent hardcoding
- **Deterministic within a day** — same map + same day = same game every time
# Grocery Bot Scoring

## Score Formula

Per game:
```
score = items_delivered × 1 + orders_completed × 5
```

- **+1 point** for each item delivered to the drop-off
- **+5 bonus** for completing an entire order (all required items delivered)

## Leaderboard

Your **leaderboard score** is the **sum of your best scores across all 21 maps**.

- Play each map as many times as you want (60s cooldown, 40/hour, 300/day)
- Only your highest score per map is saved
- Deterministic within a day — same algorithm = same score
- To maximize your rank: get good scores on ALL 21 maps

## Daily Rotation

Item placement on shelves and order contents change daily at midnight UTC. The grid structure (walls, shelf positions) stays the same. This prevents hardcoding solutions while keeping games deterministic within a single day.

## Infinite Orders

Orders never run out. When you complete the active order, the next one activates and a new preview appears. The only limit is the **300 round** cap. Score as much as you can before time runs out.

## Score Examples

| Scenario | Items | Orders | Score |
|----------|-------|--------|-------|
| Delivered 3 items, no complete orders | 3 | 0 | 3 |
| Delivered 4 items, completed 1 order | 4 | 1 | 9 |
| Delivered 15 items, completed 3 orders | 15 | 3 | 30 |
| Delivered 50 items, completed 10 orders | 50 | 10 | 100 |

## Game End Conditions

| Condition | Description |
|-----------|-------------|
| Max rounds | 300 rounds (500 for Nightmare) |
| Wall-clock timeout | 120 seconds (300 seconds for Nightmare) |
| Disconnect | Client disconnected |

# NorgesGruppen Data: Examples & Tips

## Random Baseline

Minimal `run.py` that generates random predictions (use to verify your setup):

```python
import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = []
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        for _ in range(random.randint(5, 20)):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 356),
                "bbox": [
                    round(random.uniform(0, 1500), 1),
                    round(random.uniform(0, 800), 1),
                    round(random.uniform(20, 200), 1),
                    round(random.uniform(20, 200), 1),
                ],
                "score": round(random.uniform(0.01, 1.0), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

## YOLOv8 Example

Using YOLOv8n with GPU auto-detection. **Important:** The pretrained COCO model outputs COCO class IDs (0-79), not product IDs (0-355). For correct product classification, fine-tune on the competition training data with `nc=357`. Detection-only submissions (wrong category_ids) still score up to 70%.

```python
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    predictions = []

    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        results = model(str(img), device=device, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

Include `yolov8n.pt` in your zip. This pretrained COCO model serves as a baseline — fine-tune on the competition training data for better results. With GPU available, larger models like YOLOv8m/l/x are also feasible within the timeout.

## ONNX Inference Example

ONNX works with any model framework. Use `CUDAExecutionProvider` for GPU acceleration:

**Export (on your training machine):**

```python
# From ultralytics:
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=17)

# From any PyTorch model:
import torch
model = ...  # your trained model
dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17)
```

**Inference (in your `run.py`):**

```python
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path).convert("RGB").resize((640, 640))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

        outputs = session.run(None, {input_name: arr})
        # Process outputs based on your model's output format
        # ...

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

## Common Errors

| Error | Fix |
|---|---|
| `run.py not found at zip root` | Zip the **contents**, not the folder. See "Creating Your Zip" in submission docs. |
| `Disallowed file type: __MACOSX/...` | macOS Finder resource forks. Use terminal: `zip -r ../sub.zip . -x ".*" "__MACOSX/*"` |
| `Disallowed file type: .bin` | Rename `.bin` → `.pt` (same format) or convert to `.safetensors` |
| `Security scan found violations` | Remove imports of subprocess, socket, os, etc. Use pathlib instead. |
| `No predictions.json in output` | Make sure run.py writes to the `--output` path |
| `Timed out after 300s` | Ensure GPU is used (`model.to("cuda")`), or use a smaller model |
| `Exit code 137` | Out of memory (8 GB limit). Reduce batch size or use FP16 |
| `Exit code 139` | Segfault — likely model weight version mismatch. Re-export with matching package version or use ONNX. |
| `ModuleNotFoundError` | Package not in sandbox. Export model to ONNX or include model code in your .py files. |
| `KeyError` / `RuntimeError` on model load | Version mismatch. Pin exact sandbox versions or export to ONNX. |

## Tips

- Start with the random baseline to verify your setup works
- **GPU is available** — larger models (YOLOv8m/l/x, custom transformers) are feasible within the 300s timeout
- Use `torch.cuda.is_available()` to write code that works both locally (CPU) and on the server (GPU)
- FP16 quantization is recommended — smaller weights, faster GPU inference
- ONNX with `CUDAExecutionProvider` gives good GPU performance for any framework
- Process images one at a time to stay within memory limits
- Use `torch.no_grad()` during inference
- Test your code locally before uploading
- You don't need all sandbox packages for training — only match what you use
# NorgesGruppen Data: Object Detection

Detect grocery products on store shelves. Upload your model code as a `.zip` file — it runs in a sandboxed Docker container on our servers.

## How It Works

1. Download the training data from the competition website (requires login)
2. Train your object detection model locally
3. Write a `run.py` that takes shelf images as input and outputs predictions
4. Zip your code + model weights
5. Upload at the submit page
6. Our server runs your code in a sandbox with GPU (NVIDIA L4, 24 GB VRAM) — no network access
7. Your predictions are scored: **70% detection** (did you find products?) + **30% classification** (did you identify the right product?)
8. Score appears on the leaderboard

## Downloads

Download training data and product reference images from the **Submit** page on the competition website (login required).

## Training Data

Two files are available for download:

**COCO Dataset** (`NM_NGD_coco_dataset.zip`, ~864 MB)
- 248 shelf images from Norwegian grocery stores
- ~22,700 COCO-format bounding box annotations
- 356 product categories (category_id 0-355) — detect and identify grocery products
- Images from 4 store sections: Egg, Frokost, Knekkebrod, Varmedrikker

**Product Reference Images** (`NM_NGD_product_images.zip`, ~60 MB)
- 327 individual products with multi-angle photos (main, front, back, left, right, top, bottom)
- Organized by barcode: `{product_code}/main.jpg`, `{product_code}/front.jpg`, etc.
- Includes `metadata.json` with product names and annotation counts

### Annotation Format

The COCO annotations file (`annotations.json`) contains:

```json
{
  "images": [
    {"id": 1, "file_name": "img_00001.jpg", "width": 2000, "height": 1500}
  ],
  "categories": [
    {"id": 0, "name": "VESTLANDSLEFSA TØRRE 10STK 360G", "supercategory": "product"},
    {"id": 1, "name": "COFFEE MATE 180G NESTLE", "supercategory": "product"},
    ...
    {"id": 356, "name": "unknown_product", "supercategory": "product"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 42,
      "bbox": [141, 49, 169, 152],
      "area": 25688,
      "iscrowd": 0,
      "product_code": "8445291513365",
      "product_name": "NESCAFE VANILLA LATTE 136G NESTLE",
      "corrected": true
    }
  ]
}
```

Key fields: `bbox` is `[x, y, width, height]` in pixels (COCO format). `product_code` is the barcode. `corrected` indicates manually verified annotations.

## What Annotations Look Like

A training image with all ground truth boxes (green = correctly detected product):

<img src="/docs/shelf-annotations-full.png" alt="Shelf image with all 76 products annotated in green bounding boxes — this represents a perfect mAP of 1.0" style="max-width:100%; border-radius:8px; margin:8px 0" />

Compare with a ~50% mAP result — half the products are missed entirely, and some detected boxes (red) are imprecise:

<img src="/docs/shelf-annotations-partial.png" alt="Same shelf image with only half the products detected, some with imprecise boxes shown in red — approximately 50% mAP" style="max-width:100%; border-radius:8px; margin:8px 0" />

## Submit

Upload your `.zip` at the submission page on the competition website.

## MCP Setup

Connect this docs server to your AI coding tool:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
# NorgesGruppen Data: Scoring

## Hybrid Scoring

Your final score combines detection and classification:

```
Score = 0.7 × detection_mAP + 0.3 × classification_mAP
```

Both components use mAP@0.5 (Mean Average Precision at IoU threshold 0.5).

### Detection mAP (70% of score)

Measures whether you found the products, ignoring category:

- Each prediction is matched to the closest ground truth box
- A prediction is a true positive if IoU ≥ 0.5 (category is ignored)
- This rewards accurate bounding box localization

### Classification mAP (30% of score)

Measures whether you identified the correct product:

- A prediction is a true positive if IoU ≥ 0.5 AND the `category_id` matches the ground truth
- 356 product categories (IDs 0-355) from the training data `annotations.json`

### Detection-Only Submissions

If you set `category_id: 0` for all predictions, you can score up to **0.70** (70%) from the detection component alone. Adding correct product identification unlocks the remaining 30%.

- Score range: 0.0 (worst) to 1.0 (perfect)

## Submission Limits

| Limit | Value |
|---|---|
| Submissions in-flight | 2 per team |
| Submissions per day | 3 per team |
| Infrastructure failure freebies | 2 per day (don't count against your 3) |

Limits reset at midnight UTC. If you hit an infrastructure error (our fault), it doesn't count against your daily limit — up to 2 per day. After that, infrastructure failures consume a regular submission slot.

## Leaderboard

The public leaderboard shows scores from the public test set. The final ranking uses the private test set which is never revealed to participants.

## Select for Final Evaluation

By default, your best-scoring submission is used for the final private evaluation. You can override this by clicking **Select for final** on any completed submission in your submission history. This lets you choose a submission you trust, even if it's not your highest public score. You can change your selection at any time before the competition ends.
# NorgesGruppen Data: Submission Format

## Zip Structure

Your `.zip` must contain `run.py` at the root. You may include model weights and Python helper files.

```
submission.zip
├── run.py          # Required: entry point
├── model.onnx      # Optional: model weights (.pt, .onnx, .safetensors, .npy)
└── utils.py        # Optional: helper code
```

**Limits:**

| Limit | Value |
|---|---|
| Max zip size (uncompressed) | 420 MB |
| Max files | 1000 |
| Max Python files | 10 |
| Max weight files (.pt, .pth, .onnx, .safetensors, .npy) | 3 |
| Max weight size total | 420 MB |
| Allowed file types | .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy |

## run.py Contract

Your script is executed as:

```bash
python run.py --input /data/images --output /output/predictions.json
```

### Input

`/data/images/` contains JPEG shelf images. File names use the format `img_XXXXX.jpg` (e.g., `img_00042.jpg`).

### Output

Write a JSON array to the `--output` path:

```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

| Field | Type | Description |
|---|---|---|
| `image_id` | int | Numeric ID from filename (`img_00042.jpg` → `42`) |
| `category_id` | int | Product category ID (0-355). See `categories` list in annotations.json |
| `bbox` | [x, y, w, h] | Bounding box in COCO format |
| `score` | float | Confidence score (0-1) |

## Scoring

Your score combines detection and classification:

- **70% detection mAP** — did you find the products? (bounding box IoU ≥ 0.5, category ignored)
- **30% classification mAP** — did you identify the right product? (IoU ≥ 0.5 AND correct category_id)

Detection-only submissions (`category_id: 0` for all predictions) score up to 70%. Product identification adds the remaining 30%.

## Product Categories

The training data `annotations.json` contains a `categories` list mapping integer IDs to product names:

```json
"categories": [
  {"id": 0, "name": "VESTLANDSLEFSA TØRRE 10STK 360G", "supercategory": "product"},
  {"id": 1, "name": "COFFEE MATE 180G NESTLE", "supercategory": "product"},
  ...
  {"id": 356, "name": "unknown_product", "supercategory": "product"}
]
```

Your predictions must use the same `category_id` values. When training YOLOv8 on this COCO data (with `nc=357`), the model learns the mapping and outputs the correct category_id during inference.

## Sandbox Environment

Your code runs in a Docker container with these constraints:

| Resource | Limit |
|---|---|
| Python | 3.11 |
| CPU | 4 vCPU |
| Memory | 8 GB |
| GPU | NVIDIA L4 (24 GB VRAM) |
| CUDA | 12.4 |
| Network | None (fully offline) |
| Timeout | 300 seconds |

### GPU

An NVIDIA L4 GPU is always available in the sandbox. Your code auto-detects it:

- `torch.cuda.is_available()` returns `True`
- No opt-in flag needed — GPU is always on
- For ONNX models, use `["CUDAExecutionProvider", "CPUExecutionProvider"]` as the provider list

### Pre-installed Packages

PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless 4.9.0.80, albumentations 1.3.1, Pillow 10.2.0, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.0, pycocotools 2.0.7, ensemble-boxes 1.0.9, timm 0.9.12, supervision 0.18.0, safetensors 0.4.2.

You **cannot** `pip install` at runtime.

## Training Environment

You can use **any computer vision architecture** — the sandbox supports all models via ONNX, custom PyTorch code, or the pre-installed frameworks. You don't need all sandbox packages for training — only match the versions of packages you actually use.

### Models available in the sandbox

These frameworks are pre-installed. If you train with the **exact same version**, you can submit `.pt` weights directly:

| Framework | Models | Pin this version |
|---|---|---|
| ultralytics 8.1.0 | YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR-l/x | `ultralytics==8.1.0` |
| torchvision 0.21.0 | Faster R-CNN, RetinaNet, SSD, FCOS, Mask R-CNN | `torchvision==0.21.0` |
| timm 0.9.12 | ResNet, EfficientNet, ViT, Swin, ConvNeXt, etc. (as backbones) | `timm==0.9.12` |

### Models not in the sandbox

YOLOv9, YOLOv10, YOLO11, RF-DETR, Detectron2, MMDetection, HuggingFace Transformers — these packages are not installed. Two options:

1. **Export to ONNX**: Export from any framework, load with `onnxruntime` in your `run.py`. Use opset version ≤ 20. Use `CUDAExecutionProvider` for GPU acceleration.
2. **Include model code**: Put your model class in your `.py` files + `.pt` state_dict weights. Works if the model only uses standard PyTorch ops.

**HuggingFace `.bin` files**: The `.bin` extension is not allowed, but the format is identical to `.pt` (PyTorch pickle). Rename `.bin` → `.pt`, or convert with `safetensors.torch.save_file(state_dict, "model.safetensors")`.

**Models larger than 420 MB**: Quantize to FP16 or INT8 to fit within the 420 MB weight limit. FP16 is the recommended precision for L4 GPU inference — it's both smaller and faster.

### Setting up your training environment

Only install the packages you need, with matching versions:

```bash
# YOLOv8 training
pip install ultralytics==8.1.0

# torchvision detector
pip install torch==2.6.0 torchvision==0.21.0

# Custom model with timm backbone
pip install torch==2.6.0 timm==0.9.12

# For GPU training, add the CUDA index:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

The sandbox has a GPU (NVIDIA L4 with CUDA 12.4), so GPU-trained weights run natively — no `map_location="cpu"` needed. Your code should auto-detect with `torch.cuda.is_available()`.

**Train anywhere:** You can train on any hardware — your laptop CPU, a cloud GPU, Google Colab, GCP VMs, etc. Models trained on any platform will run on the sandbox GPU. Use `state_dict` saves (not full model saves) or ONNX export for maximum compatibility.

### Version compatibility

| Risk | What happens | Fix |
|---|---|---|
| ultralytics 8.2+ weights on 8.1.0 | Model class changed, load fails | Pin `ultralytics==8.1.0` or export to ONNX |
| torch 2.7+ full model save on 2.6.0 | May reference newer operators | Use `torch.save(model.state_dict())`, not `torch.save(model)` |
| timm 1.0+ weights on 0.9.12 | Layer names changed, load fails | Pin `timm==0.9.12` or export to ONNX |
| ONNX opset > 20 | onnxruntime 1.20.0 can't load it | Export with `opset_version=17` |

### Recommended weight formats

| Approach | Format | When to use |
|---|---|---|
| ONNX export | `.onnx` | Universal — any framework, 2-3x faster on CPU |
| ultralytics .pt (pinned 8.1.0) | `.pt` | Simple YOLOv8/RT-DETR workflow |
| state_dict + model class | `.pt` | Custom architectures with standard PyTorch ops |
| safetensors | `.safetensors` | Safe loading, no pickle, fast |

## Security Restrictions

The following imports are blocked by the security scanner:
- `os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`
- `pickle`, `marshal`, `shelve`, `shutil`
- `yaml` (use `json` for config files instead)
- `requests`, `urllib`, `http.client`
- `multiprocessing`, `threading`, `signal`, `gc`
- `code`, `codeop`, `pty`

The following calls are blocked:
- `eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names

Also blocked: ELF/Mach-O/PE binaries, symlinks, path traversal.

Use `pathlib` instead of `os` for file operations. Use `json` instead of `yaml` for config files.

## Creating Your Zip

`run.py` must be at the **root** of the zip — not inside a subfolder. This is the most common submission error.

**Linux / macOS (Terminal):**
```bash
cd my_submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```

**Windows (PowerShell):**
```powershell
cd my_submission
Compress-Archive -Path .\* -DestinationPath ..\submission.zip
```

Do **not** right-click a folder and use "Compress" (macOS) or "Send to → Compressed folder" (Windows) — both nest files inside a subfolder.

**Verify your zip:**
```bash
unzip -l submission.zip | head -10
```
You should see `run.py` directly — not `my_submission/run.py`.
# Tripletex — Endpoint Specification

Your agent must expose a single HTTPS endpoint that accepts POST requests.

## `/solve` Endpoint

**Method:** POST
**Content-Type:** application/json
**Timeout:** 300 seconds (5 minutes)

## Request Format

```json
{
  "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
  "files": [
    {
      "filename": "faktura.pdf",
      "content_base64": "JVBERi0xLjQg...",
      "mime_type": "application/pdf"
    }
  ],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | The task in Norwegian natural language |
| `files` | array | Attachments (PDFs, images) — may be empty |
| `files[].filename` | string | Original filename |
| `files[].content_base64` | string | Base64-encoded file content |
| `files[].mime_type` | string | MIME type (`application/pdf`, `image/png`, etc.) |
| `tripletex_credentials.base_url` | string | Proxy API URL — use this instead of the standard Tripletex URL |
| `tripletex_credentials.session_token` | string | Session token for authentication |

## Response Format

Return this JSON when your agent has finished executing the task:

```json
{
  "status": "completed"
}
```

## Authentication

Your agent authenticates with the Tripletex API using **Basic Auth**:

- **Username:** `0` (zero)
- **Password:** the `session_token` value from the request

```python
import requests

response = requests.get(
    f"{base_url}/employee",
    auth=("0", session_token),
    params={"fields": "id,firstName,lastName,email"}
)
```

## API Key (Optional)

If you set an API key when submitting your endpoint, we send it as a Bearer token:

```
Authorization: Bearer <your-api-key>
```

Use this to protect your endpoint from unauthorized access.

## Requirements

- Endpoint must be **HTTPS**
- Must respond within **5 minutes** (300 seconds)
- Must return `{"status": "completed"}` with HTTP 200
- All Tripletex API calls must go through the provided `base_url` (proxy)

## Tripletex API Reference

All standard Tripletex v2 endpoints are available through the proxy. Common endpoints:

| Endpoint | Methods | Description |
|----------|---------|-------------|
| `/employee` | GET, POST, PUT | Manage employees |
| `/customer` | GET, POST, PUT | Manage customers |
| `/product` | GET, POST | Manage products |
| `/invoice` | GET, POST | Create and query invoices |
| `/order` | GET, POST | Manage orders |
| `/travelExpense` | GET, POST, PUT, DELETE | Travel expense reports |
| `/project` | GET, POST | Manage projects |
| `/department` | GET, POST | Manage departments |
| `/ledger/account` | GET | Query chart of accounts |
| `/ledger/posting` | GET | Query ledger postings |
| `/ledger/voucher` | GET, POST, DELETE | Manage vouchers |

## API Tips

- Use the `fields` parameter to select specific fields: `?fields=id,firstName,lastName,*`
- Use `count` and `from` for pagination: `?from=0&count=100`
- POST/PUT requests take JSON body
- DELETE requests use the ID in the URL path: `DELETE /employee/123`
- List responses are wrapped: `{"fullResultSize": N, "values": [...]}`
# Tripletex — Examples

## Minimal `/solve` Endpoint

```python
import base64
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    base_url = creds["base_url"]
    token = creds["session_token"]
    auth = ("0", token)

    for f in files:
        data = base64.b64decode(f["content_base64"])
        Path(f["filename"]).write_bytes(data)

    # TODO: Use an LLM to interpret the prompt and execute
    # the appropriate Tripletex API calls

    return JSONResponse({"status": "completed"})
```

Run with:

```bash
pip install fastapi uvicorn requests
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expose locally via HTTPS for testing:

```bash
npx cloudflared tunnel --url http://localhost:8000
```

## Tripletex API Examples

### List employees

```python
resp = requests.get(
    f"{base_url}/employee",
    auth=auth,
    params={"fields": "id,firstName,lastName,email"}
)
employees = resp.json()["values"]
```

### Create a customer

```python
resp = requests.post(
    f"{base_url}/customer",
    auth=auth,
    json={
        "name": "Acme AS",
        "email": "post@acme.no",
        "isCustomer": True
    }
)
customer_id = resp.json()["value"]["id"]
```

### Create an invoice

```python
today = "2026-03-03"
resp = requests.post(
    f"{base_url}/invoice",
    auth=auth,
    json={
        "invoiceDate": today,
        "invoiceDueDate": today,
        "customer": {"id": customer_id},
        "orders": [{"id": order_id}]
    }
)
```

### Search for a specific entity

```python
resp = requests.get(
    f"{base_url}/customer",
    auth=auth,
    params={
        "name": "Acme",
        "fields": "id,name,email",
        "count": 10
    }
)
matches = resp.json()["values"]
```

## Building an Effective Agent

1. **Parse the prompt** — Use an LLM to extract the task type, entity names, field values, and relationships from the Norwegian prompt
2. **Handle files** — Some tasks include PDFs with invoices, contracts, or expense reports. Decode from base64 and extract relevant data
3. **Map to API calls** — Determine which Tripletex endpoints to call and in what order. Some tasks require creating prerequisites first
4. **Verify your work** — After creating entities, query back to confirm they exist with correct values
5. **Handle errors** — Tripletex returns detailed error messages. Parse them to retry with corrections

## Common Task Patterns

| Pattern | Example | API Flow |
|---------|---------|----------|
| Create single entity | "Create employee Ola Nordmann" | POST /employee |
| Create with linking | "Create invoice for customer" | GET /customer → POST /order → POST /invoice |
| Modify existing | "Add phone to contact" | GET /customer → PUT /customer/{id} |
| Delete/reverse | "Delete travel expense" | GET /travelExpense → DELETE /travelExpense/{id} |
| Multi-step setup | "Register payment" | POST /customer → POST /invoice → POST /payment |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Wrong auth format | Use Basic Auth with username `0` and session token as password |
| 404 Not Found | Wrong endpoint path | Check the Tripletex v2 API docs for correct paths |
| 422 Validation Error | Missing required fields | Read error message — it specifies which fields are required |
| Empty `values` array | No results found | Check search parameters, try broader search |
| Timeout (5 min) | Agent too slow | Optimize API calls, reduce unnecessary requests |

## Tips

- The Tripletex sandbox starts empty — you may need to create prerequisites (customer, product) before creating invoices
- Use `?fields=*` to see all available fields on an entity
- Some tasks require enabling modules first (e.g., department accounting)
- Norwegian characters (æ, ø, å) work fine in API requests — send as UTF-8
- All API calls through the proxy are logged — use them for debugging in the submissions view
- Prompts come in 7 languages (nb, en, es, pt, nn, de, fr) — your agent should handle all of them

## Optimizing for Efficiency

Your score can go above 1.0 if you achieve perfect correctness with minimal API calls and zero errors. Higher-tier tasks have higher score ceilings (up to 6.0 for Tier 3). Tips:

- **Plan before calling** — Parse the prompt fully before making API calls. Understand what needs to be created/modified before starting
- **Avoid trial-and-error** — Every 4xx error (400, 404, 422) reduces your efficiency bonus. Validate inputs before sending
- **Minimize GET calls** — Don't fetch entities you don't need. If you created something, you already know its ID from the response
- **Batch where possible** — Some Tripletex endpoints accept lists. Use them instead of multiple individual calls
- **Read error messages** — If a call fails, the Tripletex error message tells you exactly what's wrong. Fix it in one retry, not several
# Tripletex — AI Accounting Agent

Build an AI agent that completes accounting tasks in Tripletex. You receive a task prompt (in one of 7 languages), use the Tripletex API to execute it, and get scored on correctness and efficiency.

## How It Works

1. Submit your HTTPS endpoint URL on the platform
2. We provision a fresh Tripletex sandbox account
3. We send a randomly selected accounting task to your `/solve` endpoint
4. Your agent reads the prompt, optionally processes attached files (PDFs, images)
5. Your agent calls the Tripletex API via a proxy to complete the task
6. We verify the result field-by-field against expected values
7. Your score updates on the rolling leaderboard

Each submission gets a brand new Tripletex account — you always start from scratch.

## Key Facts

| | |
|---|---|
| Task types | 30 different accounting tasks |
| Variants | 56 per task (7 languages × 8 data sets) |
| Language | Prompts in Norwegian, English, Spanish, Portuguese, Nynorsk, German, French |
| Timeout | 5 minutes per submission |
| API | [Tripletex v2 REST API](https://kkpqfuj-amager.tripletex.dev/v2-docs/) via authenticated proxy |
| Scoring | Field-by-field checks + efficiency bonus, best score per task kept |
| Score range | 0.0 (failed) — up to 6.0 (perfect Tier 3 + best efficiency) |
| Files | Some tasks include PDF or image attachments |

## Quick Start

1. Build a `/solve` endpoint that accepts POST requests with a task prompt and Tripletex credentials
2. Use an LLM to interpret the Norwegian prompt and decide which API calls to make
3. Call the Tripletex API using the provided proxy URL and session token
4. Return `{"status": "completed"}` when done
5. Submit your endpoint URL at `https://app.ainm.no/submit/tripletex`

## Task Categories

Your agent will encounter tasks like:

- **Employees** — Create employees, set roles, update contact info
- **Customers & Products** — Register customers, create products
- **Invoicing** — Create invoices, register payments, issue credit notes
- **Travel Expenses** — Register or delete travel expense reports
- **Projects** — Create projects linked to customers
- **Corrections** — Delete or reverse incorrect entries
- **Departments** — Create departments, enable accounting modules

Tasks range from simple single-API-call operations to multi-step workflows requiring several resources to be created and linked together.
# Tripletex — Sandbox Account

Every team gets a free Tripletex sandbox account to explore the API and web interface before submitting to the competition.

## Getting Your Sandbox

1. Go to the **Tripletex submission page** on the platform
2. Click **"Get Sandbox Account"**
3. Your sandbox is provisioned instantly

You'll receive:
- **Tripletex UI URL** — log in and explore the accounting interface
- **API base URL** — call the Tripletex v2 REST API directly
- **Session token** — authenticate your API calls

## Logging Into the Web UI

1. Go to `https://kkpqfuj-amager.tripletex.dev`
2. Enter the email shown on your sandbox card
3. Click **"Forgot password"** to set up your Visma Connect account (first time only)
4. Set a password and log in

Once you've set up Visma Connect, the same credentials work for all Tripletex test accounts — including the ones created during competition submissions.

## Using the API

Authenticate with **Basic Auth** using `0` as username and the session token as password:

```python
import requests

BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SESSION_TOKEN = "your-session-token-here"

# List employees
response = requests.get(
    f"{BASE_URL}/employee",
    auth=("0", SESSION_TOKEN),
    params={"fields": "id,firstName,lastName,email"}
)
print(response.json())

# Create a customer
response = requests.post(
    f"{BASE_URL}/customer",
    auth=("0", SESSION_TOKEN),
    json={
        "name": "Test Customer AS",
        "email": "test@example.com",
        "isCustomer": True,
    }
)
print(response.json())
```

```bash
# curl example
curl -u "0:your-session-token-here" \
  "https://kkpqfuj-amager.tripletex.dev/v2/employee?fields=id,firstName,lastName"
```

## What You Can Do

The sandbox is a full Tripletex test environment. Use it to:

- **Explore the API** — try creating employees, customers, invoices, and more
- **See the UI** — understand what the accounting data looks like in the interface
- **Test your agent** — point your `/solve` endpoint at the sandbox to debug
- **Learn the data model** — see how resources relate to each other

## Key Differences from Competition

| | Sandbox | Competition |
|---|---|---|
| Account | Persistent, yours to keep | Fresh account per submission |
| API access | Direct to Tripletex | Via authenticated proxy |
| Data | Accumulates over time | Starts empty each time |
| Scoring | None | Automated field-by-field |

## Tips

- Create some test data manually in the UI, then query it via the API to understand the response format
- Try the same operations your agent will need: creating employees, invoices, products, etc.
- The sandbox token expires March 31, 2026
- Each team gets one sandbox — all team members share it
# Tripletex — Scoring

## Field-by-Field Verification (Correctness)

After your agent responds, we query the Tripletex API to verify what was created or modified. Each task has specific checks worth different point values.

Example for a "Create employee" task (max 10 points):

| Check | Points |
|-------|--------|
| Employee found | 2 |
| Correct first name | 1 |
| Correct last name | 1 |
| Correct email | 1 |
| Administrator role assigned | 5 |

The raw score is normalized to 0–1: `correctness = points_earned / max_points` (e.g., 8/10 = 0.8).

## Tier Multiplier

Each task has a difficulty tier that multiplies your correctness score:

| Tier | Multiplier | Example tasks |
|------|-----------|---------------|
| Tier 1 | ×1 | Create employee, create customer |
| Tier 2 | ×2 | Create invoice, register payment |
| Tier 3 | ×3 | Complex multi-step workflows |

So a perfect score on a Tier 2 task = `1.0 × 2 = 2.0` base score.

## Efficiency Bonus

If your agent achieves a **perfect correctness score** (1.0), you receive an efficiency bonus that can up to **double** your tier score.

Two factors determine the bonus:

**Call efficiency** — How many API calls did your agent make compared to the best known solution for this task? Fewer calls = higher bonus.

**Error cleanliness** — How many of your API calls resulted in 4xx errors (400, 404, 422, etc.)? Errors reduce the bonus. An agent that gets it right without trial-and-error is rewarded.

| Scenario (Tier 2 task) | Score |
|------------------------|-------|
| Failed all checks | 0.0 |
| 80% of checks passed | 1.6 |
| Perfect, but many errors and extra calls | ~2.1 |
| Perfect, efficient, a few errors | ~2.6 |
| Perfect, best-in-class efficiency, zero errors | 4.0 |

The efficiency bonus only applies to perfect submissions. Non-perfect submissions score `correctness × tier`.

**Efficiency benchmarks are recalculated periodically.** As teams find more efficient solutions, the bar rises for everyone. Your best score per task is recalculated against current benchmarks every 12 hours.

## Best Score Per Task

Your score per task is your **all-time best**. Bad runs never lower your score — only improvements count.

- One good run is enough to lock in a score
- You can always improve by submitting again
- Focus on building a better agent, not grinding to recover from bad luck
- Each of the 30 tasks tracks independently

## Leaderboard

**Total leaderboard score** = sum of best scores across all task types.

The more task types your agent handles well, the higher your potential score.

## Task Assignment

Each submission receives one task, weighted toward tasks you've attempted less. Over many submissions, you'll encounter all task types. Tasks are grouped into three tiers:

- **Tier 1** — foundational tasks (e.g., create employee, create customer, create invoice)
- **Tier 2** — multi-step workflows (e.g., invoice with payment, credit notes, project billing)
- **Tier 3** — complex scenarios (e.g., bank reconciliation from CSV, error correction in ledger, year-end closing)

Each task has 56 unique variants (7 languages × 8 data sets), so you'll rarely see the same prompt twice.

### Tier Release Schedule

Tasks are released in tiers throughout the competition:

- **Tier 1** — available from competition start
- **Tier 2** — opens early Friday. Check this page for updates.
- **Tier 3** — opens early Saturday. Check this page for updates.

This gives you time to build a solid agent on simpler tasks before tackling the harder ones.

## Rate Limits

| Limit | Verified teams | Unverified teams |
|-------|---------------|-----------------|
| Concurrent submissions | 3 | 1 |
| Per task per day | 5 | 2 |
# Master Attack Plan: NM i AI 2026

This document outlines our unified strategy for dominating the Norwegian AI Championship across its three main cases. As per the repository guidelines, our overarching advantage will be leveraging GCP aggressively for training, evaluation, and orchestration. 

We assume the championship score is split equally across the three cases.

---

## 1. Tripletex — AI Accounting Agent
**Goal:** Build a high-performance, multi-lingual AI agent that maps natural language requests to Tripletex API actions efficiently.
**Task:** We host a `/solve` endpoint. The platform sends a task (in one of 7 languages), and we must execute the appropriate Tripletex API calls via a proxy within a 5-minute timeout.
**Scoring:** Field-by-field correctness + efficiency bonus (fewest necessary API calls / fastest execution).

### Attack Strategy:
* **LLM Orchestration:** Use a fast, highly capable LLM (like Gemini 1.5 Pro/Flash on GCP) to act as the reasoning engine.
* **Tool Calling / Function Calling:** Expose the Tripletex v2 REST API to the LLM as precise tools (e.g., `create_employee`, `issue_invoice`, `register_payment`). 
* **Translation / Normalization Layer:** Since prompts can be in 7 languages (including Nynorsk), the first step of the pipeline should normalize the intent into a structured JSON representation before executing API calls, ensuring the agent doesn't get confused by language nuances.
* **Sandbox Iteration:** Aggressively use the provided Tripletex sandbox account to map out the exact required JSON schemas for the 30 different accounting tasks, building a robust test suite.

---

## 2. NorgesGruppen Data — Object Detection
**Goal:** Detect and classify 356 grocery product categories on store shelves with high precision.
**Task:** Train an object detection model on the provided COCO dataset (248 images, ~22.7k bounding boxes) and product reference images (327 products). Upload a `.zip` containing `run.py` and model weights.
**Scoring:** 70% detection (finding the box) + 30% classification (identifying the exact product) using mAP@0.5.

### Attack Strategy:
* **Model Selection:** Use a state-of-the-art YOLO variant (YOLOv8, YOLOv9, or YOLOv10) for its excellent mAP@0.5 performance and ease of deployment. 
* **Data Augmentation:** The dataset is small (248 images). We must heavily augment the data using the provided product reference images (pasting them onto different backgrounds, scaling, varying lighting).
* **Two-Stage Pipeline (Optional but recommended):** If classification is difficult due to small bounding boxes, we can train YOLO strictly for *detection* (class-agnostic object localization) and use a secondary lightweight classifier (like MobileNet or ResNet) cropped on the bounding boxes to nail the 356 specific product classes.
* **Compute:** Train heavily on GCP GPUs to iterate on hyperparameter tuning before zipping and submitting.

---

## 3. Astar Island — Norse World Prediction
**Goal:** Predict the final state (50 years out) of a stochastic, procedurally generated grid world.
**Task:** For a given round, we get 50 queries (each revealing a 15x15 viewport of a 40x40 map) shared across 5 seeds. We must submit a W×H×6 probability tensor predicting the likelihood of 6 terrain classes for every cell.
**Scoring:** Entropy-weighted KL divergence between our predicted distribution and the ground truth.

### Attack Strategy:
* **Optimal Query Strategy (Active Learning):** 50 queries of 15x15 windows means we can only see a fraction of the world. We must query regions with the highest *uncertainty* or regions that contain "catalyst" terrain types (e.g., settlements or factions) that heavily influence the rest of the map.
* **Simulation/Surrogate Modeling:** We should train a neural cellular automata (NCA) or a spatio-temporal Graph Neural Network (GNN) to act as a surrogate simulator. 
* **Ensemble Predictions:** Since the game is stochastic, our surrogate model should output a probability distribution (or we run our own surrogate simulation thousands of times Monte-Carlo style) to generate the final W×H×6 probability tensor, rather than trying to predict a single deterministic outcome.

---

## Next Steps
1. **Tripletex:** Set up the FastAPI `/solve` endpoint boilerplate and connect it to the LLM tool-calling pipeline.
2. **NorgesGruppen:** Download the `NM_NGD_coco_dataset.zip` and kick off a baseline YOLO training run on GCP to establish our floor mAP score.
3. **Astar Island:** Build a script to systematically consume the 50 queries to maximize map coverage and start training a surrogate transition model.