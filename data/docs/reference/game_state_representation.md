# Game State Representation

Zèrtz3D keeps the board in two NumPy-compatible arrays shared between Python and the Rust PyO3 extension:

- **Spatial state** (`spatial_state`): 3D array `(L, H, W)`
- **Global state** (`global_state`): 1D array `(10,)`

Python (`game/zertz_board.py`) and Rust (`rust/src/game.rs`, `rust/src/board.rs`) agree on these conventions, and the data is passed back and forth via PyO3 in the `hiivelabs_zertz_mcts` crate.

## Spatial State Layout

- `L` (layers) = `4 * t + 1`
  - First 4 layers hold the current position (rings + 3 marbles)
  - Each additional block of 4 layers stores historical board states (used by training/inference pipelines that expect time windows). The Rust MCTS itself only relies on the current slice; history is preserved for parity with the Python board API.
  - The final layer is a capture flag used to enforce forced chain captures.
- `H`/`W` is the board width (7, 8, or 9 for 37/48/61 rings).

### Current-State Layers
| Layer | Meaning                          | Python accessor           | Rust equivalent   |
|-------|----------------------------------|---------------------------|-------------------|
| 0     | Ring present (1) / removed (0)   | `RING_LAYER`              | `board::BoardConfig.ring_layer` |
| 1     | White marbles                    | `MARBLE_LAYERS[0]`        | `BoardConfig::marble_to_layer["w"]` |
| 2     | Gray marbles                     |                           | `"g"` -> layer 2  |
| 3     | Black marbles                    |                           | `"b"` -> layer 3  |
| 4     | Capture flag (chain continuation)| `CAPTURE_LAYER`           | `capture_layer` (Rust) |

The capture layer mirrors Python’s behaviour: when a capture is made, the landing square gets flagged if further captures are mandatory; otherwise it remains zero.

## Global State Layout (length 10)

| Index | Meaning                       | Python constant        | Rust constant             |
|-------|-------------------------------|------------------------|---------------------------|
| 0     | White marbles in supply      | `SUPPLY_W`             | `BoardConfig.supply_w`    |
| 1     | Gray marbles in supply       | `SUPPLY_G`             | `BoardConfig.supply_g`    |
| 2     | Black marbles in supply      | `SUPPLY_B`             | `BoardConfig.supply_b`    |
| 3     | P1 captured white count      | `P1_CAP_W`             | `BoardConfig.p1_cap_w`    |
| 4     | P1 captured gray             | `P1_CAP_G`             | `BoardConfig.p1_cap_g`    |
| 5     | P1 captured black            | `P1_CAP_B`             | `BoardConfig.p1_cap_b`    |
| 6     | P2 captured white            | `P2_CAP_W`             | `BoardConfig.p2_cap_w`    |
| 7     | P2 captured gray             | `P2_CAP_G`             | `BoardConfig.p2_cap_g`    |
| 8     | P2 captured black            | `P2_CAP_B`             | `BoardConfig.p2_cap_b`    |
| 9     | Current player (0 or 1)      | `CUR_PLAYER`           | `BoardConfig.cur_player`  |

Values are floats to match `np.float32` arrays but treated as integers.

## Example Usage

```python
from game.zertz_board import ZertzBoard
board = ZertzBoard(37)

board.state[board.RING_LAYER, y, x] = 1  # ring present
board.state[board.MARBLE_LAYERS[0], y, x] = 1  # place white marble
board.global_state[board.SUPPLY_W] -= 1  # update supply
board.state[board.CAPTURE_LAYER, y, x] = 1  # forced chain capture flag

from hiivelabs_zertz_mcts import BoardState, MCTSSearch
import numpy as np

bs = BoardState(
    np.asarray(board.state, dtype=np.float32),
    np.asarray(board.global_state, dtype=np.float32),
    rings=37,
    t=board.t,
)
search = MCTSSearch()
action_type, action_payload = search.search(
    bs.get_spatial_state(),
    bs.get_global_state() if hasattr(bs, "get_global_state") else bs.get_global(),
    rings=37,
    iterations=1000,
)
```

## Coordinate Layout Example (37-Ring Board)

Y/X indexing matches the `(H, W)` axes of the spatial array. The following grid shows how `(y, x)` indices map to Zèrtz coordinates on the 37-ring board (`width = 7`). Empty array slots are unused padding (`..`).

```
y\x  0   1   2   3   4   5   6
 0  A4  B5  C6  D7  ..  ..  ..
 1  A3  B4  C5  D6  E6  ..  ..
 2  A2  B3  C4  D5  E5  F5  ..
 3  A1  B2  C3  D4  E4  F4  G4
 4  ..  B1  C2  D3  E3  F3  G3
 5  ..  ..  C1  D2  E2  F2  G2
 6  ..  ..  ..  D1  E1  F1  G1
```

- The array origin `(y=0, x=0)` corresponds to coordinate `A4`.
- Moving right (`+x`) increments the column letter; moving down (`+y`) steps toward smaller-numbered ranks.
- Positions marked `..` are outside the playable hex and remain zero across all layers.
- This layout is consistent across Python and Rust; both engines flatten indices using `dst_flat = y * width + x`.

For the 48- and 61-ring boards the same convention applies with widths `8` and `9`, respectively, yielding larger trapezoidal hex layouts.
