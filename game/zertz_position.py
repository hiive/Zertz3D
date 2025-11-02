"""Unified representations and collections for Zertz board positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from game.zertz_board import ZertzBoard


@dataclass(frozen=True)
class ZertzPosition:
    """Represents a single board coordinate across all systems."""

    board: "ZertzBoard"
    y: int
    x: int
    _label: str | None = field(default=None, init=False, repr=False)
    _axial: Tuple[int, int] | None = field(default=None, init=False, repr=False)
    _cartesian: Tuple[float, float] | None = field(default=None, init=False, repr=False)

    @property
    def yx(self) -> Tuple[int, int]:
        return self.y, self.x

    @property
    def label(self) -> str:
        if self._label is None:
            object.__setattr__(self, "_label", self.board.yx_to_label(self.yx))
        return self._label  # type: ignore[attr-defined]

    @property
    def axial(self) -> Tuple[int, int]:
        if self._axial is None:
            collection = self.board.positions
            collection.ensure()
            axial = collection.axial_for_yx(self.yx)
            if axial is None:
                raise ValueError(f"Position {self.yx} is not a valid ring coordinate")
            object.__setattr__(self, "_axial", axial)
        return self._axial  # type: ignore[attr-defined]

    @property
    def cartesian(self) -> Tuple[float, float]:
        if self._cartesian is None:
            collection = self.board.positions
            collection.ensure()
            cart = collection.cartesian_for_yx(self.yx)
            if cart is None:
                raise ValueError(f"Position {self.yx} is not a valid ring coordinate")
            object.__setattr__(self, "_cartesian", cart)
        return self._cartesian  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"ZertzPosition('{self.label}', y={self.y}, x={self.x})"

    @classmethod
    def from_yx(cls, board: "ZertzBoard", yx: Tuple[int, int]) -> "ZertzPosition":
        return cls(board=board, y=yx[0], x=yx[1])

    @classmethod
    def from_label(cls, board: "ZertzBoard", label: str) -> "ZertzPosition":
        y, x = board.label_to_yx(label)
        pos = cls.from_yx(board, (y, x))
        object.__setattr__(pos, "_label", label)
        return pos

    @classmethod
    def from_axial(cls, board: "ZertzBoard", axial: Tuple[int, int]) -> "ZertzPosition":
        board.positions.ensure()
        pos = board.positions.get_by_axial(axial)
        if pos is None:
            raise ValueError(f"Axial coordinate {axial} is not on this board")
        return pos


class ZertzPositionCollection:
    """Lazy-built caches that unify lookup across coordinate systems."""

    def __init__(self, board: "ZertzBoard") -> None:
        self.board = board
        self.invalidate()

    def invalidate(self) -> None:
        """Mark the cache dirty so it will rebuild on next access."""
        self._built = False
        self._by_yx: Dict[Tuple[int, int], ZertzPosition] = {}
        self._by_label: Dict[str, ZertzPosition] = {}
        self._by_axial: Dict[Tuple[int, int], ZertzPosition] = {}
        self._cartesian: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._yx_to_ax: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._ax_to_yx: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def ensure(self) -> None:
        """Ensure the caches are populated."""
        if self._built:
            return
        self._build()

    def _build(self) -> None:
        """Build position maps by delegating to Rust for axial coordinate computation."""
        board = self.board

        # Determine the mask of valid positions
        if board.board_layout is not None:
            mask = board.board_layout.astype(bool)
        elif board.letter_layout is not None:
            mask = board.letter_layout != ""
        else:
            mask = board.state[board.RING_LAYER] != 0

        ys, xs = np.where(mask)
        if len(ys) == 0:
            self._built = True
            return

        # Delegate axial coordinate computation to Rust
        from hiivelabs_mcts import build_axial_maps, BoardConfig

        config = BoardConfig.standard_config(board.rings, t=board.t)
        layout = mask.tolist()  # Convert numpy bool array to Python list
        yx_to_ax_rust, ax_to_yx_rust = build_axial_maps(config, layout)

        # Convert Rust maps (which use i32) to Python tuples
        # Also compute cartesian coordinates for positions
        sqrt3 = np.sqrt(3)
        scale = 3 if board.rings == board.MEDIUM_BOARD_48 else 1
        board._coord_scale = scale

        # Clear existing data
        self._by_yx.clear()
        self._by_label.clear()
        self._by_axial.clear()
        self._cartesian.clear()
        self._yx_to_ax.clear()
        self._ax_to_yx.clear()

        # Copy Rust maps to our internal dictionaries
        for (y, x), (q, r) in yx_to_ax_rust.items():
            label = board._compute_label((y, x))
            if not label:
                continue

            # Compute cartesian coordinates from centered axial coordinates
            # Note: q, r are already centered and scaled by Rust
            q_unscaled = q / scale
            r_unscaled = r / scale
            xc = sqrt3 * (q_unscaled + r_unscaled / 2.0)
            yc = 1.5 * r_unscaled
            cart = (xc, yc)

            axial = (q, r)  # Already centered and scaled by Rust

            # Create position object
            pos = ZertzPosition(board, y, x)
            object.__setattr__(pos, "_label", label)
            object.__setattr__(pos, "_axial", axial)
            object.__setattr__(pos, "_cartesian", cart)

            # Store in lookup dictionaries
            self._by_yx[(y, x)] = pos
            self._by_label[label] = pos
            self._by_axial[axial] = pos
            self._cartesian[(y, x)] = cart
            self._yx_to_ax[(y, x)] = axial
            self._ax_to_yx[axial] = (y, x)

        self._built = True

    def __iter__(self) -> Iterable[ZertzPosition]:
        return iter(self.by_yx.values())

    def get_by_yx(self, index: Tuple[int, int]) -> ZertzPosition | None:
        self.ensure()
        return self._by_yx.get(index)

    def get_by_label(self, label: str) -> ZertzPosition | None:
        self.ensure()
        return self._by_label.get(label.upper())

    def get_by_axial(self, axial: Tuple[int, int]) -> ZertzPosition | None:
        self.ensure()
        return self._by_axial.get(axial)

    @property
    def by_yx(self) -> Dict[Tuple[int, int], ZertzPosition]:
        self.ensure()
        return self._by_yx

    @property
    def by_label(self) -> Dict[str, ZertzPosition]:
        self.ensure()
        return self._by_label

    @property
    def by_axial(self) -> Dict[Tuple[int, int], ZertzPosition]:
        self.ensure()
        return self._by_axial

    def axial_for_yx(self, index: Tuple[int, int]) -> Tuple[int, int] | None:
        self.ensure()
        return self._yx_to_ax.get(index)

    def cartesian_for_yx(self, index: Tuple[int, int]) -> Tuple[float, float] | None:
        self.ensure()
        return self._cartesian.get(index)

    def label_for_yx(self, index: Tuple[int, int]) -> str:
        self.ensure()
        pos = self._by_yx.get(index)
        return pos.label if pos else ""

    @property
    def yx_to_ax(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        self.ensure()
        return self._yx_to_ax

    @property
    def ax_to_yx(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        self.ensure()
        return self._ax_to_yx
