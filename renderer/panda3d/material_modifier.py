from dataclasses import dataclass
from typing import Tuple, Iterable


@dataclass(frozen=True)
class MaterialModifier:
    highlight_color: Tuple[float, float, float, float]
    emission_color: Tuple[float, float, float, float]

    def __init__(
        self,
        highlight_color: Tuple[float, float, float, float],
        emission_color: Tuple[float, float, float, float],
    ):
        object.__setattr__(self, "highlight_color", highlight_color)
        object.__setattr__(self, "emission_color", emission_color)

    @classmethod
    def blend_mods(
        cls, this: "MaterialModifier", other: "MaterialModifier", blend_ratio: float
    ) -> "MaterialModifier":
        return MaterialModifier.blend_vectors_with_vectors(
            this.highlight_color,
            this.emission_color,
            other.highlight_color,
            other.emission_color,
            blend_ratio,
        )

    @classmethod
    def blend_mod_with_vectors(
        cls,
        this: "MaterialModifier",
        highlight_color: Iterable[float],
        emission_color: Iterable[float],
        blend_ratio: float,
    ) -> "MaterialModifier":
        return MaterialModifier.blend_vectors_with_vectors(
            this.highlight_color,
            this.emission_color,
            highlight_color,
            emission_color,
            blend_ratio,
        )

    @classmethod
    def blend_vectors_with_mod(
        cls,
        highlight_color: Iterable[float],
        emission_color: Iterable[float],
        other: "MaterialModifier",
        blend_ratio: float,
        blend_mask: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> "MaterialModifier":
        return MaterialModifier.blend_vectors_with_vectors(
            highlight_color,
            emission_color,
            other.highlight_color,
            other.emission_color,
            blend_ratio,
            blend_mask,
        )

    @classmethod
    def blend_vectors_with_vectors(
        cls,
        this_highlight_color: Iterable[float],
        this_emission_color: Iterable[float],
        other_highlight_color: Iterable[float],
        other_emission_color: Iterable[float],
        blend_ratio: float,
        blend_mask: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> "MaterialModifier":
        """Blend two color vectors with optional per-channel masking.

        Args:
            this_highlight_color: Source highlight color (RGBA)
            this_emission_color: Source emission color (RGBA)
            other_highlight_color: Target highlight color (RGBA)
            other_emission_color: Target emission color (RGBA)
            blend_ratio: Blend factor (0.0 = source, 1.0 = target)
            blend_mask: Per-channel blend mask (R, G, B, A). True = blend, False = use target
                       Default: (True, True, True, True) - blend all channels

        Returns:
            MaterialModifier with blended colors
        """
        def _convert(t, o):
            for i in range(4):
                if blend_mask[i]:
                    # Blend this channel
                    yield t[i] + (o[i] - t[i]) * blend_ratio
                else:
                    # Don't blend, use target value directly
                    yield o[i]

        new_highlight_color = tuple(
            _convert(this_highlight_color, other_highlight_color)  # noqa
        )
        new_emission_color = tuple(_convert(this_emission_color, other_emission_color))  # noqa
        return MaterialModifier(new_highlight_color, new_emission_color)  # noqa
