"""Manager class for handling entity highlights (pulsing effects and context-based highlights)."""

import math
from queue import SimpleQueue, Empty
from typing import Optional, Callable

import numpy as np
from panda3d.core import Material, LVector4

from renderer.panda3d.material_modifier import MaterialModifier
from shared.materials_modifiers import PLACEMENT_HIGHLIGHT_MATERIAL_MOD


class HighlightingManager:
    """Manages highlight animations and context-based highlights.

    TODO: Remove anim_type parameter - it's vestigial from the old unified animation system.
    HighlightingManager only handles one type (pulsing highlights), so type discrimination is unnecessary.
    Should simplify by removing queue_animation() and having queue_highlight() directly enqueue items.
    """

    def __init__(self, renderer):
        """Initialize the highlighting manager.

        Args:
            renderer: PandaRenderer instance
        """
        self.renderer = renderer
        self.animation_queue = SimpleQueue()
        self.current_animations = []  # List of active highlight animation items
        self._animation_id_counter = 0  # For generating unique IDs
        self._context_highlights: dict[str, dict] = {}

    def queue_animation(self, anim_type="highlight", defer=0, **kwargs):
        """Add a highlight animation to the queue.

        Args:
            anim_type: Should be 'highlight' (for consistency with animation_manager)
            defer: Delay before starting (seconds)
            **kwargs: Type-specific parameters
                For 'highlight': entities (or rings), duration, material_mod
        """
        # TODO: Remove this method - only used internally by queue_highlight()
        anim_item = {"type": anim_type, "defer": defer, **kwargs}
        self.animation_queue.put(anim_item)

    def queue_highlight(
        self,
        rings,
        duration,
        material_mod=None,
        defer=0,
    ):
        """Add a highlight to the queue.

        Args:
            rings: List of position strings (e.g., ['A1', 'B2', 'C3'])
            duration: How long to show highlight in seconds
            material_mod: MaterialModifier object (defaults to PLACEMENT_HIGHLIGHT_MATERIAL_MOD)
            defer: Delay before starting (seconds)
        """
        # Use default material if none provided
        if material_mod is None:
            material_mod = PLACEMENT_HIGHLIGHT_MATERIAL_MOD

        self.queue_animation(
            anim_type="highlight",
            defer=defer,
            entities=rings,
            duration=duration,
            material_mod=material_mod,
        )

    def is_active(self):
        """Check if any highlight animations are active or queued."""
        return len(self.current_animations) > 0 or not self.animation_queue.empty()

    def clear(self):
        """Clear all highlight animations from queue and active list."""
        while not self.animation_queue.empty():
            try:
                self.animation_queue.get_nowait()
            except Empty:
                break

        # Clear all active animations and restore materials
        for anim_item in self.current_animations:
            self._clear_highlight(anim_item)
        self.current_animations.clear()

    def update(self, task):
        """Update all active highlight animations.

        Args:
            task: Panda3D task object with timing information

        Returns:
            List of completed highlight animation items
        """
        # Process highlight animation queue - add new highlight animations
        while not self.animation_queue.empty():
            try:
                anim_item = self.animation_queue.get_nowait()
                anim_type = anim_item.get("type", "highlight")

                # Generate unique ID
                self._animation_id_counter += 1
                anim_item["id"] = self._animation_id_counter

                # Set timing
                defer = anim_item.get("defer", 0)
                anim_item["insert_time"] = task.time
                anim_item["start_time"] = task.time + defer
                anim_item["end_time"] = anim_item["start_time"] + anim_item["duration"]

                # For highlights, apply immediately if start time has passed
                if task.time >= anim_item["start_time"]:
                    self._apply_highlight(anim_item)

                self.current_animations.append(anim_item)
            except Empty:
                pass

        # Update all active highlight animations
        to_remove = []
        for anim_item in self.current_animations:
            # For highlights, apply when start time is reached (if not already applied)
            if "original_materials" not in anim_item:
                if task.time >= anim_item["start_time"]:
                    self._apply_highlight(anim_item)

            # Check if animation hasn't started yet
            if task.time < anim_item["start_time"]:
                continue

            # Check if animation has ended
            if task.time >= anim_item["end_time"]:
                # Clear highlight
                self._clear_highlight(anim_item)
                to_remove.append(anim_item)
                continue

            # Update highlight blend (pulsing effect)
            original_materials = anim_item.get("original_materials", {})
            target_material_mod = anim_item.get(
                "target_material_mod", PLACEMENT_HIGHLIGHT_MATERIAL_MOD
            )
            duration = anim_item["duration"]
            start_time = anim_item["start_time"]
            elapsed_time = task.time - start_time

            # Use sine wave for smooth pulsing: fade in (0 to 1), fade out (1 to 0)
            # sin goes from 0 -> 1 -> 0 over the animation duration
            pulse_factor = math.sin((elapsed_time / duration) * math.pi)

            # Update material for each entity in the highlight
            for pos_str, mat_info in original_materials.items():
                (
                    original_mat,
                    entity_type,
                    original_color,
                    original_emission,
                    original_metallic,
                    original_roughness,
                ) = mat_info

                # Get the entity based on type
                entity = None
                if entity_type == "marble" and pos_str in self.renderer.pos_to_marble:
                    entity = self.renderer.pos_to_marble[pos_str]
                elif entity_type == "ring" and pos_str in self.renderer.pos_to_base:
                    entity = self.renderer.pos_to_base[pos_str]
                elif entity_type == "supply_marble":
                    try:
                        marble_id = int(pos_str.split(":", 1)[1])
                    except (IndexError, ValueError):
                        marble_id = None
                    if marble_id is not None:
                        entity = self.renderer._marble_registry.get(marble_id)
                elif entity_type == "captured_marble":
                    try:
                        marble_id = int(pos_str.split(":", 1)[1])
                    except (IndexError, ValueError):
                        marble_id = None
                    if marble_id is not None:
                        entity = self.renderer._marble_registry.get(marble_id)

                if entity is not None:
                    # Blend between original and target colors
                    blended_material_mod = MaterialModifier.blend_vectors_with_mod(
                        original_color,
                        original_emission,
                        target_material_mod,
                        pulse_factor,
                    )
                    blended_color = LVector4(*blended_material_mod.highlight_color)
                    blended_emission = LVector4(*blended_material_mod.emission_color)
                    # Create and apply blended material
                    blended_mat = Material()
                    blended_mat.setMetallic(original_metallic)
                    blended_mat.setRoughness(original_roughness)
                    blended_mat.setBaseColor(blended_color)
                    blended_mat.setEmission(blended_emission)
                    model = entity.model if hasattr(entity, "model") else entity
                    model.setMaterial(blended_mat, 1)

        # Remove completed highlight animations
        for anim_item in to_remove:
            self.current_animations.remove(anim_item)

        return to_remove

    def _apply_highlight(self, highlight_info):
        """Apply a highlight to the specified rings and/or marbles.

        Args:
            highlight_info: Dict with 'entities', 'duration', 'material_mod',
                          'start_time', 'end_time' (timing already set by update loop)
        """

        entities_list = highlight_info.get("entities", [])

        # Store original materials and what type was highlighted
        original_materials = {}

        for pos_str in entities_list:
            # Try to highlight marble first, then ring
            entity = None
            entity_type = None

            if pos_str in self.renderer.pos_to_marble:
                # Highlight the marble at this position
                entity = self.renderer.pos_to_marble[pos_str]
                entity_type = "marble"
            elif pos_str in self.renderer.pos_to_base:
                # Highlight the ring at this position
                entity = self.renderer.pos_to_base[pos_str]
                entity_type = "ring"
            elif isinstance(pos_str, str) and pos_str.startswith("supply:"):
                try:
                    marble_id = int(pos_str.split(":", 1)[1])
                except (IndexError, ValueError):
                    marble_id = None
                if marble_id is not None:
                    entity = self.renderer._marble_registry.get(marble_id)
                    if entity is not None:
                        entity_type = "supply_marble"
            elif isinstance(pos_str, str) and pos_str.startswith("captured:"):
                try:
                    marble_id = int(pos_str.split(":", 1)[1])
                except (IndexError, ValueError):
                    marble_id = None
                if marble_id is not None:
                    entity = self.renderer._marble_registry.get(marble_id)
                    if entity is not None:
                        entity_type = "captured_marble"

            if entity is not None:
                # For supply/captured marbles, entity is a model object with .model NodePath
                # For rings/board marbles, entity is already the NodePath
                model = entity.model if hasattr(entity, "model") else entity
                original_mat = model.getMaterial()

                # Store original material properties for blending
                if original_mat is not None:
                    original_color = original_mat.getBaseColor()
                    original_emission = original_mat.getEmission()
                    original_metallic = original_mat.getMetallic()
                    original_roughness = original_mat.getRoughness()
                else:
                    # Default properties if no material exists
                    original_color = LVector4(0.8, 0.8, 0.8, 1.0)
                    original_emission = LVector4(0.0, 0.0, 0.0, 1.0)
                    original_metallic = 0.9
                    original_roughness = 0.1

                original_materials[pos_str] = (
                    original_mat,
                    entity_type,
                    original_color,
                    original_emission,
                    original_metallic,
                    original_roughness,
                )

        # Store original materials and target colors for later blending
        highlight_info["original_materials"] = original_materials

        material_mod = highlight_info.get(
            "material_mod", PLACEMENT_HIGHLIGHT_MATERIAL_MOD
        )
        highlight_info["target_material_mod"] = material_mod

    def _clear_highlight(self, highlight_info):
        """Clear a highlight and restore original materials.

        Args:
            highlight_info: Dict with 'original_materials'
        """
        original_materials = highlight_info.get("original_materials", {})

        for pos_str, mat_info in original_materials.items():
            original_mat, entity_type, _, _, _, _ = mat_info

            # Get the entity based on type
            entity = None
            if entity_type == "marble" and pos_str in self.renderer.pos_to_marble:
                entity = self.renderer.pos_to_marble[pos_str]
            elif entity_type == "ring" and pos_str in self.renderer.pos_to_base:
                entity = self.renderer.pos_to_base[pos_str]
            elif entity_type == "supply_marble":
                try:
                    marble_id = int(pos_str.split(":", 1)[1])
                except (IndexError, ValueError):
                    marble_id = None
                if marble_id is not None:
                    entity = self.renderer._marble_registry.get(marble_id)
            elif entity_type == "captured_marble":
                try:
                    marble_id = int(pos_str.split(":", 1)[1])
                except (IndexError, ValueError):
                    marble_id = None
                if marble_id is not None:
                    entity = self.renderer._marble_registry.get(marble_id)

            if entity is not None:
                model = entity.model if hasattr(entity, "model") else entity
                if original_mat is not None:
                    model.setMaterial(original_mat, 1)
                else:
                    model.clearMaterial()

    def set_context_highlights(
        self,
        context: str,
        positions: list[str] | set[str],
        material_mod: MaterialModifier | None = None,
    ) -> None:
        """Apply persistent highlights for a named context.

        Args:
            context: Context name (e.g., 'hover_primary', 'placement', etc.)
            positions: List or set of position strings to highlight
            material_mod: Optional MaterialModifier to use (uses context default if None)
        """
        self.clear_context_highlights(context)
        if not positions:
            return

        highlight_info = {
            "entities": list(positions),
            "duration": 0.0,
            "start_time": 0.0,
            "material_mod": material_mod or PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
        }
        self._context_highlights[context] = highlight_info
        self._apply_highlight(highlight_info)

        # For persistent context highlights, apply the material immediately (no pulsing)
        material_mod = highlight_info.get("material_mod", PLACEMENT_HIGHLIGHT_MATERIAL_MOD)
        original_materials = highlight_info.get("original_materials", {})

        for pos_str, mat_info in original_materials.items():
            (
                original_mat,
                entity_type,
                original_color,
                original_emission,
                original_metallic,
                original_roughness,
            ) = mat_info

            # Get the entity based on type
            entity = None
            if entity_type == "marble" and pos_str in self.renderer.pos_to_marble:
                entity = self.renderer.pos_to_marble[pos_str]
            elif entity_type == "ring" and pos_str in self.renderer.pos_to_base:
                entity = self.renderer.pos_to_base[pos_str]
            elif entity_type == "supply_marble":
                try:
                    marble_id = int(pos_str.split(":", 1)[1])
                except (IndexError, ValueError):
                    marble_id = None
                if marble_id is not None:
                    entity = self.renderer._marble_registry.get(marble_id)
            elif entity_type == "captured_marble":
                try:
                    marble_id = int(pos_str.split(":", 1)[1])
                except (IndexError, ValueError):
                    marble_id = None
                if marble_id is not None:
                    entity = self.renderer._marble_registry.get(marble_id)

            if entity is not None:
                # Apply highlight material (full intensity, no pulsing)
                highlight_color = LVector4(*material_mod.highlight_color)
                highlight_emission = LVector4(*material_mod.emission_color)

                highlight_mat = Material()
                highlight_mat.setMetallic(original_metallic)
                highlight_mat.setRoughness(original_roughness)
                highlight_mat.setBaseColor(highlight_color)
                highlight_mat.setEmission(highlight_emission)

                model = entity.model if hasattr(entity, "model") else entity
                model.setMaterial(highlight_mat, 1)

    def clear_context_highlights(self, context: str | None = None) -> None:
        """Clear context highlights for a specific context or all contexts.

        Args:
            context: Context name to clear, or None to clear all contexts
        """
        if context is None:
            contexts = list(self._context_highlights.keys())
        else:
            contexts = [context] if context in self._context_highlights else []

        for ctx in contexts:
            info = self._context_highlights.pop(ctx, None)
            if info:
                self._clear_highlight(info)