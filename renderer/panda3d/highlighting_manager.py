"""Manager class for handling entity highlights (pulsing effects and context-based highlights)."""

import math
from queue import SimpleQueue, Empty
from typing import Optional, Callable

import numpy as np
from panda3d.core import Material

from renderer.panda3d.material_modifier import MaterialModifier
from renderer.panda3d.material_manager import MaterialManager
from renderer.panda3d.entity_resolver import EntityResolver
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
            for pos_str, (saved_material, entity_type) in original_materials.items():
                entity = EntityResolver.resolve(self.renderer, pos_str, entity_type)
                if entity is not None:
                    # Update material colors in-place for pulsing effect (efficient!)
                    MaterialManager.update_material_colors_inplace(
                        entity,
                        saved_material,
                        target_material_mod,
                        pulse_factor,
                    )

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
            # Discover entity and its type from position string
            entity, entity_type = EntityResolver.discover(self.renderer, pos_str)
            if entity is not None:
                # Save original material properties for blending
                saved_material = MaterialManager.save_material(entity)
                original_materials[pos_str] = (saved_material, entity_type)

        # Store original materials and target colors for later blending
        highlight_info["original_materials"] = original_materials

        material_mod = highlight_info.get(
            "material_mod", PLACEMENT_HIGHLIGHT_MATERIAL_MOD
        )
        highlight_info["target_material_mod"] = material_mod

        # Apply a NEW highlight material to each entity (we'll modify it in-place during pulsing)
        for pos_str, (saved_material, entity_type) in original_materials.items():
            entity = EntityResolver.resolve(self.renderer, pos_str, entity_type)
            if entity is not None:
                # Extract metallic/roughness from saved material
                _, _, metallic, roughness = MaterialManager.get_material_properties(saved_material)

                # Create and apply a new material that we'll modify in-place during pulsing
                MaterialManager.apply_material(
                    entity,
                    material_mod,
                    metallic,
                    roughness,
                )

    def _clear_highlight(self, highlight_info):
        """Clear a highlight and restore original materials.

        Args:
            highlight_info: Dict with 'original_materials'
        """
        original_materials = highlight_info.get("original_materials", {})

        for pos_str, (saved_material, entity_type) in original_materials.items():
            entity = EntityResolver.resolve(self.renderer, pos_str, entity_type)
            if entity is not None:
                # Restore original material
                MaterialManager.restore_material(entity, saved_material)

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

        for pos_str, (saved_material, entity_type) in original_materials.items():
            entity = EntityResolver.resolve(self.renderer, pos_str, entity_type)
            if entity is not None:
                # Extract metallic/roughness from saved material
                _, _, metallic, roughness = MaterialManager.get_material_properties(saved_material)

                # Apply highlight material (full intensity, no pulsing)
                MaterialManager.apply_material(entity, material_mod, metallic, roughness)

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