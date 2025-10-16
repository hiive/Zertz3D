"""Manager class for handling entity animations."""

import math
from queue import SimpleQueue, Empty

import numpy as np
from panda3d.core import TransparencyAttrib


class AnimationManager:
    """Manages non-highlight animations (move animations)."""

    def __init__(self, renderer):
        """Initialize the animation manager.

        Args:
            renderer: PandaRenderer instance
        """
        self.renderer = renderer
        self.animation_queue = SimpleQueue()
        self.current_animations = []  # List of active animation items
        self._animation_id_counter = 0  # For generating unique IDs

    def queue_animation(self, anim_type="move", defer=0, **kwargs):
        """Add an animation to the queue.

        Args:
            anim_type: Animation type (currently only 'move' is supported)
            defer: Delay before starting (seconds)
            **kwargs: Type-specific parameters
                For 'move': entity, src, dst, scale, duration
        """
        anim_item = {"type": anim_type, "defer": defer, **kwargs}
        self.animation_queue.put(anim_item)

    def is_active(self):
        """Check if any animations are active or queued."""
        return len(self.current_animations) > 0 or not self.animation_queue.empty()

    def clear(self):
        """Clear all animations from queue and active list."""
        while not self.animation_queue.empty():
            try:
                self.animation_queue.get_nowait()
            except Empty:
                break
        self.current_animations.clear()

    def update(self, task):
        """Update all active animations.

        Args:
            task: Panda3D task object with timing information

        Returns:
            List of completed animation items
        """
        # Process animation queue - add new animations
        while not self.animation_queue.empty():
            try:
                anim_item = self.animation_queue.get_nowait()
                anim_type = anim_item.get("type", "move")

                # Generate unique ID
                self._animation_id_counter += 1
                anim_item["id"] = self._animation_id_counter

                # Set timing
                defer = anim_item.get("defer", 0)
                anim_item["insert_time"] = task.time
                anim_item["start_time"] = task.time + defer
                anim_item["end_time"] = anim_item["start_time"] + anim_item["duration"]

                if anim_type == "move":
                    # For move animations, store initial scale
                    entity = anim_item["entity"]
                    if "src_scale" not in anim_item:
                        anim_item["src_scale"] = entity.get_scale()
                    anim_item["dst_scale"] = anim_item.get("scale")

                self.current_animations.append(anim_item)
            except Empty:
                pass

        # Update all active animations
        to_remove = []
        for anim_item in self.current_animations:
            anim_type = anim_item.get("type", "move")

            # Check if animation hasn't started yet
            if task.time < anim_item["start_time"]:
                continue

            # Check if animation has ended
            if task.time >= anim_item["end_time"]:
                # Set final position and scale
                entity = anim_item["entity"]
                dst_scale = anim_item.get("dst_scale")
                dst = anim_item.get("dst")
                if dst_scale is not None:
                    entity.set_scale(dst_scale)
                if dst is not None:
                    entity.set_pos(dst)
                else:
                    # dst=None means this is a removal animation - hide the entity
                    entity.hide()

                to_remove.append(anim_item)
                continue

            # Update animation
            self._update_move_animation(anim_item, task.time)

        # Remove completed animations
        for anim_item in to_remove:
            self.current_animations.remove(anim_item)

        return to_remove

    def _update_move_animation(self, anim_item, current_time):
        """Update a move animation."""
        entity = anim_item["entity"]
        src = anim_item["src"]
        dst = anim_item.get("dst")
        src_scale = anim_item.get("src_scale")
        dst_scale = anim_item.get("dst_scale")
        duration = anim_item["duration"]
        start_time = anim_item["start_time"]
        elapsed_time = current_time - start_time
        anim_factor = elapsed_time / duration

        # Update scale
        if src_scale is not None and dst_scale is not None:
            sx = src_scale.x + (dst_scale - src_scale.x) * anim_factor
            entity.set_scale(sx)

        # Update position
        jump = True
        if dst is None:
            # Disappearing animation (ring removal)
            dx, dy, dz = src
            adx = -1 if dx < 0 else 1
            ady = -1 if dy < 0 else 1
            dx = max(10, abs(dx) * 10) * adx
            dy = max(4, abs(dy) * 4) * ady
            fz = 1.5
            dst = (dx, dy, dz * fz)
            jump = False

        x, y, z = self.renderer.get_current_pos(anim_factor, dst, src, jump=jump)
        entity.set_pos((x, y, z))
