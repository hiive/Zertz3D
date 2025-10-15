"""Helper class for managing mouse picking, hover detection, and user interaction."""

from typing import Callable, Optional, Any

from panda3d.core import (
    BitMask32,
    CollisionTraverser,
    CollisionNode,
    CollisionRay,
    CollisionHandlerQueue,
    NodePath,
)

from shared.materials_modifiers import HOVER_PRIMARY_MATERIAL_MOD


class InteractionHelper:
    """Manages mouse picking, collision detection, and hover state for the renderer."""

    def __init__(self, renderer):
        """Initialize the interaction helper.

        Args:
            renderer: PandaRenderer instance
        """
        self.renderer = renderer

        # Setup collision detection system
        self._picker = CollisionTraverser("zertzPicker")
        self._picker_queue = CollisionHandlerQueue()
        self._picker_ray = CollisionRay()
        self._picker_node = CollisionNode("mouseRay")
        self._picker_node.addSolid(self._picker_ray)
        self._picker_node.setFromCollideMask(BitMask32.bit(1))
        self._picker_node.setIntoCollideMask(BitMask32.allOff())
        self._picker_np = renderer.camera.attachNewNode(self._picker_node)
        self._picker.addCollider(self._picker_np, self._picker_queue)

        # Callback and state tracking
        self._selection_callback: Optional[Callable[[dict], None]] = None
        self._hover_callback: Optional[Callable[[dict | None], None]] = None
        self._hover_target_token: Optional[tuple] = None
        self._raw_hover_token: Optional[tuple] = None

    def set_selection_callback(
        self, callback: Optional[Callable[[dict], None]]
    ) -> None:
        """Set the callback for mouse click selection events.

        Args:
            callback: Function to call with selection dict when an entity is clicked
        """
        self._selection_callback = callback

    def set_hover_callback(
        self, callback: Optional[Callable[[dict | None], None]]
    ) -> None:
        """Set the callback for hover events.

        Args:
            callback: Function to call with hover dict (or None when hover ends)
        """
        self._hover_callback = callback
        self._hover_target_token = None
        self._raw_hover_token = None

    def on_mouse_click(self) -> None:
        """Handle mouse click events and trigger selection callback if applicable."""
        if self._selection_callback is None or self.renderer.mouseWatcherNode is None:
            return
        if not self.renderer.mouseWatcherNode.hasMouse():
            return

        mouse_pos = self.renderer.mouseWatcherNode.getMouse()
        self._clear_picker_queue()
        self._picker_ray.setFromLens(
            self.renderer.camNode, mouse_pos.getX(), mouse_pos.getY()
        )
        self._picker.traverse(self.renderer.render)
        if self._picker_queue.getNumEntries() == 0:
            return

        self._picker_queue.sortEntries()
        for i in range(self._picker_queue.getNumEntries()):
            entry = self._picker_queue.getEntry(i)
            node_path = entry.getIntoNodePath()
            selection = self._decode_selection(node_path)
            if selection is not None:
                self._selection_callback(selection)
                break

    def dispatch_hover_target(self) -> None:
        """Update hover state based on current mouse position.

        Called each frame from the renderer's update loop.
        """
        if self._hover_callback is None or self.renderer.mouseWatcherNode is None:
            if self._hover_target_token is not None:
                self._hover_target_token = None
                self._hover_callback(None)
            if self._raw_hover_token is not None:
                self._raw_hover_token = None
                self.renderer.clear_context_highlights("hover_pointer_raw")
            return

        if not self.renderer.mouseWatcherNode.hasMouse():
            if self._hover_target_token is not None:
                self._hover_target_token = None
                self._hover_callback(None)
            if self._raw_hover_token is not None:
                self._raw_hover_token = None
                self.renderer.clear_context_highlights("hover_pointer_raw")
            return

        mouse_pos = self.renderer.mouseWatcherNode.getMouse()
        self._clear_picker_queue()
        self._picker_ray.setFromLens(
            self.renderer.camNode, mouse_pos.getX(), mouse_pos.getY()
        )
        self._picker.traverse(self.renderer.render)
        if self._picker_queue.getNumEntries() == 0:
            if self._hover_target_token is not None:
                self._hover_target_token = None
                self._hover_callback(None)
            if self._raw_hover_token is not None:
                self._raw_hover_token = None
                self.renderer.clear_context_highlights("hover_pointer_raw")
            return

        self._picker_queue.sortEntries()
        entry = self._picker_queue.getEntry(0)

        selection = self._decode_selection(entry.getIntoNodePath())
        self._apply_direct_hover(selection)
        token = self._make_hover_token(selection)
        # print(f"_dispatch_hover_target::token: {token}")
        if token != self._hover_target_token:
            self._hover_target_token = token
            self._hover_callback(selection)

    def show_hover_feedback(
        self,
        primary: Optional[set[str] | list[str]] = None,
        secondary: Optional[set[str] | list[str]] = None,
        supply_colors: Optional[set[str] | list[str]] = None,
        captured_targets: Optional[set[tuple[int, str]] | list[tuple[int, str]]] = None,
    ) -> None:
        """Display visual feedback for hovering over entities.

        Args:
            primary: Primary positions to highlight
            secondary: Secondary positions to highlight
            supply_colors: Supply marble colors to highlight
            captured_targets: Captured marble targets to highlight (owner, color)
        """
        primary_positions = list(primary) if primary else []
        secondary_positions = list(secondary) if secondary else []

        if primary_positions:
            self.renderer.set_context_highlights("hover_primary", primary_positions)
        else:
            self.renderer.clear_context_highlights("hover_primary")

        if secondary_positions:
            self.renderer.set_context_highlights("hover_secondary", secondary_positions)
        else:
            self.renderer.clear_context_highlights("hover_secondary")

        if supply_colors:
            supply_positions: list[str] = []
            for color in supply_colors:
                supply_positions.extend(self.renderer._supply_highlight_keys(color))
            if supply_positions:
                self.renderer.set_context_highlights("hover_supply", supply_positions)
            else:
                self.renderer.clear_context_highlights("hover_supply")
        else:
            self.renderer.clear_context_highlights("hover_supply")

        if captured_targets:
            captured_positions: list[str] = []
            for owner, color in captured_targets:
                captured_positions.extend(
                    self.renderer._captured_highlight_keys(int(owner), color)
                )
            if captured_positions:
                self.renderer.set_context_highlights(
                    "hover_captured", captured_positions
                )
            else:
                self.renderer.clear_context_highlights("hover_captured")
        else:
            self.renderer.clear_context_highlights("hover_captured")

    def clear_hover_highlights(self) -> None:
        """Clear all hover-related highlights."""
        self.renderer.clear_context_highlights("hover_primary")
        self.renderer.clear_context_highlights("hover_secondary")
        self.renderer.clear_context_highlights("hover_supply")
        self.renderer.clear_context_highlights("hover_captured")
        self.renderer.clear_context_highlights("hover_pointer_raw")

    def _clear_picker_queue(self) -> None:
        """Clear all entries from the picker queue."""
        if hasattr(self._picker_queue, "clearEntries"):
            self._picker_queue.clearEntries()
        else:
            while self._picker_queue.getNumEntries():
                self._picker_queue.popEntry()

    def _decode_selection(self, node_path: NodePath) -> Optional[dict]:
        """Decode a NodePath into a selection dictionary.

        Args:
            node_path: NodePath from collision detection

        Returns:
            Dict with type, label, color, etc., or None if not a valid selection
        """
        current = node_path
        while not current.isEmpty():
            if current.hasPythonTag("zertz_entity"):
                entity = current.getPythonTag("zertz_entity")
                if entity == "ring":
                    label = current.getPythonTag("zertz_label")
                    if label:
                        return {"type": "ring", "label": label}
                elif entity == "board_marble":
                    label = current.getPythonTag("zertz_label")
                    if label:
                        return {"type": "board_marble", "label": label}
                elif entity == "supply_marble":
                    color = current.getPythonTag("zertz_color")
                    supply_key = current.getPythonTag(
                        "zertz_key"
                    ) or self.renderer._supply_key(current)
                    if color:
                        return {
                            "type": "supply_marble",
                            "color": color,
                            "supply_key": supply_key,
                        }
                elif entity == "captured_marble":
                    if not self.renderer._is_supply_empty():
                        return None
                    color = current.getPythonTag("zertz_color")
                    owner = current.getPythonTag("zertz_owner")
                    captured_key = current.getPythonTag(
                        "zertz_key"
                    ) or self.renderer._captured_key(current)
                    if color and owner is not None:
                        return {
                            "type": "captured_marble",
                            "color": color,
                            "owner": owner,
                            "captured_key": captured_key,
                        }
            current = current.getParent()
        return None

    @staticmethod
    def _make_hover_token(selection: Optional[dict]) -> Optional[tuple]:
        """Create a hashable token from a selection dict for change detection.

        Args:
            selection: Selection dict or None

        Returns:
            Tuple of selection values, or None
        """
        if selection is None:
            return None
        if not isinstance(selection, dict):
            return None
        return (
            selection.get("type"),
            selection.get("label"),
            selection.get("color"),
            selection.get("owner"),
            selection.get("index"),
            selection.get("supply_key"),
            selection.get("captured_key"),
        )

    def _apply_direct_hover(self, selection: Optional[dict]) -> None:
        """Apply visual highlight to the entity directly under the cursor.

        Args:
            selection: Selection dict from _decode_selection
        """
        token = self._make_hover_token(selection)
        if token == self._raw_hover_token:
            return
        self._raw_hover_token = token
        if selection is None:
            self.renderer.clear_context_highlights("hover_pointer_raw")
            return

        hover_positions: list[str] = []
        sel_type = selection.get("type")
        label = selection.get("label")
        if sel_type in ("ring", "board_marble") and label:
            hover_positions.append(label)
        elif sel_type == "supply_marble":
            key = selection.get("supply_key")
            if key:
                hover_positions.append(key)
        elif sel_type == "captured_marble":
            key = selection.get("captured_key")
            if key:
                hover_positions.append(key)

        if hover_positions:
            self.renderer.set_context_highlights(
                "hover_pointer_raw", hover_positions, HOVER_PRIMARY_MATERIAL_MOD
            )
        else:
            self.renderer.clear_context_highlights("hover_pointer_raw")
