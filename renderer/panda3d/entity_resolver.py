"""Helper class for resolving entity references to actual entity objects."""


class EntityResolver:
    """Resolves position strings and entity types to actual renderer entities.

    Handles two common patterns:
    1. Looking up entities when type is already known
    2. Discovering both entity and type from position string
    """

    @staticmethod
    def resolve(renderer, pos_str: str, entity_type: str):
        """Resolve an entity given its position string and type.

        Args:
            renderer: PandaRenderer instance
            pos_str: Position string (e.g., "A1", "supply:123", "captured:456")
            entity_type: Type of entity ("marble", "ring", "supply_marble", "captured_marble")

        Returns:
            Entity object or None if not found
        """
        if entity_type == "marble":
            return renderer.pos_to_marble.get(pos_str)
        elif entity_type == "ring":
            return renderer.pos_to_base.get(pos_str)
        elif entity_type in ("supply_marble", "captured_marble"):
            # Extract marble ID from "supply:123" or "captured:456" format
            try:
                marble_id = int(pos_str.split(":", 1)[1])
                return renderer._marble_registry.get(marble_id)
            except (IndexError, ValueError):
                return None
        else:
            return None

    @staticmethod
    def discover(renderer, pos_str: str) -> tuple[object | None, str | None]:
        """Discover both entity and its type from a position string.

        Tries to find the entity by checking different registries in order:
        1. Board marble positions
        2. Board ring positions
        3. Supply marbles (supply:id format)
        4. Captured marbles (captured:id format)

        Args:
            renderer: PandaRenderer instance
            pos_str: Position string

        Returns:
            tuple: (entity, entity_type) or (None, None) if not found
        """
        # Try board marble first
        if pos_str in renderer.pos_to_marble:
            return renderer.pos_to_marble[pos_str], "marble"

        # Try board ring
        if pos_str in renderer.pos_to_base:
            return renderer.pos_to_base[pos_str], "ring"

        # Try supply marble (format: "supply:123")
        if isinstance(pos_str, str) and pos_str.startswith("supply:"):
            try:
                marble_id = int(pos_str.split(":", 1)[1])
                entity = renderer._marble_registry.get(marble_id)
                if entity is not None:
                    return entity, "supply_marble"
            except (IndexError, ValueError):
                pass

        # Try captured marble (format: "captured:456")
        if isinstance(pos_str, str) and pos_str.startswith("captured:"):
            try:
                marble_id = int(pos_str.split(":", 1)[1])
                entity = renderer._marble_registry.get(marble_id)
                if entity is not None:
                    return entity, "captured_marble"
            except (IndexError, ValueError):
                pass

        return None, None