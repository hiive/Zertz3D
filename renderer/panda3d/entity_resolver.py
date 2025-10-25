"""Helper class for resolving entity references to actual entity objects."""


class EntityResolver:
    """Resolves position strings to actual renderer entities.

    Can be called directly as a function:
        entities = EntityResolver(renderer, "A1")  # Get all entities at A1
        entities = EntityResolver(renderer, "A1", "ring")  # Get only rings at A1
    """

    def __new__(cls, renderer, pos_str: str, entity_type: str | None = None) -> list[tuple[object, str]]:
        """Resolve entities at a position string, optionally filtered by type.

        Searches for entities in priority order:
        1. Board marbles (pos_to_marble)
        2. Board rings (pos_to_base)
        3. Supply marbles (supply:id format)
        4. Captured marbles (captured:id format)

        Args:
            renderer: PandaRenderer instance
            pos_str: Position string (e.g., "A1", "supply:123", "captured:456")
            entity_type: Optional filter - only return entities of this type
                        ("marble", "ring", "supply_marble", "captured_marble")
                        If None, returns all entities found at this position

        Returns:
            list: List of (entity, entity_type) tuples for all matching entities
                  Empty list if no entities found
        """
        found_entities = []

        # Try board marble
        if pos_str in renderer.pos_to_marble:
            if entity_type is None or entity_type == "marble":
                found_entities.append((renderer.pos_to_marble[pos_str], "marble"))

        # Try board ring
        if pos_str in renderer.pos_to_base:
            if entity_type is None or entity_type == "ring":
                found_entities.append((renderer.pos_to_base[pos_str], "ring"))

        # Try supply marble (format: "supply:123")
        if isinstance(pos_str, str) and pos_str.startswith("supply:"):
            try:
                marble_id = int(pos_str.split(":", 1)[1])
                entity = renderer._marble_registry.get(marble_id)
                if entity is not None:
                    if entity_type is None or entity_type == "supply_marble":
                        found_entities.append((entity, "supply_marble"))
            except (IndexError, ValueError):
                pass

        # Try captured marble (format: "captured:456")
        if isinstance(pos_str, str) and pos_str.startswith("captured:"):
            try:
                marble_id = int(pos_str.split(":", 1)[1])
                entity = renderer._marble_registry.get(marble_id)
                if entity is not None:
                    if entity_type is None or entity_type == "captured_marble":
                        found_entities.append((entity, "captured_marble"))
            except (IndexError, ValueError):
                pass

        return found_entities