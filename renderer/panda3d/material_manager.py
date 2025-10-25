"""Manager class for Panda3D material operations."""

from panda3d.core import Material, LVector4

from renderer.panda3d.material_modifier import MaterialModifier


class MaterialManager:
    """Handles Panda3D material creation, application, and restoration.

    This class centralizes all material-related operations to avoid duplication
    and provide a consistent interface for working with materials.
    """

    # Default material properties when no material exists
    DEFAULT_COLOR = LVector4(0.8, 0.8, 0.8, 1.0)
    DEFAULT_EMISSION = LVector4(0.0, 0.0, 0.0, 1.0)
    DEFAULT_METALLIC = 0.9
    DEFAULT_ROUGHNESS = 0.1

    @staticmethod
    def get_model_from_entity(entity):
        """Get the model NodePath from an entity.

        Entities can be either:
        - NodePath objects directly (rings, board marbles)
        - Objects with a .model attribute (supply/captured marbles)

        Args:
            entity: Entity object (NodePath or object with .model attribute)

        Returns:
            NodePath: The model to apply materials to
        """
        return entity.model if hasattr(entity, "model") else entity

    @staticmethod
    def save_material(entity) -> Material | None:
        """Save current material from an entity.

        Creates a copy of the original material to preserve it during animations.

        Args:
            entity: Entity to save material from

        Returns:
            Material | None: Copy of the original material, or None if no material exists
        """
        model = MaterialManager.get_model_from_entity(entity)
        original_mat = model.getMaterial()

        if original_mat is not None:
            # Make a copy so we can restore it later without side effects
            return Material(original_mat)
        else:
            return None

    @staticmethod
    def apply_material(entity, material_mod, metallic, roughness):
        """Apply a MaterialModifier to an entity.

        Creates a new Panda3D Material with the specified properties and
        applies it to the entity's model.

        Args:
            entity: Entity to apply material to (NodePath or object with .model)
            material_mod: MaterialModifier with highlight_color and emission_color
            metallic: Metallic property value (0.0-1.0)
            roughness: Roughness property value (0.0-1.0)
        """
        model = MaterialManager.get_model_from_entity(entity)

        highlight_color = LVector4(*material_mod.highlight_color)
        highlight_emission = LVector4(*material_mod.emission_color)

        material = Material()
        material.setMetallic(metallic)
        material.setRoughness(roughness)
        material.setBaseColor(highlight_color)
        material.setEmission(highlight_emission)

        model.setMaterial(material, 1)

    @staticmethod
    def restore_material(entity, saved_material: Material | None):
        """Restore a previously saved material to an entity.

        Args:
            entity: Entity to restore material to
            saved_material: Material from save_material(), or None
        """
        model = MaterialManager.get_model_from_entity(entity)

        if saved_material is not None:
            model.setMaterial(saved_material, 1)
        else:
            model.clearMaterial()

    @staticmethod
    def get_material_properties(saved_material: Material | None):
        """Extract material properties, using defaults if no material exists.

        Args:
            saved_material: Material to extract properties from, or None

        Returns:
            tuple: (color, emission, metallic, roughness)
        """
        if saved_material is not None:
            return (
                saved_material.getBaseColor(),
                saved_material.getEmission(),
                saved_material.getMetallic(),
                saved_material.getRoughness(),
            )
        else:
            return (
                MaterialManager.DEFAULT_COLOR,
                MaterialManager.DEFAULT_EMISSION,
                MaterialManager.DEFAULT_METALLIC,
                MaterialManager.DEFAULT_ROUGHNESS,
            )

    @staticmethod
    def update_material_colors_inplace(
        entity, saved_material, target_material_mod, blend_factor
    ):
        """Update an entity's material colors in-place for pulsing animations.

        Modifies the entity's existing material without creating new objects,
        which is much more efficient than creating/swapping materials each frame.

        Args:
            entity: Entity whose material to update
            saved_material: Original Material (for extracting original colors)
            target_material_mod: Target MaterialModifier to blend towards
            blend_factor: Blend ratio (0.0 = original, 1.0 = target)
        """
        # Extract original colors from saved material
        original_color, original_emission, _, _ = MaterialManager.get_material_properties(saved_material)

        # Blend all channels (RGBA) for pulsing effect
        # Alpha pulses from original material's alpha → target's score-based alpha → original alpha
        blended_material_mod = MaterialModifier.blend_vectors_with_mod(
            original_color,
            original_emission,
            target_material_mod,
            blend_factor,
            blend_mask=(True, True, True, True),  # Blend all channels including alpha
        )

        blended_color = LVector4(*blended_material_mod.highlight_color)
        blended_emission = LVector4(*blended_material_mod.emission_color)

        # Update the entity's existing material in-place
        model = MaterialManager.get_model_from_entity(entity)
        material = model.getMaterial()

        if material is not None:
            material.setBaseColor(blended_color)
            material.setEmission(blended_emission)