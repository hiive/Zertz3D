from renderer.panda3d.material_modifier import MaterialModifier

"""Material modifiers for visual highlights.

These materials define colors/emission for different highlight purposes.
They can be used with either highlighting system:
- Temporary animations: queue_highlight() with duration/defer
- Persistent contexts: set_context_highlights() via context name

The material defines WHAT is highlighted (placement, capture, hover, etc.)
The API call determines HOW it's displayed (pulsing vs static).
"""

# Heat-map materials: MAX (highest likelihood) and MIN (lowest visible likelihood)
# Used for interpolating colors based on action scores

# Placement highlights (green spectrum)
DARK_GREEN_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.0, 0.4, 0.0, 1),  # Dark green base
    emission_color=(0.0, 0.08, 0.0, 1),  # Subtle green glow
)

CYAN_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.0, 0.4, 0.4, 1),  # Cyan/turquoise - distinct from green MAX
    emission_color=(0.0, 0.08, 0.08, 1),   # Cyan glow
)

# Removal highlights (red spectrum)
DARK_RED_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.4, 0.0, 0.0, 1),  # Dark red base
    emission_color=(0.08, 0.0, 0.0, 1),  # Subtle red glow
)

MAGENTA_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.5, 0.0, 0.35, 1),  # Magenta/purple - distinct from red MAX
    emission_color=(0.12, 0.0, 0.08, 1),   # Magenta glow
)

# Capture highlights (blue spectrum)
DARK_BLUE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.0, 0.0, 0.4, 1),  # Dark blue base
    emission_color=(0.0, 0.0, 0.08, 1),  # Subtle blue glow
)

OLIVE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.5, 0.5, 0.0, 1),  # Yellow/olive - distinct from blue MAX
    emission_color=(0.12, 0.12, 0.0, 1),   # Yellow glow
)

# Selected capture highlight (bright blue - no MIN variant needed, always max)
CORNFLOWER_BLUE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.39, 0.58, 0.93, 1),  # Cornflower blue
    emission_color=(0.08, 0.12, 0.19, 1),   # Cornflower blue glow
)


# Formerly: CAPTURE_FLASH_MATERIAL_MOD
BRIGHT_YELLOW_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.9, 0.9, 0.1, 1),  # Bright yellow
    emission_color=(0.3, 0.3, 0.05, 1),  # Strong yellow glow for flash
)

# Formerly: HOVER_PRIMARY_MATERIAL_MOD
GOLD_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.95, 0.85, 0.2, 1), emission_color=(0.2, 0.18, 0.04, 1)
)

# Formerly: HOVER_SECONDARY_MATERIAL_MOD
ORANGE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.9, 0.45, 0.15, 1), emission_color=(0.18, 0.09, 0.03, 1)
)

# Formerly: HOVER_SUPPLY_MATERIAL_MOD
LIGHT_BLUE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.85, 0.85, 0.95, 1), emission_color=(0.18, 0.18, 0.22, 1)
)

# Formerly: HOVER_CAPTURED_MATERIAL_MOD
PURPLE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.75, 0.6, 0.9, 1), emission_color=(0.15, 0.12, 0.18, 1)
)

# Formerly: SUPPLY_HIGHLIGHT_WHITE_MATERIAL_MOD
LIGHT_GRAY_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.85, 0.85, 0.85, 1.0), emission_color=(0.18, 0.18, 0.18, 1.0)
)

# Formerly: SUPPLY_HIGHLIGHT_GREY_MATERIAL_MOD
MEDIUM_GRAY_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.65, 0.65, 0.65, 1.0), emission_color=(0.14, 0.14, 0.14, 1.0)
)

# Formerly: SUPPLY_HIGHLIGHT_BLACK_MATERIAL_MOD
DARK_GRAY_MATERIAL_MOD = MaterialModifier(
    highlight_color=(0.35, 0.35, 0.35, 1.0), emission_color=(0.08, 0.08, 0.08, 1.0)
)

WHITE_MATERIAL_MOD = MaterialModifier(
    highlight_color=(1.0, 1.0, 1.0, 1.0), emission_color=(0.2, 0.2, 0.2, 1.0)
)
