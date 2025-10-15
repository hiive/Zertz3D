from renderer.material_modifier import MaterialModifier

PLACEMENT_HIGHLIGHT_MATERIAL_MOD = MaterialModifier(highlight_color=(0.0, 0.4, 0.0, 1),  # Dark green base
                                                    emission_color=(0.0, 0.08, 0.0, 1)) # Subtle green glow

REMOVABLE_HIGHLIGHT_MATERIAL_MOD = MaterialModifier(highlight_color=(0.4, 0.0, 0.0, 1),  # Dark red base
                                                    emission_color=(0.08, 0.0, 0.0, 1)) # Subtle red glow

CAPTURE_HIGHLIGHT_MATERIAL_MOD = MaterialModifier(highlight_color=(0.0, 0.0, 0.4, 1),  # Dark blue base
                                                  emission_color=(0.0, 0.0, 0.08, 1))  # Subtle blue glow

SELECTED_CAPTURE_MATERIAL_MOD = MaterialModifier(highlight_color=(0.39, 0.58, 0.93, 1), # Cornflower blue
                                                 emission_color=(0.08, 0.12, 0.19, 1))  # Cornflower blue glow

ISOLATION_HIGHLIGHT_MATERIAL_MOD = MaterialModifier(highlight_color=(0.8, 0.8, 0.0, 1),  # Bright yellow
                                                    emission_color=(0.16, 0.16, 0.0, 1))  # Yellow glow

HOVER_PRIMARY_MATERIAL_MOD = MaterialModifier(highlight_color=(0.95, 0.85, 0.2, 1),
                                              emission_color=(0.2, 0.18, 0.04, 1))

HOVER_SECONDARY_MATERIAL_MOD = MaterialModifier(highlight_color=(0.9, 0.45, 0.15, 1),
                                                emission_color=(0.18, 0.09, 0.03, 1))

HOVER_SUPPLY_MATERIAL_MOD = MaterialModifier(highlight_color=(0.85, 0.85, 0.95, 1),
                                             emission_color=(0.18, 0.18, 0.22, 1))

HOVER_CAPTURED_MATERIAL_MOD = MaterialModifier(highlight_color=(0.75, 0.6, 0.9, 1),
                                               emission_color=(0.15, 0.12, 0.18, 1))

SUPPLY_HIGHLIGHT_WHITE_MATERIAL_MOD = MaterialModifier(highlight_color=(0.85, 0.85, 0.85, 1.0),
                                                       emission_color=(0.18, 0.18, 0.18, 1.0))

SUPPLY_HIGHLIGHT_GREY_MATERIAL_MOD = MaterialModifier(highlight_color=(0.65, 0.65, 0.65, 1.0),
                                                      emission_color=(0.14, 0.14, 0.14, 1.0))

SUPPLY_HIGHLIGHT_BLACK_MATERIAL_MOD = MaterialModifier(highlight_color=(0.35, 0.35, 0.35, 1.0),
                                                       emission_color=(0.08, 0.08, 0.08, 1.0))
