"""Shader-based per-node outline manager."""

from __future__ import annotations

from typing import Dict, Any

from panda3d.core import (
    CullFaceAttrib,
    DepthTestAttrib,
    DepthWriteAttrib,
    LVector4,
    Mat4,
    NodePath,
    Shader, TransparencyAttrib,
)


class OutlineManager:
    """Attach/remove extruded-outline shader copies of NodePaths."""

    _shader: Shader | None = None

    def __init__(self) -> None:
        self._entries: Dict[NodePath, Dict[str, Any]] = {}
        if OutlineManager._shader is None:
            try:
                OutlineManager._shader = Shader.load(
                    Shader.SL_GLSL,
                    "renderer/shaders/outline.vert",
                    "renderer/shaders/outline.frag",
                )
            except Exception as exc:
                OutlineManager._shader = None
                raise RuntimeError("Failed to load outline shader") from exc

    def enable_outline(
        self,
        node: NodePath,
        color: LVector4,
        thickness: float,
        threshold: float = 0.35,
        dilate: float = 1.5,
    ) -> None:
        if node.is_empty():
            return

        entry = self._entries.get(node)
        if entry:
            entry["count"] += 1
            outline = entry["outline"]
            if not outline.is_empty():
                outline.setShaderInput("outline_color", color)
                outline.setShaderInput("outline_thickness", thickness)
                outline.setShaderInput("outline_threshold", threshold)
                outline.setShaderInput("outline_dilate", dilate)
                outline.show()
            return

        if OutlineManager._shader is None:
            return

        outline = node.copy_to(node)
        outline.setMat(node, Mat4.identMat())
        outline.setName(f"{node.getName()}-outline")
        outline.setShader(self._shader, 1)
        outline.setShaderInput("outline_color", color)
        outline.setShaderInput("outline_thickness", thickness)
        outline.setShaderInput("outline_threshold", threshold)
        outline.setShaderInput("outline_dilate", dilate)
        outline.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
        outline.setAttrib(DepthWriteAttrib.make(DepthWriteAttrib.MOff))
        outline.setAttrib(DepthTestAttrib.make(DepthTestAttrib.MLessEqual))
        outline.setTransparency(TransparencyAttrib.MAlpha)
        outline.setLightOff(1)
        outline.setMaterialOff(1)
        outline.setTextureOff(1)
        outline.setBin("fixed", 250)

        self._entries[node] = {"outline": outline, "count": 1}

    def disable_outline(self, node: NodePath) -> None:
        entry = self._entries.get(node)
        if not entry:
            return

        entry["count"] -= 1
        if entry["count"] > 0:
            return

        outline = entry["outline"]
        if not outline.is_empty():
            outline.removeNode()
        del self._entries[node]

    def clear_all(self) -> None:
        for entry in self._entries.values():
            outline = entry["outline"]
            if not outline.is_empty():
                outline.removeNode()
        self._entries.clear()
