"""Paint rendered cells into an RGB image, the way a terminal would draw them.

Used by `snapshot` (one PNG) and `export` (an MP4 of the whole video). Each
character is rasterised once into a coverage atlas; a frame is then painted
with array operations, background + (foreground - background) * coverage.
"""

from __future__ import annotations

import os

import numpy as np

from .render import Cells, Renderer, _xterm256_palette

FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
)


class CellPainter:
    def __init__(self, renderer: Renderer, cell_w: int, cell_aspect: float, colors: str,
                 font: str | None = None):
        self.r = renderer
        self.cw = int(cell_w)
        self.ch = int(round(cell_w * cell_aspect))
        self.colors = colors
        self.atlas = self._atlas(font)                      # (K, ch, cw) coverage 0..1
        if colors == "256":
            pal = np.zeros((256, 3), np.float32)
            pal[16:] = _xterm256_palette()
            self.palette = pal

    def _atlas(self, font_path: str | None) -> np.ndarray:
        mode, cw, ch = self.r.mode, self.cw, self.ch
        if self.r.mode_name in ("glyph", "ramp"):
            return self._font_atlas(font_path)
        # Block and braille glyphs come from their masks: fonts draw them unevenly.
        sw, sh = mode.sub_w, mode.sub_h
        ys = (np.arange(ch) * sh // ch)[:, None]
        xs = (np.arange(cw) * sw // cw)[None, :]
        grid = mode.masks[:, ys * sw + xs]                  # (K, ch, cw)
        if self.r.mode_name == "braille":
            # Round dots instead of filled squares.
            fy = (np.arange(ch) * sh / ch) % 1 - 0.5
            fx = (np.arange(cw) * sw / cw) % 1 - 0.5
            dot = (fy[:, None] ** 2 + fx[None, :] ** 2) < 0.12
            grid = grid * dot
        return grid.astype(np.float32)

    def _font_atlas(self, font_path: str | None) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            raise SystemExit("asciivid: snapshot and export need Pillow (pip install pillow)")
        path = font_path or next((p for p in FONTS if os.path.exists(p)), None)
        if path is None:
            raise SystemExit("asciivid: no monospace font found; pass --font")
        # Size the font so one advance is one cell wide, drawn 4x and scaled down
        # for clean antialiasing.
        scale = 4
        probe = ImageFont.truetype(path, 100)
        size = 100 * self.cw * scale / probe.getlength("M")
        font = ImageFont.truetype(path, max(1, round(size)))
        ascent, descent = font.getmetrics()
        top = (self.ch * scale - (ascent + descent)) / 2
        out = []
        for c in self.r.chars:
            img = Image.new("L", (self.cw * scale, self.ch * scale), 0)
            ImageDraw.Draw(img).text((0, top), c, fill=255, font=font)
            img = img.resize((self.cw, self.ch), Image.Resampling.BOX)
            out.append(np.asarray(img, np.float32) / 255)
        return np.stack(out)

    def _rgb(self, packed: np.ndarray) -> np.ndarray:
        if self.colors == "256":
            return self.palette[packed.astype(np.int64)]
        p = packed.astype(np.uint32)
        return np.stack([(p >> 16) & 255, (p >> 8) & 255, p & 255], -1).astype(np.float32)

    def paint(self, cells: Cells) -> np.ndarray:
        rows, cols = cells.ch.shape
        cover = self.atlas[cells.ch][..., None]             # (rows, cols, ch, cw, 1)
        if self.colors == "none":
            fg = np.full((rows, cols, 3), 220, np.float32)
            bg = np.zeros((rows, cols, 3), np.float32)
        else:
            fg, bg = self._rgb(cells.fg), self._rgb(cells.bg)
        fg, bg = fg[:, :, None, None, :], bg[:, :, None, None, :]
        img = bg + (fg - bg) * cover
        img = img.transpose(0, 2, 1, 3, 4).reshape(rows * self.ch, cols * self.cw, 3)
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)
