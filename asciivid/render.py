"""Decode a video with ffmpeg and turn each frame into terminal cells.

A frame is scaled to (cols * sub_w) x (rows * sub_h) pixels, so every terminal
cell owns a small block of sub-pixels. Each cell is matched against a set of
glyph masks (how much ink each character puts on each sub-pixel) and gets the
character, foreground colour and background colour that reproduce that block
best. The Encoder then turns the cells into ANSI bytes, sending only what
changed since the previous frame.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import glyphs

# --------------------------------------------------------------------------
# Probing and decoding
# --------------------------------------------------------------------------


@dataclass
class VideoInfo:
    width: int  # display width (sample aspect and rotation applied)
    height: int
    fps: float
    duration: float
    has_audio: bool


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"asciivid: '{tool}' not found on PATH (install ffmpeg)")
    return path


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        [require("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"asciivid: cannot read {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise SystemExit(f"asciivid: no video stream in {path}")

    w, h = int(video["width"]), int(video["height"])
    sar = video.get("sample_aspect_ratio", "1:1")
    if sar not in ("0:1", "1:1", "N/A") and ":" in sar:
        n, d = sar.split(":")
        if int(d):
            w = round(w * int(n) / int(d))
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        rotation = int(side.get("rotation", rotation) or 0)
    rotation = int(video.get("tags", {}).get("rotate", rotation) or 0)
    if abs(rotation) % 180 == 90:
        w, h = h, w

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            value = Fraction(video.get(key, "0/1"))
        except (ValueError, ZeroDivisionError):
            continue
        if value > 0:
            fps = float(value)
            break
    fps = fps or 30.0

    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(w, h, fps, duration, has_audio)


def read_frames(path: str, px_w: int, px_h: int, *, fps: float, start: float = 0.0,
                duration: float | None = None, max_frames: int | None = None, brightness: float = 0.0,
                contrast: float = 1.0, gamma: float = 1.0, saturation: float = 1.0):
    """Yield frames as uint8 arrays of shape (px_h, px_w, 3)."""
    filters = [f"fps={fps}"]
    if (brightness, contrast, gamma, saturation) != (0.0, 1.0, 1.0, 1.0):
        filters.append(f"eq=brightness={brightness}:contrast={contrast}"
                       f":gamma={gamma}:saturation={saturation}")
    # "area" averages every source pixel under a target pixel: the most faithful
    # downscale for this purpose.
    filters.append(f"scale={px_w}:{px_h}:flags=area")
    cmd = [require("ffmpeg"), "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if duration:
        cmd += ["-t", str(duration)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-an", "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    size = px_w * px_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size * 4)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(px_h, px_w, 3)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()


def fit_grid(src_w: int, src_h: int, max_cols: int, max_rows: int,
             cell_aspect: float) -> tuple[int, int]:
    """Largest cols x rows that keeps the video's shape on cells that are
    `cell_aspect` times taller than wide."""
    cols = max_cols
    rows = round(cols * src_h / src_w / cell_aspect)
    if rows > max_rows:
        rows = max_rows
        cols = round(rows * cell_aspect * src_w / src_h)
    return max(1, min(cols, max_cols)), max(1, rows)


# --------------------------------------------------------------------------
# Modes: which characters, on what sub-pixel grid
# --------------------------------------------------------------------------

CHARSETS = {
    "ascii": "".join(chr(c) for c in range(32, 127)),
    "code": " {}[]()<>;:=+-*/\\|_#&!?.,'\"`~^%$@01",
    "symbols": " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "letters": " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "binary": " 01",
    "simple": " .:-=+*#%@",
    "matrix": " 0123456789abcdefABCDEF:;<>=*+-",
}

MODES = ("glyph", "ramp", "blocks", "quad", "braille")
STYLES = ("ink", "full")
COLORS = ("truecolor", "256", "gray", "none")

_QUAD_CHARS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"  # bit 0 TL, 1 TR, 2 BL, 3 BR


@dataclass
class Mode:
    sub_w: int
    sub_h: int
    chars: list[str]
    masks: np.ndarray  # (K, sub_w * sub_h) float32, row-major


def build_mode(mode: str, charset: str) -> Mode:
    if mode in ("glyph", "ramp"):
        chars = CHARSETS.get(charset, charset)
        chars = list(dict.fromkeys(chars))  # dedupe, keep order
        missing = [c for c in chars if c not in glyphs.MASKS]
        if missing:
            raise SystemExit(
                f"asciivid: no glyph data for {''.join(missing)!r}; the glyph and ramp "
                "modes take printable ASCII characters")
        if " " not in chars:
            chars.insert(0, " ")
        masks = np.array([glyphs.MASKS[c] for c in chars], np.float32)
        return Mode(glyphs.GRID_W, glyphs.GRID_H, chars, masks)
    if mode == "blocks":
        return Mode(1, 2, ["▀"], np.array([[1, 0]], np.float32))
    if mode == "quad":
        masks = [[(i >> b) & 1 for b in range(4)] for i in range(16)]
        return Mode(2, 2, list(_QUAD_CHARS), np.array(masks, np.float32))
    if mode == "braille":
        # Braille dot bits by (row, col) inside a 2 x 4 cell.
        bit = {(0, 0): 0, (1, 0): 1, (2, 0): 2, (3, 0): 6,
               (0, 1): 3, (1, 1): 4, (2, 1): 5, (3, 1): 7}
        chars, masks = [], []
        for n in range(256):
            chars.append(chr(0x2800 + n))
            masks.append([(n >> bit[(r, c)]) & 1 for r in range(4) for c in range(2)])
        return Mode(2, 4, chars, np.array(masks, np.float32))
    raise SystemExit(f"asciivid: unknown mode {mode!r} (choose from {', '.join(MODES)})")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass
class Cells:
    ch: np.ndarray  # (rows, cols) int32 index into Renderer.chars
    fg: np.ndarray  # (rows, cols) uint32 0xRRGGBB
    bg: np.ndarray  # (rows, cols) uint32 0xRRGGBB


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _xterm256_palette() -> np.ndarray:
    levels = [0, 95, 135, 175, 215, 255]
    pal = [(r, g, b) for r in levels for g in levels for b in levels]
    pal += [(v, v, v) for v in range(8, 239, 10)]
    return np.array(pal, np.float32)  # index i -> colour 16 + i


class Renderer:
    def __init__(self, cols: int, rows: int, *, mode: str = "glyph", charset: str = "ascii",
                 style: str = "ink", colors: str = "truecolor", color_step: int = 1,
                 paper: float = 1.0):
        self.cols, self.rows = cols, rows
        self.paper = min(1.0, max(0.0, float(paper)))
        self.mode_name, self.style, self.colors = mode, style, colors
        self.color_step = max(1, int(color_step))
        self.mode = build_mode(mode, charset)
        self.chars = self.mode.chars
        self.px_w, self.px_h = cols * self.mode.sub_w, rows * self.mode.sub_h
        m = self.mode.masks
        self._g = m
        self._gg = (m * m).sum(1)                       # |G|^2
        self._gc = m - m.mean(1, keepdims=True)          # G centred
        self._gcgc = (self._gc * self._gc).sum(1)        # |Gc|^2
        self._cover = m.mean(1)                          # average ink per glyph
        self._space = self.chars.index(" ") if " " in self.chars else None
        self._build_ladder()
        h = 1 - m
        self._ls_a, self._ls_b, self._ls_d = (m * m).sum(1), (m * h).sum(1), (h * h).sum(1)
        if colors == "256":
            self._pal = _xterm256_palette()
            self._pal_packed = _pack(self._pal)

    # -- one frame ------------------------------------------------------------

    def render(self, frame: np.ndarray) -> Cells:
        rows, cols, sw, sh = self.rows, self.cols, self.mode.sub_w, self.mode.sub_h
        rgb = frame.astype(np.float32)
        # (rows, sh, cols, sw, 3) -> (N, S, 3) with S = sh * sw, row-major per cell
        c = rgb.reshape(rows, sh, cols, sw, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, sh * sw, 3)

        if self.mode_name == "blocks":
            fg, bg = c[:, 0], c[:, 1]
            idx = np.zeros(len(c), np.int32)
        else:
            lum = (c @ _LUMA) / 255.0                    # (N, S)
            if self.mode_name == "ramp":
                idx, fg, bg = self._ramp(c, lum)
            elif self.style == "full":
                idx, fg, bg = self._fit_full(c, lum)
            else:
                idx, fg, bg = self._fit_ink(c, lum)

        fg, bg = self._quantize(fg), self._quantize(bg)
        return Cells(idx.reshape(rows, cols), fg.reshape(rows, cols), bg.reshape(rows, cols))

    # A cell counts as an edge when its light/dark pattern is strong enough
    # (EDGE_VAR, summed squared deviation of luminance over its sub-pixels)
    # and one glyph explains most of it (EDGE_FIT). Edges get the glyph with
    # the same shape; everything else gets a glyph from the brightness ladder.
    EDGE_VAR = 0.2
    EDGE_FIT = 0.45
    LADDER = 10

    def _build_ladder(self):
        """A few glyphs evenly spaced in ink coverage, sparse to dense. Few
        steps keep flat areas calm from one frame to the next."""
        cover = self._cover
        top = float(cover.max())
        picks = []
        for t in np.linspace(0, top, self.LADDER):
            i = int(np.abs(cover - t).argmin())
            if i not in picks:
                picks.append(i)
        self._ladder = np.array(picks, np.int32)
        self._ladder_cover = cover[self._ladder]

    def _by_brightness(self, m, ladder=True):
        """Glyph whose coverage matches brightness m (0..1), densest = white."""
        top = float(self._cover.max())
        if ladder:
            idx, cov = self._ladder, self._ladder_cover
        else:
            idx, cov = np.arange(len(self._cover)), self._cover
        return idx[np.abs(cov[None, :] - (m * top)[:, None]).argmin(1)].astype(np.int32)

    def _choose(self, lum, flat_space: bool):
        m = lum.mean(1)
        idx = self._by_brightness(m)
        if flat_space and self._space is not None:
            idx[:] = self._space
        pc = lum - m[:, None]
        var = (pc * pc).sum(1)
        score = pc @ self._gc.T                            # (N, K)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(self._gcgc > 1e-6, score * score / self._gcgc, 0.0)
        best = score.argmax(1).astype(np.int32)
        explained = score.max(1) / np.maximum(var, 1e-9)
        edge = (var > self.EDGE_VAR) & (explained > self.EDGE_FIT)
        idx[edge] = best[edge]
        return idx

    def _fit_full(self, c, lum):
        """Two colours per cell. With full paper a flat cell is plain colour
        (exact); with dimmed paper it gets a ladder glyph to carry its light."""
        idx = self._choose(lum, flat_space=self.paper >= 1)
        fg, bg = self._solve_colours(c, idx)
        if self.paper < 1:
            fg, bg = self._dim_paper(c, idx, fg, bg)
        return idx, fg, bg

    def _fit_ink(self, c, lum):
        """Characters on black: density shows brightness, the ink shows colour."""
        idx = self._choose(lum, flat_space=False)
        return idx, *self._ink_colours(c, lum, idx)

    def _ramp(self, c, lum):
        """Classic ASCII: every character of the set, ordered by real ink."""
        idx = self._by_brightness(lum.mean(1), ladder=False)
        if self.style == "full":
            fg, bg = self._solve_colours(c, idx)
            if self.paper < 1:
                fg, bg = self._dim_paper(c, idx, fg, bg)
            return idx, fg, bg
        return idx, *self._ink_colours(c, lum, idx)

    def _ink_colours(self, c, lum, idx):
        cover = self._cover[idx]
        m = lum.mean(1)
        top = float(self._cover.max())
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(cover > 1e-6, m * top / cover, 0.0)
        fg = self._ink_colour(c, lum, np.clip(a, 0.0, 1.0))
        return fg, np.zeros_like(fg)

    def _solve_colours(self, c, idx):
        """Least squares: pixel ~= G * fg + (1 - G) * bg, per channel."""
        a, b, d = self._ls_a[idx], self._ls_b[idx], self._ls_d[idx]   # per-glyph constants
        g = self._g[idx]                                  # (N, S)
        rg = np.matmul(g[:, None, :], c)[:, 0]            # sum G * pixel
        total = c.sum(1)
        rh = total - rg                                   # sum (1 - G) * pixel
        det = a * d - b * b
        mean = total / c.shape[1]
        ok = (det > 1e-6)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            fg = (d[:, None] * rg - b[:, None] * rh) / det[:, None]
            bg = (a[:, None] * rh - b[:, None] * rg) / det[:, None]
        fg = np.where(ok, fg, mean)
        bg = np.where(ok, bg, mean)
        return np.clip(fg, 0, 255), np.clip(bg, 0, 255)

    def _dim_paper(self, c, idx, fg, bg):
        """Darken the paper and brighten the ink, keeping each cell's average
        colour. The paper is only darkened as far as the ink can make up."""
        cover = self._cover[idx][:, None]
        mean = c.mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = np.where(cover < 1, (mean - 255 * cover) / (1 - cover), mean)
            bg = np.maximum(bg * self.paper, floor)
            fg = np.where(cover > 1e-6, (mean - bg * (1 - cover)) / cover, fg)
        return np.clip(fg, 0, 255), np.clip(bg, 0, 255)

    @staticmethod
    def _ink_colour(c, lum, a):
        """The cell's hue at brightness a (0..1), without clipping the hue."""
        mean = c.mean(1)
        m = lum.mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(m > 1e-6, a / m, 0.0)
        fg = mean * scale[:, None]
        peak = fg.max(1, keepdims=True)
        fg = np.where(peak > 255, fg * (255 / np.maximum(peak, 1e-6)), fg)
        return fg

    def _quantize(self, rgb: np.ndarray) -> np.ndarray:
        if self.colors == "gray":
            y = rgb @ _LUMA
            rgb = np.repeat(y[:, None], 3, 1)
        if self.colors == "256":
            d = ((rgb[:, None, :] - self._pal[None, :, :]) ** 2).sum(2)
            return (d.argmin(1) + 16).astype(np.uint32)
        if self.color_step > 1:
            s = self.color_step
            rgb = np.clip(np.rint(rgb / s) * s, 0, 255)
        return _pack(rgb)


# --------------------------------------------------------------------------
# ANSI encoding
# --------------------------------------------------------------------------


class Encoder:
    """Turn Cells into ANSI bytes, redrawing only the cells that changed.

    Cursor moves are written as CSI row;col H with 1-based positions inside
    the picture; the player shifts them to centre the picture.
    """

    GAP = 6  # unchanged cells cheaper to resend than a cursor move

    def __init__(self, renderer: Renderer):
        self.chars = renderer.chars
        self.colors = renderer.colors
        self.ink_only = renderer.mode_name != "blocks" and renderer.style == "ink"
        self.space = renderer.chars.index(" ") if " " in renderer.chars else -1
        self.full = renderer.chars.index("█") if "█" in renderer.chars else -1
        self.prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def reset(self):
        self.prev = None

    def _fg(self, v: int) -> str:
        if self.colors == "256":
            return f"\x1b[38;5;{v}m"
        return f"\x1b[38;2;{v >> 16};{(v >> 8) & 255};{v & 255}m"

    def _bg(self, v: int) -> str:
        if self.colors == "256":
            return f"\x1b[48;5;{v}m"
        return f"\x1b[48;2;{v >> 16};{(v >> 8) & 255};{v & 255}m"

    def encode(self, cells: Cells) -> bytes:
        ch, fg, bg = cells.ch, cells.fg.copy(), cells.bg.copy()
        # Colours a cell cannot show must not count as changes.
        if self.space >= 0:
            fg[ch == self.space] = 0
        if self.full >= 0:
            bg[ch == self.full] = 0
        if self.ink_only:
            bg[:] = 16 if self.colors == "256" else 0

        if self.prev is None:
            changed = np.ones(ch.shape, bool)
        else:
            pch, pfg, pbg = self.prev
            changed = (ch != pch) | (fg != pfg) | (bg != pbg)
        self.prev = (ch, fg, bg)

        use_colour = self.colors != "none"
        out: list[str] = []
        cur_fg = cur_bg = None
        chars = self.chars
        rows_changed = np.flatnonzero(changed.any(1))
        for r in rows_changed.tolist():
            cols = np.flatnonzero(changed[r]).tolist()
            ch_row, fg_row, bg_row = ch[r].tolist(), fg[r].tolist(), bg[r].tolist()
            start = cols[0]
            end = cols[0]
            spans = []
            for col in cols[1:]:
                if col - end > self.GAP:
                    spans.append((start, end))
                    start = col
                end = col
            spans.append((start, end))
            for lo, hi in spans:
                out.append(f"\x1b[{r + 1};{lo + 1}H")
                for col in range(lo, hi + 1):
                    if use_colour:
                        b = bg_row[col]
                        if b != cur_bg:
                            out.append(self._bg(b))
                            cur_bg = b
                        f = fg_row[col]
                        if f != cur_fg and ch_row[col] != self.space:
                            out.append(self._fg(f))
                            cur_fg = f
                    out.append(chars[ch_row[col]])
        return "".join(out).encode("utf-8")
