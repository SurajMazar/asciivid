"""asciivid: play any video in the terminal as characters, or bake it into a
standalone executable.

    asciivid play VIDEO [options]
    asciivid bake VIDEO -o OUT [options]
    asciivid snapshot VIDEO --at SECONDS -o frame.png [options]
    asciivid export VIDEO -o out.mp4 [--compare] [options]
    asciivid info VIDEO

Options can also come from a TOML file: ~/.config/asciivid/config.toml, or the
one --config names. Command-line flags win over the file.
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import zipfile
from pathlib import Path

import numpy as np

from . import __version__, player
from .render import (CHARSETS, COLORS, MODES, STYLES, Encoder, Renderer, fit_grid,
                     probe, read_frames, require)

DEFAULT_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "asciivid" / "config.toml"

# name -> (default, help). Every one is a --flag on play, bake and snapshot,
# and a key in the config file.
OPTIONS: dict[str, tuple[object, str]] = {
    "fps": (None, "frames per second to render (default: the video's own)"),
    "cols": (None, "width in characters (default: fit the terminal)"),
    "rows": (None, "height in characters (default: fit the terminal)"),
    "mode": ("glyph", f"how cells are drawn: {', '.join(MODES)}"),
    "charset": ("ascii", f"characters for glyph/ramp modes: {', '.join(CHARSETS)} or your own string"),
    "style": ("full", f"{' or '.join(STYLES)}: characters on black, or coloured paper behind them too"),
    "paper": (0.5, "full style: paper brightness, 0..1; below 1 the characters stand out "
                   "while each cell keeps its average colour"),
    "colors": ("truecolor", f"colour output: {', '.join(COLORS)}"),
    "color_step": (1, "round colour channels to this step; larger means less output"),
    "cell_aspect": (2.0, "character cell height / width of your terminal font"),
    "brightness": (0.0, "-1..1 added to brightness"),
    "contrast": (1.0, "contrast multiplier"),
    "gamma": (1.0, "gamma (above 1 brightens shadows)"),
    "saturation": (1.0, "colour saturation multiplier"),
    "start": (0.0, "start at this many seconds"),
    "duration": (None, "play only this many seconds"),
}
PLAY_OPTIONS: dict[str, tuple[object, str]] = {
    "loop": (False, "play forever"),
    "no_audio": (False, "play without sound"),
    "av_offset": (0.0, "delay the picture by this many seconds to line up with the sound"),
    "stats": (False, "print how many frames were drawn and skipped"),
}
TYPES = {"fps": float, "cols": int, "rows": int, "color_step": int, "cell_aspect": float, "paper": float,
         "brightness": float, "contrast": float, "gamma": float, "saturation": float,
         "start": float, "duration": float, "av_offset": float}
CHOICES = {"mode": MODES, "style": STYLES, "colors": COLORS}


def add_options(p: argparse.ArgumentParser, options: dict) -> None:
    for name, (default, help_) in options.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true", default=None, help=help_)
        else:
            p.add_argument(flag, type=TYPES.get(name, str), choices=CHOICES.get(name),
                           default=None, help=f"{help_} [{default}]" if default is not None else help_)


def resolve(args: argparse.Namespace, options: dict) -> dict:
    """Defaults < config file < command line."""
    conf = {}
    path = Path(args.config) if args.config else DEFAULT_CONFIG
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)
        conf = {k.replace("-", "_"): v for k, v in raw.items()}
        unknown = set(conf) - set(OPTIONS) - set(PLAY_OPTIONS) - {"tiers"}
        if unknown:
            print(f"asciivid: ignoring unknown config keys in {path}: {', '.join(sorted(unknown))}",
                  file=sys.stderr)
    elif args.config:
        raise SystemExit(f"asciivid: config file {path} not found")
    out = {}
    for name, (default, _) in options.items():
        value = getattr(args, name, None)
        out[name] = value if value is not None else conf.get(name, default)
    for key in ("tiers",):
        if key in conf:
            out.setdefault("_" + key, conf[key])
    return out


def grid_for(info, o: dict, term_cols: int, term_rows: int) -> tuple[int, int]:
    max_cols = o["cols"] or term_cols
    max_rows = o["rows"] or term_rows
    if o["cols"] and not o["rows"]:
        max_rows = 10**6
    if o["rows"] and not o["cols"]:
        max_cols = 10**6
    return fit_grid(info.width, info.height, max_cols, max_rows, o["cell_aspect"])


def make_renderer(cols: int, rows: int, o: dict) -> Renderer:
    return Renderer(cols, rows, mode=o["mode"], charset=o["charset"], style=o["style"],
                    colors=o["colors"], color_step=o["color_step"], paper=o["paper"])


def frames(path: str, r: Renderer, fps: float, o: dict, max_frames: int | None = None):
    return read_frames(path, r.px_w, r.px_h, fps=fps, start=o["start"], duration=o["duration"],
                       max_frames=max_frames,
                       brightness=o["brightness"], contrast=o["contrast"], gamma=o["gamma"],
                       saturation=o["saturation"])


# --------------------------------------------------------------------------
# play
# --------------------------------------------------------------------------


class LiveSource(player.Source):
    """Decode in a background thread; render only the frames we will show."""

    def __init__(self, path: str, o: dict, fps: float, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.renderer = make_renderer(cols, rows, o)
        self.encoder = Encoder(self.renderer)
        self.queue: queue.Queue = queue.Queue(maxsize=int(fps) + 2)
        self.stop = threading.Event()
        self.gen = frames(path, self.renderer, fps, o)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        self.next = 0

    def _pump(self):
        try:
            for f in self.gen:
                while not self.stop.is_set():
                    try:
                        self.queue.put(f, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    break
        finally:
            self.queue.put(None)

    def frame(self, i: int) -> bytes | None:
        f = None
        while self.next <= i:
            f = self.queue.get()
            if f is None:
                return None
            self.next += 1
        return self.encoder.encode(self.renderer.render(f))

    def close(self):
        self.stop.set()
        while self.thread.is_alive():
            try:
                self.queue.get(timeout=0.05)
            except queue.Empty:
                pass
        self.gen.close()


def cmd_play(args) -> None:
    o = resolve(args, {**OPTIONS, **PLAY_OPTIONS})
    if not sys.stdout.isatty():
        raise SystemExit("asciivid: play needs a terminal")
    info = probe(args.video)
    fps = o["fps"] or info.fps
    term_cols, term_rows = player.terminal_size()
    cols, rows = grid_for(info, o, term_cols, term_rows)

    audio = player.Audio(None)
    if info.has_audio and not o["no_audio"]:
        audio = player.Audio(player.audio_command(args.video, o["start"], o["duration"]))

    try:
        player.play(lambda: LiveSource(args.video, o, fps, cols, rows), fps, audio=audio,
                    loop=o["loop"], av_offset=o["av_offset"], stats=o["stats"])
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------
# bake
# --------------------------------------------------------------------------

DEFAULT_TIERS = [80, 120, 160, 200, 240, 300]


def bake_tier(video: str, cols: int, rows: int, o: dict, fps: float, tmpdir: str) -> dict:
    """Render one size to tier-<cols>.xz / .idx in tmpdir. Runs in a worker process."""
    r = make_renderer(cols, rows, o)
    enc = Encoder(r)
    comp = lzma.LZMACompressor(preset=2)
    data_name, index_name = f"tier-{cols}.xz", f"tier-{cols}.idx"
    lengths: list[int] = []
    t = time.monotonic()
    with open(Path(tmpdir) / data_name, "wb") as fdata:
        for f in frames(video, r, fps, o):
            b = enc.encode(r.render(f))
            lengths.append(len(b))
            fdata.write(comp.compress(b))
        fdata.write(comp.flush())
    (Path(tmpdir) / index_name).write_bytes(struct.pack(f"<{len(lengths)}I", *lengths))
    return {"cols": cols, "rows": rows, "frames": len(lengths), "data": data_name,
            "index": index_name, "_raw": sum(lengths), "_secs": time.monotonic() - t,
            "_size": (Path(tmpdir) / data_name).stat().st_size}


def cmd_bake(args) -> None:
    o = resolve(args, OPTIONS)
    info = probe(args.video)
    fps = o["fps"] or info.fps

    if args.tiers:
        widths = [int(x) for x in args.tiers.split(",")]
    elif o["cols"] or o["rows"]:
        widths = [grid_for(info, o, 10**6, 10**6)[0]]
    else:
        widths = o.get("_tiers") or DEFAULT_TIERS
    widths = sorted(set(widths))

    out_path = Path(args.output or Path(args.video).stem).expanduser()
    tmpdir = Path(tempfile.mkdtemp(prefix="asciivid-"))
    meta = {"title": Path(args.video).name, "fps": fps, "mode": o["mode"], "style": o["style"],
            "paper": o["paper"], "colors": o["colors"], "tiers": [], "audio": None}

    jobs = args.jobs or min(len(widths), os.cpu_count() or 1)
    print(f"baking {len(widths)} sizes at {fps:g} fps on {jobs} processes...", file=sys.stderr)
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = []
            for cols in widths:
                rows = round(cols * info.height / info.width / o["cell_aspect"])
                futures.append(pool.submit(bake_tier, args.video, cols, rows, o, fps, str(tmpdir)))
            for fut in as_completed(futures):
                tier = fut.result()
                print(f"  {tier['cols']:>4}x{tier['rows']:<4} {tier['frames']} frames, "
                      f"{tier.pop('_raw') / 1e6:.0f} MB of ANSI -> {tier.pop('_size') / 1e6:.1f} MB "
                      f"({tier.pop('_secs'):.0f}s)", file=sys.stderr)
                meta["tiers"].append(tier)
        meta["tiers"].sort(key=lambda x: x["cols"])

        if info.has_audio and not args.no_audio:
            audio = tmpdir / "audio.m4a"
            cmd = [require("ffmpeg"), "-v", "error", "-nostdin"]
            if o["start"]:
                cmd += ["-ss", str(o["start"])]
            cmd += ["-i", args.video]
            if o["duration"]:
                cmd += ["-t", str(o["duration"])]
            cmd += ["-vn", "-c:a", "aac", "-b:a", "160k", str(audio)]
            subprocess.run(cmd, check=True)
            meta["audio"] = "audio.m4a"

        # A zip app: python3 runs the shebanged zip's __main__.py.
        player_src = Path(player.__file__).read_text()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("__main__.py", "from asciivid_player import main_baked\nmain_baked()\n",
                       zipfile.ZIP_DEFLATED)
            z.writestr("asciivid_player.py", player_src, zipfile.ZIP_DEFLATED)
            z.writestr("meta.json", json.dumps(meta, indent=1))
            for tier in meta["tiers"]:
                z.write(tmpdir / tier["data"], tier["data"], zipfile.ZIP_STORED)
                z.write(tmpdir / tier["index"], tier["index"], zipfile.ZIP_DEFLATED)
            if meta["audio"]:
                z.write(tmpdir / meta["audio"], meta["audio"], zipfile.ZIP_STORED)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"#!/usr/bin/env python3\n")
            f.write(buf.getvalue())
        out_path.chmod(0o755)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB). Run it: ./{out_path.name}"
          if out_path.parent == Path(".") else
          f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


# --------------------------------------------------------------------------
# snapshot: render one frame to a PNG, for checking settings without a terminal
# --------------------------------------------------------------------------


def cmd_snapshot(args) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("asciivid: snapshot needs Pillow (pip install pillow)")
    from .image import CellPainter

    o = resolve(args, OPTIONS)
    info = probe(args.video)
    o["start"] = args.at
    o["duration"] = None
    cols, rows = grid_for(info, o, 160, 10**6) if not (o["cols"] or o["rows"]) else grid_for(info, o, 0, 0)
    r = make_renderer(cols, rows, o)
    f = next(frames(args.video, r, o["fps"] or info.fps, o, max_frames=1), None)
    if f is None:
        raise SystemExit(f"asciivid: no frame at {args.at}s")
    painter = CellPainter(r, args.cell_px, o["cell_aspect"], o["colors"], args.font)
    img = painter.paint(r.render(f))
    if args.compare:
        h, w = img.shape[:2]
        src = next(read_frames(args.video, w, h, fps=o["fps"] or info.fps, start=args.at,
                               max_frames=1), None)
        if src is not None:
            img = np.concatenate([src, np.full((8, w, 3), 40, np.uint8), img])
    Image.fromarray(img).save(args.output)
    print(f"wrote {args.output} ({cols}x{rows} cells)", file=sys.stderr)


# --------------------------------------------------------------------------
# export: the terminal rendering as an ordinary video file
# --------------------------------------------------------------------------


def cmd_export(args) -> None:
    from .image import CellPainter

    o = resolve(args, OPTIONS)
    info = probe(args.video)
    fps = o["fps"] or info.fps
    if not (o["cols"] or o["rows"]):
        o["cols"] = 240
    cols, rows = grid_for(info, o, 0, 0)
    r = make_renderer(cols, rows, o)
    painter = CellPainter(r, args.cell_px, o["cell_aspect"], o["colors"], args.font)
    w, h = cols * painter.cw, rows * painter.ch
    out_h = h * 2 + 8 if args.compare else h
    pad_w, pad_h = w % 2, out_h % 2                         # x264 needs even sizes

    cmd = [require("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{out_h}", "-r", str(fps), "-i", "-"]
    with_audio = info.has_audio and not args.no_audio
    if with_audio:
        if o["start"]:
            cmd += ["-ss", str(o["start"])]
        if o["duration"]:
            cmd += ["-t", str(o["duration"])]
        cmd += ["-i", args.video, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    if pad_w or pad_h:
        cmd += ["-vf", f"pad={w + pad_w}:{out_h + pad_h}"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", args.output]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    originals = None
    if args.compare:
        originals = read_frames(args.video, w, h, fps=fps, start=o["start"], duration=o["duration"])
    gap = np.full((8, w, 3), 40, np.uint8)
    total = int((o["duration"] or info.duration - o["start"]) * fps) if info.duration else 0
    t = time.monotonic()
    n = 0
    try:
        for n, f in enumerate(frames(args.video, r, fps, o), 1):
            img = painter.paint(r.render(f))
            if originals is not None:
                src = next(originals, None)
                if src is None:
                    break
                img = np.concatenate([src, gap, img])
            enc.stdin.write(img.tobytes())
            if n % 15 == 0:
                pct = f"{100 * n / total:3.0f}%" if total else str(n)
                print(f"\r  {pct}", end="", file=sys.stderr, flush=True)
    finally:
        enc.stdin.close()
        enc.wait()
        if originals is not None:
            originals.close()
    if enc.returncode:
        raise SystemExit("asciivid: ffmpeg failed to write the video")
    print(f"\rwrote {args.output}: {n} frames, {w}x{out_h} px from {cols}x{rows} cells "
          f"({time.monotonic() - t:.0f}s)", file=sys.stderr)


def cmd_info(args) -> None:
    info = probe(args.video)
    print(f"{args.video}\n  {info.width}x{info.height}, {info.fps:g} fps, "
          f"{info.duration:.2f}s, audio: {'yes' if info.has_audio else 'no'}")
    cols, rows = player.terminal_size()
    for aspect in (2.0,):
        c, r = fit_grid(info.width, info.height, cols, rows, aspect)
        print(f"  fits this {cols}x{rows} terminal as {c}x{r} characters")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="asciivid", description="Play videos in the terminal as characters.",
        epilog=f"Config file: {DEFAULT_CONFIG} (TOML; keys are the long option names).")
    parser.add_argument("--version", action="version", version=f"asciivid {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="play a video in this terminal (q quits, space pauses)")
    p.add_argument("video")
    p.add_argument("--config", help="TOML file with option defaults")
    add_options(p, OPTIONS)
    add_options(p, PLAY_OPTIONS)
    p.set_defaults(func=cmd_play)

    b = sub.add_parser("bake", help="pre-render a video into a standalone executable")
    b.add_argument("video")
    b.add_argument("-o", "--output", help="executable to write (default: the video's name)")
    b.add_argument("--tiers", help="comma-separated widths to bake; the player picks the largest "
                                   f"that fits its terminal [{','.join(map(str, DEFAULT_TIERS))}]")
    b.add_argument("--no-audio", action="store_true", help="leave the sound out")
    b.add_argument("-j", "--jobs", type=int, help="sizes to render at once [one per CPU]")
    b.add_argument("--config", help="TOML file with option defaults")
    add_options(b, OPTIONS)
    b.set_defaults(func=cmd_bake)

    s = sub.add_parser("snapshot", help="render one frame to a PNG")
    s.add_argument("video")
    s.add_argument("--at", type=float, default=0.0, help="time in seconds [0]")
    s.add_argument("-o", "--output", default="snapshot.png")
    s.add_argument("--compare", action="store_true", help="put the original frame above it")
    s.add_argument("--font", help="monospace font file to draw with")
    s.add_argument("--cell-px", type=int, default=12, help="pixel width of one character [12]")
    s.add_argument("--config", help="TOML file with option defaults")
    add_options(s, OPTIONS)
    s.set_defaults(func=cmd_snapshot)

    e = sub.add_parser("export", help="save the terminal rendering as a video file (MP4)")
    e.add_argument("video")
    e.add_argument("-o", "--output", default="asciivid.mp4")
    e.add_argument("--compare", action="store_true", help="put the original video above it")
    e.add_argument("--cell-px", type=int, default=8, help="pixel width of one character [8]")
    e.add_argument("--crf", type=int, default=16, help="x264 quality, lower is better [16]")
    e.add_argument("--font", help="monospace font file to draw with")
    e.add_argument("--no-audio", action="store_true", help="leave the sound out")
    e.add_argument("--config", help="TOML file with option defaults")
    add_options(e, OPTIONS)
    e.set_defaults(func=cmd_export)

    i = sub.add_parser("info", help="show a video's size, frame rate and length")
    i.add_argument("video")
    i.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
