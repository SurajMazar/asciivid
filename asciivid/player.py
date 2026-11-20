"""Terminal playback loop. Standard library only.

This module is also copied into every baked executable, so it must not import
numpy or anything from the asciivid package.
"""

from __future__ import annotations

import json
import lzma
import os
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zipfile

ENTER = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[0m\x1b[2J"
LEAVE = "\x1b[0m\x1b[2J\x1b[?7h\x1b[?25h\x1b[?1049l"
SYNC_BEGIN = b"\x1b[?2026h"  # synchronized output: no tearing where supported
SYNC_END = b"\x1b[?2026l"
_CUP = re.compile(rb"\x1b\[(\d+);(\d+)H")


class Terminal:
    """Alternate screen, hidden cursor, raw-ish keyboard. Restores on exit."""

    def __init__(self):
        self.fd_in = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.saved = None
        self.out = sys.stdout.buffer

    def __enter__(self):
        if self.fd_in is not None:
            import termios
            import tty
            self.saved = termios.tcgetattr(self.fd_in)
            tty.setcbreak(self.fd_in)
        self.out.write(ENTER.encode())
        self.out.flush()
        return self

    def __exit__(self, *exc):
        self.out.write(LEAVE.encode())
        self.out.flush()
        if self.saved is not None:
            import termios
            termios.tcsetattr(self.fd_in, termios.TCSADRAIN, self.saved)

    def wait_key(self, timeout: float) -> str | None:
        """Sleep up to `timeout` seconds; return a key if one was pressed."""
        timeout = max(0.0, timeout)
        if self.fd_in is None:
            time.sleep(timeout)
            return None
        ready, _, _ = select.select([self.fd_in], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd_in, 32).decode("utf-8", "ignore")
        return data[:1] if data else None


def terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size((80, 24))
    return size.columns, size.lines


def shift(data: bytes, dy: int, dx: int) -> bytes:
    if not dy and not dx:
        return data
    return _CUP.sub(lambda m: b"\x1b[%d;%dH" % (int(m[1]) + dy, int(m[2]) + dx), data)


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def audio_command(path: str, start: float = 0.0, duration: float | None = None,
                  speed: float = 1.0) -> list[str] | None:
    """A command that plays `path` (from `start`) with no window, or None."""
    if shutil.which("ffplay"):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if start:
            cmd += ["-ss", str(start)]
        if duration:
            cmd += ["-t", str(duration)]
        if speed != 1.0:
            cmd += ["-af", f"atempo={speed}"]
        return cmd + [path]
    if start or duration or speed != 1.0:
        return None
    for player, args in (("afplay", []), ("mpv", ["--no-video", "--really-quiet"]),
                         ("cvlc", ["--play-and-exit", "--quiet"])):
        if shutil.which(player):
            return [player, *args, path]
    return None


class Audio:
    def __init__(self, cmd: list[str] | None):
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None

    def start(self):
        self.stop()
        if self.cmd:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def pause(self, paused: bool):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGSTOP if paused else signal.SIGCONT)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGCONT)
            self.proc.terminate()
            try:
                self.proc.wait(1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# --------------------------------------------------------------------------
# Playback loop
# --------------------------------------------------------------------------


class Source:
    """Frames for one pass of the video.

    frame(i) returns the ANSI bytes that bring the screen to frame i (they may
    cover several skipped frames), or None at the end.
    """

    cols: int
    rows: int

    def frame(self, i: int) -> bytes | None:
        raise NotImplementedError

    def close(self):
        pass


def play(open_source, fps: float, *, audio: Audio | None = None, loop: bool = False,
         av_offset: float = 0.0, center: bool = True, stats: bool = False) -> None:
    """Play frames from `open_source()` in real time.

    Keys: q / Esc quit, space pauses.
    """
    out = sys.stdout.buffer
    audio = audio or Audio(None)
    with Terminal() as term:
        try:
            while True:
                source = open_source()
                term_cols, term_rows = terminal_size()
                dy = max(0, (term_rows - source.rows) // 2) if center else 0
                dx = max(0, (term_cols - source.cols) // 2) if center else 0
                out.write(b"\x1b[0m\x1b[2J")
                audio.start()
                t0 = time.monotonic() + av_offset
                i, shown, behind = 0, 0, 0
                paused_at = None
                quit_ = False
                while True:
                    now = time.monotonic()
                    due = max(i, int((now - t0) * fps))
                    data = source.frame(due)
                    if data is None:
                        break
                    behind += due - i
                    out.write(SYNC_BEGIN + shift(data, dy, dx) + SYNC_END)
                    out.flush()
                    shown += 1
                    i = due + 1
                    # Sleep until the next frame is due, watching the keyboard.
                    while True:
                        wait = t0 + i / fps - time.monotonic()
                        if paused_at is None and wait <= 0:
                            break
                        key = term.wait_key(wait if paused_at is None else 0.25)
                        if key in ("q", "Q", "\x1b", "\x03"):
                            quit_ = True
                            break
                        if key == " ":
                            if paused_at is None:
                                paused_at = time.monotonic()
                                audio.pause(True)
                            else:
                                t0 += time.monotonic() - paused_at
                                paused_at = None
                                audio.pause(False)
                    if quit_:
                        break
                source.close()
                audio.stop()
                if quit_ or not loop:
                    break
        finally:
            audio.stop()
    if stats:
        print(f"asciivid: drew {shown} frames, skipped {behind} to keep time")


# --------------------------------------------------------------------------
# Baked files: a zip app holding pre-rendered frames
# --------------------------------------------------------------------------
#
#   meta.json            {"fps": .., "tiers": [{"cols", "rows", "frames", "data", "index"}], "audio": name|null}
#   tier-<cols>.xz       lzma stream of every frame's ANSI bytes, back to back
#   tier-<cols>.idx      uint32 little-endian byte length of each frame


class BakedSource(Source):
    def __init__(self, archive: zipfile.ZipFile, tier: dict):
        self.cols, self.rows = tier["cols"], tier["rows"]
        raw = archive.read(tier["index"])
        self.lengths = struct.unpack(f"<{len(raw) // 4}I", raw)
        self.stream = lzma.open(archive.open(tier["data"]))
        self.pos = 0  # next frame to read from the stream

    def frame(self, i: int) -> bytes | None:
        if i >= len(self.lengths):
            return None
        parts = []
        while self.pos <= i:  # deltas build on each other: read every one up to i
            parts.append(self.stream.read(self.lengths[self.pos]))
            self.pos += 1
        return b"".join(parts)

    def close(self):
        self.stream.close()


def pick_tier(tiers: list[dict], cols: int, rows: int) -> dict:
    fitting = [t for t in tiers if t["cols"] <= cols and t["rows"] <= rows]
    if fitting:
        return max(fitting, key=lambda t: t["cols"])
    return min(tiers, key=lambda t: t["cols"])


def main_baked(argv: list[str] | None = None) -> None:
    import argparse

    archive_path = os.path.abspath(sys.argv[0])
    archive = zipfile.ZipFile(archive_path)
    meta = json.loads(archive.read("meta.json"))
    tiers = sorted(meta["tiers"], key=lambda t: t["cols"])

    parser = argparse.ArgumentParser(
        prog=os.path.basename(archive_path),
        description=f"Play {meta.get('title', 'this video')} in the terminal "
                    "(q quits, space pauses).")
    parser.add_argument("--loop", action="store_true", help="play forever")
    parser.add_argument("--no-audio", action="store_true", help="play without sound")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1.0)")
    parser.add_argument("--cols", type=int, help="use the size baked closest to this width")
    parser.add_argument("--list", action="store_true", help="list the baked sizes and exit")
    parser.add_argument("--stats", action="store_true", help="print dropped-frame count after")
    args = parser.parse_args(argv)

    if args.list:
        cols, rows = terminal_size()
        best = pick_tier(tiers, cols, rows)
        for t in tiers:
            mark = "  <- fits this terminal" if t is best else ""
            print(f"{t['cols']:>4} x {t['rows']:<4} {t['frames']} frames{mark}")
        print(f"{meta['fps']:g} fps, mode {meta.get('mode')}, style {meta.get('style')}")
        return

    if args.cols:
        tier = min(tiers, key=lambda t: abs(t["cols"] - args.cols))
    else:
        cols, rows = terminal_size()
        tier = pick_tier(tiers, cols, rows)
        if tier["cols"] > cols or tier["rows"] > rows:
            print(f"Terminal is {cols}x{rows}; the smallest baked size is "
                  f"{tier['cols']}x{tier['rows']}. Make the window bigger or the font "
                  "smaller (Cmd -), then press Enter.", file=sys.stderr)
            try:
                input()
            except EOFError:
                pass

    audio = Audio(None)
    tmp = None
    if meta.get("audio") and not args.no_audio:
        tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(meta["audio"])[1], delete=False)
        tmp.write(archive.read(meta["audio"]))
        tmp.close()
        audio = Audio(audio_command(tmp.name, speed=args.speed))

    try:
        play(lambda: BakedSource(archive, tier), meta["fps"] * args.speed,
             audio=audio, loop=args.loop, stats=args.stats)
    except KeyboardInterrupt:
        pass
    finally:
        if tmp:
            os.unlink(tmp.name)
