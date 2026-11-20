# asciivid

Play any video in your terminal, drawn with characters. Or bake a video into a
single executable file that plays it whenever you run it.

```bash
asciivid play movie.mp4
asciivid bake movie.mp4 -o movie-ascii   # then: ./movie-ascii
```

Each terminal cell holds a character, a foreground colour and a background
colour. For every cell, asciivid looks at a small block of the frame's pixels.
It picks the character whose shape matches that block, then works out the two
colours that reproduce the block most closely. With the default settings the
picture stays faithful to the video and the characters stay visible.

## Install

Needs Python 3.11+, numpy, and ffmpeg (`brew install ffmpeg`, `apt install ffmpeg`).

```bash
git clone https://github.com/SurajMazar/asciivid && cd asciivid
./install.sh              # adds an `asciivid` command to ~/.local/bin
```

Or install it as a package with `pipx install .` or `pip install --user .`.
To run it from the checkout without installing anything:

```bash
python3 -m asciivid play movie.mp4
```

## Commands

| Command | What it does |
| --- | --- |
| `asciivid play VIDEO` | Plays the video in this terminal, with sound if it has any. `q` quits, `space` pauses. |
| `asciivid bake VIDEO -o OUT` | Writes `OUT`, a standalone executable that plays the video. |
| `asciivid export VIDEO -o out.mp4` | Saves the terminal rendering as an ordinary MP4 to share. `--compare` puts the original above it. Needs Pillow. |
| `asciivid snapshot VIDEO --at 3.5 -o f.png` | Renders one frame to a PNG, to try settings without playing (needs Pillow). `--compare` puts the real frame above it. |
| `asciivid info VIDEO` | Shows the size, frame rate and length, and how it fits your terminal. |

### Baked executables

`bake` renders every frame ahead of time, at several widths (`--tiers 80,120,160,200,240,300`
by default). The output is one file: a Python zip app holding the
compressed frames, the sound, and a small player that uses only the standard
library. It needs neither numpy nor ffmpeg, only `python3`. On start it picks
the largest width that fits the window.

```bash
./movie-ascii            # play
./movie-ascii --loop     # forever
./movie-ascii --list     # the baked sizes, and which one fits this window
./movie-ascii --cols 160 # force a size
./movie-ascii --speed 0.5
```

For the most detail, make the terminal full screen and the font small (Cmd -)
before you run it.

## Options

The same options work for `play`, `bake` and `snapshot`.

| Option | Default | |
| --- | --- | --- |
| `--fps N` | the video's own | Frames per second to render. |
| `--cols N`, `--rows N` | fit the terminal | The size in characters. |
| `--mode` | `glyph` | `glyph`: each cell gets the character whose shape fits it. `ramp`: one character per brightness level, classic ASCII art. `blocks`: half blocks `▀`, two pixels per cell. `quad`: quarter blocks. `braille`: braille dots, 2x4 per cell. |
| `--charset` | `ascii` | Characters for `glyph` and `ramp`: `ascii`, `code`, `symbols`, `letters`, `binary`, `simple`, `matrix`, or your own string (printable ASCII). |
| `--style` | `full` | `full`: coloured paper behind the characters. `ink`: characters on black. |
| `--paper` | `0.5` | `full` style only. `1` gives the most exact picture: flat areas are plain colour. Lower values darken the paper and brighten the ink, so the characters stand out while each cell keeps its average colour. |
| `--colors` | `truecolor` | `truecolor`, `256`, `gray`, `none`. |
| `--color-step N` | `1` | Rounds colours to steps of N. 4-8 cuts the output for slow terminals. |
| `--cell-aspect` | `2.0` | Your font's cell height divided by its width. Adjust it if the picture looks squashed or stretched. |
| `--brightness`, `--contrast`, `--gamma`, `--saturation` | `0, 1, 1, 1` | Picture adjustments (ffmpeg `eq`). |
| `--start S`, `--duration S` | whole video | Play or bake a part of the video. |
| `--loop`, `--no-audio`, `--av-offset S`, `--stats` | | `play` only. |

### Config file

Defaults can live in `~/.config/asciivid/config.toml`, or in any TOML file
you pass with `--config`. Flags on the command line win over the file. See
[`config.example.toml`](config.example.toml).

```toml
style = "ink"
charset = "code"
fps = 24
```

## Examples

```bash
asciivid play clip.mp4 --style ink --charset code     # coloured code characters on black
asciivid play clip.mp4 --paper 1                      # the most exact picture
asciivid play clip.mp4 --mode blocks                  # pixel-perfect half blocks
asciivid play clip.mp4 --mode braille --style ink     # braille dots
asciivid play clip.mp4 --colors none --mode ramp      # plain text, no colour
asciivid bake clip.mp4 --fps 24 --tiers 120,200 -o clip-ascii
asciivid export clip.mp4 --compare -o clip-compare.mp4   # side by side with the original
```

## Terminals

Use a terminal with truecolor support: iTerm2, Ghostty, kitty, WezTerm,
Alacritty, or the VS Code terminal. A wide window at 30 fps sends several MB
per second. If a terminal falls behind, the live player skips frames to keep
time. You can also lower `--fps`, use a narrower window, or raise
`--color-step`. If colours come out wrong, for example in older versions of
Terminal.app, use `--colors 256`.

## How it works

1. ffmpeg decodes the video, applies `fps` and `eq`, and scales each frame
   (area averaging) to `cols x sub_w` by `rows x sub_h` pixels. Each cell then
   owns a `sub_w x sub_h` block: 4x8 for characters, 1x2 for half blocks,
   2x4 for braille.
2. `asciivid/glyphs.py` holds how much ink each ASCII character puts on each
   part of a 4x8 cell, measured from Menlo (`tools/gen_glyphs.py` regenerates it
   from another font). A cell with an edge gets the character with the same
   shape. A flat cell gets a character from a ladder of ink densities.
3. Least squares gives the foreground and background colours that best
   reproduce the block under that character.
4. The encoder only redraws cells that changed since the previous frame, and
   wraps each frame in synchronized output so it doesn't tear.

## License

MIT
