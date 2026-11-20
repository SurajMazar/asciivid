#!/bin/sh
# Put an `asciivid` command in ~/.local/bin (or $1) that runs this checkout.
# Edits to the checkout take effect at once; nothing is copied.
set -e
here=$(cd "$(dirname "$0")" && pwd)
bin=${1:-"$HOME/.local/bin"}
mkdir -p "$bin"

python3 -c "import numpy" 2>/dev/null || {
  echo "asciivid needs numpy: python3 -m pip install --user numpy" >&2
  exit 1
}
command -v ffmpeg >/dev/null || echo "note: install ffmpeg too (brew install ffmpeg)" >&2

cat > "$bin/asciivid" <<EOF
#!/bin/sh
PYTHONPATH="$here\${PYTHONPATH:+:\$PYTHONPATH}" exec python3 -m asciivid "\$@"
EOF
chmod +x "$bin/asciivid"
echo "installed $bin/asciivid"
case ":$PATH:" in *":$bin:"*) ;; *) echo "add $bin to your PATH" >&2 ;; esac
