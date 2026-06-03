#!/usr/bin/env bash
# Restore the committed agent-memory snapshot (.claude/memory/) into Claude Code's per-project
# memory dir, so a fresh machine picks up the project "brain" straight from the repo clone —
# no separate ~/.claude backup needed.
#
# Usage (run from anywhere; resolves the repo root from the script's own location):
#   scripts/restore_claude_memory.sh             # copy the snapshot into ~/.claude (safe, default)
#   scripts/restore_claude_memory.sh --symlink    # instead symlink the per-project memory dir AT the
#                                                  # repo's .claude/memory, so future memory edits are
#                                                  # written straight into the repo (git-tracked, and
#                                                  # the next reformat needs only the repo again)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/.claude/memory"
# Claude Code names the per-project state dir after the repo's ABSOLUTE path with every
# non-alphanumeric char replaced by '-'  (e.g. /Users/pmrj/Environment/quant/finally_quanta
# -> -Users-pmrj-Environment-quant-finally-quanta). Derived from REPO so it stays correct no
# matter where the repo is cloned. If Claude has already created the dir, this must match it.
SLUG="$(printf '%s' "$REPO" | sed 's/[^A-Za-z0-9]/-/g')"
DEST="$HOME/.claude/projects/$SLUG/memory"

[ -d "$SRC" ] || { echo "no memory snapshot at $SRC — nothing to restore" >&2; exit 1; }

if [ "${1:-}" = "--symlink" ]; then
  mkdir -p "$(dirname "$DEST")"
  if [ -e "$DEST" ] && [ ! -L "$DEST" ]; then
    mv "$DEST" "$DEST.bak.$$"          # preserve any existing live memory before linking
    echo "moved existing $DEST -> $DEST.bak.$$"
  fi
  rm -f "$DEST"
  ln -s "$SRC" "$DEST"
  echo "symlinked $DEST -> $SRC (future memory edits now land in the repo)"
else
  mkdir -p "$DEST"
  cp "$SRC"/*.md "$DEST"/
  echo "copied $(ls -1 "$SRC"/*.md | wc -l | tr -d ' ') memory files -> $DEST"
fi
