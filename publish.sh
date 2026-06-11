#!/usr/bin/env bash
# publish.sh — git wrapper for submission. NOT a GitHub API client.
set -euo pipefail
MSG="${1:-"chore: update pipeline outputs"}"
git add -A
git commit -m "$MSG"
git push
