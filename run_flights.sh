#!/usr/bin/env bash
# =============================================================================
# run_flights.sh  –  Scrape Kraków Airport departures and store in SQLite
#
# Usage:
#   ./run_flights.sh                        # process yesterday (default)
#   ./run_flights.sh --date 2026-04-25      # process a specific date
#   ./run_flights.sh --date 2026-04-25 --db /data/baza_lotow.db
#   ./run_flights.sh --html-dir /data/html --db /data/baza_lotow.db
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_DIR="${SCRIPT_DIR}/html"
DB_PATH="${SCRIPT_DIR}/baza_lotow.db"
DATE_ARG=""          # empty = yesterday (scripts decide)
KEEP_HTML=true       # set to false to delete HTML after parsing

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)      DATE_ARG="$2";   shift 2 ;;
        --db)        DB_PATH="$2";    shift 2 ;;
        --html-dir)  HTML_DIR="$2";   shift 2 ;;
        --no-keep)   KEEP_HTML=false; shift   ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "[run_flights] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$HTML_DIR"

log "========================================"
log "Starting Kraków Airport flight pipeline"
log "HTML dir : $HTML_DIR"
log "DB       : $DB_PATH"
[[ -n "$DATE_ARG" ]] && log "Date     : $DATE_ARG" || log "Date     : yesterday (auto)"
log "========================================"

# ── Step 1: Scrape the page ───────────────────────────────────────────────────
log "STEP 1 – Scraping airport website..."

SCRAPER_ARGS=(--output-dir "$HTML_DIR")
[[ -n "$DATE_ARG" ]] && SCRAPER_ARGS+=(--date "$DATE_ARG")

SCRAPER_OUTPUT=$(python3 "${SCRIPT_DIR}/page_scraper.py" "${SCRAPER_ARGS[@]}")
echo "$SCRAPER_OUTPUT"   # echo so it appears in logs

# Extract the HTML file path from the OUTPUT_FILE= line
HTML_FILE=$(echo "$SCRAPER_OUTPUT" | grep '^OUTPUT_FILE=' | cut -d'=' -f2-)

if [[ -z "$HTML_FILE" || ! -f "$HTML_FILE" ]]; then
    log "ERROR: Scraper did not produce a valid HTML file."
    exit 1
fi

log "STEP 1 done – HTML saved to: $HTML_FILE"

# ── Step 2: Parse the HTML into the database ──────────────────────────────────
log "STEP 2 – Parsing HTML into database..."

PARSER_ARGS=(--html "$HTML_FILE" --db "$DB_PATH")
[[ -n "$DATE_ARG" ]] && PARSER_ARGS+=(--date "$DATE_ARG")

python3 "${SCRIPT_DIR}/parser.py" "${PARSER_ARGS[@]}"

log "STEP 2 done – database updated: $DB_PATH"

# ── Optional cleanup ──────────────────────────────────────────────────────────
if [[ "$KEEP_HTML" == false ]]; then
    rm -f "$HTML_FILE"
    log "HTML file removed (--no-keep)."
fi

log "Pipeline finished successfully."