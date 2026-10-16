#!/usr/bin/env bash
# Wait for the wide training run to exit, then regenerate its EDA report so the
# figures match the real checkpoint instead of the earlier smoke metrics.json.
#
# Two guards, both learned the hard way:
#   * PID presence is checked with `tasklist | grep -qw`, never a command-line
#     match (that misfired and started the wide run too early).
#   * The rebuild only runs if metrics.json was written AFTER this watcher
#     started, so a crashed/killed run cannot rebuild figures from stale metrics.
set -u

WIDE_PID=14768                        # src.train --csv data/cleaned_comments_wide.csv
WIDE_CSV=data/cleaned_comments_wide.csv
WIDE_METRICS=models/parsbert-sentiment-3class-wide/metrics.json

alive() { tasklist | grep -qw "$1"; }
metrics_age() { [ -f "$WIDE_METRICS" ] && find "$WIDE_METRICS" -newermt "@$1" | wc -l || echo 0; }

START_EPOCH=$(date +%s)
echo "watch start $(date +%H:%M:%S) wide=$WIDE_PID:$(alive $WIDE_PID && echo up || echo gone)"

while alive "$WIDE_PID"; do sleep 120; done
echo "wide run exited $(date +%H:%M:%S)"
sleep 20  # let the process release its file handles

if [ "$(metrics_age "$START_EPOCH")" -eq 0 ]; then
    echo "SKIP: $WIDE_METRICS was not rewritten by this run - training did not reach a save point."
    exit 1
fi

python -X utf8 -m src.visualize --csv "$WIDE_CSV" --prefix wide
echo "wide report rebuilt $(date +%H:%M:%S)"
