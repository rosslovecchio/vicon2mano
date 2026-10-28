#!/bin/bash
# Real-data gap fill for each participant (docs/strategies/soma_gap_fill.md):
#   1. held-out check: filler trained without one HOI trial, gap_fill_check on that trial against the rigid fill
#   2. filler trained on all the participant's labelled trials
#   3. fill + fit + views (fit_trials.py, filled values weighted 0.1), OneDrive results/soma/<P>_realfill_w01
#   4. score against the rigid-fill fits on the markers really observed
# Usage (repo root): bash scripts/soma/run_realfill.sh P3 P4 ...   Logs: results/soma/real_fill/<P>_*.log
set -u
PY=.viconvenv/Scripts/python.exe
R=results/soma/real_fill
for p in "$@"; do
  L=results/soma/fullset/$p
  trials=$(for d in $L/*/; do t=$(basename $d); [ $t != static ] && [ -f $d/soma_labelled.csv ] && echo $t; done)
  test=$(echo "$trials" | grep -x Trial2_HOI || echo "$trials" | grep -x Trial_1_HOI)
  rest=$(for t in $trials; do [ $t != $test ] && echo $L/$t/soma_labelled.csv; done)
  { echo "== $p held out: $test"
    $PY -W ignore scripts/soma/train_filler.py --csv $rest --out $R/filler_${p}_no_$test.pt | tail -n 2
    for f in "" "--filler $R/filler_${p}_no_$test.pt"; do
      echo "-- ${f:-rigid}"
      $PY -W ignore scripts/soma/gap_fill_check.py --filled results/soma/movement/$p/$test/soma_labelled_filled.csv \
          --lengths 200,1000,2000,5000 --max-gap 100000 $f
    done; } > $R/${p}_heldout.log 2>&1
  $PY -W ignore scripts/soma/train_filler.py --csv $(for t in $trials; do echo $L/$t/soma_labelled.csv; done) \
      --out $R/filler_${p}_all.pt > $R/${p}_train_all.log 2>&1
  $PY -W ignore scripts/soma/fit_trials.py $p --filler $R/filler_{p}_all.pt --max-gap 100000 --w-filled 0.1 \
      --work results/soma/real_fill_w01/movement --name {p}_realfill_w01 \
      --note "gap fill learned from real frames, filled values weighted 0.1" > $R/${p}_fit.log 2>&1
  $PY -W ignore scripts/soma/score_observed.py $p rigid=results/soma/movement real_w01=results/soma/real_fill_w01/movement \
      > $R/${p}_score.log 2>&1
  echo "$p finished"
done
