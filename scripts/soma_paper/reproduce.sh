#!/bin/bash
# Reproduce the paper's pipeline (paper/main.tex, Steps 0-5) for one or more participants, using the scripts in
# scripts/soma_all. Every stage writes under results/soma/ and logs to results/soma/paper/<stage>_<P>.log.
#   smoke    synthetic quickstart (installation check; no recordings, says nothing about accuracy)
#   static   Step 1: palm-pin search + up to 7 static passes. The script only PROPOSES a pass; the paper's statics
#            were accepted by a human review of the mesh and markers, which this does not replace
#   label    Step 2: calibrate on the static, train the participant's labeller, label every trial
#   eval     labelling accuracy on the reviewed frames (Vicon vs this participant's model)
#   fill     Steps 3-5: held-out filler check, own-data filler, fill + fit (filled values weighted 0.1), score
# Usage (repo root): bash scripts/soma_paper/reproduce.sh smoke
#                    bash scripts/soma_paper/reproduce.sh "static label eval fill" P3 P4 ...
set -euo pipefail                                  # any failed stage stops the run with its exit code
PY=.viconvenv/Scripts/python.exe
S=scripts/soma_all
L=results/soma/paper
stages=$1; shift
mkdir -p $L
if [[ " $stages " == *" smoke "* ]]; then
  $PY $S/quickstart_synth.py --out results/soma/quickstart 2>&1 | tee $L/smoke.log
fi
for p in "$@"; do
  if [[ " $stages " == *" static "* ]]; then
    $PY $S/static_search.py --participants $p --palm-manual configs/surface_fit/P4_palm_pins.json \
        --radius 8 --arm palm_manual_r8 > $L/static_$p.log 2>&1
    $PY $S/static_passes.py --participant $p --start palm_manual_r8 --out results/soma/static_passes/$p \
        >> $L/static_$p.log 2>&1
  fi
  if [[ " $stages " == *" label "* ]]; then
    $PY $S/run_fullset.py --participants $p > $L/label_$p.log 2>&1
  fi
  if [[ " $stages " == *" eval "* ]]; then
    $PY $S/eval_sample.py --model results/soma/models/labeler_P10.pt --model-for $p=results/soma/models/labeler_$p.pt \
        --participants $p --out results/soma/eval/check_$p > $L/eval_$p.log 2>&1
  fi
  if [[ " $stages " == *" fill "* ]]; then
    bash $S/run_realfill.sh $p                     # logs: results/soma/real_fill/<P>_*.log
  fi
  echo "$p finished: $stages"
done
