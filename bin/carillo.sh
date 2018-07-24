#!/usr/bin/bash

# Worked well for some carillo simulations
python -m egt.main \
    --max-iterations 10000 \
    --beta 100 \
    --gamma 6 \
    --stepsize 2.5e-3 \
    --n-points 5000 \
    --plot-range -5 5 \
    --point-interval 5 10 \
    --initial-strategy standard \
    --s-rounds 10 \
    --max-animation-seconds 30 \
    --carillo
    # --stepsize 0.0001 \
    # --my-j \
    # --normalize-delta \
    # --save -s 1882876050
    # --max-animation-seconds 60
