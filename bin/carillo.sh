#!/usr/bin/bash

# Worked well for some carillo simulations
python -m egt.main \
    --carillo \
    --max-iterations 10000 \
    --beta 100 \
    --gamma 6 \
    --stepsize 2.5e-3 \
    --n-points 5000 \
    --plot-range -10 10 \
    --point-interval 5 100 \
    --max-animation-seconds 30 \
    --max-animation-seconds 10
