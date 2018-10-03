#!/usr/bin/bash

# Worked well for some carillo simulations
python -m egt.main \
       --minimizer carillo \
       --test-function double_ackley \
       --max-iterations 1000000 \
       --beta 30 \
       --gamma 6 \
       --stepsize 2.5e-3 \
       --n-points 100000 \
       --plot-range -1000 10000 \
       --point-interval 9999 10000 \
       --max-animation-seconds 30 \
