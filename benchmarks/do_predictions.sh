#!/bin/bash

set -eo pipefail

OUTDIR="one_pop_data"
if [ ! -e $OUTDIR ]
then
    mkdir -p $OUTDIR

    for L in 1e6 2e6 3e6 # 1e8 5e8 10e8
    do
        for N in 200 400 600 # 800 1000 2000 3000 4000 5000
        do
            for H in 0.1 0.25 0.5
            do
                SEED=$RANDOM$RANDOM
                OUT=$OUTDIR/run_${L}_${N}_${H}_${SEED}
                echo "$OUT  ....."
                python run_prediction.py --num_indivs $N --prop_observed 0.9 \
                    --length $L --heritability $H --tau 0.1 \
                    --seed $SEED one_pop.trees $OUT
                SEED=$((SEED + 1))
            done
        done
    done
fi

OUTDIR="one_pop_rank"
if [ ! -e $OUTDIR ]
then
    mkdir -p $OUTDIR

    L=1e6 # 10e8
    H=0.25
    for RANK in 1 10 20 30
    do
        for N in 200 400 600 # 800 1000 2000 3000 4000 5000 10000
        do
            SEED=$RANDOM$RANDOM
            OUT=$OUTDIR/run_${L}_${N}_${H}_${SEED}
            echo "$OUT  ....."
            python run_prediction.py --num_indivs $N --prop_observed 0.9 \
                --length $L --heritability $H --tau 0.1 --pcg_rank $RANK \
                --seed $SEED one_pop.trees $OUT
            SEED=$((SEED + 1))
        done
    done
fi
