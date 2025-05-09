#!/bin/bash

set -eo pipefail
if [ -e "run_py.srun" ]
then
    PYTHON="sbatch run_py.srun"
else
    PYTHON="python"
fi

TREEFILE="one_pop.trees"

OUTDIR="one_pop_data"
if [ ! -e $OUTDIR ]
then
    mkdir -p $OUTDIR
    for L in 1e8 5e8 10e8
    do
        for N in 800 1000 2000 3000 4000 5000
        do
            for H in 0.1 0.25 0.5
            do
                SEED=$RANDOM$RANDOM
                OUT=$OUTDIR/run_${L}_${N}_${H}_${SEED}
                echo "$OUT  ....."
                $PYTHON run_prediction.py --num_indivs $N --prop_observed 0.9 \
                    --length $L --heritability $H --tau 0.1 \
                    --seed $SEED $TREEFILE $OUT
                SEED=$((SEED + 1))
            done
        done
    done
    # do a few more bigger ones
    for L in 1e8 5e8 10e8
    do
        for N in 10000 20000
        do
            H=0.25
            SEED=$RANDOM$RANDOM
            OUT=$OUTDIR/run_${L}_${N}_${H}_${SEED}
            echo "$OUT  ....."
            $PYTHON run_prediction.py --num_indivs $N --prop_observed 0.9 \
                --length $L --heritability $H --tau 0.1 \
                --seed $SEED $TREEFILE $OUT
            SEED=$((SEED + 1))
        done
    done
else
    echo "${OUTDIR} already exists; not overwriting - (re)move it to run anew."
fi

OUTDIR="one_pop_rank"
if [ ! -e $OUTDIR ]
then
    mkdir -p $OUTDIR

    L=10e8
    H=0.25
    for RANK in 1 10 30 60
    do
        for N in 1000 2000 3500 5000 10000
        do
            SEED=$RANDOM$RANDOM
            OUT=$OUTDIR/run_${L}_${N}_${H}_${RANK}_${SEED}
            echo "$OUT  ....."
            $PYTHON run_prediction.py --num_indivs $N --prop_observed 0.9 \
                --length $L --heritability $H --tau 0.1 --pcg_rank $RANK \
                --seed $SEED $TREEFILE.trees $OUT
            SEED=$((SEED + 1))
        done
    done
else
    echo "${OUTDIR} already exists; not overwriting - (re)move it to run anew."
fi

TREEFILE="two_pop.trees"
OUTDIR="two_pop_data"
if [ ! -e $OUTDIR ]
then
    mkdir -p $OUTDIR
    for L in 1e8 5e8 10e8
    do
        for N in 800 1000 2000 3000 4000 5000
        do
            for H in 0.1 0.25 0.5
            do
                SEED=$RANDOM$RANDOM
                OUT=$OUTDIR/run_${L}_${N}_${H}_${SEED}
                echo "$OUT  ....."
                $PYTHON run_prediction.py --num_indivs $N --prop_observed 0.9 \
                    --length $L --heritability $H --tau 0.1 \
                    --seed $SEED $TREEFILE $OUT
                SEED=$((SEED + 1))
            done
        done
    done
else
    echo "${OUTDIR} already exists; not overwriting - (re)move it to run anew."
fi


TREEFILE="bigger_pop.trees"
OUTDIR="bigger_pop_data"
if [ ! -e $OUTDIR ]
then
    mkdir -p $OUTDIR
    for L in 1e8 5e8 10e8
    do
        for N in 800 1000 2000 3000 4000 5000
        do
            for H in 0.1 0.25 0.5
            do
                SEED=$RANDOM$RANDOM
                OUT=$OUTDIR/run_${L}_${N}_${H}_${SEED}
                echo "$OUT  ....."
                $PYTHON run_prediction.py --num_indivs $N --prop_observed 0.9 \
                    --length $L --heritability $H --tau 0.1 \
                    --seed $SEED $TREEFILE $OUT
                SEED=$((SEED + 1))
            done
        done
    done
else
    echo "${OUTDIR} already exists; not overwriting - (re)move it to run anew."
fi



