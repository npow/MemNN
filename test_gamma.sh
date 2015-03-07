#!/bin/bash
for gamma in 0.1 0.01 0.001 1 10 100; do
  echo $gamma
  python -u main.py data/en/qa2_*train.txt data/en/qa2_*test.txt $gamma > results/q2_${gamma}.txt &
done 
wait
