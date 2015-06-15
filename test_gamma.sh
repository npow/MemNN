#!/bin/bash
for gamma in 0.1 0.01 0.001 1 10 100; do
  echo $gamma
  python -u main.py --train_file data/en/qa2_*train.txt --test_file data/en/qa2_*test.txt --gamma $gamma > results/q2_${gamma}.txt &
done 
wait
