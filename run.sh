#!/bin/bash
#for x in 2 3 11 13 14 15 16 17 18; do
for x in 1 4 5 6 9 10 12 20; do
  echo $x
  time python -u main.py data/en/qa${x}_*_train.txt data/en/qa${x}_*_test.txt > results/q${x}.txt &
done
wait
