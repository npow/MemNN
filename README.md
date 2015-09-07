## This repository is deprecated in favor of End-To-End Memory Networks: [npow/MemN2N](https://github.com/npow/MemN2N)

# Description
This is an implementation of [Memory Networks](http://arxiv.org/abs/1410.3916) in Theano. The dataset can be found [here](http://fb.ai/babi).

Below are the results obtained from the current implementation on [the set of toy tasks](http://arxiv.org/abs/1502.05698). Performance is better than LSTMs in most of the tasks, but it is still not quite at the level as the MemNN results reported in the original paper. 

| Task#| N-gram Classifier | LSTM | MemNN (Weston 2014) | This Repo |
|------|-------------------|------|---------------------|-----------|
| 1    | 36                | 50   | 100                 | 76        |
| 2    | 2                 | 20   | 100                 | 70        |
| 3    | 7                 | 20   | 20                  | 18        |
| 4    | 50                | 61   | 71                  | 67        |
| 5    | 20                | 70   | 83                  | 81        |
| 6    | 49                | 48   | 47                  | 45        |
| 7    | 52                | 49   | 68                  | N/A       |
| 8    | 40                | 45   | 77                  | N/A       |
| 9    | 62                | 64   | 65                  | 67        |
| 10   | 45                | 54   | 59                  | 54        |
| 11   | 29                | 72   | 100                 | 54        |
| 12   | 9                 | 74   | 100                 | 63        |
| 13   | 26                | 94   | 100                 | 46        |
| 14   | 19                | 27   | 99                  | 100       |
| 15   | 20                | 21   | 74                  | 100       |
| 16   | 43                | 23   | 27                  | 40        |
| 17   | 46                | 51   | 54                  | 56        |
| 18   | 52                | 52   | 57                  | 53        |
| 19   | 0                 | 8    | 0                   | N/A       |
| 20   | 76                | 91   | 100                 | 100       |
| Mean | 34                | 49   | 75                  | 64.1      |


