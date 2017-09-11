## Natural Language -> Regular Expression

---

### Project Organization

- baseline_with_case: model with case encoder.
- ng_baseline: replicate model proposed by Locascio et al.
- ng_beam_search: ng_baseline + beam search, only supports beam size 1.
- error_analysis: several tools to analyze prediction result.

---

### Model architecture

- `main.py`: runtime to control the training and testing process.
- `models.py`: tensorflow Graph
- `data_provider.py`: manager of training and testing data.
- folder `configuration`: contains configuration for model
- folder `feed_tf`: data for tensorflow graph
- folder `log`: log files used for tensorboard
- folder `result_log`: log files for each training phase and test phase, containing log for each epoch.
- folder `checkpoint`: checkpoint for trained models. 


---

### Commands

For **baseline_with_case** and **ng_beam_search**:

**Train**:
```
python main.py --conf ./configuration/xx.json
```

**Test**:
```
python main.py --conf ./configuration/xx.json --checkpoint ./checkpoint/** --test --log
```

Only exact match accuracy, will be calculated in two commands above, for calculating the dfa equality is realy time consuming. 

**Test Dfa equality**:
```
python tool/test_dfa_equality.py --file test.log
```

test.log is the output of test command.

**Recalcuate the accuracy of beam search**:
```
python error_analysis/recalc_beam_search_accuracy.py --file 'output of test dfa equality'
```

--- 

### Requirement:

1. python 3+
2. Java