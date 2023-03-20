Substituted words
=================

Task description
----------------

You are given a tokenized sentence. Some tokens in it might be replaced with
a randomly picked word. For example:

```
# Original sentence:
the cat sat on the mat

# Two words replaced:
the cat apple on done mat
```


Your task is to predict which words were substituted. Specifically, for
each token in a sentence, write a probability of a replacement:

For example:

```
# Input:
the cat apple on done mat

# Output:
0.1 0.2 0.89 0.1 0.99 0.3
```


Data
----

The `data` folder contains the following files:

    data/train.src  - training dataset, source (input) sentences.
    data/train.lbl  - training dataset, token level labels.
    data/train.tgt  - training dataset, target (corrected) sentences.
    data/val.src    - validation dataset, source (input) sentences.
    data/val.lbl    - validation dataset, token level labels.
    data/val.tgt    - validation dataset, target (corrected) sentences.
    data/test.src   - test set, source (input) sentences.

The `*.src` files are what your model gets as input. `*.lbl` are per-token
labels. `*.tgt` are original sentences without any replaced words. 
It is completely up to you whether to use .tgt (or .lbl) files.

Each file contains one sentence per line. Sentences are tokenized.
Tokens are space-separated.

`*.lbl` files contain space-separated labels, 0s and 1s. Ones indicate
tokens for which replacement were made. For example:

```
# train.src:
the cat APPLE on DONE mat

# train.lbl
0 0 1 0 1 0

# train.tgt
the cat sat on the mat
```

You are free to use any additional training data, embeddings, and linguistic
resources. Just don't forget to mention it, please.


Evaluation
----------

For evaluation, we first convert model's probabilities into the hard labels.
Probabilities less than 0.5 correspond to the negative class (no substitution
made at the position), everything else is positive.

The metric we use for evaluation is F0.5. It combines precision and recall
of your predictions into a single number between 0 and 1. The higher is
the better.

To run evaluation:

    $ ./eval.py path/to/your/submission.txt

This will print number of false positives (FP), false negatives (FN),
true positives (TP), true negatives (TN), precision, recall and, finally,
F0.5.

The "predict random" baseline has F0.5 of around 0.10. We expect your solution
to produce the F0.5-score around 0.80 (or higher).


Submission
----------

When ready, send us an archive or a (private) repository link with the
following:

1. A file called `test.lbl`, your model's output for `data/test.src`.
2. Solution source code.
3. Description of your method(s), possible future work and anything else you
   want to mention.


**NOTE**

    Please, do not share the test task or your solution! When storing
    code on services like GitHub, make sure it is private. Thank you.

