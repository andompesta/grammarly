#!/usr/bin/env python
"""Evaluate submission.

Usage:

    $ ./eval.py path/to/submission.lbl --golden path/to/eval.lbl

"""
import argparse
import os
import sys

def evaluate(submission, golden):
    """Compute metrics on the validation set.

    Args:
        submission: List[str]
        golden: List[str]

    Returns:
        Dict[str, float]: metric name to value mapping

    """

    assert len(submission) == len(golden)

    tp = fp = tn = fn = 0

    for lineno, (line_sub, line_gold) in enumerate(zip(submission, golden), 1):
        try:
            sub = list(map(float, line_sub.strip().split()))
            gold = list(map(float, line_gold.strip().split()))
        except Exception as e:
            print("Error parsing line #{lineno}: {e}".format(**vars()),
                  file=sys.stderr)
            sys.exit(1)

        if len(sub) != len(gold):
            msg = "Error at line #{lineno}: number of tokens mismatch"
            print(msg.format(**vars()), file=sys.stderr)
            sys.exit(1)

        tp += sum(1 for x, y in zip(gold, sub) if x == 1 and y >= 0.5)
        tn += sum(1 for x, y in zip(gold, sub) if x == 0 and y < 0.5)
        fp += sum(1 for x, y in zip(gold, sub) if x == 0 and y >= 0.5)
        fn += sum(1 for x, y in zip(gold, sub) if x == 1 and y < 0.5)

    try:
        prec = tp / (tp + fp)
    except ZeroDivisionError:
        prec = 1.0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return {
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn,
        "Precision": prec,
        "Recall": recall,
        "F0.5": _f_measure(prec, recall, 0.5),
    }


def _f_measure(p, r, beta):
    try:
        return (1 + beta**2) * p * r / (beta**2 * p + r)
    except ZeroDivisionError:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission',
                        type=argparse.FileType('rt'),
                        help='Path to the submission file')
    parser.add_argument("--golden",
                        help=( "Path to the golden labels file. Default is "
                              "data/val.lbl"))
    args = parser.parse_args()

    if args.golden is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        golden_path = os.path.join(data_dir, 'val.lbl')
    else:
        golden_path = args.golden

    with open(golden_path):
        golden = list(open(golden_path))

    print("Evaluating...", file=sys.stderr)
    metrics = evaluate(list(args.submission.readlines()), golden)
    for metric, score in metrics.items():
        print("{}: {}".format(metric, score))


if __name__ == '__main__':
    main()
