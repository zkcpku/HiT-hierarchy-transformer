from collections import Counter


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def calculate(predict, original):
    '''
    idx already remove special token (unk,pad,...)
    :param predict: the list of list of idx
    :param original: the list of list of idx
    :return: p,r,f
    '''
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(predict, original):
        p, o = sorted(p), sorted(o)
        common = Counter(p) & Counter(o)
        true_positive += sum(common.values())
        false_positive += (len(p) - sum(common.values()))
        false_negative += (len(o) - sum(common.values()))
    return calculate_results(true_positive, false_positive, false_negative)


def old_calculate(predict, original):
    '''
    used by code2seq, code transformer
    but it will count repeat, so discarded
    :param predict:
    :param original:
    :return:
    '''
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(predict, original):
        p, o = sorted(p), sorted(o)
        if o == p:
            true_positive += len(o)
            continue
        for token in p:
            if token in o:
                true_positive += 1
            else:
                false_positive += 1
        for token in o:
            if token not in p:
                false_negative += 1
    return calculate_results(true_positive, false_positive, false_negative)
