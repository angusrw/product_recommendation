import pandas as pd
import numpy as np
import time

def evaluate(modelrecs, testdata, numrec_list):
    """ evaluate recommendations from modelrecs using testdata at number of recommendations in maxrecs

    Args:
        modelrecs -- dataframe containing some number of product recommendations per customer
        testdata -- test data in form of long table with 1 row for each customer/product combo
        numrec_list -- list of number of recomendations to evaluate model at
    """
    s = time.time()

    df_metrics = pd.DataFrame(columns=['k','recall','precision','f1'])

    # get dict of dataframes for each customer
    dict_recs = dict(zip(
        modelrecs['cid'].unique(), 
        [modelrecs[modelrecs['cid']==c] for c in modelrecs['cid'].unique()]))

    # get dict of sets of actual purchases in test set for each customer
    dict_test = dict(zip(
        testdata['cid'].unique(), 
        [set(testdata[testdata['cid']==c]['pid'].values) for c in testdata['cid'].unique()]))
    
    # for each number of recommendations, calculate metrics and add to df_metrics
    for k in numrec_list:
        print(f"Evaluating at {k} recomendations")

        # get lists of precision and recall scores for each customer
        recall_precision = list(map(list, zip(*[
            calculate_recall_precision(
                set(dict_recs[c][dict_recs[c]['rank']<=k]['pid']),
                dict_test[c])
            for i,c in enumerate(dict_test.keys())
        ])))

        avg_recall = sum(recall_precision[0])/len(recall_precision[0])
        avg_precision = sum(recall_precision[1])/len(recall_precision[1])
        f1score = 2*(avg_recall*avg_precision)/(avg_recall+avg_precision)

        df_metrics = df_metrics.append(dict(zip(df_metrics.columns, [k,avg_recall,avg_precision,f1score])), ignore_index=True)
    
    return df_metrics


def calculate_recall_precision(preds, actual):
    """ calculates recall and precision from provided sets

    Args:
        preds -- set of predicted purchases
        actual -- set of actual purchases
    """
    tp = len(preds & actual)
    fp = len(preds.difference(actual))
    fn = len(actual.difference(preds))
    return (tp/(tp+fn), tp/(tp+fp))