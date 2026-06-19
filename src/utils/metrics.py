import numpy as np
from sklearn import metrics
import pandas as pd

def confusion_matrix(df, actual_cnnx, actual_neg_cnnx, cnnx_list):
    failed_runs = []
    nonint_cols = ["Specificity", "Precision", "Recall", "F1", "Accuracy", "Balanced Accuracy"]
    temp_df = pd.DataFrame(columns=["True Positive", "False Positive", "True Negative", "False Negative", "Specificity", "Precision", "Recall", "SHD",  "F1", "Accuracy", "Balanced Accuracy"])
    temp_df["True Positive"] = df.loc[~df.index.isin(failed_runs), actual_cnnx].sum(axis=1)
    temp_df["False Positive"] = df.loc[~df.index.isin(failed_runs), actual_neg_cnnx].sum(axis=1)
    temp_df["True Negative"] = (df.loc[~df.index.isin(failed_runs), actual_neg_cnnx]==0).sum(axis=1)
    temp_df["False Negative"] = (df.loc[~df.index.isin(failed_runs), actual_cnnx]==0).sum(axis=1)

    # Check denominators for division by zero error
    temp_df.loc[(temp_df["True Negative"] + temp_df["False Positive"]) == 0, 
        "Specificity"] = 0
    temp_df.loc[(temp_df["True Negative"] + temp_df["False Positive"]) != 0, 
        "Specificity"] = temp_df["True Negative"]/(temp_df["True Negative"] + temp_df["False Positive"])
    
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Positive"]) == 0,
        "Precision"] = 0
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Positive"]) != 0,
        "Precision"]  = temp_df["True Positive"]/(temp_df["True Positive"] + temp_df["False Positive"])

    temp_df.loc[(temp_df["True Positive"] + temp_df["False Negative"]) == 0,
        "Recall"] = 0
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Negative"]) != 0,
        "Recall"] = temp_df["True Positive"]/(temp_df["True Positive"] + temp_df["False Negative"])
    
    temp_df["SHD"] = temp_df["False Positive"] + temp_df["False Negative"]

    
    denom = temp_df["Precision"] + temp_df["Recall"]
    temp_df["F1"] = 0
    mask = denom != 0
    temp_df.loc[mask, "F1"] = ( 2 * temp_df.loc[mask, "Precision"] * temp_df.loc[mask, "Recall"]) / denom[mask]

    
    temp_df["Accuracy"] = (temp_df["True Positive"] + temp_df["True Negative"] )/len(cnnx_list)
    temp_df["Balanced Accuracy"] = (temp_df["Recall"] + temp_df["Specificity"] )/2

    df = pd.concat([df, temp_df], axis=1)
    df = df.astype({col: 'Int64' for col in df.columns if col not in nonint_cols})
    means = df.loc[~df.index.isin(failed_runs), cnnx_list + ["Specificity", "Precision", "Recall", "SHD", "F1", "Accuracy", "Balanced Accuracy"]].mean()
    medians = df.loc[~df.index.isin(failed_runs), cnnx_list + ["Specificity", "Precision", "Recall", "SHD", "F1", "Accuracy", "Balanced Accuracy"]].median()
    variances = df.loc[~df.index.isin(failed_runs), cnnx_list + ["Specificity", "Precision", "Recall", "SHD", "F1", "Accuracy", "Balanced Accuracy"]].var()
    stddev = df.loc[~df.index.isin(failed_runs), cnnx_list + ["Specificity", "Precision", "Recall", "SHD", "F1", "Accuracy", "Balanced Accuracy"]].std()

    df.loc['mean'] = means
    df.loc['median'] = medians
    df.loc['variance'] = variances
    df.loc['stddev'] = stddev

    return df

"""
Computes the confusion matrix for contemporaneous datasets. Metrics are separated by lagged vs contemporanoeus relationships. 
"""
def confusion_matrix_contemporaneous(df, actual_cnnx, actual_neg_cnnx, cnnx_list):
    nonint_cols = ["Specificity", "Precision", "Recall", "SHD", "F1", "Balanced Accuracy", "Conflict Rate",
                   "Specificity_CAdj", "Precision_CAdj", "Recall_CAdj", "F1_CAdj", "Balanced Accuracy_CAdj",
                   "Precision_COr", "Recall_COr", "F1_COr",]
    temp_df = pd.DataFrame(columns=["True Positive", "False Positive", "True Negative", "False Negative", 
                                    "True Positive CAdj", "False Positive CAdj", 
                                    "True Negative CAdj", "False Negative CAdj",
                                    "True Positive COr", 
                                    "Conflicts", "Conflict Rate",
                                    "Specificity", "Precision", "Recall", "SHD",  "F1", "Balanced Accuracy",
                                    "Specificity_CAdj", "Precision_CAdj", "Recall_CAdj", "F1_CAdj", "Balanced Accuracy_CAdj",
                                    "Precision_COr", "Recall_COr", "F1_COr",
                                    ])
    
    temp_df["True Positive"] = df.loc[:, [a for a in actual_cnnx if a[2] > 0]].sum(axis=1)
    temp_df["False Positive"] = df.loc[:, [a for a in actual_neg_cnnx if a[2] > 0]].sum(axis=1)
    temp_df["True Negative"] = (df.loc[:, [a for a in actual_neg_cnnx if a[2] > 0]]==0).sum(axis=1)
    temp_df["False Negative"] = (df.loc[:, [a for a in actual_cnnx if a[2] > 0]]==0).sum(axis=1)

    # Check denominators for division by zero error
    temp_df.loc[(temp_df["True Negative"] + temp_df["False Positive"]) == 0, 
        "Specificity"] = 0
    temp_df.loc[(temp_df["True Negative"] + temp_df["False Positive"]) != 0, 
        "Specificity"] = temp_df["True Negative"]/(temp_df["True Negative"] + temp_df["False Positive"])
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Positive"]) == 0,
        "Precision"] = 0
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Positive"]) != 0,
        "Precision"]  = temp_df["True Positive"]/(temp_df["True Positive"] + temp_df["False Positive"])
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Negative"]) == 0,
        "Recall"] = 0
    temp_df.loc[(temp_df["True Positive"] + temp_df["False Negative"]) != 0,
        "Recall"] = temp_df["True Positive"]/(temp_df["True Positive"] + temp_df["False Negative"])
    temp_df["SHD"] = temp_df["False Positive"] + temp_df["False Negative"]
    denom = temp_df["Precision"] + temp_df["Recall"]
    temp_df["F1"] = 0
    mask = denom != 0
    temp_df.loc[mask, "F1"] = ( 2 * temp_df.loc[mask, "Precision"] * temp_df.loc[mask, "Recall"]) / denom[mask]
    temp_df["Balanced Accuracy"] = (temp_df["Recall"] + temp_df["Specificity"] )/2
    
    # Contemporaneous Adjacency - Include conflicts & undirected
    contemp_adj = [(a, b, c) for (a, b, c) in actual_cnnx if c == 0] + [(b, a, c) for (a, b, c) in actual_cnnx if c == 0]
    not_contemp_adj = [(a, b, c) for (a, b, c) in cnnx_list if (a, b, c) not in contemp_adj and a < b and c == 0]

    # Get only contemporaneous pairs (unordered)
    contemp_pairs = [(a, b, c) for (a, b, c) in actual_cnnx if c == 0]

    # Initialize storage
    tp_list, fp_list, tn_list, fn_list = [], [], [], []

    # --- For true contemporaneous edges ---
    for (a, b, c) in contemp_pairs:
        ab = df[(a, b, c)]
        ba = df[(b, a, c)]
        
        tp_mask = ((ab.isin([1,2,3])) | (ba.isin([1,2,3])))
        fn_mask = ((ab == 0) & (ba == 0))
        
        tp_list.append(tp_mask.astype(int))
        fn_list.append(fn_mask.astype(int))

    # --- For contemporaneous non-edges ---
    for (a, b, c) in not_contemp_adj:
        ab = df[(a, b, c)]
        ba = df[(b, a, c)]
        
        fp_mask = ((ab.isin([1,2,3])) | (ba.isin([1,2,3])))
        tn_mask = ((ab == 0) & (ba == 0))
        
        fp_list.append(fp_mask.astype(int))
        tn_list.append(tn_mask.astype(int))

    # Combine results per row
    temp_df["True Positive CAdj"] = pd.concat(tp_list, axis=1).sum(axis=1)
    temp_df["False Negative CAdj"] = pd.concat(fn_list, axis=1).sum(axis=1)
    temp_df["False Positive CAdj"] = pd.concat(fp_list, axis=1).sum(axis=1)
    temp_df["True Negative CAdj"] = pd.concat(tn_list, axis=1).sum(axis=1)
    

    # Contemporaneous Orientation 
    directed_cnnx = [(a, b, c) for (a, b, c) in actual_cnnx if c == 0 and (b, a, c) not in actual_cnnx]
    temp_df["True Positive COr"] = df[directed_cnnx].eq(1).sum(axis=1) 
    
    temp_df["Conflicts"] = (df==2).sum(axis=1)/2 # Conflicts always 2n (a,b)/(b,a)
    
    temp_df.loc[(temp_df["True Negative CAdj"] + temp_df["False Positive CAdj"]) == 0, 
        "Specificity_CAdj"] = 0
    temp_df.loc[(temp_df["True Negative CAdj"] + temp_df["False Positive CAdj"]) != 0, 
        "Specificity_CAdj"] = temp_df["True Negative CAdj"]/(temp_df["True Negative CAdj"] + temp_df["False Positive CAdj"])
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"]) == 0,
        "Precision_CAdj"] = 0
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"]) != 0,
        "Precision_CAdj"]  = temp_df["True Positive CAdj"]/(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"])
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Negative CAdj"]) == 0,
        "Recall_CAdj"] = 0
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Negative CAdj"]) != 0,
        "Recall_CAdj"] = temp_df["True Positive CAdj"]/(temp_df["True Positive CAdj"] + temp_df["False Negative CAdj"])
    denom = temp_df["Precision_CAdj"] + temp_df["Recall_CAdj"]
    temp_df["F1_CAdj"] = 0
    mask = denom != 0
    temp_df.loc[mask, "F1_CAdj"] = ( 2 * temp_df.loc[mask, "Precision_CAdj"] * temp_df.loc[mask, "Recall_CAdj"]) / denom[mask]
    temp_df["Balanced Accuracy_CAdj"] = (temp_df["Recall_CAdj"] + temp_df["Specificity_CAdj"] )/2
 
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"]) == 0,
        "Conflict Rate"] = 0
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"]) != 0,
        "Conflict Rate"] = temp_df["Conflicts"]/(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"])

    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"]) == 0,
        "Precision_COr"] = 0
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"]) != 0,
        "Precision_COr"]  = temp_df["True Positive COr"]/(temp_df["True Positive CAdj"] + temp_df["False Positive CAdj"])
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Negative CAdj"]) == 0,
        "Recall_COr"] = 0
    temp_df.loc[(temp_df["True Positive CAdj"] + temp_df["False Negative CAdj"]) != 0,
        "Recall_COr"] = temp_df["True Positive COr"]/(temp_df["True Positive CAdj"] + temp_df["False Negative CAdj"])
    denom = temp_df["Precision_COr"] + temp_df["Recall_COr"]
    temp_df["F1_COr"] = 0
    mask = denom != 0
    temp_df.loc[mask, "F1_COr"] = ( 2 * temp_df.loc[mask, "Precision_COr"] * temp_df.loc[mask, "Recall_COr"]) / denom[mask]

    df = pd.concat([df, temp_df], axis=1)
    means = df.loc[:, cnnx_list + nonint_cols].mean()
    medians = df.loc[:, cnnx_list + nonint_cols].median()
    variances = df.loc[:, cnnx_list + nonint_cols].var()
    stddev = df.loc[:, cnnx_list + nonint_cols].std()

    df.loc['mean'] = means
    df.loc['median'] = medians
    df.loc['variance'] = variances
    df.loc['stddev'] = stddev

    return df