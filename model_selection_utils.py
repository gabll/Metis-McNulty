import sklearn as sk
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import bisect
import pandas as pd

def print_model_evaluation(estimator, Y_test, X_test, Y_all, X_all, cross_val_samples=3):
    """Print learning curve, ROC, confusion matrix, cross-validation scores given a estimator
    Y has to be binarized in advance
    If cross_val_samples is 1, metrics are based only on the test set provided
    """
    if cross_val_samples == 1:
        print 'Accuracy:\t%.4f' % accuracy_score(Y_test, estimator.predict(X_test))
        print 'Precision:\t%.4f' % precision_score(Y_test, estimator.predict(X_test))
        print 'Recall:\t\t%.4f' % recall_score(Y_test, estimator.predict(X_test))
        print 'F1:\t\t%.4f' % f1_score(Y_test, estimator.predict(X_test))
    else:
        print print_cross_val_score(estimator, X_all, Y_all, cv=cross_val_samples)
        print print_cross_val_score(estimator, X_all, Y_all, cv=cross_val_samples, scoring='precision')
        print print_cross_val_score(estimator, X_all, Y_all, cv=cross_val_samples, scoring='recall')
        print print_cross_val_score(estimator, X_all, Y_all, cv=cross_val_samples, scoring='f1')
    plt.figure(figsize=(18,5))
    plt.subplot(1, 2, 1)
    plot_learning_curve(estimator,'Learning curve', X_all, Y_all)
    plt.subplot(1, 2, 2)
    plot_roc(estimator,'ROC', X_test, Y_test)
    plt.show()
    print '\nConfusion Matrix'
    print format_confusion_matrix(Y_test, estimator.predict(X_test))

def print_cross_val_score(estimator, X_all, Y_all, cv=3, scoring=None):
    score = cross_val_score(estimator, X_all, Y_all, cv=cv, scoring=scoring)
    score_round = ['%.4f' % (i, ) for i in score]
    if scoring == 'f1':
        scoring += '\t'
    elif not scoring:
        scoring = 'accuracy'
    return 'Cross-validation ' + scoring + ':\t%.4f' % np.mean(score) + ', mean of' + str(score_round)

def format_confusion_matrix(Y_true, Y_predicted):
    cm = confusion_matrix(Y_true, Y_predicted)
    cm_str = [['TP: ', 'FP: '], ['FN: ', 'TN: ']]  
    cm_str[0][0] += str(cm[0][0])
    cm_str[0][1] += str(cm[0][1])
    cm_str[1][0] += str(cm[1][0])
    cm_str[1][1] += str(cm[0][1])
    return '\n'.join([''.join(['{:12}'.format(item) for item in row]) for row in cm_str])

def convert_features(df, out_var, dummies=True, scaling=True, only_important_features=True, null_value=np.nan):
    """Convert features in a dataframe.
    df:      pandas dataframe (including outcome variable)
    out_var: outcome label (string)
    dummies: if True, convert all categorical features (string) to dummies
    scaling: if True, convert all continuous features to float type and then to z-score
    only_important_features: if True, it will keep the null_values as baseline. If false, it will drop the null_value rows
    null_value: null value used in the dataframe - string (ex: 'unknown')"""
    df = df.replace(null_value, np.nan)
    for i in df.columns:
    	if only_important_features == False:
    	    df = df.dropna(subset=[i])
        if df[i].dtypes == np.object and i != out_var and dummies:
            i_dummies = pd.get_dummies(df[i])
            i_dummies = i_dummies.add_prefix(i + '_')
            if df[i].nunique() == 2 and not np.any(df[i].isnull()):
                i_dummies = i_dummies.drop(i_dummies.columns[1], 1)
            df = pd.concat([df,i_dummies], axis=1)
            df = df.drop([i],1)   
        else:
            if scaling and i != out_var:
                df[i] = scale(df[i].astype('float'))
    return df.dropna()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.logspace(np.log10(0.1),np.log10(1.0),30)):
    """Plot a learning curve for an estimator"""
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training samples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #plt.grid() uncomment if you don't want to use seaborn
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

    plt.legend(loc="best")
    return plt
    
def plot_roc(estimator, title, X_test, Y_test, ylim=None, pos_label=None):
    """Plot a roc curve for an estimator"""
   #if type(estimator) != sk.svm.classes.SVC:
    Y_pred_prob = estimator.predict_proba(X_test)[:,1]
    #else:
    #    Y_pred_prob = estimator.decision_function(X_ts)
    fpr, tpr, thres = roc_curve(Y_test, Y_pred_prob, pos_label=pos_label)
    plt.plot(fpr, tpr, 'o-')
    #plt.grid() uncomment if you don't want to use seaborn
    plt.title(title + ' (AUC=%.2f)' % auc(fpr, tpr))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("False Positive Ratio (FPR)")
    plt.ylabel("True Positive Ratio (TPR)")
    #plt.legend(loc="best")
    return plt

def plot_roc_svc(estimator, title, X_test, Y_test, ylim=None, pos_label=None):
    """Plot a roc curve for an estimator"""
   #if type(estimator) != sk.svm.classes.SVC:
    Y_pred_prob = estimator.decision_function(X_test)
    #else:
    #    Y_pred_prob = estimator.decision_function(X_ts)
    fpr, tpr, thres = roc_curve(Y_test, Y_pred_prob, pos_label=pos_label)
    plt.plot(fpr, tpr, 'o-')
    #plt.grid() uncomment if you don't want to use seaborn
    plt.title(title + ' (AUC=%.2f)' % auc(fpr, tpr))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("False Positive Ratio (FPR)")
    plt.ylabel("True Positive Ratio (TPR)")
    #plt.legend(loc="best")
    return plt
    
def __barplot(labels,data):
    pos=np.arange(len(data))
    plt.xticks(pos+0.4,labels)
    plt.bar(pos,data)

def histplot(data,bins=None,nbins=5):
    if not bins:
        minx,maxx=min(data),max(data)
        space=(maxx-minx)/float(nbins)
        bins=np.arange(minx,maxx,space)
    binned=[bisect.bisect(bins,x) for x in data]
    l=['%.1f'%x for x in list(bins)+[maxx]] if space<1 else [str(int(x)) for x in list(bins)+[maxx]]
    displab=[x+'-'+y for x,y in zip(l[:-1],l[1:])]
    __barplot(displab,[binned.count(x+1) for x in range(len(bins))])