import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'TWITTER': self.__eval_twitter_regression,
                'TWITTER2015': self.__eval_twitter2015_regression
            }
        else:
            self.metrics_dict = {
                'TWITTER': self.____eval_twitter_classification,
                'TWITTER2015': self.____eval_twitter_classification
            }


    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()



        binary_truth = (test_truth >= 0)

        binary_preds = (test_preds >= 0)

        acc2 = accuracy_score(binary_preds, binary_truth)

        f_score = f1_score(binary_truth, binary_preds, average='weighted')
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            #"Non0_acc_2":  round(non_zeros_acc2, 4),
            #"Non0_F1_score": round(non_zeros_f1_score, 4),
            #"Mult_acc_5": round(mult_a5, 4),
            #"Mult_acc_7": round(mult_a7, 4),
            #"MAE": round(mae, 4),
            #"Corr": round(corr, 4)
        }
        return eval_results

    def __eval_twitter_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)
    def __eval_twitter2015_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]