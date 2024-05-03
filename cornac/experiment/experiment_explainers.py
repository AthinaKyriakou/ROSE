import os
import time
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cornac.models import Recommender
from cornac.explainer import Explainer
from cornac.metrics_explainer.metric_exp import Metric_Exp as Metrics


NUM_FMT = "{:.4f}"
VALID_COMBO = [("fm_regressor", "LIMERS"), ("EFM", "Exp_EFM"), ("MTER", "Exp_MTER"), ("EFM", "Exp_EFM_Mod"),
               ("ALS", "ALS"), ("MF", "PHI4MF"), ("EMF", "PHI4MF"), ("NEMF", "PHI4MF"), ("EMF", "EMF"), ("NEMF", "EMF")]

class MetricError(ValueError):
    pass

def _table_format(data, headers=None, index=None, extra_spaces=0, h_bars=None):
    if headers is not None:
        data.insert(0, headers)
    if index is not None:
        index.insert(0, "recommender:explainer")
        for idx, row in zip(index, data):
            row.insert(0, idx)

    column_widths = np.asarray([[len(str(v)) for v in row] for row in data]).max(axis=0)

    row_fmt = (
        " | ".join(["{:>%d}" % (w + extra_spaces) for w in column_widths][1:]) + "\n"
    )
    if index is not None:
        row_fmt = "{:<%d} | " % (column_widths[0] + extra_spaces) + row_fmt

    output = ""
    for i, row in enumerate(data):
        if h_bars is not None and i in h_bars:
            output += row_fmt.format(
                *["-" * (w + extra_spaces) for w in column_widths]
            ).replace("|", "+")
        output += row_fmt.format(*row)
    return output

class Explainers_Experiment:
    """
    Create experiment to evaluate explainers and output evaluation results
    Inherits from base class cornac.experiment.Experiment
    """
    def __init__(self, eval_method, models, metrics, distribution=True, rec_k = 10, feature_k = 10, eval_train=True, verbose=True, num_threads=0, save_dir="./experiment_plots", **kwargs):
        """"
        Args:
            eval_method: evaluation method (e.g. ratiosplit) 
            models: list of (recommender, explainer) tuples to be evaluated
            metrics: list of explainer metrics
            distribution: if True, histogram of result distribution are saved
            rec_k: default 10, number of recommendations created by recommender
            feature_k: default 10, number of features in explanations created by explainer
            eval_train: default True, if True, evaluate train_set, else evaluate test_set
            verbose: default True
            num_threads: default 0
            save_dir: default None (current folder), used to save experiment result
        """
        self.eval_method = eval_method
        self.dataset = self.eval_method.train_set if eval_train else self.eval_method.test_set
        self.models = self._validate_models(models)
        self.distribution = distribution
        self.rec_k = rec_k
        self.feature_k = feature_k
        self.metrics = self._validate_metrics(metrics)
        self.verbose = verbose
        self.num_threads = num_threads
        self.save_dir = save_dir
        self.kwargs = kwargs

        self.current_rec = None
        self.current_exp = None
        self.current_metric = None
        self.rec_name = None
        self.exp_name = None
        self.recommendations1 = None
        self.recommendations2 = None
        self.explanations1 = None
        self.explanations2 = None
        self.result = None
    
    @staticmethod
    def _validate_models(models):
        if not hasattr(models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(models))
            )

        valid_models = []
        for model in models:
            if (isinstance(model[0], Recommender) and isinstance(model[1], Explainer)):
                if (model[0].name, model[1].name) in VALID_COMBO:
                    valid_models.append(model)
                else:
                    print(f'{model[0].name}:{model[1].name} removed from list of models since the combination is not valid!')
            elif (isinstance(model[0][0], Recommender) and isinstance(model[0][1], Recommender) and isinstance(model[1][0], Explainer) and isinstance(model[1][1], Explainer)):
                if (model[0][0].name, model[1][0].name) in VALID_COMBO and (model[0][1].name, model[1][1].name) in VALID_COMBO:
                    valid_models.append(model)
                else:
                    print(f'({model[0][0].name},{model[0][1].name}):({model[1][0].name},{model[1][1].name}) removed from list of models since the combination is not valid!')
        return valid_models
    
    @staticmethod
    def _validate_metrics(metrics):
        if not hasattr(metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(type(metrics))
            )
        valid_metrics = []
        for metric in metrics:
            if isinstance(metric, Metrics):
                valid_metrics.append(metric)
        return valid_metrics


    def _get_metric_explainer(self, metric):
        """
        Check if metric can be used for current (recommender, explainer) pair
        Args: 
            metric: metric used to evaluate explainer
        Return:
            metric 
        """
        metrics_support = False
        if self.pair_exp:
            if metric.name in ["FA", "RA"]:
                metrics_support = True
        else:
            if metric.name == 'Metric_Exp_DIV':
                metrics_support = True 
            elif metric.name == 'Metric_Exp_FPR':
                if self.current_exp.name in ['LIMERS', 'Exp_EFM', 'Exp_MTER']:
                    metrics_support = True 
            elif metric.name == 'PSPNFNS':
                if self.current_exp.name in ['LIMERS', 'Exp_EFM', 'Exp_MTER']:
                    metrics_support = True
            elif metric.name in ["FA", "RA"]:
                pass
            elif metric.name in ['EnDCG', 'MEP']:
                if self.current_exp.name in ['EMF', 'PHI4MF'] and self.current_rec.name in ['EMF', 'NEMF']:
                    metrics_support = True
            elif metric.name in ['PGF']:
                if self.current_exp.name in ['EMF', 'PHI4MF', 'ALS']:
                    metrics_support = True
            else:
                print(f'Metric {metric.name} not found')
                return None

        if not metrics_support:
            raise MetricError(f'Metric {metric.name} does not support {self.exp_name}.')
            
        return metric

    def _evaluate_explainer(self):
        """
        Run evaluation steps using one metric for one (recommender, explainer) pair: 
            step 1: create recommendations
            step 2: create explanations
            step 3: evaluate explanations
        Return:
            evaluation result
        """
        name = "placeholder" #to be addressed by each condition
        print(f'Step 3/3: Metric {self.current_metric.name} starts evaluation...')
        if self.current_metric.name == 'PSPNFNS':
            print(f"self.current_rec: {self.current_rec.name}, self.current_exp: {self.current_exp.name}")
            (pn, ps, fns),(pn_d, ps_d, fns_d) = self.current_metric.compute(self.current_rec, self.current_exp, self.explanations1)
            print(f'Result: Probability of Necessity: {pn}; Probability of Sufficiency: {ps}; Harmonic Mean: {fns}')
            self._plot_distribution("PN", pn_d)
            self._plot_distribution("PS", ps_d)
            self._plot_distribution("FNS", fns_d)
            return fns  
        elif self.current_metric.name == 'Metric_Exp_DIV':
            fd, fd_d = self.current_metric.compute(self.explanations1)
            print(f'Result: Feature diversity: {fd}')
            self._plot_distribution("FDIV", fd_d, groupby="explanation")
            return fd
        elif self.current_metric.name == 'Metric_Exp_FPR':
            (precision, recall, ff1), (precision_d, recall_d, ff1_d) = self.current_metric.compute(self.current_rec, self.current_exp)
            print(f'Result: Feature Precision: {precision}; Feature Recall: {recall}; Harmonic Mean: {ff1}')
            self._plot_distribution("FP", precision_d, groupby="explanation")
            self._plot_distribution("FR", recall_d, groupby="explanation")
            self._plot_distribution("FF1", ff1_d, groupby="explanation")
            return ff1
        elif self.current_metric.name in ["FA", "RA"]:
            (result, result_d)  = self.current_metric.compute(self.explanations1, self.explanations2)
            print(f'Result: Average {self.current_metric.name}: {result}')
            self._plot_distribution(self.current_metric.name, result_d)
            return result
        elif self.current_metric.name in ['EnDCG', 'MEP']:
            (result, result_d) = self.current_metric.compute(recommender=self.current_rec, recommendations=self.recommendations1)
            print(f'Result: {self.current_metric.name}: {result}')
            self._plot_distribution(self.current_metric.name, result_d)
            return result
        elif self.current_metric.name in ['PGF']:
            (result, result_d) = self.current_metric.compute(recommender=self.current_rec, explainer=self.current_exp, explanations=self.explanations1)
            print(f'Result: {self.current_metric.name}: {result}')
            self._plot_distribution(self.current_metric.name, result_d)
            return result
        else:
            ##TODO: to be verified
            print(f"Metric {self.current_metric.name} has not be implemented")
            return None  

    def _get_recommendations(self, current_rec):
        """
        Create recommendations using current recommender model. 
        Return:
            recommendation dataframe [user_id, item_id] 
        """
        users = [k for k in self.dataset.uid_map.keys()] 
        rec_df = current_rec.recommend_to_multiple_users(users, k=self.rec_k)
        rec_df = rec_df[['user_id', 'item_id']]
        return rec_df

    def _get_explanations(self, current_exp, recommendations):
        """Create explanations for all [user_id, item_id] pair.
        Return:
            Numpy array of user_id, item_id, explanations. Return only non-zero explanations
        """
        exp = current_exp.explain_recommendations(recommendations=recommendations, num_features=self.feature_k)[['user_id', 'item_id', 'explanations']]
        exp = exp[exp['explanations'] != {}] #remove records with empty explanation
        if not current_exp.name in ['EMF', 'PHI4MF']:
            exp['explanations'] = exp['explanations'].apply(lambda x: [v for v in x.keys()])
        exp = exp[['user_id', 'item_id', 'explanations']].values
        return exp
    
    def _plot_distribution(self, name, values, groupby="user"):
        """
        Args:
            name: metric name to be used for x-label and histogram title
            values: list of values to be plotted
            groupby: user/item/explanation
        """
        if self.distribution:
            plt.hist(values)
            plt.xlabel(name)
            plt.ylabel("count")
            plt.title(f'{name} distribution by {groupby} - {self.rec_name}:{self.exp_name}')
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            save_dir = "." if self.save_dir is None else self.save_dir
            output_file = os.path.join(save_dir, "CornacMetricExp-{} distribution-{}-{}-{}.png".format(name, self.rec_name, self.exp_name, timestamp))
            try: 
                plt.savefig(output_file)
                plt.close() 
            except FileNotFoundError:
                os.mkdir(save_dir)
                plt.savefig(output_file)
                plt.close() 

    def run(self):
        data = []
        models = []
        metrics = []
        for model in self.models:
            self.current_rec, self.current_exp = model
            if isinstance(self.current_rec, Recommender) and isinstance(self.current_exp, Explainer):
                self.rec_name = self.current_rec.name
                self.exp_name =  self.current_exp.name
                model_name = self.current_rec.name + ":" + self.current_exp.name
                self.pair_exp = False
            else:
                self.rec_name = (self.current_rec[0].name, self.current_rec[1].name)
                self.exp_name = (self.current_exp[0].name, self.current_exp[1].name)
                model_name = self.current_rec[0].name + ":" + self.current_exp[0].name + "'vs'" + self.current_rec[1].name + ":" + self.current_exp[1].name
                self.pair_exp = True


            start_time = time.time()
            if self.pair_exp:
                print(f'Start training Recommender1 {self.current_rec[0].name}...')
                self.current_rec[0].fit(self.eval_method.train_set)
                print(f'Start training Recommender2 {self.current_rec[1].name}...')
                self.current_rec[1].fit(self.eval_method.train_set)
            else:
                print(f'Start training Recommender {self.current_rec.name}...')
                self.current_rec.fit(self.eval_method.train_set)
            end_time = time.time()
            train_time = end_time - start_time

            result = []
            start_time = time.time()
            print(f"*****Start evaluating model-explainer: '{model_name}'...")
            if self.pair_exp:
                print(f'Step 1/3: Creates fake recommendations from dataset for common used')
                users = np.array(list(self.dataset.uid_map.keys()))
                items = np.array(list(self.dataset.iid_map.keys()))
                min_num = min(len(users), len(items))
                self.recommendations = pd.DataFrame({'user_id': users[:min_num], 'item_id': items[:min_num]})
                print(f'Step 2/3: Explainer1 {self.current_exp[0].name} create explanation for all recommendations')
                self.explanations1 = self._get_explanations(self.current_exp[0], self.recommendations)
                print(f'Step 2/3: Explainer2 {self.current_exp[1].name} create explanation for all recommendations')
                self.explanations2 = self._get_explanations(self.current_exp[1], self.recommendations)
            else:
                print(f'Step 1/3: Recommender {self.current_rec.name} creates recommendations')
                self.recommendations1 = self._get_recommendations(self.current_rec)
                print(f'Step 2/3: Explainer {self.current_exp.name} create explanation for all recommendations')
                self.explanations1 = self._get_explanations(self.current_exp, self.recommendations1)

            for metric in self.metrics:
                if metric.name not in metrics:
                    metrics.append(metric.name)
                try:
                    self.current_metric = self._get_metric_explainer(metric)
                    temp = self._evaluate_explainer()
                except MetricError:
                    temp = "N/A"
                    print(f'Metric {metric.name} does not support {self.exp_name}.')
                result.append(temp)

            end_time = time.time()
            evaluate_time = end_time - start_time
            
            result.append(train_time)
            result.append(evaluate_time)

            data.append(result)
            models.append(model_name)
        metrics.extend(["Train(s)", "Evaluate(s)"])
        print(f"experiment data: {data}")
        self.result = data.copy()
        result = _table_format(data=data, headers = metrics, index=models)
        print("\n")
        print(result)

        #save result
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_dir = "." if self.save_dir is None else self.save_dir
        output_file = os.path.join(save_dir, "CornacMetricExp-{}-{}-{}.log".format(self.rec_name, self.exp_name, timestamp))
        with open(output_file, "w") as f:
            f.write(result)
            
