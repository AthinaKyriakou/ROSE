import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualization(object):
    """Data Visualization class   """
    
    def __init__(self):
        """Initialize the Visualization class

        Args:
            
        """
        pass
                
    def visualize_experiemnt_result(self, EE, kind = "bar"):
        """Visualize the result of the runing experiment

        Args:
            EE (_class_): the class Object of the Explainers_Experiment
            kind (_string_): the type of the chart to be plotted, default is "bar", options: "bar", "line";
        """
        if EE.result is None:
            EE.run()
        # result = EE.data # list: the result of the experiment
        # pairs = EE.models # list: [model:explainer, model:explainer, ..]
        # metrics = EE.metrics # list: [metric1, metric2, .., train_cost, evaluate_cost]
        result = [list[1:] for list in EE.result]
        columns = [metric.name for metric in EE.metrics]
        columns.extend(["train_cost", "evaluate_cost"])
        pair_name_list = [list[0] for list in EE.result]
        data_df = pd.DataFrame(result, columns=columns, index=pair_name_list)
        data_df.fillna(0, inplace=True)
        metrics_data = data_df.iloc[:,:-2]
        cost_data = data_df.iloc[:,-2:]
        print("=======Start Visualization=======")
        if len(metrics_data.columns)>0:
            self._plot_exp_vs_metric(metrics_data, kind)
        if len(cost_data.columns)>0:
            self._plot_cost(cost_data)
        print("=======Visualization Done=======")
        
    def _plot_exp_vs_metric(self, df, kind = "bar"):
        """Plot the bar chart: compare the performance of model/explainer pairs on a specific metric

        Args:
            df (_dataframe_): a dataframe with the model/explainer pairs as rows and the metrics as columns
            kind (_string_): the type of the chart to be plotted, default is "bar", options: "bar", "line";
        """
        
        print(f"Plot the {kind} chart for the metrics:")
        if kind == "bar":
            plt.figure(figsize=(6, 4))
            df.plot(kind=kind, ax=plt.gca())
            plt.xticks(rotation=0)
            plt.show()
        print(df)
        if kind == "line":
            plt.figure(figsize=(6, 4))
            ax = df.plot(kind=kind, marker='o', linestyle='-')
            # Set x-axis labels to be the DataFrame index
            ax.set_xticks(range(len(df.index)))
            ax.set_xticklabels(df.index)
            plt.xticks(rotation=0)
            plt.show()
        
        # for column in df.columns:
        #     plt.bar(df.index, df[column], label=column)
        #     plt.ylabel(column)
        #     plt.show()

    def _plot_cost(self, df):
        """Plot the bar chart: compare the cost of training and evaluating the model/explainer pairs

        Args:
            df (_dataframe_): a dataframe with the model/explainer pairs as rows and the cost as columns
        """
        print("Plot the bar chart for the cost:")
        # Plotting the stacked bar chart
        # plt.bar(df.index, df.iloc[:, -2], label=df.columns[-2])
        # plt.bar(df.index, df.iloc[:, -1], bottom=df.iloc[:, -2], label=df.columns[-1])
        df.plot(kind='bar', stacked=True)
        # Adding labels and title
        plt.xticks(rotation=0)
        #plt.xlabel("model:explainer")
        plt.ylabel("time(s)")
        plt.legend()
        plt.show()
