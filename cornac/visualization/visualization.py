import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class Visualization(object):
    """Class for visualizing the result of the experiment, and the feature importance.
  
    """
    
    def __init__(self, save_dir:str = None):
        """Initialize the Visualization class
  
        """
        self.save_dir = "." if save_dir is None else save_dir
        self.target_k_name = None
                
    def visualize_experiemnt_result(self, EE_list, kind = "bar", rotate_x = 90):
        """Visualize the result of the runing experiment

        Parameters
        ----------
            EE: class 
                the class Object of the Explainers_Experiment
            kind: string
                the type of the chart to be plotted, default is "bar", options: "bar", "line"
            rotate_x: int
                the degree of rotation of x-axis labels, default is 90
                
        """
        df_list = []
        for EE in EE_list:
            if not hasattr(EE, "result"):
                raise ValueError("The input object is not an instance of Explainers_Experiment")
            # result = EE.data # list: the result of the experiment
            # pairs = EE.models # list: [model:explainer, model:explainer, ..]
            # metrics = EE.metrics # list: [metric1, metric2, .., train_cost, evaluate_cost]
            result = [list[1:] for list in EE.result]
            columns = [metric.name for metric in EE.metrics]
            columns.extend(["train_cost", "evaluate_cost"])
            pair_name_list = [list[0] for list in EE.result]
            data_df = pd.DataFrame(result, columns=columns, index=pair_name_list)
            df_list.append(data_df)
        df_all = pd.concat(df_list, axis=0)
        df_all.fillna(0, inplace=True)
        metrics_data = df_all.drop(columns=["train_cost", "evaluate_cost"])
        cost_data = df_all[["train_cost", "evaluate_cost"]]
        
        if len(metrics_data.columns)>0:
            self._plot_exp_vs_metric(metrics_data, kind, rotate_x)
        if len(cost_data.columns)>0:
            self._plot_cost(cost_data, rotate_x)
        
    def _plot_exp_vs_metric(self, df, kind, rotate_x = 90):
        """Plot the bar chart: compare the performance of model/explainer pairs on a specific metric"""
        
        print(f"Plot the {kind} chart for the metrics:")
        if kind == "bar":
            plt.figure(figsize=(6, 4))
            df.plot(kind=kind, ax=plt.gca())
            plt.xticks(rotation=90)
            plt.show()
        print(df)
        if kind == "line":
            plt.figure(figsize=(6, 4))
            ax = df.plot(kind=kind, marker='o', linestyle='-')
            # Set x-axis labels to be the DataFrame index
            ax.set_xticks(range(len(df.index)))
            ax.set_xticklabels(df.index)
            plt.xticks(rotation=rotate_x)
            plt.show()
    

    def _plot_cost(self, df, rotate_x = 90):
        """Plot the bar chart: compare the cost of training and evaluating the model/explainer pairs"""
        print("Plot the bar chart for the cost:")
        # Plotting the stacked bar chart
        # plt.bar(df.index, df.iloc[:, -2], label=df.columns[-2])
        # plt.bar(df.index, df.iloc[:, -1], bottom=df.iloc[:, -2], label=df.columns[-1])
        df.plot(kind='bar', stacked=True)
        # Adding labels and title
        plt.xticks(rotation=rotate_x)
        #plt.xlabel("model:explainer")
        plt.ylabel("time(s)")
        plt.legend()
        plt.show()
        
    def visualize_target_k(self, EE_list, target_k_name="feature_k", kind = "bar"):
        """plot the change of metrics' value along with the change of target k parameter

        PARAMETERS
        ----------
            EE_list: list
                a list of Explainers_Experiment instances
            target_k_name: str
                optional, the name of target parameter. Defaults to "feature_k", options: "feature_k", "rec_k"
            kind: str
                optional, the wanted type of chart. Defaults to "bar". Options: "bar", "line", "scatter"
                
        """
        self.target_k_name = target_k_name
        df_list = self._transform_format(EE_list)
        columns = df_list[0].columns[:-1]
        feature_k_list = [df_list[i][self.target_k_name].unique()[0] for i in range(len(df_list))]
        fig, ax = plt.subplots(len(columns), 1, figsize=(8, 4*len(columns)))
        for i, col in enumerate(columns):
            data = pd.concat([d[col] for d in df_list], axis=1)
            data.columns = feature_k_list
            data = data.T
            if kind == "bar":
                data.plot(kind=kind, ax=ax[i], title=col)
            elif kind == "line":
                sns.lineplot(data=data, ax=ax[i])
            elif kind == "scatter":
                sns.scatterplot(data=data, ax=ax[i])
            else:
                raise ValueError("kind should be one of ['bar', 'line', 'scatter']")
            ax[i].set_xlabel(self.target_k_name)
            ax[i].set_title(col)
        plt.tight_layout()
        plt.show()
    
    def _transform_format(self, EE_list):
        """Transform the result of the experiment to a list of dataframes
        
        PARAMETERS
        ----------
            EE_list: list
                a list of Explainers_Experiment instances
                
        RETURNS
        -------
            df_list: list
                a list of dataframes, each dataframe contains the metrics' value for each pair of model:explainer
        """
        
        df_list = []
        for EE in EE_list:
            if not hasattr(EE, "result"):
                raise ValueError("The input object is not an instance of Explainers_Experiment")
            
            result = [list[1:] for list in EE.result]
            #result.append(EE.feature_k)
            pair_name_list = [list[0] for list in EE.result]
            columns = [metric.name for metric in EE.metrics]
            columns.extend(["train_cost", "evaluate_cost"])
            data_df = pd.DataFrame(result, columns=columns, index = pair_name_list)
            data_df[self.target_k_name] = EE.feature_k if self.target_k_name == "feature_k" else EE.rec_k
            data_df.fillna(0, inplace=True)
            df_list.append(data_df)
        return df_list
    
    def create_individual_feature_importance_plot(
            self,
            df: pd.DataFrame, 
            user_id: str, 
            item_id: int, 
            type="bar", 
            top_k:int = 10,
            save_plot:bool = True,
        ):
        """Plot feature importance plot for individual recommendation
        
        Parameters
        ----------
            df: pd.DataFrame
                the dataframe containing the explanation data
            user_id: str
                the user id
            item_id: int
                the item id
            type: str, optional, default: "bar"
                the type of the plot, options: "bar", "line", "scatter"
            top_k: int, optional, default: 10
                the number of top features to be plotted
            save_plot: bool, optional, default: True
                whether to save the plot or not
                
        Returns
        -------
            pd.DataFrame
                the filtered dataframe only containing the explanation data for the given user_id and item_id    
        """
        filtered_df = df[(df["user_id"] == user_id) & (df["item_id"] == item_id)]

        if not filtered_df.empty:
            curr_explanation = filtered_df["explanations"].iloc[0]
            top_k = min(top_k, len(curr_explanation.keys()))
            x = list(curr_explanation.keys())[:top_k]
            y = list(curr_explanation.values())[:top_k]
            fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            if type=="line":
                axe.plot(x,y)
            elif type=="scatter":
                axe.scatter(x,y)
            elif type=="bar":
                axe.bar(x,y)
            else:
                raise ValueError("Does not support other plot type.")
            axe.set_title(f"Top {top_k} Explanatory Features \n for user {user_id[:6]}... and item {item_id}")
            axe.set_xticklabels(x, rotation=45)
            if save_plot:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                fig_name = os.path.join(self.save_dir, f"{timestamp}_feature_importance_for_item{item_id}_user_{user_id}")
                fig.savefig(fig_name)
            
        return filtered_df

    def create_aggregate_feature_importance_plot(
            self,
            df: pd.DataFrame, 
            user_id: str = None, 
            item_id: int = None,
            type:str="bar", 
            top_k:int = 10,
            save_plot:bool = True,
        ):
        """Plot by count and weighted coefficient for whole explanation dataframe.
        Can filter by user_id and/or item_id
        
        Parameters
        ----------
            df: pd.DataFrame
                the dataframe containing the explanation data
            user_id: str, optional, default: None
                the user id, if None, will use all users
            item_id: int, optional, default: None
                the item id, if None, will use all items
            type: str, optional, default: "bar"
                the type of the plot, options: "bar", "line", "scatter"
            top_k: int, optional, default: 10
                the number of top features to be plotted
            save_plot: bool, optional, default: True
                whether to save the plot or not
        
        Returns
        -------
            pd.DataFrame
                the filtered dataframe only containing the explanation data for the given user_id and item_id
        
        """
        filter_condition = pd.Series([True] * len(df), index=df.index)
        plot_title = ""
        if user_id:
            filter_condition &= (df["user_id"] == user_id)
            user_id = user_id[:6] + "..."
            plot_title += f"for user {user_id} "
        if item_id:
            filter_condition &= (df["item_id"] == item_id)
            plot_title += f"for item {item_id} "

        filtered_df = df[filter_condition]

        if not filtered_df.empty:
            agg_count = {}
            agg_coeff = {}
            size = len(filtered_df)
            for i, (_, _, exp, _) in filtered_df.iterrows():
                for feat, coeff in exp.items():
                    if feat in agg_count.keys():
                        agg_count[feat] += 1
                        agg_coeff[feat] += coeff
                    else:
                        agg_count[feat] = 1
                        agg_coeff[feat] = coeff

            agg_coeff = {feat: coeff/size for feat, coeff in agg_coeff.items()}
            top_k = min(top_k, len(agg_count.keys()))
            x_count = list(agg_count.keys())[:top_k]
            y_count = list(agg_count.values())[:top_k]
            x_coeff = list(agg_coeff.keys())[:top_k]
            y_coeff = list(agg_coeff.values())[:top_k]
            user_id = user_id[:6] + "..." if user_id else None
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            if type=="line":
                axes[0].plot(x_count, y_count)
                axes[1].plot(x_coeff, y_coeff)
            elif type=="scatter":
                axes[0].scatter(x_count, y_count)
                axes[1].scatter(x_coeff, y_coeff)    
            elif type=="bar":   
                axes[0].bar(x_count, y_count)
                axes[1].bar(x_coeff, y_coeff)     
            else:
                raise ValueError("Does not support plot type other than line, scatter and bar.")
            axes[0].set_title(f"Top {top_k} Explanatory Features (by count) \n {plot_title}")
            axes[0].set_xticklabels(x_count, rotation=45)
            axes[1].set_title(f"Top {top_k} Explanatory Features (weighted coefficient) \n {plot_title}")
            axes[1].set_xticklabels(x_coeff, rotation=45)
            if save_plot:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                fig_name = os.path.join(self.save_dir, f"{timestamp}_agg_feature_importance")
                fig.savefig(fig_name)
        return filtered_df
