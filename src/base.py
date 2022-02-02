import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from   collections import Counter
from   collections import defaultdict

color_1 = cm.get_cmap("Set2")(2) # blue color
color_2 = cm.get_cmap("Set2")(1) # orange color

'''
Class for Preliminary EDA
'''
class PrelimEDA():
    
    def __init__(self):
        pass
    
    def fit(self, X_df):
        """
        Creates lists of numeric and categorical columns
        """
        
        # Check that X_df is a DataFrame
        assert type(X_df) == pd.DataFrame
        
        self.categorical_vars = X_df.select_dtypes(exclude = np.number).columns.to_list()
        self.numeric_vars     = X_df.select_dtypes(include = np.number).columns.to_list()
        
        return self
    
    def _get_plotting_specs(self):
        
        """
        Defines label/title specifications for plotting
        """
        
        # Label specifications
        self.label_fontsize   = 13
        self.title_fontsize   = 15
        self.title_fontweight = "bold"
        self.title_y = 1.03
        
        return self

        
    def plot_numeric_vars(self, X_df, columns_to_log = [], cols = 3):
    
        """
        Plots the distribution of numeric variables
        """
 
        # Check that X_df is a DataFrame
        assert type(X_df) == pd.DataFrame
        
        # Get list of cat/num vars
        self.fit(X_df)
        
        # Number of numeric variables to plot
        nbr_numeric_vars = len(self.numeric_vars)

        # Define number of rows for plt.subplots()
        # (By default: cols = 3)
        if np.mod(nbr_numeric_vars,cols) == 0:
            rows = nbr_numeric_vars//cols
        else:
            rows = nbr_numeric_vars//cols + 1

        # Plot distributions of numeric variables
        fig, axes = plt.subplots(rows, cols, figsize=(18,rows*3))
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()
        
        fig.suptitle('Distribution of numeric variables',
                     fontsize = self.title_fontsize,
                     fontweight = self.title_fontweight,
                     y = self.title_y
                    )
        
        for c, ax in zip(self.numeric_vars,axes.ravel()):
            ax.hist(X_df[c], bins = 40, histtype=u'step')
            ax.set_xlabel(c, fontsize=self.label_fontsize)
            ax.set_ylabel('Counts', fontsize=self.label_fontsize)
            
            if c in columns_to_log:
                ax.set_yscale('log')

        # Take care of empty plots by removings the axes
        nbr_empty_plots = (rows * cols) - nbr_numeric_vars
        
        if nbr_empty_plots != 0:
            #print(nbr_empty_plots)
            for ax in axes.ravel()[::-1][0:nbr_empty_plots]:
                ax.axis('off')

        plt.tight_layout()
        
    def plot_missing_values(self, X_df):
        
        """
        Plots the percentage of missing in variables
        """
    
        # Check that X_df is a DataFrame
        assert type(X_df) == pd.DataFrame

        # Get label/title specifications for plotting
        self._get_plotting_specs()
        
        #Get lists of cat/num  vars
        self.fit(X_df)

        plt.figure(figsize=(12,3))
        plt.suptitle('Percentage of missing values',
                     fontsize = self.title_fontsize,
                     fontweight = self.title_fontweight,
                     y = self.title_y
                    )

        w = 100*X_df.isnull().mean().sort_values(ascending=False)
        w.plot(kind='bar', edgecolor='grey', color = color_1);
        plt.xticks(rotation = 80);
        plt.xlabel('Columns', fontsize=self.label_fontsize)
        plt.ylabel('Percentage of missing values (%)', fontsize=self.label_fontsize)

        
    def target_sample_distribution(self, X_df, target):
        """
        Plots the distribution of samples in target variable (categorical)
        """
        
        # Check that the target is a categorical variable
        assert X_df[target].dtype != np.number, "target_sample_distribution can only be used with categorical target variables."

        w = X_df[target].value_counts()
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()

        plt.title('Distribution of samples in target variable',
                  fontsize = self.title_fontsize,
                  fontweight = self.title_fontweight,
                  y = self.title_y
                 )
        plt.bar(w.index, w.values, edgecolor='gray',color = color_1);
        plt.xticks(list(w.index), labels = X_df[target].unique(), rotation=45)
        plt.xlabel(target,fontsize=self.label_fontsize)
        plt.ylabel('Counts',fontsize=self.label_fontsize);


    def target_distribution_num(self, X_df, target, log_y_scale = False):
        """
        Plots the distribution of target variable (numeric)
        """
        
        # Check that the target is a numeric variable
        assert X_df[target].dtype == np.number, "target_distribution_num can only be used with numeric target variables."
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()
        
        # Get lists of cat/num vars
        self.fit(X_df)

        plt.title('Distribution of target variable',
                  fontsize = self.title_fontsize,
                  fontweight = self.title_fontweight,
                  y = self.title_y
                 )
        plt.hist(X_df[target], edgecolor='gray',bins = 50, color = color_1);
        plt.xlabel(target,fontsize=self.label_fontsize)
        plt.ylabel('Counts',fontsize=self.label_fontsize);
        
        if log_y_scale:
            plt.yscale('log')
            
            
    def plot_most_common_words(self, df, N):       
    
        '''This function computes the N most common words in 
        class 0/1 text samples hand plots the results in a common
        histogram.

        Parameters
        ----------
        df : pandas DataFrame
            Input data
        N  : int
            Nbr of most common words to plot
        '''

        # Join cleaned tokens for classes 0/1 into a single string and split on whitespace
        tokens_0 = ' '.join([' '.join([tok for tok in tweet]) for tweet in df[df['label']== 0]['clean_tweet']]).split()
        tokens_1 = ' '.join([' '.join([tok for tok in tweet]) for tweet in df[df['label']== 1]['clean_tweet']]).split()

        # Create dictionaries with word frequencies in classes 0/1
        freq_words_dict_0 = dict(Counter(tokens_0).most_common(N))
        freq_words_dict_1 = dict(Counter(tokens_1).most_common(N))

        # Create lists of most common words for each class
        words_0 = list(freq_words_dict_0.keys())
        words_1 = list(freq_words_dict_1.keys())

        # Join most common words in classes 0/1 in a single set
        common_words = set(words_0+words_1)

        # Collect words frequencies in classes 0/1
        counts_0  = [freq_words_dict_0[w] if w in words_0 else 0 for w in common_words]
        counts_1  = [freq_words_dict_1[w] if w in words_1 else 0 for w in common_words]

        # Store results in DataFrame and sort values
        df_counts = pd.DataFrame(list(zip(common_words, counts_0, counts_1)), 
                                 columns =['Word', 'Counts_class_0','Counts_class_1'])
        df_counts = df_counts.sort_values(by='Counts_class_0',ascending=False)
        df_counts.reset_index(drop=True, inplace=True)

        # Plot most common words in classes 0/1
        # -------------------------------------
        plt.figure(figsize=(18,5))
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()

        plt.bar(x = df_counts.Word,height=df_counts.Counts_class_0,edgecolor='grey',label='Class 0',alpha=0.6, color = color_1)
        plt.bar(x = df_counts.Word,height=df_counts.Counts_class_1,edgecolor='grey',label='Class 1',alpha=0.6, color = color_2)
        plt.ylabel('Word counts',fontsize=self.label_fontsize)
        plt.title('Top '+str(N)+' most frequent words in classes 0/1',
                  fontsize = self.title_fontsize,
                  fontweight = self.title_fontweight,
                  y = self.title_y)
        plt.legend()
        plt.xticks(rotation=90,fontsize=self.label_fontsize); 
        plt.xlim([-1,len(list(common_words))+0.5])
        
    def most_common_negation_words(self, df, N): 
        
        # Create dictionaries with negation word frequencies in classes 0/1
        neg_words_dict = defaultdict(int)
        neg_words_dict_0 = defaultdict(int)
        neg_words_dict_1 = defaultdict(int)

        # Dictionary for negation words in all classes
        for tweet in df.loc[:, 'clean_tweet']:
            for w in tweet:
                if any(ele in w for ele in ['not_', 'no_', 'never_']):
                    neg_words_dict[w] += 1

        # Dictionary for negation words in class 0
        for tweet in df.loc[df['label']==0, 'clean_tweet']:
            for w in tweet:
                if any(ele in w for ele in ['not_', 'no_', 'never_']):
                    neg_words_dict_0[w] += 1

        # Dictionary for negation words in class 1
        for tweet in df.loc[df['label']==1, 'clean_tweet']:
            for w in tweet:
                if any(ele in w for ele in ['not_', 'no_', 'never_']):
                    neg_words_dict_1[w] += 1

        # Sort frequency dicts in descending order
        neg_words_dict = {k:v for k,v in sorted(neg_words_dict.items(), 
                                                key=lambda item:item[1], reverse=True)}

        neg_words_dict_0 = {k:v for k,v in sorted(neg_words_dict_0.items(), 
                                                key=lambda item:item[1], reverse=True)}

        neg_words_dict_1 = {k:v for k,v in sorted(neg_words_dict_1.items(), 
                                                key=lambda item:item[1], reverse=True)}

        # Create list with most common negation words
        common_words = list(neg_words_dict.keys())[0:N]

        # Collect words frequencies in classes 0/1
        counts_0  = [neg_words_dict_0[w] if w in list(neg_words_dict_0.keys()) else 0 for w in common_words]
        counts_1  = [neg_words_dict_1[w] if w in list(neg_words_dict_1.keys()) else 0 for w in common_words]

        # Store results in DataFrame and sort values
        df_counts = pd.DataFrame(list(zip(common_words, counts_0, counts_1)), 
                                         columns =['Negation word', 'Counts_class_0','Counts_class_1'])
        df_counts = df_counts.sort_values(by='Counts_class_0',ascending=False)
        df_counts.reset_index(drop=True, inplace=True)

        # Plot most common words in classes 0/1
        # -------------------------------------
        plt.figure(figsize=(18,5))

        # Get label/title specifications for plotting
        self._get_plotting_specs()

        plt.bar(x = df_counts['Negation word'],height=df_counts.Counts_class_0,edgecolor='grey',label='Class 0',alpha=0.6, color = color_1)
        plt.bar(x = df_counts['Negation word'],height=df_counts.Counts_class_1,edgecolor='grey',label='Class 1',alpha=0.6, color = color_2)
        plt.ylabel('Word counts',fontsize=self.label_fontsize)
        plt.title('Top '+str(N)+' most frequent negation words in corpus',
                  fontsize = self.title_fontsize,
                  fontweight = self.title_fontweight,
                  y = self.title_y)
        plt.legend()
        plt.xticks(rotation=90, fontsize=self.label_fontsize); 
        plt.xlim([-1,len(list(common_words))+0.5]);

      
        
    def plot_numeric_vars_vs_categorical_target(self,X_df,target):
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()
        
        # Get lists of num/cat vars
        self.fit(X_df)
        
        # Get lists of categorical/numeric variables 
        self.fit(X_df)
        
        fig,axes = plt.subplots(4,2,figsize=(18,4*4.5))

        fig.suptitle('Numeric variables vs. target',
                     fontsize = self.title_fontsize,
                     fontweight = self.title_fontweight,
                     y = self.title_y
                    )
        
        for i, c in enumerate(self.numeric_vars):

            # Boxplots
            if c in ['log_capital-gain','log_capital-loss']:
                data = X_df[X_df[c]>0]
            else:
                data = X_df

            sns.violinplot(x=c,
                           y=target,
                           data=data, ax = axes[i,0]
                          )
            axes[i,0].set_xlabel(c, fontsize=self.label_fontsize)
            axes[i,0].set_ylabel(target, fontsize=self.label_fontsize)


            # Histograms
            sns.histplot(x=c, data = X_df, hue=target, bins = 40, ax = axes[i,1])
            axes[i,1].set_ylabel('Counts', fontsize=self.label_fontsize)
            axes[i,1].set_xlabel(c, fontsize=self.label_fontsize)

            if (c == 'log_capital-gain') or (c == 'log_capital-loss') :
                axes[i,1].set_yscale('log')


        plt.tight_layout()
        
    def plot_numeric_vars_vs_numeric_target(self, X_df, target, columns_to_log = [], hue = None, title = True, cols = 3):
    
        """
        Plots the numeric variables vs numeric target
        """
 
        # Check that X_df is a DataFrame
        assert type(X_df) == pd.DataFrame
        
        # Check that the target is a numeric variable
        assert X_df[target].dtype == np.number, "target_distribution_num can only be used with numeric target variables."

        # Get label/title specifications for plotting
        self._get_plotting_specs()        
        
        # Get lists of cat/num variables
        self.fit(X_df)
        
        # Number of numeric variables to plot
        cols_to_plot = self.numeric_vars
        cols_to_plot.remove(target)
        nbr_numeric_vars = len(cols_to_plot)
        
        # Sort cols_to_plot by the corr coefficient
        cols_to_plot_sorted = list(np.abs(X_df.corr()[target]).sort_values(ascending=False).index[1:])
        
        # Define number of rows for plt.subplots()
        # (By default: cols = 3)
        if np.mod(nbr_numeric_vars,cols) == 0:
            rows = nbr_numeric_vars//cols
        else:
            rows = nbr_numeric_vars//cols + 1

        # Plot distributions of numeric variables
        fig, axes = plt.subplots(rows, cols, figsize=(18,rows*3))
        
        if title:
            fig.suptitle('Numeric variables vs target variable',
                         fontsize = self.title_fontsize,
                         fontweight = self.title_fontweight,
                         y = self.title_y
                        )
        
        for c, ax in zip(cols_to_plot_sorted,axes.ravel()):
            sns.scatterplot(data = X_df, x = c, y = target, alpha = 0.3, hue = hue, ax = ax)
            ax.set_xlabel(c, fontsize=self.label_fontsize)
            ax.set_ylabel(target, fontsize=self.label_fontsize)
            ax.set_title('corr: '+str(np.round(X_df[c].corr(X_df[target]),2)),fontsize=self.label_fontsize)
            
            if c in columns_to_log:
                ax.set_yscale('log')

        # Take care of empty plots by removings the axes
        nbr_empty_plots = (rows * cols) - nbr_numeric_vars
        
        if nbr_empty_plots != 0:
            #print(nbr_empty_plots)
            for ax in axes.ravel()[::-1][0:nbr_empty_plots]:
                ax.axis('off')

        plt.tight_layout()
   
        
    def plot_categorical_variables_unique(self,X_df,log_y_scale=False):
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()
        
        # Get lists of categorical/numeric variables 
        self.fit(X_df)
        
        plt.figure(figsize=(13,5))
        plt.suptitle('Number of unique values in categorical variables',
                     fontsize = self.title_fontsize,
                     fontweight = self.title_fontweight,
                     y = self.title_y)
        plt.ylabel('Number of unique labels', fontsize=self.label_fontsize)
        plt.xlabel('Column name', fontsize=self.label_fontsize)
        X_df[self.categorical_vars].nunique().sort_values(ascending=False).plot(kind='bar', edgecolor='grey', color = color_1);
    
        if log_y_scale:
            plt.yscale('log')
            
            
    def compute_rank(self, df_X, target):

        if target in df_X.columns:
            # Drop target variable
            df_colinear = df_X.drop(target, axis=1)

            # Remove missing values
            df_colinear.dropna(inplace=True)

            # Keep only numeric variables
            df_colinear = df_colinear.select_dtypes(include = np.number)

            # Rank of the matrix
            X1 = np.c_[np.ones(df_colinear.shape[0]), df_colinear]
            print('X1 shape: ',X1.shape)
            rank = np.linalg.matrix_rank(X1)

            if rank == X1.shape[1]:
                print("Number of independent features: {}. The matrix is full rank.".format(rank))
            elif rank < X1.shape[1]:
                print("Number of independent features: {}. The matrix is rank-deficient.".format(rank))
                
                
    def compute_condition_number(self, df_X, target):
        
        if target in df_X.columns:
            # Drop target variable
            df_colinear = df_X.drop(target, axis=1)

        # Remove missing values
        df_colinear.dropna(inplace=True)

        # Keep only numeric variables
        df_colinear = df_colinear.select_dtypes(include = np.number)

        # Condition number of matrix X1
        X1 = np.c_[np.ones(df_colinear.shape[0]), df_colinear]

        cn = np.linalg.cond(X1)
        print("Condition number:", cn)  # Depends on the noise value
        
        
    def addlabels(self, x, y, ax):
        for i in range(len(x)):
            ax.text(i, y[i], '{:,.0f}'.format(y[i]), ha = 'center', 
                     bbox = dict(facecolor = 'white', alpha =.8, )
                    )
            
        return self


    def plot_top_levels_in_cat_vars(self, X_df, columns_to_log = [], title = True, cols = 3):
        
        # Get lists of cat/num vars
        self.fit(X_df)
        
        # Get label/title specifications for plotting
        self._get_plotting_specs()

        # Number of categorical variables to plot
        cols_to_plot = self.categorical_vars
        nbr_vars = len(cols_to_plot)

        # Define number of rows for plt.subplots()
        # (By default: cols = 3)
        if np.mod(nbr_vars,cols) == 0:
            rows = nbr_vars//cols
        else:
            rows = nbr_vars//cols + 1

        # Plot top levels in categorical variables
        fig, axes = plt.subplots(rows, cols, figsize=(18,rows*5))
        
        if title:
            fig.suptitle('Top levels in categorical variables',
                         fontsize = self.title_fontsize,
                         fontweight = self.title_fontweight,
                         y = self.title_y)

        N = 30 # Choose how many levels to include in plots

        for c, ax in zip(cols_to_plot,axes.ravel()):
            w = X_df[c].value_counts()
            ax.bar(w.index[0:N], w.values[0:N], edgecolor='gray', color = color_1);
            self.addlabels(w.index[0:N],w.values[0:N],ax)
            ax.set_xticks(list(w.index[0:N]))
            ax.set_xticklabels(list(w.index[0:N]), rotation=45)
            ax.set_xlabel(c,fontsize=self.label_fontsize)
            ax.set_ylabel('Counts',fontsize=self.label_fontsize)

            if c in columns_to_log :
                ax.set_yscale('log') 

        # Take care of empty plots by removings the axes
        nbr_empty_plots = (rows * cols) - nbr_vars

        if nbr_empty_plots != 0:
            for ax in axes.ravel()[::-1][0:nbr_empty_plots]:
                ax.axis('off')

        plt.tight_layout()