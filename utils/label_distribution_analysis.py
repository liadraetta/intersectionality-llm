import re
import os 
import pandas as pd
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score

def filter_full_dataset_by_socio_demographic_levels(df, socio_demographic_variables):
    """
    Filter the DataFrame by specifying the demographic variables
    """
    filtered_df = df[df['demographics'] == socio_demographic_variables].reset_index(drop=True)
    return filtered_df 

def read_all_model_datasets(model_name, cot_name, results_base_path, results_socdem_base_path):
    """
    Read all datasets for the specified model and CoT setting.
    Returns a dictionary with key: dict, where key is the socio-demographic variable combination
    """
    all_output_dfs = {}
    # Read the dataset for the baseline.
    df_baseline_path = results_base_path / f'cleaned_predictions_{model_name}_{cot_name}_baseline.csv'
    all_output_dfs['baseline'] = pd.read_csv(df_baseline_path)
    # Read the datasets for the socio-demographic variables.
    all_output_df_names = os.listdir(results_socdem_base_path)
    all_output_df_names = [file for file in all_output_df_names if file.startswith(f'cleaned_predictions_{model_name}_{cot_name}') and file.endswith('.csv')]
    for df_name in all_output_df_names:
        match = re.search(r'cleaned_predictions_(.*)_(CoT|noCoT)_(.*).csv', df_name)
        if match:
            socio_demographic_variables = match.group(3)
            df_path = results_socdem_base_path / df_name
            all_output_dfs[socio_demographic_variables] = pd.read_csv(df_path)
    return all_output_dfs



def filter_out_missing_rows(all_model_datasets, print_all=False):
    # Create a deep copy to avoid modifying the original datasets
    all_model_datasets_filtered = deepcopy(all_model_datasets)
    
    # Dictionary to store missing rows for each dataset
    missing_rows_dict = {}
    
    # For each dataset, identify (postId, annId) pairs that contain prediction -1
    all_missing_rows = set()
    for key, df in all_model_datasets.items():
        # Find rows where prediction is -1 and extract postId, annId as tuples
        missing_tuples = list(zip(df[df['prediction'] == -1]['postId'], 
                                  df[df['prediction'] == -1]['annId']))
        
        # Store the missing tuples for this dataset
        missing_rows_dict[key] = missing_tuples
        if print_all:
            print(f"Missing rows in {key}: {len(missing_tuples)}")
        # Add to the global set of missing rows
        all_missing_rows.update(missing_tuples)
        
    print(f"Total unique missing rows across all datasets: {len(all_missing_rows)} and valid rows: {len(all_model_datasets_filtered['baseline']) - len(all_missing_rows)}")
    
    # Remove rows with missing predictions from all datasets
    for key, df in all_model_datasets_filtered.items():
        # Filter out rows where (postId, annId) is in the set of all missing rows
        df_filtered = df[~df.set_index(['postId', 'annId']).index.isin(all_missing_rows)]
        all_model_datasets_filtered[key] = df_filtered.reset_index(drop=True)
    
    return all_model_datasets_filtered

def safe_cohen_kappa(y_true, y_pred):
    """
    Modify cohen kappa so that if one labeller returns all 1s, then return nan instead of 0.
    """
    if len(set(y_true)) == 1 or len(set(y_pred)) == 1:
        return float('nan')
    return cohen_kappa_score(y_true, y_pred)


def compute_intersectionality_cohen_kappa(all_model_datasets_filtered, socio_demographic_variables):
    """
    Compute Cohen Kappa for the dataset containing all socio-demographic variables to all other datasets.
    """
    # Identify the intersectionality dataset
    all_socdem_string = '_'.join(socio_demographic_variables)
    intersectionality_dataset = all_model_datasets_filtered[all_socdem_string]
    all_model_datasets_filtered = {key: df for key, df in all_model_datasets_filtered.items() if key != all_socdem_string}
    
    # Extract all combinations of socio-demographic variables
    all_combinations = intersectionality_dataset['demographics'].unique()
    
    # Initialize a dictionary to store Cohen Kappa values
    all_results = {} # ('socio-demographic variable combination': {'all': {'kappa': kappa_value, 'exact_match': exact_match_count}, 'white_male_left': {'kappa': kappa_value, 'exact_match': exact_match_count}, ......})
    
    # Compute overall Cohen Kappa and Cohen Kappa for each socio-demographic variable combination. Also report number of instances and exact matches
    # Iterate through each dataset and compute Cohen Kappa.
    for key, df in all_model_datasets_filtered.items():
        # Extract the predictions from both datasets, ensuring they are aligned using postId and annId
        merged_df = pd.merge(intersectionality_dataset[['postId', 'annId', 'prediction']], 
                            df[['postId', 'annId', 'prediction']], 
                            on=['postId', 'annId'], 
                            suffixes=('_intersectional', '_current'))
        
        intersectionality_predictions = merged_df['prediction_intersectional']
        df_predictions = merged_df['prediction_current']        
        # Count number of items where we have exact match 
        exact_match_count = int((intersectionality_predictions == df_predictions).sum())
        exact_match_count_perc = exact_match_count / len(intersectionality_predictions) * 100
        kappa_value = cohen_kappa_score(intersectionality_predictions, df_predictions)
        all_results[key] = {}
        all_results[key]['all'] = {'kappa': kappa_value, 
                                   'exact_match': exact_match_count, 
                                   'exact_match_perc': round(exact_match_count_perc, 1), 
                                   'count': len(intersectionality_predictions)}
        # Now consider all combinations of socio-demographic variables, filter the dataset and compute Cohen Kappa for them as well.
        for socdem_combination in all_combinations:
            # Filter the intersectionality dataset by the socio-demographic variable combination
            filtered_intersectionality_df = filter_full_dataset_by_socio_demographic_levels(intersectionality_dataset, socdem_combination)
            
            # Merge the filtered datasets
            merged_filtered_df = pd.merge(filtered_intersectionality_df[['postId', 'annId', 'prediction']], 
                                          df[['postId', 'annId', 'prediction']], 
                                          on=['postId', 'annId'], 
                                          suffixes=('_intersectional', '_current'))
            if not merged_filtered_df.empty:
                intersectionality_predictions_filtered = merged_filtered_df['prediction_intersectional']
                df_predictions_filtered = merged_filtered_df['prediction_current']
                
                exact_match_count_filtered = int((intersectionality_predictions_filtered == df_predictions_filtered).sum())
                exact_match_count_perc_filtered = exact_match_count_filtered / len(intersectionality_predictions_filtered) * 100
                kappa_value_filtered = safe_cohen_kappa(intersectionality_predictions_filtered, df_predictions_filtered)
                all_results[key][f"{socdem_combination}"] = {'kappa': kappa_value_filtered, 
                                                              'exact_match': exact_match_count_filtered, 
                                                              'exact_match_perc': round(exact_match_count_perc_filtered, 1), 
                                                              'count': len(intersectionality_predictions_filtered)}
                            
    # Now consider filtered versions of the datasets and compute Cohen Kappa for them as well
    return all_results


### PLOTTING FUNCTIONS ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def parse_filter_key(key):
    """Parse the filter key to create a readable column name"""
    if key == 'all':
        return 'Overall'
    # Parse the dictionary string
    try:
        # Remove quotes and braces, split by commas
        key_clean = key.strip("{}").replace("'", "").replace('"', '')
        parts = [part.strip() for part in key_clean.split(',')]
        # Extract values
        values = []
        for part in parts:
            if ':' in part:
                value = part.split(':')[1].strip()
                values.append(value)
        return ' '.join(values).title()
    except:
        return key

def create_visualization_table(data, metric='kappa'):
    """Create a table visualization of the results"""
    # Get all unique filter keys across all trait sets
    all_filters = set()
    for trait_set in data.keys():
        all_filters.update(data[trait_set].keys())
    
    # Create readable column names and store total counts
    filter_mapping = {f: parse_filter_key(f) for f in all_filters}
    total_counts = {}
    
    # Get total counts for each filter across all trait sets
    for filter_key in all_filters:
        col_name = filter_mapping[filter_key]
        # Find the total count for this filter (should be same across trait sets)
        for trait_set in data.keys():
            if filter_key in data[trait_set] and 'count' in data[trait_set][filter_key]:
                total_counts[col_name] = data[trait_set][filter_key]['count']
                break
    
    # Create the dataframe
    rows = []
    for trait_set in data.keys():
        row = {'Trait Set': trait_set}
        for filter_key in all_filters:
            col_name = filter_mapping[filter_key]
            if filter_key in data[trait_set]:
                row[col_name] = data[trait_set][filter_key][metric]
            else:
                row[col_name] = np.nan
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Trait Set')
    # Change row names in df
    df.index = [trait_set.replace('_', ' ').title() for trait_set in df.index]
    df.index = [trait_set.replace('Political', 'Politics') for trait_set in df.index]
    # Change the order of the rows to be Baseline, Race, Gender, Politics
    desired_order = ['Baseline', 'Gender', 'Race', 'Politics', 'Gender Race', 'Gender Politics', 'Race Politics']
    df = df.reindex(desired_order)

    # Order df columns based on total counts
    ordered_columns = sorted(df.columns, key=lambda x: total_counts.get(x, 0), reverse=True)
    df = df[ordered_columns]
    
    return df, filter_mapping, total_counts

def visualize_heatmap(data, 
                      vmin=-0.1, 
                      vmax=1.0, 
                      save_path=None, 
                      model_name=None,
                      plot_var='kappa'): # exact_match_perc
    kappa_df, filter_mapping, exact_counts = create_visualization_table(data, plot_var)
    # Create a proper heatmap visualization
    plt.figure(figsize=(16, 10))  # Increased height to accommodate two-line labels
    
    # Set appropriate bounds for kappa scores
    vmin, vmax = vmin, vmax
    cmap = 'RdYlGn'
    
    # Create the heatmap
    cbar_kws = {'label': 'Kappa Score'} if plot_var == 'kappa' else {'label': 'Exact Match Percentage'}
    mask = kappa_df.isnull()
    ax = sns.heatmap(kappa_df,
                     annot=True,
                     fmt='.3f',
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     mask=mask,
                     cbar_kws=cbar_kws,
                     square=False)
    
    # Add a vertical line to separate "Overall" from other columns
    if 'Overall' in kappa_df.columns:
        plt.axvline(x=1, color='black', linewidth=2)
    
    # Create custom x-axis labels with exact match counts
    x_labels = []
    for col in kappa_df.columns:
        if col in exact_counts:
            label = f"{col}\n({exact_counts[col]})"
        else:
            label = col
        x_labels.append(label)
    # Set the custom labels
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    plt.title(f'{model_name}', fontsize=20, pad=20)

    plt.xlabel('Input values in intersection', fontsize=16)
    plt.ylabel('Trait variables', fontsize=16)
    
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the figure if save_path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def visualize_combined_heatmap(all_models_data, vmin=-0.1, vmax=1.0, save_path=None, plot_var='kappa'):
    """
    Create a combined heatmap with all 4 models in subplots with shared color axis
    """
    model_names = list(all_models_data.keys())
    # Create subplots - 2x2 grid for 4 models
    fig, axes = plt.subplots(2, 2, figsize=(22, 18), dpi=150)
    axes = axes.flatten()
    
    # Common colormap and normalization
    cmap = 'RdYlGn'
    
    # Create visualization tables for each model
    model_dfs = {}
    all_exact_counts = {}
    for model_name, data in all_models_data.items():
        kappa_df, filter_mapping, exact_counts = create_visualization_table(data, plot_var)
        model_dfs[model_name] = kappa_df
        all_exact_counts[model_name] = exact_counts
    
    # Create heatmaps for each model
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        kappa_df = model_dfs[model_name]
        exact_counts = all_exact_counts[model_name]
        
        # Create mask for missing values
        mask = kappa_df.isnull()
        
        # Create heatmap (without individual colorbars)

        sns.heatmap(kappa_df,
                   annot=True,
                   fmt='.3f' if plot_var == 'kappa' else '.1f',
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   mask=mask,
                   cbar=False,  # No individual colorbar
                   square=False,
                   ax=ax)
        
        # Add vertical line to separate "Overall" from other columns
        if 'Overall' in kappa_df.columns:
            ax.axvline(x=1, color='black', linewidth=2)
        
        # Create custom x-axis labels with exact match counts
        x_labels = []
        for col in kappa_df.columns:
            if col in exact_counts:
                label = f"{col}\n({exact_counts[col]})"
            else:
                label = col
            x_labels.append(label)
        
        # Set labels and title
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_title(f'{model_name}', fontsize=20, pad=10)
        ax.set_xlabel('Input values in intersection', fontsize=16)
        ax.set_ylabel('Trait variables', fontsize=16)
        ax.tick_params(axis='y', rotation=0)
    
    # Add a shared colorbar
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Create normalization and scalar mappable for colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Adjust layout to make room for colorbar and title
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.92, 
                       wspace=0.3, hspace=0.4)
    
    # Add colorbar - positioned to the right of the plots
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    if plot_var == 'kappa':
        cbar.set_label('Kappa Score', fontsize=20)
    elif plot_var == 'exact_match_perc':
        cbar.set_label('Exact Match Percentage', fontsize=20)
    
    # Overall title
    # fig.suptitle('Cohen\'s Kappa Scores Across All Models', fontsize=18, y=0.96)
    
    # Save the figure if save_path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved to: {save_path}")
    else:
        plt.show()


def visualize_heatmap_combined_metrics(data, 
                                       vmin=50.0, 
                                       vmax=100.0, 
                                       save_path=None, 
                                       model_name=None):
    """
    Create a heatmap showing exact match percentages with kappa scores in brackets below
    """
    kappa_df, filter_mapping, exact_counts = create_visualization_table(data, 'kappa')
    perc_df, _, _ = create_visualization_table(data, 'exact_match_perc')
    
    # Create a proper heatmap visualization
    plt.figure(figsize=(16, 10))
    
    # Create combined annotation matrix
    combined_annot = perc_df.copy().astype(str)
    for i in range(len(perc_df)):
        for j in range(len(perc_df.columns)):
            kappa_val = kappa_df.iloc[i, j]
            perc_val = perc_df.iloc[i, j]
            
            if pd.isna(perc_val):
                combined_annot.iloc[i, j] = ''
            else:
                combined_annot.iloc[i, j] = f'{perc_val:.1f}%\n({kappa_val:.3f})'
    
    # Set appropriate bounds for exact match percentages (0-100%)
    cmap = 'RdYlGn'
    
    # Create the heatmap using exact match percentages for color scheme
    cbar_kws = {'label': 'Exact Match Percentage (%)'}
    mask = perc_df.isnull()
    ax = sns.heatmap(perc_df,
                     annot=combined_annot,
                     fmt='',  # Empty format since we're providing custom annotations
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     mask=mask,
                     cbar_kws=cbar_kws,
                     square=False)
    
    # Add a vertical line to separate "Overall" from other columns
    if 'Overall' in perc_df.columns:
        plt.axvline(x=1, color='black', linewidth=2)
    
    # Create custom x-axis labels with exact match counts
    x_labels = []
    for col in perc_df.columns:
        if col in exact_counts:
            label = f"{col}\n({exact_counts[col]})"
        else:
            label = col
        x_labels.append(label)
    
    # Set the custom labels
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    plt.title(f'{model_name} - Exact Match % with Kappa Scores', fontsize=20, pad=20)
    plt.xlabel('Input values in intersection', fontsize=16)
    plt.ylabel('Trait variables', fontsize=16)
    
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the figure if save_path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def visualize_combined_heatmap_with_metrics(all_models_data, 
                                           vmin=50.0, 
                                           vmax=100.0, 
                                           save_path=None):
    """
    Create a combined heatmap with all 4 models showing exact match percentages with kappa scores in brackets
    """
    model_names = list(all_models_data.keys())
    # Create subplots - 2x2 grid for 4 models
    fig, axes = plt.subplots(2, 2, figsize=(24, 20), dpi=150)
    axes = axes.flatten()
    
    # Common colormap and normalization
    cmap = 'RdYlGn'
    
    # Create visualization tables for each model
    model_kappa_dfs = {}
    model_perc_dfs = {}
    all_exact_counts = {}
    
    for model_name, data in all_models_data.items():
        kappa_df, filter_mapping, exact_counts = create_visualization_table(data, 'kappa')
        perc_df, _, _ = create_visualization_table(data, 'exact_match_perc')
        model_kappa_dfs[model_name] = kappa_df
        model_perc_dfs[model_name] = perc_df
        all_exact_counts[model_name] = exact_counts
    
    # Create heatmaps for each model
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        kappa_df = model_kappa_dfs[model_name]
        perc_df = model_perc_dfs[model_name]
        exact_counts = all_exact_counts[model_name]
        
        # Create combined annotation matrix
        combined_annot = perc_df.copy().astype(str)
        for row in range(len(perc_df)):
            for col in range(len(perc_df.columns)):
                kappa_val = kappa_df.iloc[row, col]
                perc_val = perc_df.iloc[row, col]
                
                if pd.isna(perc_val):
                    combined_annot.iloc[row, col] = ''
                else:
                    combined_annot.iloc[row, col] = f'{perc_val:.0f}%\n({kappa_val:.2f})'
        
        # Create mask for missing values
        mask = perc_df.isnull()
        
        # Create heatmap using exact match percentages for color scheme (without individual colorbars)
        sns.heatmap(perc_df,
                   annot=combined_annot,
                   fmt='',  # Empty format since we're providing custom annotations
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   mask=mask,
                   cbar=False,  # No individual colorbar
                   square=False,
                   ax=ax)
        
        # Add vertical line to separate "Overall" from other columns
        if 'Overall' in perc_df.columns:
            ax.axvline(x=1, color='black', linewidth=2)
        
        # Create custom x-axis labels with exact match counts
        x_labels = []
        for col in perc_df.columns:
            if col in exact_counts:
                label = f"{col}\n({exact_counts[col]})"
            else:
                label = col
            x_labels.append(label)
        
        # Set labels and title
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_title(f'{model_name}', fontsize=20, pad=10)
        ax.set_xlabel('Input values in intersection', fontsize=16)
        ax.set_ylabel('Trait variables', fontsize=16)
        ax.tick_params(axis='y', rotation=0)
    
    # Add a shared colorbar
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Create normalization and scalar mappable for colorbar (using exact match percentage range)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Adjust layout to make room for colorbar and title
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.92, 
                       wspace=0.3, hspace=0.4)
    
    # Add colorbar - positioned to the right of the plots
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Exact Match Percentage (%)', fontsize=20)
    
    # Save the figure if save_path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Combined plot with metrics saved to: {save_path}")
    else:
        plt.show()