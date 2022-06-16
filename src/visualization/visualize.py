import logging
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import find_dotenv, load_dotenv
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from constants import *

# Use the pgf backend
mpl.use('pgf')


# use latex pgf format instead of png
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

logger = logging.getLogger(__name__)


def _load_data(filepath: str) -> pd.DataFrame:
  """Read the final data into a pandas dataframe

  Args:
      filepath (str): filepat of the final data

  Returns:
      pd.DataFrame:
  """
  df = pd.read_feather(filepath)
  return df


def _create_heat_map(buy_type_probability_and_count: list(), columns: str, index: str, values: str, bar_label: str,  max: int, min: int):
    # read data as df
    df = pd.DataFrame(buy_type_probability_and_count)

    # adjust table order for better visualization
    df["ct_buy_type"] = df["ct_buy_type"].astype(
        pd.api.types.CategoricalDtype(categories=buy_types_reversed))
    df["t_buy_type"] = df["t_buy_type"].astype(
        pd.api.types.CategoricalDtype(categories=buy_types))

    # pivot table
    data = df.pivot_table(columns=columns, index=index, values=values)

    # adjust color scheme
    cmap = sns.cm.rocket_r

    # plot heatmap
    ax = sns.heatmap(data.T, vmin=min, vmax=max, annot=True,
                     cmap=cmap, cbar_kws={'label': bar_label}, fmt='g')

    # annotate axis
    ax.set(xlabel='CT Buy Type', ylabel='T Buy Type')
    return plt


def _generate_buy_type_probability_and_count(df: pd.DataFrame) -> list():
  buy_type_probability_and_count = list()

  # check every possible combination if ct and t buy types
  for ct_btype in buy_types:
    for t_btype in buy_types:
      # generate a 'buy frame' on each given buy combination
      buy_frame = df[(df['t_buy_type'] == t_btype) &
                     (df['ct_buy_type'] == ct_btype)]

      count_games = len(buy_frame)
      count_games_ct_won = len(buy_frame[buy_frame['ct_round_win'] == 1])

      # measure probability as the fraction of games where t won over total games with that buy observation
      ct_win_prob = count_games_ct_won / count_games

      # generate a lookup object
      probability_and_count = {
          't_buy_type': t_btype,
          'ct_buy_type': ct_btype,
          'ct_win_probability': ct_win_prob,
          'count_rounds_with_buy_observation': count_games
      }
      buy_type_probability_and_count.append(probability_and_count)

  return buy_type_probability_and_count


def visualize_buy_type_probability(buy_type_probability_and_count: list()) -> None:
    """Generate heatmap plot based on how buy types affect the probability of winning a round

    Args:
        buy_type_probability_and_count (list): a preproccessed aggregation
    """
    plt = _create_heat_map(buy_type_probability_and_count, columns='t_buy_type', index='ct_buy_type',
                           values='ct_win_probability', bar_label='CT Win Probability', min=0, max=1)
    plt.savefig("./reports/figures/buy_type_probability.pgf")
    logger.info(
        f"💾 Plot stored as: './reports/figures/buy_type_probability.pgf'")
    plt.clf()


def visualize_buy_type_count(buy_type_probability_and_count: list()) -> None:
    """Visualize heatmap of number of observations occuring in the buy_type_probability plot

    Args:
        buy_type_probability_and_count (list): a preproccessed aggregation
    """
    plt = _create_heat_map(buy_type_probability_and_count, columns='t_buy_type', index='ct_buy_type',
                           values='count_rounds_with_buy_observation', bar_label='Number of Rounds', min=0, max=60000)
    plt.savefig('./reports/figures/buy_type_count.pgf')
    logger.info(
        f"💾 Plot stored as: './reports/figures/buy_type_count.pgf'")
    plt.clf()


def visualize_game_win_probability_by_rounds(df: pd.DataFrame) -> None:
    """Summarize the probability of winning a game depending on the round score shown as a heatmap

    Args:
        df (pd.DataFrame): the processed data as a pandas dataframe
    """
    dfs = list()

    # group individual matches
    groups = df.groupby('match_name')

    # generate lookup list
    for _, df in groups:
        ct_starting_team = df.iloc[0]['ct_team']
        t_starting_team = df.iloc[0]['t_team']
        for index_label, _ in df.iterrows():

            if df.at[index_label, 'ct_team'] == ct_starting_team and df.at[index_label, 't_team'] == t_starting_team:
                df.at[index_label,
                      'ct_starting_team_score'] = df.at[index_label, 'ct_score']
                df.at[index_label,
                      't_starting_team_score'] = df.at[index_label, 't_score']
            else:
                df.at[index_label,
                      'ct_starting_team_score'] = df.at[index_label, 't_score']
                df.at[index_label,
                      't_starting_team_score'] = df.at[index_label, 'ct_score']

        df['ct_starting_team_win'] = 1 if df.iloc[len(
            df) - 1]['end_t_score'] > df.iloc[len(df) - 1]['end_ct_score'] else 0
        df['t_starting_team_win'] = 1 if df.iloc[len(
            df) - 1]['end_t_score'] < df.iloc[len(df) - 1]['end_ct_score'] else 0
        dfs.append(df)

    df = pd.concat(dfs)

    buy_type_probability_and_count = list()
    scores = range(0, 16)

    for ct_score in scores:
        for t_score in scores:
            score_frame = df[(df['ct_starting_team_score'] == ct_score) & (
                df['t_starting_team_score'] == t_score)]
            count_games = len(score_frame)
            count_games_ct_won = len(
                score_frame[score_frame['ct_starting_team_win'] == 1])

            ct_win_prob = count_games_ct_won / count_games
            if ct_score == 15 and t_score == 15:
                ct_win_prob = None
            probability_and_count = {
                't_starting_team_score': t_score,
                'ct_starting_team_score': ct_score,
                'ct_win_probability': ct_win_prob,
                'count_games_with_buy_observation': count_games
            }
            buy_type_probability_and_count.append(probability_and_count)

    # get dataframe from lookup and processed data
    df = pd.DataFrame(buy_type_probability_and_count)

    # pivot table
    data = df.pivot_table(columns='ct_starting_team_score', index='t_starting_team_score',
                          values='ct_win_probability')

    # adjust column and axis order
    data = data.iloc[::-1]

    # # plot heatmap
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(data, annot=True, cmap=cmap, fmt='.1f', cbar_kws={
                     'label': 'CT Starting Team Win Probability'})

    # set axis labels
    ax.set(xlabel='CT Starting Team Score', ylabel='T Starting Team Score')

    # save figure
    plt.savefig('./reports/figures/game_win_probability_by_rounds.pgf')
    logger.info(
        f"💾 Plot stored as: './reports/figures/game_win_probability_by_rounds.pgf'")
    plt.clf()

def visualize_pairplot_to_discover_relevant_features(df: pd.DataFrame) -> None:
    """Plot a pairplot to identify all corallations in the dataset

    Args:
        df (pd.DataFrame): the data as a dataframe
    """
    sns.pairplot(df.sample(5000), hue="current_side", kind='reg', plot_kws={'scatter_kws': {'alpha': 0.1}})
    plt.savefig('./reports/figures/pairplot.pgf')
    logger.info(
        f"💾 Plot stored as: './reports/figures/pairplot.pgf'")
    plt.clf()


def _prepare_train_test_for_base_model_comparison(df: pd.DataFrame):
    label_encoder = preprocessing.LabelEncoder()

    categorical_value_columns = [
        'buy_type', 'starting_side', 'current_side']

    unnecessary_columns = ['match_name', 'client_name', 'round_end_reason', 'round_win', 'team',
                            'map_name',
                           'round_start_eq_val', 'spend', '_data_id', '_match_id']

    # print(df.columns)
    df.drop(labels=unnecessary_columns, inplace=True, axis=1)

    for c in categorical_value_columns:
        df[c] = label_encoder.fit_transform(df[c])

    col = df.columns

    standscl = preprocessing.StandardScaler()
    
    
    standscl.fit(df)
    standscl.transform(df)
    df = standscl.transform(df)
    df = pd.DataFrame(df, columns=col)

    df['match_win'] = label_encoder.fit_transform(df['match_win'])
    X = df.drop('match_win', axis=1)
    y = df['match_win']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def _prepare_train_test_for_map_model_comparison(df: pd.DataFrame):
    label_encoder = preprocessing.LabelEncoder()

    categorical_value_columns = [
        'buy_type', 'starting_side', 'current_side', 'map_name']

    unnecessary_columns = ['match_name', 'client_name', 'round_end_reason', 'round_win', 'team',
                           'round_start_eq_val', 'spend', '_data_id', '_match_id']

    df.drop(labels=unnecessary_columns, inplace=True, axis=1)

    for c in categorical_value_columns:
        df[c] = label_encoder.fit_transform(df[c])

    col = df.columns

    standscl = preprocessing.StandardScaler()
    standscl.fit(df)
    standscl.transform(df)
    df = standscl.transform(df)
    df = pd.DataFrame(df, columns=col)

    df['match_win'] = label_encoder.fit_transform(df['match_win'])
    X = df.drop('match_win', axis=1)
    y = df['match_win']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def visualize_auc_curves_base_models(df: pd.DataFrame) -> None:
    """Plot the auroc curves for the base models

    Args:
        df (pd.DataFrame): the test data as a dataframe
    """

    _, X_test, _, y_test = _prepare_train_test_for_base_model_comparison(df)

    with open('./models/logistic_regression.pkl', 'rb') as f:
        regression_model = pickle.load(f)

    with open('./models/decision_tree_classifier.pkl', 'rb') as f:
        decision_tree_classifier = pickle.load(f)

    with open('./models/mlp_classifier.pkl', 'rb') as f:
        mlp_classifier = pickle.load(f)

    y_pred = regression_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label="Logistic Regression, AUC="+str(auc))

    y_pred = decision_tree_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label="Decsion Tree Classifier, AUC="+str(auc))

    y_pred = mlp_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label="Multilayer Perceptron, AUC="+str(auc))

    plt.legend()

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.savefig("auroc_curves_maps.pgf")

    # save figure
    plt.savefig('./reports/figures/auroc_curves_model_comparison.pgf')
    logger.info(
        f"💾 Plot stored as: './reports/figures/auroc_curves_model_comparison.pgf'")
    plt.clf()


def visualize_auc_curves_map_included(df: pd.DataFrame) -> None:
    """Plot the auroc curves for the MLP Classifier with the map as an encoded feature

    Args:
        df (pd.DataFrame): the test data as a dataframe
    """
    _, X_test, _, y_test = _prepare_train_test_for_map_model_comparison(df.copy())

    with open('./models/mlp_classifier_with_maps.pkl', 'rb') as f:
        mlp_maps = pickle.load(f)

    with open('./models/mlp_classifier.pkl', 'rb') as f:
        mlp_no_maps = pickle.load(f)

    y_pred = mlp_maps.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label="MLP Classifier with maps, AUC="+str(auc))


    _, X_test, _, y_test = _prepare_train_test_for_base_model_comparison(df)

    y_pred = mlp_no_maps.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label="MLP Classifier without maps, AUC="+str(auc))

    plt.legend()

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # save figure
    plt.savefig('./reports/figures/auroc_curves_model_maps.pgf')
    logger.info(
        f"💾 Plot stored as: './reports/figures/auroc_curves_model_maps.pgf'")
    plt.clf()


def main():
    """ Runs data visualization scripts to turn generate figures and graphs from the cleaned data
    """

    logger.info('Generating Visualizations 📊')
    logger.info('Loading data ...')

    #  If it doesn't exist generate a refined dataset from the base dataset for training purposes'
    if not os.path.exists('./data/processed/team_score_and_buy__dataset__training.feather'):
      logger.info(
          "Couldn't find dataset suited for training constructing the datset first ...")
      FILEPATH = './data/processed/team_score_and_buy__dataset.feather'
    #   df_refined = _create_dataset(FILEPATH)
      df_refined.to_feather(
          './data/processed/team_score_and_buy__dataset__training.feather')
    else:
      logger.info(
          "Found refined training dataset using now for model building.")
      df_refined = pd.read_feather(
          './data/processed/team_score_and_buy__dataset__training.feather')

    DATA_FILEPATH = './data/processed/team_score_and_buy__dataset.feather'
    df = _load_data(DATA_FILEPATH)

    logger.info('Generating lookup for buy combinations ...')
    # Plot how Round Wins Are influenced by buy type
    observations = _generate_buy_type_probability_and_count(df)

    # Plot round win probability
    logger.info('Plotting round based win probability by buy type ...')
    visualize_buy_type_probability(observations)

    # Plot how many rounds fit each specific buy combination
    logger.info('Plotting observation count ...')
    visualize_buy_type_count(observations)

    # Plot how rounds affect game win probability
    logger.info('Plotting rounds based game win probability ...')
    visualize_game_win_probability_by_rounds(df)
    
    # Create Pairplot to identify Correlations
    logger.info('Constructing pairplot ...')
    visualize_pairplot_to_discover_relevant_features(df.refined.copy())

    # Plot auroc curves to compare the models
    logger.info('Plotting auroc curves to compare the base models ...')
    visualize_auc_curves_base_models(df_refined.copy())

    # Plot auroc curves to compare the models
    logger.info(
        'Plotting auroc curves to compare the models with the maps as encoded feature ...')
    visualize_auc_curves_map_included(df_refined.copy())

    # done
    logger.info('✅ Visualizations created')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
