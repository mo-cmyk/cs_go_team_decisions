import logging
import os
import pickle

import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def _create_dataset(filepath: str) -> pd.DataFrame:
  """Generate a clean dataset in the right schema for model building purposes

  Args:
      filepath (str): the filepath of the source dataset

  Returns:
      pd.DataFrame: the cleaned data as a dataframe 
  """
  df = pd.read_feather(filepath)

  # filter columns to clean data
  column_filter = ['match_name', 'client_name', 'map_name', 'team',
                   'round_num',  'round_end_reason',
                   'team_score', 'current_side', 'starting_side', 'buy_type',
                   'round_start_eq_val', 'spend', 'round_win', 'match_win', '_data_id', '_match_id']
  dfs = list()

  # group by game to process games individually
  groups = df.groupby('match_name')
  for _, ct_df in tqdm(groups, desc='Building dataset', unit='matches'):

      # Asses Starting Teams
      ct_starting_team = df.iloc[0]['ct_team']
      t_starting_team = df.iloc[0]['t_team']

      # copy dataframe to view each match from both team's sides
      t_df = ct_df.copy()

      # process dataframe to add necessary columns
      for index_label, row_series in ct_df.iterrows():

        if ct_df.at[index_label, 'ct_team'] == ct_starting_team and ct_df.at[index_label, 't_team'] == t_starting_team:
            ct_df.at[index_label,
                     'team_score'] = ct_df.at[index_label, 'ct_score']
            ct_df.at[index_label, 'current_side'] = 'CT'
            ct_df.at[index_label, 'starting_side'] = 'CT'
            ct_df.at[index_label,
                     'buy_type'] = ct_df.at[index_label, 'ct_buy_type']
            ct_df.at[index_label, 'round_start_eq_val'] = ct_df.at[index_label,
                                                                   'ct_round_start_eq_val']
            ct_df.at[index_label, 'spend'] = ct_df.at[index_label,
                                                      'ct_round_spend_money']
            ct_df.at[index_label,
                     'round_win'] = ct_df.at[index_label, 'ct_round_win']

        else:
          ct_df.at[index_label, 'team_score'] = ct_df.at[index_label, 't_score']
          ct_df.at[index_label, 'current_side'] = 'T'
          ct_df.at[index_label, 'starting_side'] = 'CT'
          ct_df.at[index_label, 'buy_type'] = ct_df.at[index_label, 't_buy_type']
          ct_df.at[index_label, 'round_start_eq_val'] = ct_df.at[index_label,
                                                                 't_round_start_eq_val']
          ct_df.at[index_label, 'spend'] = ct_df.at[index_label,
                                                    't_round_spend_money']
          ct_df.at[index_label, 'round_win'] = ct_df.at[index_label, 't_round_win']

        ct_df['team'] = 0
        # encode game winner (target)
        ct_df['match_win'] = 1 if ct_df.iloc[len(
            ct_df) - 1]['end_t_score'] < ct_df.iloc[len(ct_df) - 1]['end_ct_score'] else 0

      # process dataframe copy to add necessary columns
      for index_label, row_series in t_df.iterrows():
          if t_df.at[index_label, 'ct_team'] == ct_starting_team and t_df.at[index_label, 't_team'] == t_starting_team:
              t_df.at[index_label, 'team_score'] = t_df.at[index_label, 't_score']
              t_df.at[index_label, 'current_side'] = 'T'
              t_df.at[index_label, 'starting_side'] = 'T'
              t_df.at[index_label, 'buy_type'] = t_df.at[index_label, 't_buy_type']
              t_df.at[index_label, 'round_start_eq_val'] = t_df.at[index_label,
                                                                   't_round_start_eq_val']
              t_df.at[index_label, 'spend'] = t_df.at[index_label,
                                                      't_round_spend_money']
              t_df.at[index_label,
                      'round_win'] = t_df.at[index_label, 't_round_win']

          else:
            t_df.at[index_label, 'team_score'] = t_df.at[index_label, 'ct_score']
            t_df.at[index_label, 'current_side'] = 'CT'
            t_df.at[index_label, 'starting_side'] = 'T'
            t_df.at[index_label, 'buy_type'] = t_df.at[index_label, 'ct_buy_type']
            t_df.at[index_label, 'round_start_eq_val'] = t_df.at[index_label,
                                                                 'ct_round_start_eq_val']
            t_df.at[index_label, 'spend'] = t_df.at[index_label,
                                                    'ct_round_spend_money']
            t_df.at[index_label, 'round_win'] = t_df.at[index_label, 'ct_round_win']

          # encode game winner (target)
          t_df['team'] = 1
          t_df['match_win'] = 1 if t_df.iloc[len(
              t_df) - 1]['end_t_score'] < t_df.iloc[len(t_df) - 1]['end_ct_score'] else 0

      # build a dataframe compositied of the ct and t side (both team sides)
      final_df = pd.concat([ct_df, t_df])

      # apply filter to only keep necessary columns
      final_df = final_df[column_filter]

      # adjust datatypes
      final_df = final_df.astype({"team_score": int, "round_win": int})

      # the unique row identifier now serves as a unique round  identifier since the data is now duplicated
      final_df.rename({'_data_id': '_round_id'})

      # add proccessed datframe from one game to list of all games
      dfs.append(final_df)

  # build dataframe from single games
  df = pd.concat(dfs)

  df.reset_index(inplace=True, drop=True)

  return df


def build_logistic_regression(df: pd.DataFrame) -> None:
  """Generates our Logistic Regression Baseline Model

  Args:
      df (pd.DataFrame): the training data as a dataframe
  """
  label_encoder = preprocessing.LabelEncoder()
  categorical_value_columns = [
      'buy_type', 'starting_side', 'current_side']

  unnecessary_columns = ['match_name', 'client_name', 'round_end_reason', 'round_win', 'team',
                         'map_name',
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

  model = LogisticRegression()
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  report = classification_report(y_test, prediction)
  print(report)
  with open('./models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(model, f)


def build_decision_tree_classifier(df: pd.DataFrame) -> None:
  """Generates our Decision Tree Classifier Model

  Args:
      df (pd.DataFrame): the training data as a dataframe
  """
  label_encoder = preprocessing.LabelEncoder()
  categorical_value_columns = [
      'buy_type', 'starting_side', 'current_side']

  unnecessary_columns = ['match_name', 'client_name', 'round_end_reason', 'round_win', 'team',
                         'map_name',
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

  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  report = classification_report(y_test, prediction)
  print(report)
  with open('./models/decision_tree_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)


def build_ml_perceptron(df: pd.DataFrame) -> None:
  """Generates our MLP Classifier

  Args:
      df (pd.DataFrame): the training data as a dataframe
  """
  label_encoder = preprocessing.LabelEncoder()
  categorical_value_columns = [
      'buy_type', 'starting_side', 'current_side']

  unnecessary_columns = ['match_name', 'client_name', 'round_end_reason', 'round_win', 'team',
                         'map_name',
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

  model = MLPClassifier(verbose=True, early_stopping=True,
                        hidden_layer_sizes=(20, 10))
  model.fit(X_train.values, y_train)
  prediction = model.predict(X_test)
  report = classification_report(y_test, prediction)
  print(report)
  with open('./models/mlp_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)


def build_ml_perceptron_with_maps(df: pd.DataFrame) -> None:
  """Generates our MLP Classifier but also taking maps into account

  Args:
      df (pd.DataFrame): the training data as a dataframe
  """
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

  model = MLPClassifier(verbose=True, early_stopping=True,
                        hidden_layer_sizes=(20, 10))
  model.fit(X_train.values, y_train)
  prediction = model.predict(X_test)
  report = classification_report(y_test, prediction)
  print(report)
  with open('./models/mlp_classifier_with_maps.pkl', 'wb') as f:
    pickle.dump(model, f)


def main():
    """ Runs scripts to build all mentioned models and if it doesn't exists a refined training dataset from the base dataset as mentioned in the paper.
    """
    logger = logging.getLogger(__name__)

    #  If it doesn't exist generate a refined dataset from the base dataset for training purposes'
    if not os.path.exists('./data/processed/team_score_and_buy__dataset__training.feather'):
      logger.info(
          "Couldn't find dataset suited for training constructing the datset first ...")
      FILEPATH = './data/processed/team_score_and_buy__dataset.feather'
      df = _create_dataset(FILEPATH)
      df.to_feather(
          './data/processed/team_score_and_buy__dataset__training.feather')
    else:
      logger.info(
          "Found refined training dataset useing now for model building.")
      df = pd.read_feather(
          './data/processed/team_score_and_buy__dataset__training.feather')

    # Base Models:

    # Model 1: Build Logistic Regression
    logger.info('Model 1/3: Generating Baseline Model')
    build_logistic_regression(df.copy())

    # Model 2: Build Decision Tree Classifier
    logger.info('Model 2/3: Building Decision Tree Classifier')
    build_decision_tree_classifier(df.copy())

    # Model 3: Build Multilayer Perceptron Classifier
    logger.info('Model 3/3: Building MLP Classifier')
    build_ml_perceptron(df.copy())

    # ================================================================
    # Taking Maps into account
    build_ml_perceptron_with_maps(df.copy())

    logger.info('✅ All Models created')
    logger.info('💾 Models Sotred in ./models')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
