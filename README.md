# CS:GO Aanalyzing Team level Decisions and Modeling Win Probability in COUNTER-STRIKE: GLOBAL OFFENSIVE 🔫

This is the code and a little demodata as well as a guide to go for the extra mile 🏃‍♀️ to test the code with demofiles at a large scale.

## TL;DR
This is an overview of all functions in this project

```bash
❯ make help
Available rules:

clean               Delete all compiled Python files
create_environment  Set up python interpreter environment
data                Make Dataset
lint                Lint using flake8
parse               Parse raw demofiles
requirements        Install Python Dependencies
test_environment    Test python environment is setup correctly
train               Train Model
visuals             Prepare all reports and figures
(END)
```

## Get more data
Although in this repository for illustration some demofiles are included as sample data or the parsed datasets are provided you have to download cs:go demofiles first to be able to use the pipeline to its full extent. The Python package [GoScrape](https://github.com/mo-cmyk/goscrape) is suitable for this.

Start by installing the package using [pip](https://pypi.org/project/goscrape/)

```bash
pip install goscrape
```

To learn more about other functions of the package it makes sense to read the documentation. In the following only the functions relevant for this project will be explained.

First a json lookup needs to be constructed to store all relevant information for matches and games.
I used CS:GO matches hosted by [htlv.org](https://htlv.org) that took place between 2021/04/01 and 2022/04/01. All of those matches were online.
This lookup can be achieved by the following command:

```bash
goscrape events -s 2021-04-01 -e 2022-04-01 -m -t ONLINE -f data/lookups/

```

Afterwards the demofiles have to be downloaded.
```bash
goscrape fetch -f event_lookup__2021_04_01__2022_04_01__ONLINE.json -o data/raw/
```

They can either be directly put into the /raw subdirectory of the data folder or all manually extracted and put there.


## Getting Started

### 1. Setup a python 🐍 or conda enviroment
```bash
make create_environment
```
```bash
make test_environment
```
If conda is installed it will be preferred

### 2. Install all relevant requirements 📦
```bash
pip install -r requirements.txt
```

### 3. Parse raw demofiles ⚙️
```bash
make parse
```
This will parse the raw files under /data/raw and put the structured json files in /data/interim

### 4. Construct initial dataset from parsed demofiles 🧩
```bash
make data
```
This will concatenate the parsed json files into a single [feather file](https://arrow.apache.org/docs/python/feather.html#:~:text=Feather%20is%20a%20portable%20file,Python%20(pandas)%20and%20R.) stored under /data/processed

### 4. Build training dataset and features to construct and train models 🏋️‍♂️
```bash
make train
```
Build the Training Dataset and build all relevant Models. These will be saved under /models

### 5. Plot all relevant figures and reports 📊
```bash
make visuals
```
Run the command to get all relevant plots, charts and figures in the LaTeX compatible .pgf format

## Project Structure
The project is structured in the following way:
```bash
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── interim
│   │   ├── 9z-vs-meta-py-nuke.json
│   │   ├── b8-vs-run-or-die-m4-vertigo.json
│   │   ├── bears-vs-isurus-m1-ancient_68896.json
│   │   ├── black-dragons-vs-intz-m2-vertigo.json
│   │   ├── bth-vs-boca-juniors-m2-dust2.json
│   │   ├── coscu-army-vs-isurus-nuke.json
│   │   ├── croatia-vs-czech-republic-mirage.json
│   │   ├── enterprise-vs-kappab-m2-overpass.json
│   │   ├── exploit-vs-souldazz-nuke.json
│   │   ├── furious-vs-paqueta-m3-vertigo.json
│   │   ├── gaijin-vs-prifu-m1-mirage.json
│   │   ├── gaijin-vs-prifu-m2-overpass.json
│   │   ├── iberian-family-vs-offset-m2-vertigo.json
│   │   ├── isurus-vs-eqole-inferno.json
│   │   ├── isurus-vs-intz-dust2.json
│   │   ├── isurus-vs-leviatan-m2-nuke.json
│   │   ├── kappab-vs-biiceps-m2-mirage_66795.json
│   │   ├── nexus-vs-portugal-m1-dust2.json
│   │   ├── order-vs-paradox-m1-dust2.json
│   │   ├── sestri-vs-ec-kyiv-m2-inferno.json
│   │   └── virtus-pro-vs-gambit-vertigo.json
│   ├── lookups
│   │   └── event_lookup__2021_04_01__2022_04_01__ONLINE.json
│   ├── processed
│   │   ├── team_score_and_buy__dataset.feather
│   │   └── team_score_and_buy__dataset__training.feather
│   └── raw
│       ├── 9z-vs-meta-py-nuke.dem
│       ├── b8-vs-run-or-die-m4-vertigo.dem
│       ├── bears-vs-isurus-m1-ancient_68896.dem
│       ├── black-dragons-vs-intz-m2-vertigo.dem
│       ├── bth-vs-boca-juniors-m2-dust2.dem
│       ├── coscu-army-vs-isurus-nuke.dem
│       ├── croatia-vs-czech-republic-mirage.dem
│       ├── enterprise-vs-kappab-m2-overpass.dem
│       ├── exploit-vs-souldazz-nuke.dem
│       ├── furious-vs-paqueta-m3-vertigo.dem
│       ├── gaijin-vs-prifu-m1-mirage.dem
│       ├── gaijin-vs-prifu-m2-overpass.dem
│       ├── iberian-family-vs-offset-m2-vertigo.dem
│       ├── isurus-vs-eqole-inferno.dem
│       ├── isurus-vs-intz-dust2.dem
│       ├── isurus-vs-leviatan-m2-nuke.dem
│       ├── kappab-vs-biiceps-m2-mirage_66795.dem
│       ├── nexus-vs-portugal-m1-dust2.dem
│       ├── order-vs-paradox-m1-dust2.dem
│       ├── sestri-vs-ec-kyiv-m2-inferno.dem
│       └── virtus-pro-vs-gambit-vertigo.dem
├── models
│   ├── decision_tree_classifier.pkl
│   ├── logistic_regression.pkl
│   ├── mlp_classifier.pkl
│   └── mlp_classifier_with_maps.pkl
├── notebooks
│   ├── DataExploration.ipynb
│   └── HyperparameterTuning.ipynb
├── reports
│   └── figures
│       ├── auroc_curves_model_comparison.pgf
│       ├── auroc_curves_model_maps.pgf
│       ├── buy_type_count-img0.png
│       ├── buy_type_count.pgf
│       ├── buy_type_probability-img0.png
│       ├── buy_type_probability.pgf
│       ├── feature_importance.pgf
│       ├── game_win_probability_by_rounds-img0.png
│       ├── game_win_probability_by_rounds.pgf
│       └── pairplot.pgf
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── make_dataset.py
│   │   └── parse_data.py
│   ├── models
│   │   ├── __init__.py
│   │   └── train_model.py
│   └── visualization
│       ├── __init__.py
│       ├── constants.py
│       └── visualize.py
├── test_environment.py
└── tox.ini

13 directories, 78 files

```
## License

The source code for the site is licensed under the MIT license, which you can find in
the MIT-LICENSE.txt file.

## Notes  
⚠️ Warning the demofiles provided in this repository do not correlate to the event lookup nor to the complete training dataset provided. They are mereley for demonstrations purposes and are not enough to provide relevant data.

I am not affiliated with hltv.org in any way and all rights and ownership for the provided demofiles are belong to hltv.org.