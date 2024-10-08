# ML Poisonous Mushrooms

### Run

In order to run the hyper optimization you can configure the file `ml_poisonous_mushrooms/hyper_task.py` and run the command below:

```bash
poetry run poe hyper
```

In order to monitor resources, you can install `htop` (if on linux) and run it.

```bash
sudo apt install htop
```

```bash
htop
```

### Analysis Ensemble

The ensemble method will take all possible models and find the best combination of them for the ensemble model.

The script will produce `ensemble_optimization_<limit_data_percentage>_<model_run>.csv` file that contains the cross validation score for each ensemble combination.

In order to compare the best ensemble model to the singular models within the ensemble, you can run the following command on the produced file:

```python
df = df.sort_values(['score'], ascending=[False])

df_top = df.head(1)

top_combination_names = df_top["combination"].values[0].split("-")


df_singular = df.loc[df["combination"].isin(top_combination_names) & ~df["combination"].str.contains("-")]

df = pd.concat(
    [
        df_singular,
        df_top
    ], axis=0
)
```
