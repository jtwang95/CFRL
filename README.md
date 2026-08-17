# Counterfactually Fair Reinforcement Learning via Sequential Data Preprocessing

This is the official Python implementation for **Counterfactually Fair Reinforcement Learning via Sequential Data Preprocessing**. The paper proposes sequential data preprocessing (SDP), a model-agnostic approach that transforms longitudinal data before offline policy learning to promote counterfactual fairness while retaining information useful for decision making. The repository contains synthetic, semi-synthetic, and real-data experiments comparing SDP with standard and fairness-oriented baselines.

## Setup

Create and activate the Conda environment from the repository root:

```bash
conda env create --file environment.yml
conda activate fair_rl
```

The main implementation of SDP is in `core/preprocessing.py`; offline policy-learning code is in `policy_learning/`.

## Reproducing the experiments

Run each command from the indicated directory because the scripts use relative paths. The reported experiments used 20-core parallelization; users can adjust `CORES` or `N_CORES` in each script according to their available computational resources. Full runs can take substantial time.

### Synthetic experiments

```bash
cd simulation/code
python -u run_exp.py
```

Results are written to `simulation/out/`. The three experiment settings in `run_exp1()` reproduce Figures 4(a)–(c) in the main paper (sample-size effect), Figure I.1 in the supplement (horizon effect), and Figure 4(d) in the main paper (sensitive-variable effect). Comment or uncomment the corresponding setting in `run_exp.py` to run each experiment. The plots and tables can be recreated with `simulation/code/plots_tables.ipynb`.

### Semi-synthetic experiments

```bash
cd semi_synthetic/code
python -u run_analysis.py
```

The semi-synthetic data are based on the PowerED study. The trained neural-network weights for the PowerED transition-and-reward environment are included in `semi_synthetic/models/`, allowing the semi-synthetic data to be generated without access to the original PowerED data. Example generated datasets are provided under `semi_synthetic/results/<attribute>_tmnn_rmnn_N1000_1000_T12/data/`. Select `education_b`, `ethnicity_b`, `race_b`, or `sex_b` in the `__main__` block of `run_analysis.py`. Results are written to `semi_synthetic/results/`, and figures can be recreated with `semi_synthetic/code/plots.ipynb`.

To reproduce the preprocessing from authorized PowerED data, place the demographic file `PowerED RL Data_Table 1.dta` and trajectory file `PowerED data.xlsx` in `semi_synthetic/data/`, then run:

```bash
cd semi_synthetic/code
Rscript preprocess_data.R
Rscript imputation.R
```

The PowerED data are available upon request rather than through a public repository; see the [PowerED study website](https://cmcd.sph.umich.edu/research-program-areas/using-advances-in-mobile-health-to-make-opioid-use-safer/) for contact information.

### Real-data experiments

The PowerED dataset used in this analysis contains 207 patients discharged from the emergency department and followed for 12 weeks. Each week, patients received one of three interventions: brief interactive voice response (IVR) calls, longer IVR calls, or calls with a counselor. Weekly outcomes include self-reported pain, pain interference, and opioid misuse scores, along with baseline characteristics collected before the study. See the [PowerED study publication](https://www.jmir.org/2023/1/e44165) for further details.

```bash
cd real_data_analysis/code
python -u run_analysis.py
```

The real-data analysis reuses `PowerEDEnv` from `semi_synthetic/code/powered_env.py`, which loads the imputed PowerED data from `semi_synthetic/data/impute_data.csv` and the trained environment weights from `semi_synthetic/models/`. The script evaluates all four sensitive attributes and saves a timestamped CSV to `real_data_analysis/results/`. Summaries can be recreated with `real_data_analysis/code/summary_results.ipynb`.

## Citation

If you use this work, please cite:

```bibtex
@article{wang2025counterfactually,
  title={Counterfactually fair reinforcement learning via sequential data preprocessing},
  author={Wang, Jitao and Shi, Chengchun and Piette, John D and Loftus, Joshua R and Zeng, Donglin and Wu, Zhenke},
  journal={arXiv preprint arXiv:2501.06366},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
