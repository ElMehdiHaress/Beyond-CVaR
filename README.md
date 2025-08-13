## Beyond-CVaR

Python rework of experiments around **risk measures beyond CVaR**, including:
- **CPT (Cumulative Prospect Theory)** prediction error;
- **SRM (Spectral Risk Measures)** prediction error;
- **SRM excess risk** vs. sample size `n` and time horizon `T`.

The package is structured for reproducible tests
# Authors
Matthew J.Holland and El Mehdi Haress

## Beyond-CVaR

Tools and experiments around prediction error (CPT/SRM) and excess risk (SRM).

## Install
```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
# pip install pytest numpy scipy scikit-learn statsmodels matplotlib
import beyond_cvar as bc
# Modules available:
# - bc.prediction_error_cpt
# - bc.prediction_error_srm
# - bc.excess_risk_over_t_srm
# - bc.excess_risk_over_n_srm
#Test:
#pytest -q
```

## Project layout

Beyond-CVaR/

├── src/

│   └── beyond_cvar/

│       ├── __init__.py

│       ├── prediction_error_cpt.py

│       ├── prediction_error_srm.py

│       ├── excess_risk_over_t_srm.py

│       └── excess_risk_over_n_srm.py

├── tests/

│   ├── test_prediction_error_cpt_examples.py

│   ├── test_prediction_error_srm_examples.py

│   ├── test_excess_risk_over_t_srm_examples.py

│   └── test_excess_risk_over_n_srm_examples.py

├── pyproject.toml

└── README.md


## License
MIT

