# Beyond-CVaR
In this work, we incorporate a new set of risks (spectral risk measures and CPT value). into learning algorithms, and we deal with the case of heavy tailed losses.  First of all,we study a general-purpose estimator of theses risks for potentially heavy tailed randomvariables, one which is easy to implement in practice.  This estimator requires nothingmore than a finite variance. We provide high-probability excess bounds and compare themwith already existing ones. Once this is established, we then derive learning algorithms which consists of a stage-wise robust gradient descent. For this procedure we provide againhigh-probability excess-risk bounds. To complement the theory we conduct empirical testsof the underlying spectral risk measures estimator and the learning algorithm derived fromit.
# Authors
Matthew J.Holland and El Mehdi Haress

# Beyond-CVaR

Tools and experiments around prediction error (CPT/SRM) and excess risk (SRM).

## Install

```bash
pip install -e .
import beyond_cvar as bc
# Modules available:
# - bc.prediction_error_cpt
# - bc.prediction_error_srm
# - bc.excess_risk_over_t_srm
# - bc.excess_risk_over_n_srm
#Test:
#pytest -q
```
## License
MIT

