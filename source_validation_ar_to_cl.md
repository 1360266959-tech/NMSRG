# NMSRG source-only hyperparameter-selection record

## Scope

**Formal development task:** Office-Home `Ar → Cl`.

This concise audit record summarizes the formal source-only parameter-selection run. The complete raw log is retained separately in `output_question6.txt`.

## Protocol

```text
Source domain: Ar, Target domain: Cl
========== Source-validation parameter selection ==========
Stratified source split: 80/20; seed: 2024
```

| Item | Specification |
|---|---|
| Labels used for selection | Labeled source samples only |
| Source split | Stratified 80% development training and 20% development validation |
| Random seed | 2024 |
| Selection metric | Source-validation mean class accuracy, MCA |
| $q$ | $\{0.5, 0.666667\}$ |
| $\lambda$ | $\{0.0001, 0.001, 0.01\}$ |
| $\beta$ | $\{0.00001, 0.0001, 0.001\}$ |
| Candidate tuples | $2\times3\times3=18$ |

Each candidate uses the same random seed, source split, graph sampling, and clustering initialization. Target labels are excluded during parameter selection.

## Selected configuration

```text
Selected tuple: q=0.666667, lambda=0.001, beta=0.0001, source-val MCA=0.9926

Selected by source validation for Ar -> Cl:
q = 0.666667, lambda = 0.001, beta = 0.0001, source-val MCA = 0.9926
```

The fixed formal configuration is therefore:

$$
(q,\lambda,\beta)=\left(\frac{2}{3},10^{-3},10^{-4}\right).
$$

## Use after selection

The selected tuple is fixed for formal complete-model comparisons, class-wise analyses, and complete-model ablation rows. Final evaluation reuses all labeled source samples and unlabeled target features. Target labels are used only to calculate evaluation metrics after prediction.

## Associated reproducibility materials

- `daima/zzh_txt/select_params_source_validation.txt`: source-only stratified-selection implementation and the 18-tuple grid.
- `daima/zzh_txt/office_home.txt`: driver that performs source validation before final target evaluation.
- `output_question6.txt`: complete raw source-validation log, including the formal `Ar → Cl` selection record.
- `output_question6_2.txt`: fixed-configuration evaluation log.
