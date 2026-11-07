# Methods

Analytical methods accompanying the manuscript. Reference numbers follow the
[reference list](REFERENCES.md); runnable commands are in [Analysis workflows](WORKFLOW.md).

## STAR Methods

### Key resources table

| Resource | Source | Identifier |
|---|---|---|
| Tahoe-100M expression and annotations | Zhang et al. [1] | 10.1016/j.cell.2026.08.035 |
| L1000 Level 3 expression and RNA-well annotations | Subramanian et al. [2] | 10.1016/j.cell.2017.10.049 |
| DRUG-seq counts and plate annotations | Ye et al. [3] | 10.1038/s41467-018-06500-x |
| sciPlex3 counts and replicate annotations | Srivatsan et al. [4] | 10.1126/science.aax6234 |
| CIGS counts and compound catalog | Xiang et al. [5] | 10.1038/s41592-025-02781-5 |
| MIX-seq counts and cell assignments | McFarland et al. [6] | 10.1038/s41467-020-17440-w |
| KPGT molecular features | Li et al. [9] | 10.1038/s41467-023-43214-1 |
| Control-expression baseline and G5 decoder | This study | DrugPerturbDB: data/config/models.json |
| Analysis code, source tables, and CPU environment | This study | https://github.com/baijinming97/DrugPerturbDB |

File-level download URLs, sizes, SHA256 checksums, gene mappings, and fixed data splits are supplied with the repository. The CPU environment specifies Python and package versions, CPU PyTorch, and the R dependencies needed for DRUG-seq preparation.

### Resource availability

#### Correspondence

Further information and requests should be directed to Sharon Prince (sharon.prince@uct.ac.za) or Geoff S. Nitschke (gnitschke@cs.uct.ac.za).

#### Materials availability

This study did not generate new biological materials.

#### Data and code availability

Source data are available from the studies listed above. Analysis code, numerical results, identity and condition annotations, gene mappings, and ten frozen models are available in [DrugPerturbDB](https://github.com/baijinming97/DrugPerturbDB). The repository provides download, preparation, and analysis commands with a locked CPU environment. Version-matched CIGS count inputs and fixed feature matrices for the exploratory comparisons are included. Predictive analyses use the supplied frozen weights. Code and study-generated materials are distributed under CC BY-NC 4.0; external resources retain their original terms.

### Experimental model and study participant details

We reanalyzed six drug-perturbation datasets from human cell lines [1-6], without new participant, animal, or cell-culture experiments. Source-wide analyses included all observations meeting annotation, control, and quality requirements in the specified releases, without first intersecting drugs with Tahoe. Cross-source analyses additionally required matched identities and treatment conditions.

Tahoe descriptions covered 379 drug IDs, 47 cell lines, and 52,886 qualified conditions at 0.05, 0.5, and 5 μM. Predictive analyses retained the fixed five-fold chemical-group protocol: a 261-drug candidate panel and 153 unique development-VALID drugs. Descriptive inclusion was independent of model roles. The model TEST partition was not evaluated, and expanded descriptions were not used to retrain or select models.

### Method details

#### Identity, gene axes, and response construction

Drug matching used source structures, official compound identifiers, CAS and PubChem information, and parent-structure reconciliation preserving stereochemistry. Conflicting or unresolved identities were not forcibly merged. Cell identities were reconciled to Cellosaurus; unresolved assignments were excluded from same-cell comparisons. Response similarity was not used for matching.

Each source was analyzed on native measured genes and a fixed set of 869 common Entrez genes. The latter retained the eight-source intersection defined during analysis design. The 882-gene intersection of the six current sources was evaluated separately on identical matched samples. Missing or ambiguous gene mappings were masked, not assigned zero. L1000's native axis contained 978 measured landmark genes, excluding inferred expression. Duplicate sciPlex3 Ensembl entries were consolidated and nonhuman genes excluded from its native human axis.

Count-based sources were normalized by the original full-library total before gene subsetting and transformed as log2(CPM + 1); log2(CPM + 100) was the processing sensitivity. Tahoe retained author-provided linear normalized expression, and L1000 L3 retained source normalization. Responses were treated expression minus a matched control from the same cell and experimental batch.

Tahoe plate-level pseudobulks required at least 50 full-QC cells and retained the source-qualified 47-cell panel. Descriptive condition profiles pooled eligible repeat plates by full-QC cell count; reconstruction of the frozen predictor instead retained its original equal-plate aggregation. sciPlex3 and MIX-seq pseudobulks were defined by cell, drug, dose, time, and independent replicate, required at least 20 cells, and required a corresponding control. DRUG-seq used nonempty DMSO wells on the same physical plate and excluded zero-library wells. L1000 controls were matched within RNA plate and condition; technical rereads of the same physical RNA well were merged.

#### Independent repeatability

Replicate pairs matched drug, cell, dose, and time but came from distinct physical plates or independent biological replicate units with nonshared control pools. Same-plate pairs sharing controls were excluded. Tahoe independence was verified at the physical-plate control-pool level.

Across-gene Pearson correlations were computed separately for normalized treated expression and treated-minus-control responses, then averaged across repeat pairs within condition and across conditions within drug. External sources used equal drug weight; Tahoe used equal chemical-group weight and equal drug weight within group. Within each endpoint, native and common gene analyses used the same valid pairs: a pair with undefined PCC on either axis was excluded from both. This rule was applied separately to expression and response. Repeatability was not used for attenuation correction or converted into a prediction ceiling.

#### Cross-source agreement and same-drug retrieval

Primary comparisons used 24 h treatments and all positive-dose pairs differing by no more than twofold. Dose-pair estimates were averaged within drug and cell before cell and drug aggregation. Responses averaged across each source's available cells and responses in confirmed matching cells were reported separately. Comparisons involving Tahoe used equal chemical-group weight, with equal drug weight within group; other source pairs used equal drug weight.

Same-drug and wrong-drug scores used candidate pools sharing treatment time and the two source-dose settings, with at least two drugs. Retrieval AUC was the mean of I(same > wrong) + 0.5 × I(same = wrong) over alternative drugs. Matched-minus-mismatched PCC and retrieval AUC used identical queries.

For the six source pairs with independent repeats on both sides, within-source and cross-source correlations were computed on identical drugs, cells, doses, and 869 genes. All estimates used single-repeat profiles, averaging eligible combinations within condition. Source pairs without joint repeat support were not scored in this comparison. Sensitivities included 882 genes, exact doses, the larger pseudocount, equal-drug weighting, and plate deletion or repeat averaging where feasible.

#### Response decomposition

Complete weighted panels were decomposed for each gene as Y(c,d) = μ + α(c) + β(d) + γ(c,d). The grand mean μ, cell deviation α, and drug deviation β were obtained from weighted marginal means; γ was the remaining interaction residual.[8] Measurements and predictions were decomposed separately using identical weights.

In Tahoe, drugs were grouped by their complete 47-cell control-expression pattern. Subtracting the weighted drug mean within these groups defined the drug main-effect component not represented by the baseline. The corresponding interaction component was obtained by constrained projection, with zero weighted mean across cells for each drug and across drugs within each cell-context group. Predictive corrections fixed contexts containing fewer than three drugs to zero; descriptive decomposition used the full constraint space. Projection therefore depends on reference drugs in the evaluation panel.

Source-wide component energies assigned equal total weight to each drug, shared equally among its observed cell, dose, time, and library conditions. Incomplete panels were stratified by dose, time, and library and fitted by weighted least squares using observed entries only. Cell and drug contributions were averaged over their two orders of entry; stratum energies were combined using their weight mass. These contributions and the residual sum to the observed response sum of squares. Designs unable to distinguish drug and interaction effects, including single-cell-line panels, were marked separately. The decomposition describes observed variation, including measurement error.

#### Fixed baseline and exploratory predictions

The frozen control-expression baseline was a seed1337 multilayer perceptron with three 2,048-unit hidden layers, ReLU activations, dropout 0.1, and 978 inputs and outputs. Matched control expression X1 predicted complete ΔX without drug identity or structure. Its original loss was 0.5 × MSE(X1 + predicted ΔX, X2) + 0.5 × (1 - condition-wise ΔX PCC). Checkpoints had been selected on outer VALID; these weights remained fixed. Component scores were derived from the same complete prediction, not from separately trained branches.

Figure S2 reports ten fixed configurations under chemical-group holdout or joint chemical-group and known-mechanism holdout. Inputs comprised molecular structure, target information, and external measurements; the target was the drug main-effect component not represented by the baseline. Inputs, model forms, response ranks, and drug support are specified in Data S2 and the supplied configurations. Annotated-target priors used target-linked L1000 genetic-perturbation signatures, distinct from same-drug chemical-treatment signatures.

Preprocessing, target bases, hyperparameters, and shrinkage were determined within the corresponding TRAIN fit blocks. The Ridge/three-program configurations used up to 32 input PCs, four chemical-group holdout blocks, and three-fold inner selection of the regularization parameter from {0.1, 1, 10, 100}. Each configuration was compared with three whole-drug training-label permutations (seeds 1337, 2026, and 3407), exchanging labels between equally sized chemical groups; held-out inputs remained unpermuted. Primary exploratory scores used the outer-TRAIN context-centered target; held-block centering was reported separately. Missing inputs retained the baseline in complete-response comparisons.

#### Drug-main-effect-to-interaction mapping

The G5 decoder used measured drug main effects to predict interaction responses not represented by the baseline. Selection and out-of-fold shrinkage used four complete control-context blocks, excluding fitting drugs whose chemical groups overlapped the held-out block. Hyperparameter selection, when needed, was nested within those blocks.

Projected 5 μM TRAIN interactions were arranged as drug-by-(cell × gene) matrices. A chemical-group-weighted rank-20 basis was fitted by iterative low-rank reconstruction of missing eligible entries, with at most 250 iterations and a relative objective-change tolerance of 1e-5. Drug main effects were centered within control-context groups; PCA used the unweighted TRAIN mean and up to 32 components, standardized by TRAIN score standard deviations. Available 0.5 and 0.05 μM TRAIN panels used the same basis and preprocessing, with dose covariate 1 + log10(dose/5). Each dose block's weights were normalized before pooling for Ridge fitting. Predicted interactions were constrained and shrunk using TRAIN calibration before addition to the baseline; measured drug main effects themselves were not added.

Cross-plate validation fixed Ridge regularization at 100 and the correction coefficient at 1, without a new search. Evaluation drugs and their chemical groups were excluded at every training dose. All four within- and cross-plate directions for plates 6 and 14 used the same 47 drugs, 47 cell lines, 978 genes, and 5 μM conditions. Source-plate drug means used all 47 cells.

Three controls permuted TRAIN inputs by whole drug within joint dose-coverage patterns, using a shared mapping across doses (seeds 20261103, 20261104, and 20261105). Only the input-to-code Ridge map was refitted; basis and preprocessing remained fixed, and evaluation inputs were unpermuted. Both transfer directions were retained and weighted equally when summarized.

### Quantification and statistical analysis

PCC was computed across genes within the specified condition or component and aggregated with the stated support weights; MSE used the same weights. Predictive analyses used equal chemical-group weight, equal drug weight within group, and equal cell and gene weights. Repeated VALID roles shared each drug's weight.

The 95% intervals were the 2.5th and 97.5th percentiles of 2,000 paired bootstrap draws. External source-wide and external-to-external comparisons resampled drug IDs; Tahoe and comparisons involving Tahoe resampled chemical groups. Model and control contrasts used the same draws. Cross-plate analyses re-estimated centering and projection in each draw while holding models fixed. Single-drug estimates had no drug-level interval; cross-source intervals required at least three resampling units. Panels with fewer than five drugs were marked descriptive.

For complete balanced panels, component MSEs were checked to sum to complete MSE. Pure-interaction corrections were required to preserve μ, α, and β and to produce equal changes in complete and interaction MSE. Undefined measurement and complete-response correlations were omitted rather than filled with zero. The retained increment-score convention assigned zero to degenerate corrections; the single-plate baseline's undefined interaction PCC was not treated as zero.

Intervals describe uncertainty conditional on the fixed samples and analysis, not historical model search or retraining. The three permutations are fixed negative controls rather than a permutation-test null distribution. Cross-plate intervals condition on the tested plate pair. Reported analyses can be recomputed in the locked CPU environment; independent checks cover representative repeat correlations, cross-plate prediction, sample and gene axes, role isolation, and energy closure. Numerical results, configurations, and pairing lists are supplied in Data S1, Data S2, and the repository.

