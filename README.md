Simulate the manual annotation of a dataset, with independent control over how much
disagreement comes from:

  * SUBJECTIVITY  -> individual annotator differences (a stable "personal lens" each
                     annotator applies, per Basile et al. 2021 / Basile 2020)
  * AMBIGUITY     -> stimulus/data-inherent ambiguity (flattens the true label
                     distribution, so even one person would be inconsistent on repeats)

Theoretical grounding
---------------------
Basile, Fell, Fornaciari, Hovy, Paun, Plank, Poesio, Uma (2021),
"We Need to Consider Disagreement in Evaluation" (ACL Anthology 2021.bppf-1.3),
identify three sources of annotation disagreement:

  1. Individual differences (the annotator's own perception/background)
  2. Stimulus characteristics (ambiguity in the instance itself)
  3. Context (moment-to-moment noise, attention slips, etc.)

and cite a simulation (Basile, 2020) where synthetic annotations are generated as a
function of two knobs: "difficulty" (general ambiguity of the task) and "subjectivity"
(an annotator-background-linked bias). This module implements and extends that idea:
instances and annotators are both embeddings in a vector space, so their *distribution*
(class separation, clustering, polarized annotator subgroups, ...) can also be
controlled directly.

Generative model
-----------------
1. `n_classes` concept prototypes are placed in an `embedding_dim`-dimensional space.
2. Each instance is assigned a true class and embedded near that class's prototype
   (within-class spread = how "typical" vs "borderline" instances are).
3. Each annotator gets a personal, FIXED perturbation of the class prototypes
   (their own subjective "concept space"), drawn from one of several possible
   subgroups (to simulate polarization / cultural clusters of annotators). The
   magnitude of this perturbation is scaled by `subjectivity`.
4. For an (instance, annotator) pair, per-class logits are the negative squared
   distance from the instance to that annotator's (perturbed) prototypes. These
   logits are passed through a softmax with a temperature controlled by
   `ambiguity` (higher ambiguity -> flatter distribution -> more stochastic,
   instance-driven disagreement, independent of who's annotating).
5. A label is sampled from that softmax distribution. Annotators can optionally
   repeat-annotate the same instance, which lets you empirically decompose
   disagreement into an "inter-annotator" (subjectivity-driven) component and an
   "intra-annotator" (ambiguity-driven) component.