# 🍝 PASTA simulator

Simulate the manual annotation of a dataset, with independent control over how much
disagreement comes from:

  * Subjectivity (individual annotator differences)
  * Ambiguity (stimulus/data-inherent difficulty of the task

## Parameters

#### main parameters
- **ambiguity**: (float, default: 0.3) how ambiguous is the annotation task, from 0 (fully deterministic) to 1 (random noise)
- **subjectivity** (float, default: 0.3) how subjective is the task, from 0 (all annotators share one objective view) to 1 (the labeling depends entirely on the individual perception)

#### annotation parameters
- **n_instances** (int, default: 1000) number of instances to generate
- **n_annotators** (int, default: 10) number of simulated annotators
- **n_classes** (int, default: 2) number of possible labels
- **embedding_dim** (int, default: 64) number of dimensions of the instance/annotator embedding space
- **n_annotator_subgroups** (int, default: 2) how many groups are the annotators polarized into
- **annotations_per_instance** (int, default: None) how many annotations should each instance receive (if None, all annotators label every instance
- **repeats** (int, default: 1) how many times each annotator repeats the labeling

#### tweaks
- **class_separation** (float, default: 6.0) distance between class prototypes
- **instance_spread** (float, default: 1.0) how tightly instances cluster around their class
- **annotator_subgroup_separation** (float, default: 6.0) distance between annotator groups' centers
- **annotator_spread** (float, default: None) individual variation within a subgroup; *None* defaults to **class_separation**
- **seed** (int, default: 42) seed for the random number generator

## Usage

### Installation

``pip install git+https://github.com/CCC-Unito/pasta-sim.git``

### Example

````
from pasta import PastaConfig, PastaSimulator

cfg = PastaConfig(ambiguity=0.5, subjectivity=0.5)
sim = PastaSimulator(cfg)
df = sim.annotate()
df.to_csv("example_annotations.csv", index=False)
print (df.head())

   instance_id  annotator_id  repeat_id  true_class  label  p_true_class
0            0             0          0           1      1      0.967583
1            0             1          0           1      1      0.961386
2            0             2          0           1      1      0.825421
3            0             3          0           1      1      0.996422
4            0             4          0           1      1      0.995316
````

## Theoretical Background

Basile et al. (2021), [We Need to Consider Disagreement in Evaluation](https://aclanthology.org/2021.bppf-1.3/),
identify three sources of annotation disagreement:

  1. Individual differences (the annotator's own perception/background)
  2. Stimulus characteristics (ambiguity in the instance itself)
  3. Context (moment-to-moment noise, attention slips, etc.)

In [Basile (2020)](https://www.researchgate.net/publication/351798920_It's_the_End_of_the_Gold_Standard_as_We_Know_It_Leveraging_Non-aggregated_Data_for_Better_Evaluation_and_Explanation_of_Subjective_Tasks), synthetic annotations are generated as a
function of two parameters: "difficulty" (general ambiguity of the task) and "subjectivity"
(an annotator-background-linked bias). This module implements and extends that idea:
instances and annotators are both embeddings in a vector space, so their *distribution*
(class separation, clustering, polarized annotator subgroups, ...) can also be controlled directly.

**Instances** are defined as vectors in a $d$-dimensional embedding space. A set of $C$ class prototypes $\mu_1, \dots, \mu_C \in \mathbb{R}^d$ define the task's objective category structure, placed at mutual distance controlled by a **class separation** parameter $k$:

$$
\mu_c = k \cdot \hat{u}_c
$$

where $g_c \sim \mathcal{N}(0, I_D)$ for each class $c = 1, \dots, C$ and ${\hat{u}_c = \frac{g_c}{\lVert g_c \rVert_2}}$. With this construction, every prototype lies exactly on the sphere of radius $k$ in $\mathbb{R}^D$.

The directions $\hat{u}_c$ are iid uniformly on the unit sphere, i.e., not deliberately separated from each other.

Each instance $i$ is assigned a ground-truth class $y_i^* \in \{1,\dots,C\}$ and embedded as perturbations of the class prototypes by isotropic Gaussian noise: $x_i \sim \mathcal{N}(\mu_{y_i^*}, \tau_x^2 I_d)$, where the within-class variance $\tau_x$ (**instance spread**) governs how much instances are geometrically ambiguous by construction (i.e., how close to a class boundary they naturally fall), independent of any global disagreement parameter.

**Annotators** are not represented as points in this space, but as *transformations* of it. Each annotator $j$ is tied to a perturbation $\delta_{j,c} \in \mathbb{R}^d$ for every class $c$, drawn once and held fixed for the duration of the simulation, sampled from one of $K$ subgroup distributions to model polarized populations of annotators, inspired by the annotator polarization discussed in [Akhtar et al. (2019)](https://link.springer.com/chapter/10.1007/978-3-030-35166-3_41). Annotator $j$'s transformed prototype for class $c$ is then

$$
\mu_{c,j} = \mu_c + s \cdot \delta_{j,c}
$$

where $s \in [0,1]$ is a global **subjectivity** parameter. At $s=0$ all annotators share the objective prototypes, while as $s \to 1$, each annotator's internal category structure diverges from the objective one and from every other annotator's.

### Annotation

Given instance embedding $x_i$ and annotator $j$'s personal prototypes, we define per-class logits via a Radial Basis Function (RBF) kernel similarity to the personal prototypes with **bandwidth** $\frac{k}{2}$:

$$
z_{i,c,j} = -\frac{\lVert x_i - \mu_{c,j} \rVert^2}{k^2/2}
$$

and obtain an annotation distribution through a softmax with temperature $\tau(a) = \frac{1}{(1-a)^2}$:

$$
p(y_{i,j} = c) = \frac{e^{z_{i,c,j}/\tau(a)}}{\sum_{c'} e^{z_{i,c',j}/\tau(a)}}
$$

where $a \in [0,1]$ is the global **ambiguity** parameter.

At $a=0$, labeling is near-deterministic, governed only by geometric proximity; as $a \to 1$, the distribution flattens toward uniform, injecting stochastic uncertainty that is a property of the instance, not the annotator. An observed label is then simply sampled as argmax over $p(y_{i,j})$.
````
