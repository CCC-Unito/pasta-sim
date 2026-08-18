"""
pasta.py

Perspectivist Annotation of Subjective Tasks
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy.special import softmax

@dataclass
class PastaConfig:
    # --- scale of the simulation ---
    n_instances: int = 1000
    n_annotators: int = 10
    n_classes: int = 2
    embedding_dim: int = 64

    # --- main parameters ---
    subjectivity: float = 0.3   # 0 = all annotators share one objective view
    ambiguity: float = 0.3      # 0 = fully deterministic, crisp labels

    # --- how "hard" the true label boundary is intrinsically (independent of the
    #     ambiguity knob, which scales this globally) ---
    class_separation: float = 6.0     # distance between class prototypes
    instance_spread: float = 1.0      # how tightly instances cluster around their class

    # --- annotator population structure (polarization) ---
    n_annotator_subgroups: int = 2
    annotator_subgroup_separation: float = 6.0
    annotator_spread: Optional[float] = None  # individual variation within a subgroup;
                                               # None -> defaults to class_separation, so
                                               # subjectivity=1 gives personal perturbations
                                               # comparable in scale to the inter-class distance

    # --- annotation coverage / reliability ---
    annotations_per_instance: Optional[int] = None  # None = all annotators label it
    repeats: int = 1                                # repeat labelings per assigned annotator

    seed: Optional[int] = 42

    def __post_init__(self):
        if not (0.0 <= self.subjectivity <= 1.0):
            raise ValueError("subjectivity must be in [0, 1]")
        if not (0.0 <= self.ambiguity <= 1.0):
            raise ValueError("ambiguity must be in [0, 1]")
        if self.annotations_per_instance is not None:
            if not (1 <= self.annotations_per_instance <= self.n_annotators):
                raise ValueError("annotations_per_instance must be between 1 and n_annotators")
        if self.annotator_spread is None:
            self.annotator_spread = self.class_separation


def sample_clustered_points(
    n_points: int,
    n_clusters: int,
    dim: int,
    cluster_spread: float,
    cluster_separation: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample `n_points` in `dim`-D space, arranged around `n_clusters` centers.

    Centers are spread out on a random configuration scaled by `cluster_separation`;
    points are drawn from an isotropic Gaussian of std `cluster_spread` around their
    assigned center.

    Returns (points, cluster_assignment, centers).
    """
    if n_clusters == 1:
        centers = np.zeros((1, dim))
    else:
        centers = rng.normal(size=(n_clusters, dim))
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centers = centers / norms * cluster_separation

    assignment = rng.integers(0, n_clusters, size=n_points)
    noise = rng.normal(scale=cluster_spread, size=(n_points, dim))
    points = centers[assignment] + noise
    return points, assignment, centers

class PastaSimulator:
    """Generates a synthetic annotated dataset given a SimulationConfig."""

    def __init__(self, config: PastaConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self._build_class_prototypes()
        self._build_instances()
        self._build_annotators()

    # ---- construction steps ---- #

    def _build_class_prototypes(self):
        cfg = self.config
        # one fixed point per class, spread apart by class_separation
        if cfg.n_classes == 1:
            self.class_prototypes = np.zeros((1, cfg.embedding_dim))
        else:
            centers = self.rng.normal(size=(cfg.n_classes, cfg.embedding_dim))
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.class_prototypes = centers / norms * cfg.class_separation

        # bandwidth of the RBF-style classifier, tied to how far apart classes are
        self.bandwidth = max(cfg.class_separation / 2.0, 1e-6)

    def _build_instances(self):
        cfg = self.config
        true_classes = self.rng.integers(0, cfg.n_classes, size=cfg.n_instances)
        noise = self.rng.normal(
            scale=cfg.instance_spread, size=(cfg.n_instances, cfg.embedding_dim)
        )
        embeddings = self.class_prototypes[true_classes] + noise

        self.instances = pd.DataFrame({
            "instance_id": np.arange(cfg.n_instances),
            "true_class": true_classes,
        })
        self.instance_embeddings = embeddings  # (n_instances, dim)

    def _build_annotators(self):
        cfg = self.config
        flat_dim = cfg.n_classes * cfg.embedding_dim

        flat_shifts, subgroup, subgroup_centers = sample_clustered_points(
            n_points=cfg.n_annotators,
            n_clusters=cfg.n_annotator_subgroups,
            dim=flat_dim,
            cluster_spread=cfg.annotator_spread,
            cluster_separation=cfg.annotator_subgroup_separation,
            rng=self.rng,
        )
        # reshape each annotator's flat vector into a (n_classes, dim) perturbation
        # of the class prototypes -- i.e. "how this person personally sees each class"
        self.annotator_shifts = flat_shifts.reshape(
            cfg.n_annotators, cfg.n_classes, cfg.embedding_dim
        )
        self.annotators = pd.DataFrame({
            "annotator_id": np.arange(cfg.n_annotators),
            "subgroup": subgroup,
        })
        self._annotator_subgroup_centers = subgroup_centers

    # ---- core generative step ---- #

    def _label_probabilities(self, instance_idx: int, annotator_idx: int) -> np.ndarray:
        """P(label = c) for a given instance/annotator pair."""
        cfg = self.config
        x = self.instance_embeddings[instance_idx]
        personal_prototypes = (
            self.class_prototypes + cfg.subjectivity * self.annotator_shifts[annotator_idx]
        )
        sq_dist = np.sum((personal_prototypes - x) ** 2, axis=1)
        logits = -sq_dist / (2 * self.bandwidth ** 2)

        temperature = (1 / (1 - cfg.ambiguity+0.0001))**2
        return softmax(logits / temperature)
        
    def annotate(self) -> pd.DataFrame:
        """Run the annotation process and return a long-format DataFrame with columns:
        instance_id, annotator_id, repeat_id, true_class, label, p_true_class
        (p_true_class = the probability the annotator's own distribution assigned
        to the objectively true class -- useful for diagnostics).
        """
        cfg = self.config
        rows = []

        for i in range(cfg.n_instances):
            true_class = self.instances.loc[i, "true_class"]

            if cfg.annotations_per_instance is None:
                assigned = np.arange(cfg.n_annotators)
            else:
                assigned = self.rng.choice(
                    cfg.n_annotators, size=cfg.annotations_per_instance, replace=False
                )

            for a in assigned:
                probs = self._label_probabilities(i, a)
                for r in range(cfg.repeats):
                    label = self.rng.choice(cfg.n_classes, p=probs)
                    rows.append({
                        "instance_id": i,
                        "annotator_id": int(a),
                        "repeat_id": r,
                        "true_class": int(true_class),
                        "label": int(label),
                        "p_true_class": float(probs[true_class]),
                    })

        return pd.DataFrame(rows)

