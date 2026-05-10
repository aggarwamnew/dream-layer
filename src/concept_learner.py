"""concept_learner.py — Autonomous Concept Learning via LLM Exploration (v6).

The core Phase 1 experiment: a persistent outer layer that learns concepts
by exploring a frozen language model. Uses MLX + Llama 3 8B (4-bit) as
the environment and sentence-transformers for response embeddings.

v1 Architecture (Projection + GRU):
    Experience (text from Llama 3 8B)
        -> MiniLM embedding [384]           (frozen sensory system)
        -> Linear projection [384 -> 64]    (LEARNED: skip-gram insight)
        -> Concept space [64]               (where concepts live)
        -> GRU cell [64 -> 64]              (self-state: Elman insight)
        -> Hidden state [64]                (THE SELF)
        -> Concept classifier [64 -> N]     (maps state to concept probs)

    Learning signal: contrastive loss (online, per-experience).
    Parameters: ~25K (projection + GRU + classifier).

Run locally on Mac M3. All models cached locally.

    pip install mlx-lm sentence-transformers torch
    python concept_learner.py                      # first session
    python concept_learner.py                      # resumes from saved state
    python concept_learner.py --steps 100          # longer session
    python concept_learner.py --reset              # start fresh

State persists to: ./results/concept_learner/dream_state.json
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── Configuration ──────────────────────────────────────────────────

DEFAULT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_STEPS = 30
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
NOVELTY_THRESHOLD = 0.25      # similarity below this = novel experience
FEATURE_TOP_K = 8             # top surprise tokens extracted as features

# v1 Neural Architecture
CONCEPT_DIM = 64              # learned concept space dimensionality
LEARNING_RATE = 0.005         # online learning rate
CONTRASTIVE_MARGIN = 0.3      # margin for contrastive loss

# v2 Anti-Collapse (self-regulation)
MAX_HIDDEN_NORM = 2.0         # clamp GRU hidden state norm (neural homeostasis)
HABITUATION_THRESHOLD = 2     # consecutive same-concept activations before decay
HABITUATION_DECAY = 0.5       # multiply score by this when habituated
ENTROPY_WEIGHT = 0.5          # weight for entropy regularization loss
DIVERSITY_WEIGHT = 0.2        # weight for anchor diversity loss
EXPLORATION_BONUS = 0.3       # score bonus for concepts with 0 activations

# v4 Passive Perception
PASSIVE_OBSERVE_TOKENS = 120  # tokens per passive observation chunk
SURPRISE_ACCUMULATOR_WINDOW = 5  # rolling window for surprise accumulation
WAKEUP_THRESHOLD = 1.5        # mean surprise over window to trigger active mode
PASSIVE_RATIO = 0.6           # fraction of steps that are passive (vs active)
OBSERVE_SEEDS = [             # seeds for free generation (diverse starting points)
    "In the beginning",
    "The nature of",
    "When the sun",
    "All living things",
    "Deep in the forest",
    "The river flows",
    "Birds fly because",
    "Fire is",
    "What makes something alive",
    "The world is made of",
    # v5: More diverse seeds to push beyond seed concepts
    "Mathematics is the language of",
    "The city was built on",
    "Music can make people",
    "The ocean is deeper than",
    "Time passes differently when",
    "Love is not the same as",
    "Gravity pulls everything",
    "Language gives humans the ability to",
    "Stars are born when",
    "Memory is how we",
]

# v5 Neurogenesis
GROWTH_SUSTAINED_WINDOW = 3       # (legacy, unused in v5.1)
CONSOLIDATION_INTERVAL = 10       # consolidate (replay) every N passive steps
CONSOLIDATION_REPLAY_K = 5        # replay this many recent experiences during consolidation
MAX_CONCEPTS = 20                 # cap: do not grow beyond this many concepts

# v5.1 Feature-Frequency Growth
FEATURE_PROMOTE_THRESHOLD = 4     # feature must appear in this many experiences to promote
FEATURE_CONCEPT_SPREAD = 3        # feature must appear across this many DIFFERENT concepts
FEATURE_PROMOTE_CHECK_INTERVAL = 8  # check for promotable features every N steps
FEATURE_MIN_LENGTH = 3            # min token length for promotion (isalpha filters artifacts)

# Common stopwords to filter from features
# ONLY true grammatical function words. Content words like "own", "world",
# "life" are NOT filtered — the child decides what matters through frequency.
# v5.1 insight: a child asks "what is own?" and learns about possession.
STOPWORDS = {
    "the", "and", "but", "for", "are", "was", "were", "been", "being",
    "have", "has", "had", "not", "you", "your", "they", "them", "this",
    "that", "with", "from", "will", "would", "could", "should", "what",
    "which", "where", "when", "who", "how", "can", "its", "all", "also",
    "about", "more", "some", "very", "just", "even", "here", "there",
    "than", "then", "each", "only", "into", "over", "such", "most",
    "other", "because", "does", "did", "our", "their",
    "many", "these", "those",
}
RECALL_TOP_K = 3              # memories recalled per step
STATE_DIR = Path(__file__).parent.parent / "results" / "feedback"
WEIGHTS_PATH = STATE_DIR / "concept_net.pt"

# Ablation modes
ABLATION_MODES = ["random", "template", "structure", "feedback"]
# Global mode — set at startup from --mode arg
_ABLATION_MODE = "feedback"  # default = full v6

# Seed concepts — intentionally incomplete, to be refined through learning
SEED_CONCEPTS = {
    "bird":   {"wings", "flies", "feathers", "alive", "sky"},
    "fire":   {"hot", "burns", "bright", "light", "red"},
    "water":  {"wet", "flows", "liquid", "cold", "river"},
    "metal":  {"hard", "shiny", "heavy", "strong", "iron"},
    "animal": {"alive", "moves", "eats", "breathes", "legs"},
    "tree":   {"tall", "leaves", "wood", "roots", "grows"},
}

# Prompt templates for different exploration modes
EXPLORE_TEMPLATES = [
    "Tell me about {concept}.",
    "What is {feature}?",
    "Describe {concept} in detail.",
    "What do you know about {feature}?",
]

ASSOCIATE_TEMPLATES = [
    "Is {concept_a} related to {concept_b}?",
    "What do {concept_a} and {concept_b} have in common?",
    "Can {concept_a} be like {concept_b}?",
]

CREATIVE_TEMPLATES = [
    "Can {concept_a} {feature_from_b}?",
    "What if {concept_a} could {feature_from_b}?",
    "Is there something that is both {concept_a} and {concept_b}?",
]

FOLLOWUP_TEMPLATE = "Tell me more about {tokens}."


# ── Data Structures ────────────────────────────────────────────────

@dataclass
class Concept:
    name: str
    features: dict  # feature_name → confidence (float 0-1)
    experience_ids: list = field(default_factory=list)
    created_at_step: int = 0
    times_activated: int = 0

    def top_features(self, n=5):
        sorted_f = sorted(self.features.items(), key=lambda x: -x[1])
        return [f for f, _ in sorted_f[:n]]


@dataclass
class Experience:
    id: int
    step: int
    session: int
    prompt: str
    response: str
    mean_surprise: float
    max_surprise: float
    top_surprise_tokens: list  # [{token, surprisal}]
    concept_labels: list       # concept names assigned
    features_extracted: list   # raw feature strings
    novelty_score: float       # how poorly it fit existing concepts
    embedding: list = field(default_factory=list)  # response embedding
    timestamp: str = ""

    def feature_set(self):
        return set(self.features_extracted)


@dataclass
class DreamState:
    session_count: int = 0
    total_steps: int = 0
    concepts: dict = field(default_factory=dict)       # name → Concept
    experiences: list = field(default_factory=list)     # Experience list
    self_summary: str = "I have just been born. I know nothing yet."
    exploration_log: list = field(default_factory=list) # recent topics
    concept_graph_edges: list = field(default_factory=list)  # (c1, c2, reason)
    # v6: Question strategy learning (persists across sessions)
    strategy_weights: dict = field(default_factory=lambda: {
        'feature_gap': 1.0, 'shared_feat': 1.0, 'connection': 1.0,
        'unvisited': 1.0, 'combine': 1.0,
    })
    strategy_scores: dict = field(default_factory=lambda: {
        'feature_gap': [], 'shared_feat': [], 'connection': [],
        'unvisited': [], 'combine': [],
    })

    def save(self, path):
        data = {
            "session_count": self.session_count,
            "total_steps": self.total_steps,
            "self_summary": self.self_summary,
            "exploration_log": self.exploration_log[-100:],
            "concept_graph_edges": self.concept_graph_edges,
            # v6: persist strategy learning
            "strategy_weights": self.strategy_weights,
            "strategy_scores": {k: v[-50:] for k, v in
                                self.strategy_scores.items()},
            "concepts": {
                name: {
                    "name": c.name,
                    "features": c.features,
                    "experience_ids": c.experience_ids[-50:],
                    "created_at_step": c.created_at_step,
                    "times_activated": c.times_activated,
                }
                for name, c in self.concepts.items()
            },
            "experiences": [
                {
                    "id": e.id,
                    "step": e.step,
                    "session": e.session,
                    "prompt": e.prompt,
                    "response": e.response[:500],
                    "mean_surprise": e.mean_surprise,
                    "max_surprise": e.max_surprise,
                    "top_surprise_tokens": e.top_surprise_tokens,
                    "concept_labels": e.concept_labels,
                    "features_extracted": e.features_extracted,
                    "novelty_score": e.novelty_score,
                    "timestamp": e.timestamp,
                }
                for e in self.experiences[-500:]
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        state = cls()
        state.session_count = data["session_count"]
        state.total_steps = data["total_steps"]
        state.self_summary = data["self_summary"]
        state.exploration_log = data.get("exploration_log", [])
        state.concept_graph_edges = data.get("concept_graph_edges", [])
        # v6: restore strategy learning
        if "strategy_weights" in data:
            state.strategy_weights = data["strategy_weights"]
        if "strategy_scores" in data:
            state.strategy_scores = data["strategy_scores"]
        for name, cdata in data["concepts"].items():
            state.concepts[name] = Concept(
                name=cdata["name"],
                features=cdata["features"],
                experience_ids=cdata.get("experience_ids", []),
                created_at_step=cdata.get("created_at_step", 0),
                times_activated=cdata.get("times_activated", 0),
            )
        for edata in data.get("experiences", []):
            state.experiences.append(Experience(
                id=edata["id"],
                step=edata["step"],
                session=edata["session"],
                prompt=edata["prompt"],
                response=edata["response"],
                mean_surprise=edata["mean_surprise"],
                max_surprise=edata["max_surprise"],
                top_surprise_tokens=edata["top_surprise_tokens"],
                concept_labels=edata["concept_labels"],
                features_extracted=edata["features_extracted"],
                novelty_score=edata.get("novelty_score", 0.0),
                timestamp=edata.get("timestamp", ""),
            ))
        return state


# ── Environment (Frozen Language Model via MLX) ───────────────────

class Environment:
    """Wraps an MLX language model as an explorable world.

    Uses Llama 3 8B (4-bit) for generation and surprise measurement.
    Uses sentence-transformers (MiniLM) for response embeddings.
    """

    def __init__(self, model_name: str):
        import mlx_lm
        from sentence_transformers import SentenceTransformer

        print(f"  Loading environment: {model_name}")
        self.model, self.tokenizer = mlx_lm.load(model_name)
        self.model_name = model_name
        self._mlx_lm = mlx_lm

        # Load embedding model (cached locally)
        print(f"  Loading embedder: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        print(f"  Environment ready: {model_name}")
        print(f"  Embedding dim: {self.embedding_dim}")

    def query(self, prompt: str):
        """Send a prompt to the environment and get response + surprise.

        Returns:
            response_text: generated text
            surprisals: list of {token, surprisal} for each generated token
            mean_surprise: mean surprisal across generated tokens
            max_surprise: max surprisal
            response_embedding: sentence embedding of response [dim]
        """
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        # Stream generate to collect per-token logprobs
        surprisals = []
        response_tokens = []
        sampler = make_sampler(temp=TEMPERATURE)

        for response in self._mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=MAX_NEW_TOKENS,
            sampler=sampler,
        ):
            token_id = response.token
            token_text = response.text.strip() if response.text else ""
            logprobs = response.logprobs

            # Compute surprisal from logprobs
            if logprobs is not None:
                # logprobs is a vector of log probabilities
                log_prob_array = np.array(logprobs, copy=False)
                token_log_prob = float(log_prob_array[token_id])
                surprisal = -token_log_prob

                if token_text and len(token_text) > 0:
                    surprisals.append({
                        "token": token_text,
                        "surprisal": round(surprisal, 4),
                    })

            response_tokens.append(response.text or "")

            if response.finish_reason:
                break

        response_text = "".join(response_tokens).strip()

        # Compute response embedding via sentence-transformers
        if response_text:
            emb_np = self.embedder.encode(response_text, convert_to_numpy=True)
            embedding = torch.from_numpy(emb_np).float()
        else:
            embedding = torch.zeros(self.embedding_dim)

        mean_s = sum(s["surprisal"] for s in surprisals) / max(len(surprisals), 1)
        max_s = max((s["surprisal"] for s in surprisals), default=0.0)

        return response_text, surprisals, mean_s, max_s, embedding

    def observe(self, seed: str = None):
        """v4: Passive observation. The world speaks, the child listens.

        Generates free text from a seed without an explicit question.
        Returns the same tuple as query() for uniform processing.
        This is Layer 0: peripheral perception, always on.
        """
        if seed is None:
            seed = random.choice(OBSERVE_SEEDS)

        # Use a minimal prompt to get free-form generation
        return self.query(seed)


# ── ConceptNet (v1 Neural Architecture) ────────────────────────────

class ConceptNet(nn.Module):
    """The learned outer layer brain.

    Architecture:
        MiniLM embedding [384] → projection [64] → GRU [64] → classifier [N]

    The projection compresses sensory input into concept space.
    The GRU maintains a self-state across sequential experiences.
    The classifier maps the self-state to concept probabilities.
    """

    def __init__(self, input_dim: int, concept_dim: int, num_concepts: int):
        super().__init__()
        self.concept_dim = concept_dim

        # Projection: MiniLM space → concept space (skip-gram insight)
        self.projection = nn.Linear(input_dim, concept_dim)

        # Self-state: GRU maintains hidden state across experiences (Elman)
        self.gru = nn.GRUCell(concept_dim, concept_dim)

        # Classifier: maps self-state → concept probabilities
        self.classifier = nn.Linear(concept_dim, num_concepts)

        # Concept anchors: learned prototype vectors in concept space
        self.concept_anchors = nn.Parameter(
            torch.randn(num_concepts, concept_dim) * 0.1
        )

        # Persistent hidden state (THE SELF)
        self.register_buffer(
            "hidden_state", torch.zeros(1, concept_dim)
        )

        self._num_concepts = num_concepts

    def forward(self, embedding: torch.Tensor):
        """Process one experience through the outer layer.

        Args:
            embedding: MiniLM sentence embedding [384]

        Returns:
            concept_logits: [num_concepts] — classification scores
            projected: [concept_dim] — position in concept space
            new_hidden: [concept_dim] — updated self-state
        """
        # Project into concept space
        projected = self.projection(embedding.unsqueeze(0))  # [1, 64]

        # Update self-state through GRU
        new_hidden = self.gru(projected, self.hidden_state)  # [1, 64]

        # v2: Neural homeostasis — clamp hidden state norm
        h_norm = new_hidden.norm(dim=-1, keepdim=True)
        if h_norm.item() > MAX_HIDDEN_NORM:
            new_hidden = new_hidden * (MAX_HIDDEN_NORM / h_norm)

        self.hidden_state = new_hidden.detach()  # detach to prevent BPTT

        # Classify using updated state
        concept_logits = self.classifier(new_hidden)  # [1, N]

        return concept_logits.squeeze(0), projected.squeeze(0), new_hidden.squeeze(0)

    def concept_similarity(self, projected: torch.Tensor):
        """Compute similarity between projected experience and concept anchors.

        Returns:
            similarities: [num_concepts] cosine similarities
        """
        return F.cosine_similarity(
            projected.unsqueeze(0),  # [1, 64]
            self.concept_anchors,    # [N, 64]
        )

    def add_concept(self, initial_embedding: torch.Tensor = None):
        """Grow the network to accommodate a new concept.

        Extends classifier and concept_anchors by one.
        """
        old_n = self._num_concepts
        new_n = old_n + 1

        # Extend classifier
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        self.classifier = nn.Linear(self.concept_dim, new_n)
        self.classifier.weight.data[:old_n] = old_weight
        self.classifier.bias.data[:old_n] = old_bias
        nn.init.xavier_uniform_(self.classifier.weight.data[old_n:old_n+1])
        self.classifier.bias.data[old_n] = 0.0

        # Extend concept anchors
        old_anchors = self.concept_anchors.data
        if initial_embedding is not None:
            # Use the projected embedding as the initial anchor
            new_anchor = initial_embedding.detach().unsqueeze(0)
        else:
            new_anchor = torch.randn(1, self.concept_dim) * 0.1
        new_anchors = torch.cat([old_anchors, new_anchor], dim=0)
        self.concept_anchors = nn.Parameter(new_anchors)

        self._num_concepts = new_n
        return old_n  # index of new concept

    def get_self_vector(self):
        """Return the current self-state as a 1D vector."""
        return self.hidden_state.squeeze(0).detach()

    def save_weights(self, path: Path):
        """Save network weights and hidden state."""
        torch.save({
            "model_state": self.state_dict(),
            "num_concepts": self._num_concepts,
        }, path)

    @classmethod
    def load_weights(cls, path: Path, input_dim: int, concept_dim: int):
        """Load network from saved weights."""
        checkpoint = torch.load(path, weights_only=False)
        num_concepts = checkpoint["num_concepts"]
        net = cls(input_dim, concept_dim, num_concepts)
        net.load_state_dict(checkpoint["model_state"])
        return net


# ── Outer Layer (The Self) ─────────────────────────────────────────

class OuterLayer:
    """Persistent concept learner with self-referential loop (v2).

    v1: Uses ConceptNet (projection + GRU) for learned representations.
    v2: Adds self-regulation (entropy, habituation, diversity, norm clamping).
    The network's hidden state IS the self.
    """

    def __init__(self, state: DreamState, embedding_dim: int, embedder=None):
        self.state = state
        self.embedding_dim = embedding_dim
        self.embedder = embedder

        # Concept embeddings (v0 compatibility + seed initialization)
        self.concept_embeddings = {}  # name -> tensor

        # v1: concept name <-> index mapping
        self.concept_names = list(state.concepts.keys()) if state.concepts else []
        self.concept_to_idx = {n: i for i, n in enumerate(self.concept_names)}

        # v1: Neural network
        num_concepts = max(len(self.concept_names), 1)
        if WEIGHTS_PATH.exists() and not getattr(state, '_reset', False):
            print(f"  Loading ConceptNet from {WEIGHTS_PATH}")
            self.net = ConceptNet.load_weights(
                WEIGHTS_PATH, embedding_dim, CONCEPT_DIM
            )
        else:
            self.net = ConceptNet(embedding_dim, CONCEPT_DIM, num_concepts)
            print(f"  Initialized fresh ConceptNet "
                  f"({sum(p.numel() for p in self.net.parameters()):,} params)")

        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.training_losses = []  # track loss per step

        # v2: Habituation state (boredom mechanism)
        self._last_concept = None
        self._consecutive_count = 0

        # v5: Sustained novelty tracker (for neurogenesis)
        self._high_novelty_streak = 0
        self._pending_novel_embeddings = []  # embeddings that could not be classified

        # v5.1: Feature frequency tracker
        # Maps feature_name -> list of (concept_name, experience_id, embedding)
        self._feature_occurrences = defaultdict(list)

        # v6: Strategy learning (references state for persistence)
        self._strategy_weights = state.strategy_weights
        self._strategy_scores = state.strategy_scores

    def initialize_seeds(self):
        """Bootstrap with seed concepts (only on first session)."""
        if self.state.concepts:
            return  # already initialized

        for name, features in SEED_CONCEPTS.items():
            self.state.concepts[name] = Concept(
                name=name,
                features={f: 0.5 for f in features},
                created_at_step=0,
            )

            # Create initial embedding from concept description
            if self.embedder is not None:
                desc = f"{name}: {', '.join(features)}"
                emb_np = self.embedder.encode(desc, convert_to_numpy=True)
                self.concept_embeddings[name] = torch.from_numpy(emb_np).float()

        # Build concept index
        self.concept_names = list(self.state.concepts.keys())
        self.concept_to_idx = {n: i for i, n in enumerate(self.concept_names)}

        # Reinitialize network with correct number of seed concepts
        num_seeds = len(self.concept_names)
        self.net = ConceptNet(self.embedding_dim, CONCEPT_DIM, num_seeds)

        # Initialize concept anchors from seed embeddings
        if self.concept_embeddings:
            with torch.no_grad():
                for name, emb in self.concept_embeddings.items():
                    idx = self.concept_to_idx[name]
                    projected = self.net.projection(emb.unsqueeze(0))
                    self.net.concept_anchors.data[idx] = projected.squeeze(0)

        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        print(f"  Seeded {num_seeds} concepts: "
              f"{', '.join(self.concept_names)}")
        print(f"  ConceptNet: {sum(p.numel() for p in self.net.parameters()):,} params, "
              f"concept_dim={CONCEPT_DIM}")
        if self.concept_embeddings:
            print(f"  Anchors initialized from seed embeddings")

    def extract_features(self, surprisals: list) -> list:
        """Extract features from surprise tokens.

        Features = the tokens themselves, filtered for content words.
        High-surprise tokens are edge features.
        Low-surprise tokens are established associations.
        """
        # Filter to meaningful content tokens
        meaningful = [
            s for s in surprisals
            if len(s["token"]) >= 3
            and s["token"].isalpha()
            and s["token"].lower() not in STOPWORDS
        ]

        # Take top-K by surprise as primary features
        sorted_by_surprise = sorted(meaningful, key=lambda x: -x["surprisal"])
        features = [s["token"].lower() for s in sorted_by_surprise[:FEATURE_TOP_K]]

        # Also include some low-surprise tokens (established knowledge)
        sorted_by_familiarity = sorted(meaningful, key=lambda x: x["surprisal"])
        familiar = [s["token"].lower() for s in sorted_by_familiarity[:3]]

        return list(set(features + familiar))

    def classify(self, features: list, embedding: torch.Tensor):
        """Classify using the neural network (v2: with habituation).

        Returns:
            best_concept: name of best matching concept (or None)
            similarity: similarity score to best concept
            novelty: 1 - similarity
            all_scores: dict of concept -> similarity
        """
        if not self.concept_names:
            return None, 0.0, 1.0, {}

        # Forward pass through ConceptNet
        logits, projected, hidden = self.net(embedding)

        # Compute similarity to concept anchors in learned space
        anchor_sims = self.net.concept_similarity(projected)

        # Build score dict
        scores = {}
        for i, name in enumerate(self.concept_names):
            if i < len(anchor_sims):
                scores[name] = anchor_sims[i].item()

        if not scores:
            return None, 0.0, 1.0, {}

        # v2: Habituation — decay scores for over-activated concepts
        if (self._last_concept is not None
                and self._consecutive_count >= HABITUATION_THRESHOLD):
            habituated = self._last_concept
            if habituated in scores:
                scores[habituated] *= HABITUATION_DECAY

        # v2: Exploration bonus — boost unvisited concepts
        for name in scores:
            if name in self.state.concepts:
                if self.state.concepts[name].times_activated == 0:
                    scores[name] += EXPLORATION_BONUS

        best = max(scores, key=scores.get)
        similarity = scores[best]
        novelty = 1.0 - max(similarity, 0.0)

        # v2: Update habituation counter
        if best == self._last_concept:
            self._consecutive_count += 1
        else:
            self._last_concept = best
            self._consecutive_count = 1

        # Store projected embedding for learning
        self._last_projected = projected
        self._last_logits = logits

        return best, similarity, novelty, scores

    def learn_from_experience(self, concept_name: str, embedding: torch.Tensor,
                              is_novel: bool):
        """Online learning step (v2: entropy + diversity).

        Loss = CE + 0.5*contrastive + 0.3*entropy + 0.2*diversity

        Entropy reg: penalizes confident classifications (attention spreading).
        Diversity: pushes concept anchors apart (lateral inhibition).
        """
        if concept_name not in self.concept_to_idx:
            return 0.0

        self.optimizer.zero_grad()

        # Forward pass (fresh, with gradients)
        logits, projected, hidden = self.net(embedding)

        target_idx = self.concept_to_idx[concept_name]

        # 1. Classification loss (cross-entropy)
        target = torch.tensor([target_idx])
        ce_loss = F.cross_entropy(logits.unsqueeze(0), target)

        # 2. Contrastive loss on concept anchors
        anchor = self.net.concept_anchors[target_idx]
        pos_sim = F.cosine_similarity(projected.unsqueeze(0), anchor.unsqueeze(0))
        contrastive_loss = (1.0 - pos_sim).mean()

        # 3. v2: Entropy regularization (attention spreading)
        # High entropy = spread attention. Penalize low entropy.
        probs = F.softmax(logits, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(len(self.concept_names))))
        entropy_loss = 1.0 - (entropy / max_entropy)  # 0 when uniform, 1 when certain

        # 4. v2: Anchor diversity loss (lateral inhibition)
        # Push all anchors apart from each other
        anchors = self.net.concept_anchors
        if anchors.shape[0] >= 2:
            anchor_sims = F.cosine_similarity(
                anchors.unsqueeze(0),  # [1, N, D]
                anchors.unsqueeze(1),  # [N, 1, D]
                dim=2,
            )  # [N, N]
            # Mask diagonal (self-similarity = 1.0, not interesting)
            mask = 1.0 - torch.eye(anchors.shape[0])
            diversity_loss = (anchor_sims * mask).mean()
        else:
            diversity_loss = torch.tensor(0.0)

        # Total loss
        loss = (ce_loss
                + 0.5 * contrastive_loss
                + ENTROPY_WEIGHT * entropy_loss
                + DIVERSITY_WEIGHT * diversity_loss)

        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def update_concept(self, concept_name: str, features: list,
                       embedding: torch.Tensor, experience_id: int):
        """Update an existing concept with new evidence (v0 + v1 learning)."""
        concept = self.state.concepts[concept_name]
        concept.times_activated += 1
        concept.experience_ids.append(experience_id)

        # v0: Update feature confidences
        for feat in features:
            if feat in concept.features:
                concept.features[feat] = min(concept.features[feat] + 0.1, 1.0)
            else:
                concept.features[feat] = 0.2

        # v0: Update concept embedding (running average)
        if concept_name in self.concept_embeddings:
            n = concept.times_activated
            old_emb = self.concept_embeddings[concept_name]
            self.concept_embeddings[concept_name] = (
                old_emb * (n - 1) / n + embedding / n
            )
        else:
            self.concept_embeddings[concept_name] = embedding.clone()

        # v1: Online learning step
        loss = self.learn_from_experience(concept_name, embedding, is_novel=False)
        return loss

    def create_concept(self, features: list, embedding: torch.Tensor,
                       experience_id: int, response: str,
                       preferred_name: str = None) -> str:
        """Create a new concept from a novel experience (v0 + v1 network growth).

        v6: Uses preferred_name if provided (from feature promotion),
        otherwise picks the most distinguishing single-word feature.
        """
        if preferred_name and preferred_name not in self.state.concepts:
            name = preferred_name
        else:
            # Pick the best single-word feature as name
            good_names = [f for f in features
                          if len(f) >= 3 and f not in STOPWORDS
                          and f not in self.state.concepts]
            name = good_names[0] if good_names else \
                   f"concept_{len(self.state.concepts)}"

        if name in self.state.concepts:
            name = f"{name}_{self.state.total_steps}"

        self.state.concepts[name] = Concept(
            name=name,
            features={f: 0.5 for f in features},
            experience_ids=[experience_id],
            created_at_step=self.state.total_steps,
            times_activated=1,
        )
        self.concept_embeddings[name] = embedding.clone()

        # v1: Grow the network
        with torch.no_grad():
            projected = self.net.projection(embedding.unsqueeze(0)).squeeze(0)
        new_idx = self.net.add_concept(initial_embedding=projected)

        # Update concept index
        self.concept_names.append(name)
        self.concept_to_idx[name] = new_idx

        # Rebuild optimizer to include new parameters
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        return name

    def maybe_grow(self, novelty: float, embedding: torch.Tensor,
                   features: list, response: str,
                   concept_label: str = None) -> str:
        """v5.1: Check if a feature should be promoted to a concept.

        Growth trigger: a feature appears frequently AND across multiple
        different existing concepts. This means the feature is a cross-cutting
        pattern that deserves its own representation.

        Example: 'fly' appears in bird (3x), insect (2x), aircraft (1x)
        → 'fly' is promoted to a concept because it spans 3 different concepts.

        The child sees a bird fly. What is flying? The child sees a plane fly.
        What is flying? Eventually 'flying' becomes its own thing.

        Returns: new concept name if grown, None otherwise.
        """
        if len(self.concept_names) >= MAX_CONCEPTS:
            return None

        # Track every feature's occurrence with its concept context
        for feat in features:
            # Skip features that are already concept names
            if feat in self.state.concepts:
                continue
            self._feature_occurrences[feat].append((
                concept_label or "unknown",
                len(self.state.experiences),
                embedding.clone(),
            ))

        # Only check for promotion periodically (not every step)
        if self.state.total_steps % FEATURE_PROMOTE_CHECK_INTERVAL != 0:
            return None

        # Find features ready for promotion
        best_candidate = None
        best_score = 0

        for feat, occurrences in self._feature_occurrences.items():
            if feat in self.state.concepts:
                continue  # already a concept
            if len(feat) < FEATURE_MIN_LENGTH:
                continue  # too short to be meaningful

            # Count total occurrences
            total_count = len(occurrences)
            if total_count < FEATURE_PROMOTE_THRESHOLD:
                continue

            # Count unique concepts this feature appeared across
            unique_concepts = len(set(c for c, _, _ in occurrences))
            if unique_concepts < FEATURE_CONCEPT_SPREAD:
                continue

            # Score: total count * concept spread
            score = total_count * unique_concepts
            if score > best_score:
                best_score = score
                best_candidate = feat

        if best_candidate is None:
            return None

        # Promote the best candidate feature to a concept
        occurrences = self._feature_occurrences[best_candidate]
        # Average embedding from all experiences containing this feature
        avg_emb = torch.stack([e[2] for e in occurrences]).mean(dim=0)

        # The new concept's initial features come from co-occurring features
        co_features = Counter()
        for exp in self.state.experiences:
            if best_candidate in exp.features_extracted:
                for f in exp.features_extracted:
                    if f != best_candidate:
                        co_features[f] += 1
        top_co = [f for f, _ in co_features.most_common(5)]
        initial_features = [best_candidate] + top_co[:4]

        new_name = self.create_concept(
            initial_features, avg_emb,
            len(self.state.experiences), response,
            preferred_name=best_candidate,  # v6: use the feature as name
        )

        # Clear this feature from the tracker (it's now a concept)
        del self._feature_occurrences[best_candidate]

        return new_name

    def consolidate(self):
        """v5: Consolidation phase — the child sleeps and replays.

        Replays recent experiences with gradient updates to strengthen
        concept anchors. Like sleep consolidation in the brain.
        """
        if len(self.state.experiences) < CONSOLIDATION_REPLAY_K:
            return 0.0

        recent = self.state.experiences[-CONSOLIDATION_REPLAY_K:]
        total_loss = 0.0
        replayed = 0

        for exp in recent:
            if not exp.concept_labels:
                continue
            concept_name = exp.concept_labels[0]
            if concept_name not in self.concept_to_idx:
                continue
            if concept_name not in self.concept_embeddings:
                continue

            # Replay: run a learning step on this past experience
            emb = self.concept_embeddings[concept_name]
            loss = self.learn_from_experience(concept_name, emb, is_novel=False)
            total_loss += loss
            replayed += 1

        avg_loss = total_loss / max(replayed, 1)
        return avg_loss

    def recall(self, features: list) -> list:
        """Recall past experiences with overlapping features."""
        feature_set = set(features)
        scores = Counter()

        for exp in self.state.experiences:
            overlap = len(feature_set & set(exp.features_extracted))
            if overlap > 0:
                scores[exp.id] = overlap

        top_ids = [eid for eid, _ in scores.most_common(RECALL_TOP_K)]
        return [e for e in self.state.experiences if e.id in top_ids]

    def detect_connections(self, features: list, concept_labels: list):
        """Detect co-activation between concepts (connection mechanism A)."""
        if len(concept_labels) < 2:
            return

        for i in range(len(concept_labels)):
            for j in range(i + 1, len(concept_labels)):
                c1, c2 = concept_labels[i], concept_labels[j]
                shared = set(features) & set(self.state.concepts[c1].features.keys()) & \
                         set(self.state.concepts[c2].features.keys())
                if shared:
                    edge = (c1, c2, list(shared)[:3])
                    if edge not in self.state.concept_graph_edges:
                        self.state.concept_graph_edges.append(edge)
                        print(f"    💡 Connection: {c1} ↔ {c2} via {shared}")

    def _get_content_tokens(self, experience) -> list:
        """Get clean content tokens from an experience's surprise tokens."""
        return [
            t["token"] for t in experience.top_surprise_tokens
            if len(t["token"]) >= 3
            and t["token"].isalpha()
            and t["token"].lower() not in STOPWORDS
        ]

    def choose_prompt_random(self, env=None):
        """Ablation: Random mode — no concept structure, just random prompts."""
        topics = [
            "Tell me about something interesting.",
            "What is the meaning of life?",
            "Describe the universe.",
            "What happens when things change?",
            "Why do things exist?",
            "Tell me a story.",
            "What is the most important thing?",
            "Explain how the world works.",
            "What is knowledge?",
            "Describe something beautiful.",
            "What is the nature of reality?",
            "How do things grow?",
            "What makes something alive?",
            "Tell me about movement.",
            "What is time?",
        ]
        return random.choice(topics), "random"

    def choose_prompt_template(self, env=None):
        """Ablation: Template mode — fixed templates using concepts, equal weights."""
        concepts = self.state.concepts
        names = list(concepts.keys())
        if not names:
            return "What exists?", "bootstrap"

        # Pick a random template type
        roll = random.random()
        if roll < 0.4 and len(names) >= 1:
            # Explore template
            c = random.choice(names)
            templates = EXPLORE_TEMPLATES
            feats = list(concepts[c].features.keys())
            feat = random.choice(feats) if feats else c
            tmpl = random.choice(templates)
            return tmpl.format(concept=c, feature=feat), "template_explore"
        elif roll < 0.7 and len(names) >= 2:
            # Associate template
            a, b = random.sample(names, 2)
            tmpl = random.choice(ASSOCIATE_TEMPLATES)
            return tmpl.format(concept_a=a, concept_b=b), "template_associate"
        elif len(names) >= 2:
            # Creative template
            a, b = random.sample(names, 2)
            b_feats = list(concepts[b].features.keys())
            fb = random.choice(b_feats) if b_feats else b
            tmpl = random.choice(CREATIVE_TEMPLATES)
            return tmpl.format(concept_a=a, concept_b=b, feature_from_b=fb), "template_creative"
        else:
            c = random.choice(names)
            return f"Tell me about {c}.", "template_fallback"

    def choose_prompt(self, env=None):
        """v6: Structure-driven questions with feedback-weighted strategy.

        The child's OWN learned structure decides WHAT to ask about:
        - Feature gaps: bird has 'fly', metal does not -> ask about metal+fly
        - Shared features: 'gravity' appears in bird AND metal -> ask why
        - Concept connections: bird<->animal via 'alive' -> explore the link
        - Unvisited concepts: pure curiosity

        The base model only ARTICULATES the question naturally.

        v6 addition: strategies are weighted by past success.
        Strategies that produce more learning get asked more often.

        In ablation mode 'structure', all weights are equal (no feedback).
        In ablation modes 'random' and 'template', this method is not called.

        Returns: (question_string, strategy_name) tuple.
        """
        # Ablation: delegate to mode-specific methods
        if _ABLATION_MODE == "random":
            return self.choose_prompt_random(env)
        elif _ABLATION_MODE == "template":
            return self.choose_prompt_template(env)

        concepts = self.state.concepts
        names = list(concepts.keys())

        if not names:
            return "What exists?", "bootstrap"

        # ── Strategy selection (weighted by past success) ─────────────
        # Weights are persisted in DreamState and initialized in __init__.

        # ── The child's structure identifies WHAT to ask ──────────────

        # Strategy 1: Feature gap (A has X, B does not -> can B do X?)
        seed = None
        # In 'structure' mode, use equal weights (no feedback learning)
        if _ABLATION_MODE == "structure":
            w = {s: 1.0 for s in self._strategy_weights}
        else:
            w = self._strategy_weights
        total_w = sum(w.values())

        # Weighted random strategy selection
        strategy_roll = random.random() * total_w
        cumulative = 0
        chosen_strategies = []
        for s, sw in w.items():
            cumulative += sw
            if cumulative >= strategy_roll and not chosen_strategies:
                chosen_strategies.append(s)

        # Try the chosen strategy first, then others as fallback
        if 'feature_gap' in chosen_strategies or (not chosen_strategies
                and len(names) >= 2 and random.random() < 0.35):
            a, b = random.sample(names, 2) if len(names) >= 2 else (None, None)
            a_feats = set(f for f in concepts[a].features
                         if f not in STOPWORDS and len(f) >= 3)
            b_feats = set(f for f in concepts[b].features
                         if f not in STOPWORDS and len(f) >= 3)
            gap = a_feats - b_feats
            if gap:
                feat = random.choice(list(gap))
                seed = f"feature_gap:{a}:{b}:{feat}"

        # Strategy 2: Shared feature across concepts (why does X do Y?)
        if seed is None and random.random() < 0.3:
            feature_owners = {}
            for name, concept in concepts.items():
                for feat in concept.features:
                    if feat not in STOPWORDS and len(feat) >= 3:
                        feature_owners.setdefault(feat, []).append(name)
            shared = {f: o for f, o in feature_owners.items()
                      if len(o) >= 2}
            if shared:
                feat = random.choice(list(shared.keys()))
                owners = shared[feat]
                seed = f"shared_feat:{feat}:{','.join(owners[:3])}"

        # Strategy 3: Explore a connection the child found
        if seed is None and self.state.concept_graph_edges \
                and random.random() < 0.25:
            c1, c2, via = random.choice(self.state.concept_graph_edges)
            seed = f"connection:{c1}:{c2}:{','.join(via)}"

        # Strategy 4: Unvisited concept (pure curiosity)
        if seed is None:
            unvisited = [n for n in names
                         if concepts[n].times_activated == 0]
            if unvisited and random.random() < 0.4:
                c = random.choice(unvisited)
                seed = f"unvisited:{c}"

        # Strategy 5: Combine two random concept features
        if seed is None and len(names) >= 2:
            a, b = random.sample(names, 2)
            a_feats = [f for f in concepts[a].features
                       if len(f) >= 3 and f not in STOPWORDS]
            b_feats = [f for f in concepts[b].features
                       if len(f) >= 3 and f not in STOPWORDS]
            if a_feats and b_feats:
                fa = random.choice(a_feats)
                fb = random.choice(b_feats)
                seed = f"combine:{a}:{fa}:{b}:{fb}"

        # ── The base model ARTICULATES the question ───────────────────
        if env is not None and seed is not None:
            # Convert structural seed into a natural question
            parts = seed.split(":")
            strategy = parts[0]

            if strategy == "feature_gap":
                _, concept_a, concept_b, feat = parts
                articulation_prompt = (
                    f"{concept_a} can {feat}. "
                    f"A curious child wonders about {concept_b} and asks:"
                )
            elif strategy == "shared_feat":
                _, feat, owners_str = parts
                owners = owners_str.split(",")
                articulation_prompt = (
                    f"Both {owners[0]} and {owners[1]} involve {feat}. "
                    f"A curious child asks:"
                )
            elif strategy == "connection":
                _, c1, c2, via_str = parts
                articulation_prompt = (
                    f"{c1} and {c2} are connected through {via_str}. "
                    f"A curious child asks:"
                )
            elif strategy == "unvisited":
                concept = parts[1]
                articulation_prompt = (
                    f"A curious child has never seen {concept} and asks:"
                )
            elif strategy == "combine":
                _, ca, fa, cb, fb = parts
                articulation_prompt = (
                    f"A child who knows that {ca} involves {fa} "
                    f"and {cb} involves {fb} asks:"
                )
            else:
                articulation_prompt = (
                    f"A curious child who knows about "
                    f"{', '.join(names[:6])} asks:"
                )

            try:
                response, _, _, _, _ = env.query(articulation_prompt)
                question = response.strip().split("\n")[0].strip()
                question = question.lstrip("#-*`| ")
                if not question.endswith("?"):
                    question = question.rstrip(".!") + "?"
                if len(question) > 100:
                    question = question[:100].rsplit(" ", 1)[0] + "?"
                words = [w for w in question.split() if w.isalpha()]
                if len(words) >= 3:
                    return question, strategy
            except Exception:
                pass

        # ── Fallback: direct structural question (no base model) ──────
        if seed:
            parts = seed.split(":")
            strategy = parts[0]
            if strategy == "feature_gap":
                return f"Can {parts[2]} {parts[3]}?", strategy
            elif strategy == "shared_feat":
                return f"What is {parts[1]}?", strategy
            elif strategy == "unvisited":
                return f"What is {parts[1]}?", strategy
            elif strategy == "combine":
                return f"What is {parts[2]} and {parts[4]}?", strategy

        c = random.choice(names)
        return f"Tell me about {c}.", "fallback"

    def update_self_summary(self):
        """Compress accumulated experience into a self-narrative."""
        n_concepts = len(self.state.concepts)
        n_experiences = len(self.state.experiences)
        n_sessions = self.state.session_count
        n_connections = len(self.state.concept_graph_edges)

        top_concepts = sorted(
            self.state.concepts.values(),
            key=lambda c: c.times_activated,
            reverse=True,
        )[:5]

        recent_novelties = [
            e for e in self.state.experiences[-20:]
            if e.novelty_score > 0.5
        ]

        self.state.self_summary = (
            f"I have explored {n_experiences} experiences across "
            f"{n_sessions} sessions. I know {n_concepts} concepts. "
            f"My most familiar concepts are: "
            f"{', '.join(c.name for c in top_concepts)}. "
            f"I have found {n_connections} connections between concepts. "
            f"In my recent exploration, I encountered "
            f"{len(recent_novelties)} novel experiences."
        )


# ── Dream Loop ─────────────────────────────────────────────────────

def _do_active_step(env, outer, session, session_log):
    """v6: Full conscious engagement with feedback loop.

    1. Snapshot concept state BEFORE asking
    2. Ask structure-driven question
    3. Learn from answer
    4. Measure what changed (information gain)
    5. Score the strategy that produced the question
    6. Update strategy weights so the child gets better at asking
    """
    state = outer.state
    state.total_steps += 1
    step = state.total_steps
    t0 = time.time()

    # ── v6: SNAPSHOT state before asking ──────────────────────────
    n_concepts_before = len(state.concepts)
    n_connections_before = len(state.concept_graph_edges)
    total_features_before = sum(
        len(c.features) for c in state.concepts.values()
    )

    # Choose what to explore (structure-driven, feedback-weighted)
    prompt, strategy = outer.choose_prompt(env=env)
    response, surprisals, mean_s, max_s, embedding = env.query(prompt)
    features = outer.extract_features(surprisals)
    best_concept, similarity, novelty, all_scores = outer.classify(
        features, embedding
    )

    # Full learning (gradient update)
    exp_id = len(state.experiences)
    concept_labels = []

    if novelty > (1 - NOVELTY_THRESHOLD):
        new_name = outer.create_concept(features, embedding, exp_id, response)
        concept_labels.append(new_name)
        step_loss = 0.0
        event = f"🆕 NEW CONCEPT: {new_name}"
    else:
        step_loss = outer.update_concept(best_concept, features, embedding, exp_id)
        concept_labels.append(best_concept)
        event = f"📎 {best_concept} (sim={similarity:.2f})"

        for cname, score in all_scores.items():
            if cname != best_concept and score > 0.3:
                concept_labels.append(cname)

    outer.detect_connections(features, concept_labels)
    recalled = outer.recall(features)

    # Store experience
    top_tokens = sorted(surprisals, key=lambda x: -x["surprisal"])[:5]
    experience = Experience(
        id=exp_id, step=step, session=session,
        prompt=prompt, response=response[:300],
        mean_surprise=round(mean_s, 4), max_surprise=round(max_s, 4),
        top_surprise_tokens=top_tokens, concept_labels=concept_labels,
        features_extracted=features, novelty_score=round(novelty, 4),
        timestamp=datetime.now().isoformat(),
    )
    state.experiences.append(experience)
    state.exploration_log.append(prompt[:60])

    # v5.1: Track features for growth (active steps contribute too)
    growth = outer.maybe_grow(novelty, embedding, features, response,
                              concept_label=best_concept)

    # ── v6: MEASURE learning delta (information gain) ────────────
    n_concepts_after = len(state.concepts)
    n_connections_after = len(state.concept_graph_edges)
    total_features_after = sum(
        len(c.features) for c in state.concepts.values()
    )

    # Information gain = what the child learned from this question
    new_concepts = n_concepts_after - n_concepts_before
    new_connections = n_connections_after - n_connections_before
    new_features = max(0, total_features_after - total_features_before)
    # Novelty bonus: answer was genuinely surprising
    surprise_bonus = max(0, mean_s - 0.5)  # above baseline surprise

    info_gain = (
        new_concepts * 3.0       # new concept = high value
        + new_connections * 2.0  # new connection = good
        + new_features * 0.5     # new features = some value
        + surprise_bonus         # surprise = found something new
        + len(features) * 0.1    # extracted features = some content
    )

    # ── v6: UPDATE strategy weights (only in feedback mode) ─────
    if _ABLATION_MODE == "feedback" and strategy in outer._strategy_scores:
        outer._strategy_scores[strategy].append(info_gain)
        # Recompute weights from rolling average (last 20 scores)
        for s, scores in outer._strategy_scores.items():
            recent = scores[-20:] if scores else []
            if recent:
                outer._strategy_weights[s] = max(0.2,
                    sum(recent) / len(recent) + 0.5)  # floor of 0.2
    elif strategy in outer._strategy_scores:
        # Still track scores for reporting, just don't update weights
        outer._strategy_scores[strategy].append(info_gain)

    elapsed = time.time() - t0
    feat_str = ", ".join(features[:4])
    recalled_str = f" | recalled {len(recalled)} memories" if recalled else ""
    loss_str = f" | L={step_loss:.3f}" if step_loss > 0 else ""
    gain_str = f" | IG={info_gain:.1f}" if info_gain > 0 else ""
    print(
        f"  Step {step:3d}: 🔴 ACTIVE S={mean_s:.3f} | {event} | "
        f"feats=[{feat_str}]{recalled_str}{loss_str}{gain_str} | "
        f"[{strategy}] | {elapsed:.1f}s"
    )
    print(f"           Q: {prompt[:70]}")
    print(f"           A: {response[:70]}...")

    session_log.append({
        "step": step, "mode": "active", "prompt": prompt,
        "mean_surprise": mean_s, "concept_labels": concept_labels,
        "novelty": novelty, "features": features,
        "event": event, "loss": step_loss if step_loss > 0 else None,
        "strategy": strategy, "info_gain": round(info_gain, 2),
    })
    return mean_s


def _do_passive_step(env, outer, session, session_log, seed=None):
    """Layer 0: Peripheral perception. Watch the world, no questions asked.

    Light processing: feature extraction + concept lookup, NO gradient update.
    The child's eyes are open but it is not asking anything.
    v5: Also tracks novelty for potential neurogenesis.
    """
    state = outer.state
    state.total_steps += 1
    step = state.total_steps
    t0 = time.time()

    # Observe the world (free generation)
    response, surprisals, mean_s, max_s, embedding = env.observe(seed)

    # Layer 0: feature extraction only
    features = outer.extract_features(surprisals)

    # Layer 1: concept lookup (NO gradient, just recognition)
    with torch.no_grad():
        best_concept, similarity, novelty, all_scores = outer.classify(
            features, embedding
        )

    # v5: Check if the architecture should grow
    growth_event = outer.maybe_grow(novelty, embedding, features, response,
                                     concept_label=best_concept)

    # Passive: store as experience but DO NOT update weights
    exp_id = len(state.experiences)
    concept_labels = [best_concept] if best_concept else []
    if growth_event:
        concept_labels = [growth_event]

    # Still detect connections (passive observation can notice patterns)
    outer.detect_connections(features, concept_labels)

    # Store experience (lighter: no gradient, but memory forms)
    top_tokens = sorted(surprisals, key=lambda x: -x["surprisal"])[:5]
    # Use the seed as the "prompt" for passive observations
    passive_prompt = seed or "[passive observation]"
    experience = Experience(
        id=exp_id, step=step, session=session,
        prompt=f"👁 {passive_prompt}", response=response[:300],
        mean_surprise=round(mean_s, 4), max_surprise=round(max_s, 4),
        top_surprise_tokens=top_tokens, concept_labels=concept_labels,
        features_extracted=features, novelty_score=round(novelty, 4),
        timestamp=datetime.now().isoformat(),
    )
    state.experiences.append(experience)
    state.exploration_log.append(f"👁 {passive_prompt[:50]}")

    elapsed = time.time() - t0
    feat_str = ", ".join(features[:4])
    if growth_event:
        concept_str = f"🌱 GREW: {growth_event}"
    else:
        concept_str = f"📎 {best_concept} (sim={similarity:.2f})" if best_concept else "❓ unrecognized"
    print(
        f"  Step {step:3d}: 👁 PASSIVE S={mean_s:.3f} | {concept_str} | "
        f"feats=[{feat_str}] | {elapsed:.1f}s"
    )
    print(f"           Saw: {response[:70]}...")

    session_log.append({
        "step": step, "mode": "passive", "prompt": passive_prompt,
        "mean_surprise": mean_s, "concept_labels": concept_labels,
        "novelty": novelty, "features": features,
        "event": concept_str, "loss": None,
        "growth": growth_event,
    })
    return mean_s


def run_session(env: Environment, outer: OuterLayer, num_steps: int):
    """v4: Two-phase dream session with passive perception.

    The child alternates between:
    - 👁 PASSIVE: watching the world (no questions, no gradient)
    - 🔴 ACTIVE: asking questions and learning (full gradient update)

    Surprise accumulation determines when to switch from passive to active.
    Like a child sitting in a room, watching, absorbing, and occasionally
    asking "what is THAT?" when something surprising happens.
    """
    state = outer.state
    state.session_count += 1
    session = state.session_count

    print(f"\n{'='*60}")
    print(f"  Session {session} | Steps {state.total_steps + 1}"
          f"-{state.total_steps + num_steps}")
    print(f"  Concepts: {len(state.concepts)} | "
          f"Experiences: {len(state.experiences)} | "
          f"Connections: {len(state.concept_graph_edges)}")
    mode_label = f"ABLATION={_ABLATION_MODE.upper()}" if _ABLATION_MODE != "feedback" else "v6 FEEDBACK"
    print(f"  Mode: {mode_label} (👁 passive / 🔴 active / 🌱 growth / 💤 consolidation)")
    print(f"  Self: {state.self_summary[:100]}...")
    print(f"{'='*60}\n")

    session_log = []
    surprise_buffer = []  # rolling window of recent surprise values
    passive_count = 0
    active_count = 0

    for step_in_session in range(num_steps):
        # Decide mode: passive or active?
        # Rule: accumulate surprise passively. When surprise exceeds threshold,
        # the child "wakes up" and asks a question.
        if step_in_session == 0:
            # First step is always passive — the child opens its eyes
            mode = "passive"
        elif len(surprise_buffer) >= SURPRISE_ACCUMULATOR_WINDOW:
            recent_mean = sum(surprise_buffer[-SURPRISE_ACCUMULATOR_WINDOW:]) / SURPRISE_ACCUMULATOR_WINDOW
            if recent_mean > WAKEUP_THRESHOLD:
                mode = "active"  # surprise accumulated — child interrupts
            else:
                # v5.1: Curiosity impulse — the child asks questions
                # even when calm, driven by knowledge not surprise.
                # More concepts = more curiosity = more questions.
                n_concepts = len(state.concepts)
                curiosity_chance = min(0.15, 0.03 * n_concepts)
                if random.random() < curiosity_chance:
                    mode = "active"  # calm curiosity — not startled, just thinking
                else:
                    mode = "passive"  # keep watching
        elif random.random() < PASSIVE_RATIO:
            mode = "passive"
        else:
            mode = "active"

        if mode == "passive":
            mean_s = _do_passive_step(env, outer, session, session_log)
            passive_count += 1

            # v5: Consolidation — replay recent experiences periodically
            if passive_count > 0 and passive_count % CONSOLIDATION_INTERVAL == 0:
                cons_loss = outer.consolidate()
                if cons_loss > 0:
                    print(f"           \U0001f4a4 CONSOLIDATION (replay {CONSOLIDATION_REPLAY_K} memories, loss={cons_loss:.3f})")
        else:
            mean_s = _do_active_step(env, outer, session, session_log)
            active_count += 1
            # After asking, reset surprise accumulator (curiosity satisfied)
            surprise_buffer.clear()

        surprise_buffer.append(mean_s)

    # Update self-summary at end of session
    outer.update_self_summary()

    # Save state
    state_path = STATE_DIR / "dream_state.json"
    state.save(state_path)

    # v1: Save network weights
    outer.net.save_weights(WEIGHTS_PATH)

    # Save session log
    log_path = STATE_DIR / f"session_{session:04d}.json"
    with open(log_path, "w") as f:
        json.dump(session_log, f, indent=2)

    # Print session summary
    avg_loss = (
        sum(outer.training_losses) / len(outer.training_losses)
        if outer.training_losses else 0.0
    )
    self_norm = outer.net.get_self_vector().norm().item()
    n_params = sum(p.numel() for p in outer.net.parameters())

    print(f"\n{'─'*60}")
    print(f"  Session {session} complete")
    print(f"  Total concepts: {len(state.concepts)}")
    print(f"  Total experiences: {len(state.experiences)}")
    print(f"  Connections found: {len(state.concept_graph_edges)}")
    print(f"  👁 Passive: {passive_count} | 🔴 Active: {active_count}")
    print(f"  v1 avg loss: {avg_loss:.4f} | self-state norm: {self_norm:.3f} | params: {n_params:,}")
    print(f"  Self: {state.self_summary}")
    print(f"  State saved to: {state_path}")
    print(f"  Weights saved to: {WEIGHTS_PATH}")
    print(f"{'─'*60}")

    # Print concept map
    print(f"\n  📊 Concept Map:")
    for name, concept in sorted(
        state.concepts.items(),
        key=lambda x: -x[1].times_activated,
    ):
        top_feats = concept.top_features(6)
        bar = "█" * concept.times_activated
        print(f"    {name:20s} ({concept.times_activated:3d}x) "
              f"[{', '.join(top_feats)}] {bar}")

    if state.concept_graph_edges:
        print(f"\n  🔗 Connections:")
        for c1, c2, shared in state.concept_graph_edges[-10:]:
            print(f"    {c1} ↔ {c2} (via {', '.join(shared)})")

    # v6: Strategy effectiveness report
    if hasattr(outer, '_strategy_scores'):
        print(f"\n  🧠 Question Strategy Effectiveness (v6 feedback):")
        for strategy, scores in sorted(
            outer._strategy_scores.items(),
            key=lambda x: -sum(x[1]) / max(len(x[1]), 1),
        ):
            if scores:
                avg = sum(scores) / len(scores)
                w = outer._strategy_weights.get(strategy, 1.0)
                print(f"    {strategy:15s}: avg IG={avg:.2f} "
                      f"| {len(scores)} uses | weight={w:.2f}")
            else:
                print(f"    {strategy:15s}: (unused)")



# ── Main ───────────────────────────────────────────────────────────

def main():
    global _ABLATION_MODE, STATE_DIR, WEIGHTS_PATH

    parser = argparse.ArgumentParser(description="Dream Layer Phase 1: Concept Learner")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Steps per session")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument(
        "--mode", type=str, default="feedback",
        choices=ABLATION_MODES,
        help="Ablation mode: random|template|structure|feedback (default: feedback)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility in ablation runs"
    )
    args = parser.parse_args()

    # Set ablation mode
    _ABLATION_MODE = args.mode

    # Mode-specific output directory (ablation runs save separately)
    if args.mode != "feedback":
        STATE_DIR = Path(__file__).parent.parent / "results" / f"ablation_{args.mode}"
        WEIGHTS_PATH = STATE_DIR / "concept_net.pt"
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("=" * 60)
    print("  Dream Layer — Phase 1: Concept Learner")
    print("  The outer layer learns. The world just responds.")
    print(f"  Environment: {DEFAULT_MODEL}")
    print(f"  Ablation mode: {args.mode.upper()}")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")
    print("=" * 60)

    # Load or initialize state
    state_path = STATE_DIR / "dream_state.json"

    if args.reset:
        if state_path.exists():
            os.remove(state_path)
        if WEIGHTS_PATH.exists():
            os.remove(WEIGHTS_PATH)
        print("\n  State and weights reset.")

    if state_path.exists():
        state = DreamState.load(state_path)
        print(f"\n  Resuming from session {state.session_count}, "
              f"step {state.total_steps}")
        print(f"  Concepts: {len(state.concepts)} | "
              f"Experiences: {len(state.experiences)}")
    else:
        state = DreamState()
        print("\n  First session — initializing fresh state")

    # Load environment (Llama 3 8B via MLX + MiniLM embedder)
    print()
    env = Environment(DEFAULT_MODEL)

    # Initialize outer layer
    outer = OuterLayer(state, env.embedding_dim, embedder=env.embedder)
    outer.initialize_seeds()

    # Run session
    run_session(env, outer, args.steps)

    print(f"\n✅ Session complete. Run again to continue learning.")
    print(f"   python {__file__} --steps {args.steps} --mode {args.mode}")


if __name__ == "__main__":
    main()
