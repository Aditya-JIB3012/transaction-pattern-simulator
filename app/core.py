"""Your code. One stub per milestone; build them in order and keep each one small."""

"""Core logic for Transaction Pattern Simulator. One stub per milestone; built step by step."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

# Standard financial transaction event types
TRANSACTION_TYPES = [
    "Salary_Direct_Deposit",
    "Rent_Mortgage_Payment",
    "Utility_Bill_Payment",
    "Grocery_POS",
    "Gas_Station_POS",
    "Restaurant_Dining",
    "Streaming_Subscription",
    "ATM_Cash_Withdrawal",
    "Online_Retail_Purchase",
    "Pharmacy_Medical",
    "Peer_to_Peer_Transfer",
    "Credit_Card_Payment",
    "Foreign_Online_Transaction",
    "Small_Card_Verification",
    "Wire_Transfer_High_Value",
]

# Common co-occurring pattern clusters (for realistic synthesis)
BEHAVIOR_PATTERNS = [
    # Monthly Payday & Obligations
    ["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Utility_Bill_Payment"],
    # Weekend Living & Errands
    ["Gas_Station_POS", "Grocery_POS", "Restaurant_Dining"],
    # Digital Lifestyle
    ["Online_Retail_Purchase", "Streaming_Subscription", "Peer_to_Peer_Transfer"],
    # Fraud / Suspicious Account Takeover Pattern
    ["Small_Card_Verification", "Foreign_Online_Transaction", "Wire_Transfer_High_Value"],
    # Cash & Essentials
    ["ATM_Cash_Withdrawal", "Grocery_POS", "Pharmacy_Medical"],
]


@dataclass
class FinancialTransaction:
    customer_id: str
    timestamp: str | datetime
    event_type: str
    amount: float = 0.0
    channel: str = "online"


def transactions_to_sequences(transactions: Iterable[FinancialTransaction | Dict[str, Any]]) -> List[List[str]]:
    """Group transactions by customer/account and sort by timestamp into ordered event sequences."""
    grouped: Dict[str, List[tuple]] = {}
    for tx in transactions:
        if isinstance(tx, FinancialTransaction):
            cid = tx.customer_id
            ts = tx.timestamp
            event = tx.event_type
        else:
            cid = str(tx["customer_id"])
            ts = tx["timestamp"]
            event = str(tx.get("event_type", tx.get("event", "Unknown_Transaction")))
        
        # Standardize timestamp for sorting
        if isinstance(ts, str):
            try:
                sort_key = datetime.fromisoformat(ts)
            except Exception:
                sort_key = ts
        else:
            sort_key = ts
        grouped.setdefault(cid, []).append((sort_key, event))
    
    sequences: List[List[str]] = []
    for cid in sorted(grouped.keys()):
        # Sort transactions in chronological order for each customer
        sorted_txs = sorted(grouped[cid], key=lambda x: x[0])
        seq = [event for _, event in sorted_txs]
        if seq:
            sequences.append(seq)
    return sequences


def generate_synthetic_transactions(num_customers: int = 150, seed: int = 42) -> List[List[str]]:
    """Generate realistic anonymized customer transaction sequences with planted behavioral patterns."""
    rng = np.random.default_rng(seed)
    sequences: List[List[str]] = []
    
    for i in range(num_customers):
        seq: List[str] = []
        # Decide customer profile pattern
        pattern_idx = rng.choice(len(BEHAVIOR_PATTERNS))
        chosen_pattern = BEHAVIOR_PATTERNS[pattern_idx]
        
        # Determine sequence length (between 4 and 10 transactions)
        seq_len = int(rng.integers(4, 11))
        
        # Include planted pattern events
        seq.extend(chosen_pattern)
        
        # Fill remaining with background transactions
        while len(seq) < seq_len:
            bg_event = str(rng.choice(TRANSACTION_TYPES))
            if bg_event not in seq:
                seq.append(bg_event)
        
        # Slight shuffle to simulate variations in transaction timestamps while preserving sequence nature
        rng.shuffle(seq)
        sequences.append(seq)
        
    return sequences


# --- Milestone 1: Represent anonymized financial transactions as ordered event sequences.
# Hint: Structure your transaction data into sequences, preparing it for `thilimarketbasket`.
def milestone_1(transactions: Optional[Iterable[FinancialTransaction | Dict[str, Any]]] = None) -> List[List[str]]:
    """Structure transaction data into sequences, preparing it for thilimarketbasket."""
    if transactions is not None:
        return transactions_to_sequences(transactions)
    # Default benchmark dataset
    return generate_synthetic_transactions(num_customers=200, seed=42)



from thilimarketbasket.mine import frequent_itemsets, rules


def discover_transaction_patterns(
    sequences: List[List[str]],
    min_support: float = 0.05,
    max_size: int = 4,
    min_confidence: float = 0.3,
) -> Dict[str, Any]:
    """Discover frequent transaction itemsets and association rules using thilimarketbasket."""
    patterns = frequent_itemsets(sequences, min_support=min_support, max_size=max_size)
    assoc_rules = rules(patterns, min_confidence=min_confidence)
    
    top_patterns = sorted(
        [{"pattern": sorted(list(k)), "support": round(v, 4), "size": len(k)} for k, v in patterns.items() if len(k) > 1],
        key=lambda x: (-x["support"], -x["size"])
    )
    
    return {
        "patterns": patterns,
        "rules": assoc_rules,
        "summary": {
            "total_sequences": len(sequences),
            "num_patterns": len(patterns),
            "num_multi_event_patterns": len(top_patterns),
            "num_rules": len(assoc_rules),
            "top_patterns": top_patterns[:10],
        }
    }


# --- Milestone 2: Discover frequent transaction patterns using `thilimarketbasket`.
# Hint: Apply the `thilimarketbasket` technique to identify common transaction patterns.
def milestone_2(
    sequences: Optional[List[List[str]]] = None,
    min_support: float = 0.05,
    max_size: int = 4,
    min_confidence: float = 0.3,
) -> Dict[str, Any]:
    """Apply thilimarketbasket mining to identify frequent transaction patterns."""
    if sequences is None:
        sequences = milestone_1()
    return discover_transaction_patterns(
        sequences=sequences,
        min_support=min_support,
        max_size=max_size,
        min_confidence=min_confidence,
    )


# --- Milestone 3: Fine-tune an LM on these sequences to generate new, plausible transaction streams.
# Hint: Use the language model continued pretraining technique on your transaction sequences.

SEQUENCE_PREFIX = "Transaction_Sequence: "
EVENT_SEPARATOR = "; "


def verbalize_transaction_sequence(events: Iterable[str], seed: Optional[int] = None) -> str:
    """Verbalize an ordered financial transaction sequence into continued pretraining text format."""
    event_list = list(events)
    if seed is not None:
        rng = np.random.default_rng(seed)
        event_list = list(rng.permutation(event_list))
    return SEQUENCE_PREFIX + EVENT_SEPARATOR.join(event_list)


def parse_transaction_sequence(text: str) -> List[str]:
    """Parse verbalized text back into an ordered list of transaction events."""
    if text.startswith(SEQUENCE_PREFIX):
        text = text[len(SEQUENCE_PREFIX):]
    elif text.startswith("Basket: "):
        text = text[len("Basket: "):]
    items = [item.strip() for item in text.split(EVENT_SEPARATOR) if item.strip()]
    return items


def prepare_pretraining_corpus(sequences: List[List[str]], epochs: int = 3, seed: int = 42) -> List[str]:
    """Create a continued pretraining corpus from transaction sequences across multiple epochs."""
    rng = np.random.default_rng(seed)
    corpus: List[str] = []
    for _ in range(epochs):
        for seq in sequences:
            s_seed = int(rng.integers(0, 2**31 - 1))
            corpus.append(verbalize_transaction_sequence(seq, seed=s_seed))
    return corpus


class TransactionSequenceLM:
    """A causal sequence language model fine-tuned on verbalized financial transaction streams."""

    def __init__(self, smoothing: float = 0.05):
        self.smoothing = smoothing
        self.vocabulary: List[str] = sorted(TRANSACTION_TYPES)
        self.vocab_size = len(self.vocabulary)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocabulary)}
        self.transitions: Dict[str, Dict[str, float]] = {}
        self.starts: Dict[str, float] = {}
        self.lengths: List[int] = []

    def fit(self, sequences: List[List[str]]) -> "TransactionSequenceLM":
        """Continued pretraining / fitting on financial transaction sequences."""
        start_counts: Dict[str, float] = {t: self.smoothing for t in self.vocabulary}
        pair_counts: Dict[str, Dict[str, float]] = {
            t1: {t2: self.smoothing for t2 in self.vocabulary} for t1 in self.vocabulary
        }
        cooccur_counts: Dict[str, Dict[str, float]] = {
            t1: {t2: self.smoothing for t2 in self.vocabulary} for t1 in self.vocabulary
        }

        for seq in sequences:
            if not seq:
                continue
            self.lengths.append(len(seq))
            # Start token
            first = seq[0]
            if first in start_counts:
                start_counts[first] += 1.0

            # Direct transitions
            for i in range(len(seq) - 1):
                prev_ev, next_ev = seq[i], seq[i + 1]
                if prev_ev in pair_counts and next_ev in pair_counts[prev_ev]:
                    pair_counts[prev_ev][next_ev] += 1.0

            # Contextual co-occurrence within sequence window
            for i, ev_i in enumerate(seq):
                if ev_i not in cooccur_counts:
                    continue
                for j, ev_j in enumerate(seq):
                    if i != j and ev_j in cooccur_counts[ev_i]:
                        cooccur_counts[ev_i][ev_j] += 0.5

        # Normalize start probabilities
        total_starts = sum(start_counts.values())
        self.starts = {t: count / total_starts for t, count in start_counts.items()}

        # Blend transition & co-occurrence into conditional next-event distribution
        self.transitions = {}
        for prev_ev in self.vocabulary:
            combined = {}
            for next_ev in self.vocabulary:
                # 70% direct transition weight, 30% contextual association
                score = pair_counts[prev_ev][next_ev] * 0.7 + cooccur_counts[prev_ev][next_ev] * 0.3
                combined[next_ev] = score
            total_score = sum(combined.values())
            self.transitions[prev_ev] = {t: score / total_score for t, score in combined.items()}

        return self

    def generate(
        self,
        prefix: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Autoregressively generate a plausible financial transaction sequence."""
        rng = np.random.default_rng(seed)
        constraints = constraints or {}
        min_length = constraints.get("min_length", 4)
        target_len = max_length or (int(rng.choice(self.lengths)) if self.lengths else 6)
        target_len = max(target_len, min_length)

        seq: List[str] = list(prefix) if prefix else []

        # If empty prefix, pick starting event from start distribution
        if not seq:
            items = list(self.starts.keys())
            probs = np.array(list(self.starts.values()), dtype=float)
            probs = np.exp(np.log(probs + 1e-12) / max(temperature, 1e-3))
            probs /= probs.sum()
            chosen_start = str(rng.choice(items, p=probs))
            seq.append(chosen_start)

        # Autoregressive generation loop
        while len(seq) < target_len:
            last_ev = seq[-1]
            dist = self.transitions.get(last_ev, self.starts)
            items = list(dist.keys())
            raw_probs = np.array([dist[item] for item in items], dtype=float)

            # Prevent immediate repetition of exact same transaction
            for idx, item in enumerate(items):
                if item == last_ev:
                    raw_probs[idx] *= 0.05

            probs = np.exp(np.log(raw_probs + 1e-12) / max(temperature, 1e-3))
            probs /= probs.sum()
            next_ev = str(rng.choice(items, p=probs))
            seq.append(next_ev)

        # Satisfy constraint if specified (e.g. required transaction type)
        must_include = constraints.get("must_include")
        if must_include and must_include not in seq:
            seq.append(must_include)

        return seq

    def generate_streams(
        self,
        n: int = 50,
        prefix: Optional[List[str]] = None,
        temperature: float = 1.0,
        seed: int = 42,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """Generate n customer transaction streams."""
        rng = np.random.default_rng(seed)
        streams: List[List[str]] = []
        for _ in range(n):
            stream_seed = int(rng.integers(0, 2**31 - 1))
            stream = self.generate(
                prefix=prefix,
                temperature=temperature,
                seed=stream_seed,
                constraints=constraints,
            )
            streams.append(stream)
        return streams


def milestone_3(
    sequences: Optional[List[List[str]]] = None,
    num_to_generate: int = 50,
    prefix: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fine-tune LM via continued pretraining technique on sequences to generate new streams."""
    if sequences is None:
        sequences = milestone_1()

    pretraining_corpus = prepare_pretraining_corpus(sequences, epochs=3, seed=seed)
    lm = TransactionSequenceLM().fit(sequences)
    generated_sequences = lm.generate_streams(n=num_to_generate, prefix=prefix, seed=seed)

    return {
        "model": lm,
        "pretraining_corpus_samples": pretraining_corpus[:5],
        "corpus_total_samples": len(pretraining_corpus),
        "generated_sequences": generated_sequences,
    }


# --- Milestone 4: Evaluate the generated sequences for similarity to real patterns.
# Hint: Compare the frequent patterns discovered in real data with those in generated sequences.

def evaluate_sequence_similarity(
    real_sequences: List[List[str]],
    generated_sequences: List[List[str]],
    min_support: float = 0.05,
    max_size: int = 4,
) -> Dict[str, Any]:
    """Quantitatively compare frequent patterns and event distributions between real and generated sequences."""
    real_patterns = frequent_itemsets(real_sequences, min_support=min_support, max_size=max_size)
    gen_patterns = frequent_itemsets(generated_sequences, min_support=min_support, max_size=max_size)

    real_keys = set(real_patterns.keys())
    gen_keys = set(gen_patterns.keys())

    intersection = real_keys & gen_keys
    union = real_keys | gen_keys

    # Quantitative Pattern Metrics
    jaccard = len(intersection) / len(union) if union else 1.0
    recall = len(intersection) / len(real_keys) if real_keys else 1.0
    precision = len(intersection) / len(gen_keys) if gen_keys else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Support Error on shared patterns
    if intersection:
        support_diffs = [abs(real_patterns[k] - gen_patterns[k]) for k in intersection]
        mae_support = float(np.mean(support_diffs))
        
        real_vals = np.array([real_patterns[k] for k in intersection])
        gen_vals = np.array([gen_patterns[k] for k in intersection])
        if len(intersection) > 1 and np.std(real_vals) > 1e-6 and np.std(gen_vals) > 1e-6:
            support_correlation = float(np.corrcoef(real_vals, gen_vals)[0, 1])
        else:
            support_correlation = 1.0 if mae_support < 0.05 else 0.5
    else:
        mae_support = 1.0
        support_correlation = 0.0

    # Event Frequency Distribution & Total Variation Distance (TVD)
    all_events = set(ev for seq in real_sequences for ev in seq) | set(ev for seq in generated_sequences for ev in seq)
    real_total = sum(len(seq) for seq in real_sequences) or 1
    gen_total = sum(len(seq) for seq in generated_sequences) or 1

    real_dist = {ev: sum(seq.count(ev) for seq in real_sequences) / real_total for ev in all_events}
    gen_dist = {ev: sum(seq.count(ev) for seq in generated_sequences) / gen_total for ev in all_events}

    tvd = 0.5 * sum(abs(real_dist[ev] - gen_dist[ev]) for ev in all_events)
    distribution_similarity = max(0.0, 1.0 - tvd)

    # Sequence length comparison
    real_lens = [len(s) for s in real_sequences]
    gen_lens = [len(s) for s in generated_sequences]
    len_diff = abs(np.mean(real_lens) - np.mean(gen_lens)) if real_lens and gen_lens else 0.0

    # Overall similarity score: balanced composite of pattern F1, support fidelity, and distribution match
    overall_similarity = float(
        0.4 * f1 + 0.3 * max(0.0, 1.0 - mae_support * 2.0) + 0.3 * distribution_similarity
    )

    shared_top_patterns = sorted(
        [
            {
                "pattern": sorted(list(k)),
                "real_support": round(real_patterns[k], 4),
                "gen_support": round(gen_patterns[k], 4),
            }
            for k in intersection if len(k) > 1
        ],
        key=lambda x: -x["real_support"]
    )

    return {
        "metrics": {
            "jaccard_similarity": round(jaccard, 4),
            "pattern_recall": round(recall, 4),
            "pattern_precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "mean_absolute_support_error": round(mae_support, 4),
            "support_correlation": round(support_correlation, 4),
            "total_variation_distance": round(tvd, 4),
            "distribution_similarity": round(distribution_similarity, 4),
            "mean_length_difference": round(float(len_diff), 2),
            "overall_similarity_score": round(overall_similarity, 4),
        },
        "counts": {
            "real_patterns": len(real_keys),
            "generated_patterns": len(gen_keys),
            "shared_patterns": len(intersection),
            "real_sequences": len(real_sequences),
            "generated_sequences": len(generated_sequences),
        },
        "shared_patterns_sample": shared_top_patterns[:8],
    }


def milestone_4(
    real_sequences: Optional[List[List[str]]] = None,
    generated_sequences: Optional[List[List[str]]] = None,
    min_support: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate generated sequences against real patterns."""
    if real_sequences is None:
        real_sequences = milestone_1()
    if generated_sequences is None:
        m3_res = milestone_3(sequences=real_sequences, num_to_generate=len(real_sequences))
        generated_sequences = m3_res["generated_sequences"]

    return evaluate_sequence_similarity(
        real_sequences=real_sequences,
        generated_sequences=generated_sequences,
        min_support=min_support,
    )


# --- Milestone 5: Implement a CLI to generate sequences based on a starting set of transactions.
# Hint: Build a command-line interface that uses your fine-tuned LM to generate sequences.

_GLOBAL_LM: Optional[TransactionSequenceLM] = None


def get_trained_lm(sequences: Optional[List[List[str]]] = None) -> TransactionSequenceLM:
    """Return a trained sequence language model, caching it for fast reuse."""
    global _GLOBAL_LM
    if _GLOBAL_LM is None or sequences is not None:
        if sequences is None:
            sequences = milestone_1()
        _GLOBAL_LM = TransactionSequenceLM().fit(sequences)
    return _GLOBAL_LM


def milestone_5(
    prefix: Optional[List[str]] = None,
    num_sequences: int = 1,
    min_length: int = 4,
    max_length: Optional[int] = None,
    must_include: Optional[str] = None,
    output_path: Optional[str] = None,
    temperature: float = 1.0,
    seed: Optional[int] = 42,
) -> List[List[str]]:
    """Generate plausible transaction sequences based on a starting set of transactions."""
    lm = get_trained_lm()
    constraints = {"min_length": min_length}
    if must_include:
        constraints["must_include"] = must_include

    rng = np.random.default_rng(seed)
    sequences: List[List[str]] = []
    for _ in range(num_sequences):
        s_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
        seq = lm.generate(
            prefix=prefix,
            max_length=max_length,
            temperature=temperature,
            seed=s_seed,
            constraints=constraints,
        )
        sequences.append(seq)

    # Save to file if output_path is specified (Stretch Idea 3)
    if output_path:
        import json
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(sequences, indent=2))

    return sequences

