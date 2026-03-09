"""MSKG (Manufacturing Service Knowledge Graph) data processor.

Processes the MSKG dataset from Li & Starly (2024) into our standard
RecDataset format. Since MSKG is a knowledge graph of manufacturers and
their services (not a user-item interaction dataset), we construct
synthetic client-manufacturer interactions based on service matching.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.loader import RecDataset


# Size thresholds based on typical manufacturing classifications
SIZE_BINS = {"small": (0, 50), "medium": (50, 500), "large": (500, float("inf"))}

# US Census regions for geographic grouping
US_REGIONS = {
    "northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX", "DC"],
    "west": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_TO_REGION = {s: r for r, states in US_REGIONS.items() for s in states}


def _assign_size_group(row: pd.Series) -> str:
    emp = row.get("num_employees", row.get("employee_count", 0))
    if pd.isna(emp):
        emp = 0
    for label, (lo, hi) in SIZE_BINS.items():
        if lo <= emp < hi:
            return label
    return "large"


def _assign_geo_group(row: pd.Series) -> str:
    state = row.get("state", row.get("location_state", ""))
    if pd.isna(state):
        return "unknown"
    state = str(state).strip().upper()
    return STATE_TO_REGION.get(state, "unknown")


def load_mskg_from_json(mskg_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MSKG data from JSON/CSV files in the given directory."""
    mskg_dir = Path(mskg_dir)

    # Try different file formats
    manufacturers = None
    for fname in ["manufacturers.csv", "manufacturers.json", "nodes.csv", "companies.csv"]:
        fpath = mskg_dir / fname
        if fpath.exists():
            if fname.endswith(".json"):
                manufacturers = pd.read_json(fpath)
            else:
                manufacturers = pd.read_csv(fpath)
            break

    services_rel = None
    for fname in ["relationships.csv", "edges.csv", "manufacturer_services.csv", "services.csv"]:
        fpath = mskg_dir / fname
        if fpath.exists():
            services_rel = pd.read_csv(fpath)
            break

    certifications = None
    for fname in ["certifications.csv", "certs.csv"]:
        fpath = mskg_dir / fname
        if fpath.exists():
            certifications = pd.read_csv(fpath)
            break

    return manufacturers, services_rel, certifications


def generate_synthetic_interactions(
    manufacturers: pd.DataFrame,
    services_rel: pd.DataFrame,
    num_clients: int = 500,
    interactions_per_client: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic client-manufacturer interactions based on service similarity.

    Since MSKG doesn't contain actual client interaction data, we simulate
    realistic interactions where clients have service needs and interact with
    manufacturers that match those needs.
    """
    rng = np.random.RandomState(seed)
    manufacturer_ids = manufacturers.index.values if "manufacturer_id" not in manufacturers.columns \
        else manufacturers["manufacturer_id"].values

    n_mfg = len(manufacturer_ids)

    # Build service profile for each manufacturer
    if services_rel is not None and len(services_rel) > 0:
        # Aggregate services per manufacturer
        mfg_col = [c for c in services_rel.columns if "manufacturer" in c.lower() or "source" in c.lower() or "company" in c.lower()]
        svc_col = [c for c in services_rel.columns if "service" in c.lower() or "target" in c.lower() or "capability" in c.lower()]

        if mfg_col and svc_col:
            mfg_col, svc_col = mfg_col[0], svc_col[0]
            svc_text = services_rel.groupby(mfg_col)[svc_col].apply(lambda x: " ".join(x.astype(str))).reset_index()
            svc_text.columns = ["manufacturer_id", "service_text"]
        else:
            svc_text = pd.DataFrame({"manufacturer_id": manufacturer_ids, "service_text": ["manufacturing"] * n_mfg})
    else:
        svc_text = pd.DataFrame({"manufacturer_id": manufacturer_ids, "service_text": ["manufacturing"] * n_mfg})

    # TF-IDF similarity for service matching
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(svc_text["service_text"].fillna("unknown"))

    # Generate client profiles and interactions
    interactions = []
    for client_id in range(num_clients):
        # Client has a random service need vector
        client_need = rng.rand(tfidf_matrix.shape[1])
        client_need = client_need / (np.linalg.norm(client_need) + 1e-8)

        # Score manufacturers by similarity to client need
        scores = tfidf_matrix.dot(client_need)
        scores = np.asarray(scores).flatten()

        # Add noise and sample proportional to score
        noisy_scores = scores + rng.exponential(0.1, size=len(scores))
        probs = np.maximum(noisy_scores, 0)
        probs = probs / (probs.sum() + 1e-8)

        # Sample interactions (with replacement allowed)
        n_interact = rng.poisson(interactions_per_client)
        n_interact = max(3, min(n_interact, n_mfg))
        chosen = rng.choice(n_mfg, size=n_interact, replace=False, p=probs)

        for idx in chosen:
            interactions.append({
                "user_id": client_id,
                "item_id": idx,
                "rating": 1.0,  # implicit feedback
            })

    return pd.DataFrame(interactions)


def process_mskg(
    mskg_dir: str = "data/raw/mskg",
    num_clients: int = 500,
    interactions_per_client: int = 10,
    seed: int = 42,
) -> RecDataset:
    """Process MSKG into a RecDataset.

    If raw data is not available, generates a realistic synthetic dataset
    that mirrors the structure and properties of MSKG.
    """
    mskg_path = Path(mskg_dir)

    if mskg_path.exists() and any(mskg_path.iterdir()):
        manufacturers, services_rel, certifications = load_mskg_from_json(mskg_path)

        if manufacturers is not None:
            # Assign protected attributes
            manufacturers["size_group"] = manufacturers.apply(_assign_size_group, axis=1)
            manufacturers["geo_group"] = manufacturers.apply(_assign_geo_group, axis=1)

            # Build interactions
            interactions = generate_synthetic_interactions(
                manufacturers, services_rel, num_clients, interactions_per_client, seed
            )

            # Build feature/attribute DataFrames
            item_features = manufacturers.copy()
            if "manufacturer_id" not in item_features.columns:
                item_features = item_features.reset_index().rename(columns={"index": "item_id"})
            else:
                item_features = item_features.rename(columns={"manufacturer_id": "item_id"})

            protected_attrs = item_features[["item_id", "size_group", "geo_group"]].copy()

            return RecDataset(
                interactions=interactions,
                item_features=item_features,
                protected_attrs=protected_attrs,
                name="MSKG",
            )

    # Fallback: generate fully synthetic dataset mimicking MSKG properties
    print("[mskg_processor] Raw MSKG not found. Generating synthetic dataset.")
    return _generate_synthetic_mskg(num_manufacturers=2000, num_clients=num_clients, seed=seed)


def _generate_synthetic_mskg(
    num_manufacturers: int = 2000,
    num_clients: int = 500,
    seed: int = 42,
) -> RecDataset:
    """Generate a synthetic dataset that mimics MSKG's structure."""
    rng = np.random.RandomState(seed)

    # Manufacturer features
    services = [
        "cnc_machining", "3d_printing", "injection_molding", "sheet_metal",
        "casting", "welding", "laser_cutting", "edm", "grinding",
        "surface_treatment", "assembly", "quality_inspection",
        "prototyping", "powder_coating", "anodizing", "heat_treatment",
    ]

    certifications_list = ["ISO9001", "ISO14001", "AS9100", "IATF16949", "NADCAP", "ISO13485"]
    states = list(STATE_TO_REGION.keys())

    # Size distribution: ~60% small, ~30% medium, ~10% large (realistic for manufacturing)
    size_probs = [0.60, 0.30, 0.10]
    size_labels = ["small", "medium", "large"]
    sizes = rng.choice(size_labels, size=num_manufacturers, p=size_probs)

    manufacturers = []
    for i in range(num_manufacturers):
        n_services = rng.poisson(3) + 1
        n_services = min(n_services, len(services))
        mfg_services = rng.choice(services, size=n_services, replace=False).tolist()
        n_certs = rng.poisson(1)
        n_certs = min(n_certs, len(certifications_list))
        mfg_certs = rng.choice(certifications_list, size=n_certs, replace=False).tolist() if n_certs > 0 else []
        state = rng.choice(states)

        manufacturers.append({
            "item_id": i,
            "services": " ".join(mfg_services),
            "certifications": " ".join(mfg_certs),
            "state": state,
            "size_group": sizes[i],
            "geo_group": STATE_TO_REGION[state],
            "num_employees": {
                "small": rng.randint(1, 50),
                "medium": rng.randint(50, 500),
                "large": rng.randint(500, 5000),
            }[sizes[i]],
        })

    mfg_df = pd.DataFrame(manufacturers)

    # Build TF-IDF for service matching
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(mfg_df["services"])

    # Popularity bias: large manufacturers get more interactions
    popularity_boost = {"small": 1.0, "medium": 2.0, "large": 5.0}
    base_pop = np.array([popularity_boost[s] for s in sizes])
    base_pop = base_pop / base_pop.sum()

    interactions = []
    for client_id in range(num_clients):
        # Client need: random subset of services
        need_services = rng.choice(services, size=rng.poisson(2) + 1, replace=False)
        need_text = " ".join(need_services)
        need_vec = vectorizer.transform([need_text])

        # Score by content match
        content_scores = cosine_similarity(need_vec, tfidf).flatten()

        # Combine content match with popularity (simulates real-world bias)
        combined = 0.6 * content_scores + 0.4 * base_pop
        combined = np.maximum(combined, 1e-8)
        probs = combined / combined.sum()

        n_interact = max(3, min(rng.poisson(8), 30))
        chosen = rng.choice(num_manufacturers, size=n_interact, replace=False, p=probs)

        for idx in chosen:
            interactions.append({"user_id": client_id, "item_id": int(idx), "rating": 1.0})

    interactions_df = pd.DataFrame(interactions)
    protected_df = mfg_df[["item_id", "size_group", "geo_group"]].copy()

    return RecDataset(
        interactions=interactions_df,
        item_features=mfg_df,
        protected_attrs=protected_df,
        name="MSKG-synthetic",
    )
