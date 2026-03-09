"""DataCo Smart Supply Chain dataset processor.

Transforms the DataCo dataset into a supplier recommendation format:
- Users = customers (by customer segment + market)
- Items = products supplied by specific suppliers
- Protected attribute = supplier market region
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import RecDataset


def process_dataco(
    dataco_dir: str = "data/raw/dataco",
    min_interactions: int = 3,
    seed: int = 42,
) -> RecDataset:
    """Process DataCo supply chain data into RecDataset format."""
    dataco_path = Path(dataco_dir)

    # Try to find the actual CSV
    csv_files = list(dataco_path.glob("*.csv"))
    df = None
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="latin-1")
            if len(df) > 1000:
                break
        except Exception:
            continue

    if df is not None and len(df) > 0:
        return _process_real_dataco(df, min_interactions, seed)

    print("[supply_chain_processor] DataCo not found. Generating synthetic dataset.")
    return _generate_synthetic_supply_chain(seed=seed)


def _process_real_dataco(df: pd.DataFrame, min_interactions: int, seed: int) -> RecDataset:
    """Process the real DataCo CSV."""
    # Identify relevant columns (DataCo has varying column names)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "customer id" in cl:
            col_map["customer_id"] = col
        elif "product" in cl and "id" in cl:
            col_map["product_id"] = col
        elif "order region" in cl:
            col_map["region"] = col
        elif "market" in cl and "market" not in col_map:
            col_map["market"] = col
        elif "category" in cl and "name" in cl:
            col_map["category"] = col
        elif "department" in cl and "name" in cl:
            col_map["department"] = col

    if "customer_id" not in col_map or "product_id" not in col_map:
        # Fallback: use first two integer columns
        print("[supply_chain_processor] Could not map columns, using synthetic.")
        return _generate_synthetic_supply_chain(seed=seed)

    # Create user-item interactions
    interactions = df[[col_map["customer_id"], col_map["product_id"]]].copy()
    interactions.columns = ["user_id_raw", "item_id_raw"]

    # Re-index
    user_map = {u: i for i, u in enumerate(interactions["user_id_raw"].unique())}
    item_map = {it: i for i, it in enumerate(interactions["item_id_raw"].unique())}
    interactions["user_id"] = interactions["user_id_raw"].map(user_map)
    interactions["item_id"] = interactions["item_id_raw"].map(item_map)
    interactions["rating"] = 1.0
    interactions = interactions[["user_id", "item_id", "rating"]].drop_duplicates()

    # Filter by minimum interactions
    user_counts = interactions["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    interactions = interactions[interactions["user_id"].isin(valid_users)]

    item_counts = interactions["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_interactions].index
    interactions = interactions[interactions["item_id"].isin(valid_items)]

    # Build item features and protected attributes
    item_df = df.drop_duplicates(subset=[col_map["product_id"]]).copy()
    item_df["item_id"] = item_df[col_map["product_id"]].map(item_map)
    item_df = item_df.dropna(subset=["item_id"])
    item_df["item_id"] = item_df["item_id"].astype(int)

    # Protected: use market/region as geo group, estimate size from order volume
    order_volume = df.groupby(col_map["product_id"]).size().reset_index(name="volume")
    order_volume["item_id"] = order_volume[col_map["product_id"]].map(item_map)
    q33, q66 = order_volume["volume"].quantile([0.33, 0.66])
    order_volume["size_group"] = pd.cut(
        order_volume["volume"],
        bins=[-1, q33, q66, float("inf")],
        labels=["small", "medium", "large"],
    )

    if "region" in col_map:
        region_mode = df.groupby(col_map["product_id"])[col_map["region"]].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"
        ).reset_index()
        region_mode.columns = ["product_id_raw", "geo_group"]
        region_mode["item_id"] = region_mode["product_id_raw"].map(item_map)
    else:
        region_mode = pd.DataFrame({
            "item_id": list(item_map.values()),
            "geo_group": "unknown",
        })

    protected_attrs = order_volume[["item_id", "size_group"]].merge(
        region_mode[["item_id", "geo_group"]], on="item_id", how="left"
    ).dropna(subset=["item_id"])
    protected_attrs["item_id"] = protected_attrs["item_id"].astype(int)

    item_features = item_df[["item_id"]].copy()
    if "category" in col_map:
        item_features["category"] = item_df[col_map["category"]].values[:len(item_features)]

    return RecDataset(
        interactions=interactions.reset_index(drop=True),
        item_features=item_features.reset_index(drop=True),
        protected_attrs=protected_attrs.reset_index(drop=True),
        name="DataCo",
    )


def _generate_synthetic_supply_chain(
    num_suppliers: int = 1000,
    num_customers: int = 300,
    seed: int = 42,
) -> RecDataset:
    """Generate a synthetic supply chain dataset."""
    rng = np.random.RandomState(seed)

    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Africa"]
    categories = ["electronics", "machinery", "chemicals", "metals", "textiles", "plastics"]

    # Suppliers
    size_labels = ["small", "medium", "large"]
    sizes = rng.choice(size_labels, size=num_suppliers, p=[0.55, 0.30, 0.15])
    geo_groups = rng.choice(regions, size=num_suppliers, p=[0.3, 0.25, 0.25, 0.12, 0.08])
    cats = rng.choice(categories, size=num_suppliers)

    item_features = pd.DataFrame({
        "item_id": range(num_suppliers),
        "category": cats,
        "region": geo_groups,
    })

    protected_attrs = pd.DataFrame({
        "item_id": range(num_suppliers),
        "size_group": sizes,
        "geo_group": geo_groups,
    })

    # Popularity bias by size
    pop_boost = {"small": 1.0, "medium": 2.5, "large": 6.0}
    base_pop = np.array([pop_boost[s] for s in sizes])
    base_pop = base_pop / base_pop.sum()

    interactions = []
    for cust_id in range(num_customers):
        pref_cat = rng.choice(categories)
        cat_match = np.array([1.5 if c == pref_cat else 0.5 for c in cats])
        probs = cat_match * base_pop
        probs = probs / probs.sum()

        n = max(3, min(rng.poisson(6), 25))
        chosen = rng.choice(num_suppliers, size=n, replace=False, p=probs)
        for idx in chosen:
            interactions.append({"user_id": cust_id, "item_id": int(idx), "rating": 1.0})

    return RecDataset(
        interactions=pd.DataFrame(interactions),
        item_features=item_features,
        protected_attrs=protected_attrs,
        name="DataCo-synthetic",
    )
