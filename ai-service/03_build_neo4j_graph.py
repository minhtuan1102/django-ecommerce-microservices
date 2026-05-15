import argparse
import time
from pathlib import Path

import pandas as pd
try:
    from neo4j import GraphDatabase
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: neo4j. Run: pip install -r requirements_assignment.txt"
    ) from exc


def wait_for_neo4j(driver, timeout_seconds: int = 90, interval_seconds: int = 3) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with driver.session() as session:
                session.run("RETURN 1").consume()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(interval_seconds)

    raise RuntimeError(
        f"Neo4j is not reachable after {timeout_seconds} seconds. "
        f"Last error: {last_error}"
    )


# Define action to category and impact mapping
ACTION_METADATA = {
    "view": {"category": "browse", "impact_score": 1},
    "click": {"category": "browse", "impact_score": 2},
    "add_to_cart": {"category": "transaction", "impact_score": 3},
    "remove_from_cart": {"category": "transaction", "impact_score": 2},
    "wishlist": {"category": "preference", "impact_score": 2},
    "share": {"category": "social", "impact_score": 3},
    "purchase": {"category": "transaction", "impact_score": 5},
    "review": {"category": "social", "impact_score": 4},
}


def create_constraints(session) -> None:
    session.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    session.run("CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
    session.run("CREATE CONSTRAINT behavior_id_unique IF NOT EXISTS FOR (b:Behavior) REQUIRE b.id IS UNIQUE")
    session.run("CREATE CONSTRAINT action_type_unique IF NOT EXISTS FOR (a:Action) REQUIRE a.name IS UNIQUE")


def clear_graph(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def insert_batch(tx, rows: list[dict]) -> None:
    # First, create Action nodes for all action types
    tx.run(
        """
        UNWIND $actions AS action
        MERGE (a:Action {name: action.name})
        SET a.category = action.category, a.impact_score = action.impact_score
        """,
        actions=[
            {"name": act, "category": meta["category"], "impact_score": meta["impact_score"]}
            for act, meta in ACTION_METADATA.items()
        ],
    )
    
    # Then create User, Product, Behavior nodes and relationships
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (u:User {id: row.user_id})
        MERGE (p:Product {id: row.product_id})
        MERGE (a:Action {name: row.action})
        
        // Merge Behavior node (unique per interaction)
        MERGE (b:Behavior {id: row.behavior_id})
        SET b.action_type = row.action,
            b.timestamp = row.timestamp,
            b.category = row.category,
            b.impact_score = row.impact_score
        
        // Create relationships (MERGE to avoid duplicates)
        MERGE (u)-[:PERFORMED {timestamp: row.timestamp}]->(b)
        MERGE (b)-[:ON]->(p)
        MERGE (b)-[:IS_TYPE_OF]->(a)
        
        // Keep original INTERACTED for backward compatibility
        MERGE (u)-[:INTERACTED {action: row.action, timestamp: row.timestamp}]->(p)
        """,
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load data_user500.csv into Neo4j graph with Behavior nodes")
    parser.add_argument("--data", type=str, default="data_user500.csv", help="Input CSV path")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for insert")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing graph data before inserting",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)

    required_cols = {"user_id", "product_id", "action", "timestamp"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Prepare rows with behavior_id and metadata
    rows = []
    for idx, row in df[list(required_cols)].iterrows():
        action = row["action"]
        behavior_id = f"{row['user_id']}-{action}-{row['product_id']}-{idx}"
        metadata = ACTION_METADATA.get(action, {"category": "unknown", "impact_score": 0})
        
        rows.append({
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "action": action,
            "timestamp": row["timestamp"],
            "behavior_id": behavior_id,
            "category": metadata["category"],
            "impact_score": metadata["impact_score"],
        })

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    wait_for_neo4j(driver)
    with driver.session() as session:
        if args.reset:
            print("Clearing existing graph data...")
            clear_graph(session)

        create_constraints(session)

        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            session.execute_write(insert_batch, chunk)

        # Query statistics
        result = session.run(
            """
            CALL {
                MATCH (u:User)
                RETURN count(u) AS users
            }
            CALL {
                MATCH (p:Product)
                RETURN count(p) AS products
            }
            CALL {
                MATCH (b:Behavior)
                RETURN count(b) AS behaviors
            }
            CALL {
                MATCH (a:Action)
                RETURN count(a) AS actions
            }
            RETURN users, products, behaviors, actions
            """
        ).single()

    driver.close()

    print("Graph created successfully with Behavior nodes!")
    print(
        f"Users: {result['users']}, Products: {result['products']}, "
        f"Behaviors: {result['behaviors']}, Actions: {result['actions']}"
    )


if __name__ == "__main__":
    main()
