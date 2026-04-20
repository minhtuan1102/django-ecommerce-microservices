"""
Neo4j Graph Service for Behavior Analysis
Handles interaction logging and graph-based recommendations
"""
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings

# Try to import neo4j, but don't fail if not installed (for robustness)
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)

class GraphService:
    """
    Service to interact with Neo4j Graph Database
    """
    
    def __init__(self):
        self.uri = getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        self.user = getattr(settings, 'NEO4J_USER', 'neo4j')
        self.password = getattr(settings, 'NEO4J_PASSWORD', 'password')
        self.driver = None
        self._initialized = False
        
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password)
                )
                self._initialized = True
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
        else:
            logger.warning("Neo4j driver not installed. GraphService will operate in dummy mode.")

    def close(self):
        if self.driver:
            self.driver.close()

    def log_interaction(self, user_id: str, product_id: str, event_type: str, metadata: Dict = None):
        """
        Log an interaction between a user and a product in the graph
        """
        if not self._initialized:
            return
            
        # Standardize relationship type (VIEW, CART, PURCHASE, SEARCH)
        rel_type = event_type.upper().replace('PRODUCT_', '')
        if rel_type not in ['VIEW', 'CART', 'PURCHASE', 'SEARCH']:
            rel_type = 'INTERACTED'
            
        query = f"""
        MERGE (u:User {{id: $user_id}})
        MERGE (p:Product {{id: $product_id}})
        MERGE (u)-[r:{rel_type}]->(p)
        ON CREATE SET r.weight = 1, r.created_at = datetime()
        ON MATCH SET r.weight = r.weight + 1, r.updated_at = datetime()
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, user_id=str(user_id), product_id=str(product_id))
        except Exception as e:
            logger.error(f"Error logging interaction to Neo4j: {e}")

    def get_recommendations(self, user_id: str, limit: int = 5) -> List[str]:
        """
        Collaborative filtering recommendation via graph:
        'Users who interacted with things you interacted with also interacted with these'
        """
        if not self._initialized:
            return []
            
        query = """
        MATCH (u:User {id: $user_id})-[r1:VIEW|CART|PURCHASE]->(p1:Product)
        MATCH (other:User)-[r2:VIEW|CART|PURCHASE]->(p1)
        WHERE other <> u
        MATCH (other)-[r3:VIEW|CART|PURCHASE]->(rec:Product)
        WHERE NOT (u)-[:VIEW|CART|PURCHASE]->(rec)
        RETURN rec.id AS product_id, COUNT(*) AS strength
        ORDER BY strength DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=str(user_id), limit=limit)
                return [record["product_id"] for record in result]
        except Exception as e:
            logger.error(f"Error getting recommendations from Neo4j: {e}")
            return []

    def get_related_products(self, product_id: str, limit: int = 5) -> List[str]:
        """
        Content-based/Co-occurrence recommendation:
        'Products frequently bought/viewed together'
        """
        if not self._initialized:
            return []
            
        query = """
        MATCH (p:Product {id: $product_id})<-[:VIEW|CART|PURCHASE]-(u:User)-[:VIEW|CART|PURCHASE]->(other:Product)
        WHERE other <> p
        RETURN other.id AS product_id, COUNT(*) AS strength
        ORDER BY strength DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, product_id=str(product_id), limit=limit)
                return [record["product_id"] for record in result]
        except Exception as e:
            logger.error(f"Error getting related products from Neo4j: {e}")
            return []

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get structured context about a user for RAG
        """
        if not self._initialized:
            return {}
            
        query = """
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[r:VIEW|CART|PURCHASE]->(p:Product)
        WITH u, p, r ORDER BY r.updated_at DESC
        RETURN 
            u.id as user_id,
            collect({product_id: p.id, type: type(r), weight: r.weight})[0..5] as recent_interactions
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=str(user_id))
                record = result.single()
                if record:
                    return {
                        "user_id": record["user_id"],
                        "recent_interactions": record["recent_interactions"]
                    }
        except Exception as e:
            logger.error(f"Error getting user context from Neo4j: {e}")
            
        return {}

# Singleton instance
_graph_instance: Optional[GraphService] = None

def get_graph_service() -> GraphService:
    """Get or create GraphService instance"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = GraphService()
    return _graph_instance
