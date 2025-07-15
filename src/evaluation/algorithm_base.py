from abc import ABC, abstractmethod
from .performance_metrics import PerformanceMetrics

class AlgorithmBase(ABC):

    def __init__(self, config, driver):
        self.config = config
        self.driver = driver
        self.name = self.config.get('name', 'Unknown Algorithm')

    @abstractmethod
    def insert_relationships(self, relationships) -> PerformanceMetrics:
        """Insert relationships and return performance metrics"""
        pass

    def clear_database(self):
        with self.driver.session() as session:

            count_result = session.run("""
                MATCH (n:Entity) 
                WHERE n.isBase IS NULL OR n.isBase = false
                RETURN count(n) as node_count
            """)
            node_count = count_result.single()["node_count"]

            session.run("""
                MATCH (n:Entity) 
                WHERE n.isBase IS NULL OR n.isBase = false
                DETACH DELETE n
            """)

            print(f"Cleared {node_count} non-base Entity nodes and their relationships")

    def get_database_stats(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(DISTINCT n) as nodes, COUNT(r) as relationships
            """)
            return result.single()
