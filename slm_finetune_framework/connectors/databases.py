"""Connector for relational databases via SQLAlchemy."""
from typing import Iterable, List
from sqlalchemy import create_engine, text
from slm_finetune_framework.core.interfaces import BaseConnector, RawChunk
from slm_finetune_framework.core.registry import ConnectorRegistry


class DatabaseConnector(BaseConnector):
    """Basic database connector that streams table rows or queries as text."""

    def __init__(self, conn_str: str, tables: List[str] | None = None, queries: List[str] | None = None):
        self.engine = create_engine(conn_str)
        self.tables = tables or []
        self.queries = queries or []

    def list_resources(self) -> Iterable[str]:
        for t in self.tables:
            yield f"table:{t}"
        for i, q in enumerate(self.queries):
            yield f"query:{i}"

    def load_resource(self, resource_id: str) -> Iterable[RawChunk]:
        kind, ident = resource_id.split(":", 1)
        with self.engine.connect() as conn:
            if kind == "table":
                sql = text(f"SELECT * FROM {ident}")
            elif kind == "query":
                sql = text(self.queries[int(ident)])
            else:
                raise ValueError(f"Unknown resource kind {kind}")

            # Use streaming / batching if needed for big tables
            rows = conn.execute(sql).fetchmany(10000)
            for i, row in enumerate(rows):
                content = " ".join(str(v) for v in row._mapping.values())
                yield RawChunk(
                    id=f"{resource_id}:{i}",
                    source_type="database",
                    source_uri=resource_id,
                    metadata={"row_index": i},
                    content=content,
                )


ConnectorRegistry.register("database", DatabaseConnector)
