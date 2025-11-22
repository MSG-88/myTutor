"""Connector for various server/application sources."""
from typing import Iterable
from slm_finetune_framework.core.interfaces import BaseConnector, RawChunk
from slm_finetune_framework.core.registry import ConnectorRegistry


class ServerConnector(BaseConnector):
    def __init__(self, server_type: str, config: dict):
        """
        server_type: 'ftp' | 'sftp' | 'sharepoint' | 'jira' | 'confluence' | 'http'
        config: server-specific settings
        """
        self.server_type = server_type
        self.config = config
        self.client = self._init_client()

    def _init_client(self):
        # TODO: implement per server_type, or set up stub for Codex
        return None

    def list_resources(self) -> Iterable[str]:
        # TODO: return file paths/issue keys/pages/etc
        return []

    def load_resource(self, resource_id: str) -> Iterable[RawChunk]:
        # TODO: fetch content from the server, parse into text
        return []


ConnectorRegistry.register("server", ServerConnector)
