"""Registries for connectors and task builders."""
from typing import Dict, Type
from .interfaces import BaseConnector, BaseTaskBuilder


class ConnectorRegistry:
    """Registry mapping connector names to implementations."""

    _registry: Dict[str, Type[BaseConnector]] = {}

    @classmethod
    def register(cls, name: str, connector_cls: Type[BaseConnector]):
        cls._registry[name] = connector_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseConnector:
        if name not in cls._registry:
            raise ValueError(f"Unknown connector type: {name}")
        return cls._registry[name](**kwargs)


class TaskBuilderRegistry:
    """Registry mapping task builder names to implementations."""

    _registry: Dict[str, Type[BaseTaskBuilder]] = {}

    @classmethod
    def register(cls, name: str, tb_cls: Type[BaseTaskBuilder]):
        cls._registry[name] = tb_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTaskBuilder:
        if name not in cls._registry:
            raise ValueError(f"Unknown task builder type: {name}")
        return cls._registry[name](**kwargs)
