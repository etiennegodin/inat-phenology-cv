import logging
from typing import Union

from .clients import BatchEndpointClient, ParametrizedEndpointClient, PhotoClient
from .config import EndpointConfig, PhotoConfig
from .protocols import (
    BinaryFetcherProtocol,
    BinaryWriterProtocol,
    JsonFetcherProtocol,
    JsonWriterProtocol,
)

logger = logging.getLogger(__name__)


def make_client(
    config: Union[EndpointConfig, PhotoConfig, None] = None,
    fetcher: Union[JsonFetcherProtocol, BinaryFetcherProtocol, None] = None,
    writer: Union[JsonWriterProtocol, BinaryWriterProtocol, None] = None,
) -> Union[BatchEndpointClient, ParametrizedEndpointClient, PhotoClient]:
    """Pick the right client based on the configuration and components.

    - PhotoConfig (or binary fetcher/writer) -> PhotoClient
    - EndpointConfig with id_param -> ParametrizedEndpointClient
    - EndpointConfig without id_param -> BatchEndpointClient
    """
    if isinstance(config, PhotoConfig) or (
        config is None and isinstance(fetcher, BinaryFetcherProtocol)
    ):
        cfg = config if isinstance(config, PhotoConfig) else PhotoConfig()
        return PhotoClient(cfg, fetcher, writer)

    if isinstance(config, EndpointConfig):
        if config.id_param is not None:
            return ParametrizedEndpointClient(config, fetcher, writer)
        return BatchEndpointClient(config, fetcher, writer)

    raise ValueError(f"Unsupported config or client type: {type(config)}")
