from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plant_pheno.config import IngestPhotosParams
from plant_pheno.inference import InatInferenceClient
from plant_pheno.inference.inat_client import (
    BatchEndpointClient,
    BinaryFetcher,
    EndpointConfig,
    LocalBinaryWriter,
    ParametrizedEndpointClient,
    PhotoClient,
    PhotoConfig,
    make_client,
)


class DummyBinaryFetcher:
    def __init__(self, data: bytes = b"dummy_image_bytes"):
        self.data = data
        self.fetched_urls = []

    async def fetch(self, session, url: str) -> bytes:
        self.fetched_urls.append(url)
        return self.data


class DummyBinaryWriter:
    def __init__(self):
        self.written = []
        self.closed = False

    async def write(self, data: bytes, filename: str) -> None:
        self.written.append((data, filename))

    def close(self):
        self.closed = True


def test_make_client_dispatch_endpoint_batch():
    config = EndpointConfig(endpoint="observations")
    fetcher = MagicMock()
    writer = MagicMock()
    client = make_client(config, fetcher, writer)
    assert isinstance(client, BatchEndpointClient)


def test_make_client_dispatch_endpoint_parametrized():
    config = EndpointConfig(endpoint="observations", id_param="taxon_id")
    fetcher = MagicMock()
    writer = MagicMock()
    client = make_client(config, fetcher, writer)
    assert isinstance(client, ParametrizedEndpointClient)


def test_make_client_dispatch_photo():
    config = PhotoConfig(size="small")
    fetcher = DummyBinaryFetcher()
    writer = DummyBinaryWriter()
    client = make_client(config, fetcher, writer)
    assert isinstance(client, PhotoClient)
    assert client.config.size == "small"


def test_make_client_dispatch_photo_implicit():
    fetcher = BinaryFetcher(fallback_extensions=[".jpg"])
    writer = DummyBinaryWriter()
    client = make_client(None, fetcher, writer)
    assert isinstance(client, PhotoClient)


def test_make_client_invalid():
    with pytest.raises(ValueError):
        make_client(config=123, fetcher=None, writer=None)


def test_photo_config_from_params():
    params = IngestPhotosParams(extensions=[".png", ".jpg"], size="large")
    cfg = PhotoConfig.from_params(params)
    assert cfg.size == "large"
    assert cfg.extensions == [".png", ".jpg"]
    assert cfg.item_id_key == "photo_id"
    assert "photos" in cfg.base_url


@pytest.mark.anyio
async def test_photo_client_execute_ids():
    fetcher = DummyBinaryFetcher(b"img_data")
    writer = DummyBinaryWriter()
    config = PhotoConfig(
        size="medium",
        extensions=[".jpg"],
        base_url="https://test.photos",
    )
    client = PhotoClient(config, fetcher, writer)
    await client.execute([101, "102"])

    assert len(fetcher.fetched_urls) == 2
    assert "https://test.photos/101/medium.jpg" in fetcher.fetched_urls
    assert "https://test.photos/102/medium.jpg" in fetcher.fetched_urls
    assert len(writer.written) == 2
    assert (b"img_data", "101.jpg") in writer.written
    assert (b"img_data", "102.jpg") in writer.written


@pytest.mark.anyio
async def test_photo_client_execute_dict_records():
    fetcher = DummyBinaryFetcher(b"img_data")
    writer = DummyBinaryWriter()
    config = PhotoConfig(
        size="original",
        extensions=[".jpeg"],
        base_url="https://test.photos",
        item_id_key="photo_id",
    )
    client = PhotoClient(config, fetcher, writer)
    await client.execute([{"photo_id": 999, "other_col": "val"}])

    assert fetcher.fetched_urls == ["https://test.photos/999/original.jpeg"]
    assert writer.written == [(b"img_data", "999.jpeg")]


@pytest.mark.anyio
async def test_photo_client_execute_empty():
    fetcher = DummyBinaryFetcher()
    writer = DummyBinaryWriter()
    client = PhotoClient(PhotoConfig(), fetcher, writer)
    await client.execute([])
    assert len(fetcher.fetched_urls) == 0
    assert len(writer.written) == 0


def test_photo_client_context_manager():
    fetcher = DummyBinaryFetcher()
    writer = DummyBinaryWriter()
    with PhotoClient(PhotoConfig(), fetcher, writer) as client:
        assert client is not None
    assert writer.closed is True


@pytest.mark.anyio
async def test_local_binary_writer(tmp_path: Path):
    target_dir = tmp_path / "images"
    writer = LocalBinaryWriter(target_dir)

    with writer:
        await writer.write(b"content123", "subdir/test.jpg")

    saved_file = target_dir / "subdir" / "test.jpg"
    assert saved_file.exists()
    assert saved_file.read_bytes() == b"content123"


def test_inat_inference_client_format_ids():
    with patch.object(InatInferenceClient, "_load_model"):
        mock_params = MagicMock()
        client = InatInferenceClient(mock_params)

        # Single string normalization in format or list
        ids = client._format_observations_ids(
            ["https://www.inaturalist.org/observations/12345", "https://site/obs/67890"]
        )
        assert ids == [12345, 67890]


def test_inat_inference_client_download_photos():
    with patch.object(InatInferenceClient, "_load_model"):
        mock_params = MagicMock()
        mock_params.photo_target_dir = "/tmp/fake_dir"
        client = InatInferenceClient(mock_params)

        with patch("plant_pheno.inference.clients.make_client") as mock_make_client:
            mock_client_instance = MagicMock()
            mock_client_instance.execute = AsyncMock()
            mock_make_client.return_value = mock_client_instance

            photos_params = IngestPhotosParams(size="small")
            client.download_photos(
                [101, 102], "/tmp/fake_dir", rate=5, params=photos_params
            )

            assert mock_make_client.called
            config_arg = mock_make_client.call_args[0][0]
            assert isinstance(config_arg, PhotoConfig)
            assert config_arg.size == "small"
            assert mock_client_instance.execute.called
