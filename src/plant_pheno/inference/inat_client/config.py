from dataclasses import asdict, dataclass, field
from typing import Union


@dataclass
class EndpointConfig:
    endpoint: str
    fields: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    per_page: Union[int, None] = 200
    chunk_size: int = 200
    id_param: Union[str, None] = None
    api_version: int = 2
    url: str = field(init=False)
    id_fields: list[str] = field(default_factory=lambda: ["uuid", "id"])
    write_empty_rows: bool = False

    def __post_init__(self):
        print("post")
        if self.id_param is None:
            self.url = (
                f"https://api.inaturalist.org/v{self.api_version}/{self.endpoint}/"
            )
        else:
            self.url = (
                f"https://api.inaturalist.org/v{self.api_version}/{self.endpoint}"
            )

        # Convert fields dict to request format
        self.fields = f"({_fields_to_string(self.fields)})"

    def to_dict(self) -> dict:
        """Serialize config for logging to MLflow."""
        return asdict(self)


def _fields_to_string(fields_dict: dict, level=0):
    parts = []
    for key, value in fields_dict.items():
        if isinstance(value, dict):
            nested = _fields_to_string(value, level + 1)
            parts.append(f"{key}:({nested})")
        elif value is True:
            parts.append(f"{key}:!t")
    return ",".join(parts)
