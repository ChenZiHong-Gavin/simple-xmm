import importlib
from typing import Dict, Any


class _LazyClass:
    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._cls = None

    def _get_cls(self):
        if self._cls is None:
            if self.module_name.startswith("."):
                package = __package__
            else:
                package = None
            module = importlib.import_module(self.module_name, package=package)
            self._cls = getattr(module, self.class_name)
        return self._cls

    def __call__(self, *args, **kwargs):
        cls = self._get_cls()
        return cls(*args, **kwargs)

    def __repr__(self):
        return f"<LazyClass {self.module_name}.{self.class_name}>"


class _LazyDict(dict):
    def __init__(self, mapping: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.mapping = mapping
        self._loaded = {}

    def __getitem__(self, key: str):
        if key not in self.mapping:
            raise KeyError(key)

        if key not in self._loaded:
            sub_map = self.mapping[key]
            self._loaded[key] = {k: _LazyClass(m, c) for k, (m, c) in sub_map.items()}
        return self._loaded[key]

    def __contains__(self, key):
        return key in self.mapping

    def items(self):
        for k in self.mapping:
            yield k, self[k]

    def keys(self):
        return self.mapping.keys()

    def values(self):
        for k in self.mapping:
            yield self[k]


_PROCESSORS_MAPPING = {
    "image": {
        "clip": (".image.clip_image_processor", "ImageModalProcessor"),
    },
    "audio": {
        "whisper": (".audio.whisper_audio_processor", "AudioModalProcessor"),
    },
    "protein": {
        "esm": (".protein.esm_protein_processor", "ProteinModalProcessor"),
    },
}

_ENCODERS_MAPPING = {
    "image": {
        "clip": (".image.clip_image_encoder", "ImageModalEncoder"),
    },
    "audio": {
        "whisper": (".audio.whisper_audio_encoder", "AudioModalEncoder"),
    },
    "protein": {
        "esm": (".protein.esm_protein_processor", "ProteinModalEncoder"),
    },
}

MODALITY_PROCESSORS = _LazyDict(_PROCESSORS_MAPPING)
MODALITY_ENCODERS = _LazyDict(_ENCODERS_MAPPING)
