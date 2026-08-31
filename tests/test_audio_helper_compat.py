from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_nodes_module():
    spec = importlib.util.spec_from_file_location("minimax_h3_extend_nodes", ROOT / "nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nodes = load_nodes_module()


class ExplodingReferenceNode:
    def __getattribute__(self, name):
        raise AssertionError(f"class-level fallback was evaluated eagerly: {name}")


class ResolveEncodeRefAudioTests(unittest.TestCase):
    def test_prefers_comfyui_0330_module_level_helper_without_eager_fallback(self):
        def module_helper(audio_vae, audio):
            return audio_vae, audio

        native = SimpleNamespace(
            _encode_ref_audio=module_helper,
            MiniMaxH3ReferenceToVideo=ExplodingReferenceNode(),
        )

        self.assertIs(nodes._resolve_encode_ref_audio(native), module_helper)

    def test_falls_back_to_comfyui_0331_class_level_helper(self):
        def class_helper(audio_vae, audio):
            return audio_vae, audio

        native = SimpleNamespace(
            MiniMaxH3ReferenceToVideo=SimpleNamespace(_encode_ref_audio=class_helper)
        )

        self.assertIs(nodes._resolve_encode_ref_audio(native), class_helper)

    def test_falls_back_when_module_attribute_exists_but_is_not_callable(self):
        def class_helper(audio_vae, audio):
            return audio_vae, audio

        native = SimpleNamespace(
            _encode_ref_audio=None,
            MiniMaxH3ReferenceToVideo=SimpleNamespace(_encode_ref_audio=class_helper),
        )

        self.assertIs(nodes._resolve_encode_ref_audio(native), class_helper)

    def test_fails_with_actionable_error_when_both_layouts_are_missing(self):
        with self.assertRaisesRegex(AttributeError, "no compatible MiniMax H3"):
            nodes._resolve_encode_ref_audio(SimpleNamespace())

    def test_image_only_ref_blocks_work_with_comfyui_0331_class_layout(self):
        def class_helper(audio_vae, audio):
            raise AssertionError("image-only references must not encode audio")

        native = ModuleType("comfy_extras.nodes_minimax_h3")
        native.MiniMaxH3ReferenceToVideo = SimpleNamespace(_encode_ref_audio=class_helper)
        native._resize = lambda image, width, height, mode: image

        comfy_extras = ModuleType("comfy_extras")
        comfy_extras.__path__ = []
        comfy_extras.nodes_minimax_h3 = native

        vae = SimpleNamespace(encode=lambda image: "encoded-image")
        image = nodes.torch.zeros((1, 64, 64, 3))

        with patch.dict(
            sys.modules,
            {
                "comfy_extras": comfy_extras,
                "comfy_extras.nodes_minimax_h3": native,
            },
        ):
            ref_items, ref_blocks = nodes._build_ref_blocks(
                vae=vae,
                audio_vae=None,
                width=64,
                height=64,
                frame_count=124,
                ref_image_size="match",
                ref_images={"ref_image_1": image},
                ref_videos=None,
                ref_video_audios=None,
                ref_audios=None,
            )

        self.assertEqual(ref_items[0]["type"], "image")
        self.assertEqual(ref_blocks[0]["kind"], "image")
        self.assertEqual(ref_blocks[0]["latent"], "encoded-image")


if __name__ == "__main__":
    unittest.main()
