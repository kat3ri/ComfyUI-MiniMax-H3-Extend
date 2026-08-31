"""The actual MiniMax H3 Video Extend node, backed by patch.py's PackedLayout/
extra_conds patches. See patch.py's module docstring for the full mechanism
and its README.md for verification status -- this is a real feature port,
reasoned through against the fork's actual model code, but not yet confirmed
against a live reference render.

Exposes two things:
  MiniMaxH3VideoExtendPatched  A normal, directly-draggable ComfyUI node
                                (classic dict-based API, for broad
                                compatibility with older ComfyUI versions).
  _inject_into_native()        Adds MiniMaxH3VideoExtend to
                                comfy_extras.nodes_minimax_h3's own namespace
                                (only if genuinely absent), so anything that
                                looks it up by that name there -- e.g.
                                ComfyUI-H3-Cast's H3CastToVideoExtend --
                                transparently finds a working implementation
                                without needing any changes of its own.
"""

import math

import torch


def _resolve_encode_ref_audio(native):
    """Return ComfyUI's reference-audio encoder across core layouts.

    ComfyUI 0.33.0 exposed the helper at module level, while 0.33.1 moved it
    onto ``MiniMaxH3ReferenceToVideo``. Resolve lazily so accessing a missing
    class attribute cannot break versions where the module-level helper is
    already available.
    """
    encode_ref_audio = getattr(native, "_encode_ref_audio", None)
    if callable(encode_ref_audio):
        return encode_ref_audio

    reference_node = getattr(native, "MiniMaxH3ReferenceToVideo", None)
    encode_ref_audio = getattr(reference_node, "_encode_ref_audio", None)
    if callable(encode_ref_audio):
        return encode_ref_audio

    raise AttributeError(
        "ComfyUI exposes no compatible MiniMax H3 reference-audio encoder "
        "(expected comfy_extras.nodes_minimax_h3._encode_ref_audio or "
        "MiniMaxH3ReferenceToVideo._encode_ref_audio)"
    )


def _context_span(n_frames):
    """Cursor-axis duration spanned by n_frames trailing latent frames ending
    at a target origin -- pure position math, no model weights involved."""
    import comfy.ldm.minimax.model as h3model
    return sum(h3model.FRAME_RESCALE * h3model.FRAME_PER_TOKEN[k % 5] for k in range(-n_frames, 0))


def _context_keyframes(context_latent, context_frames):
    """Build context/context_audio keyframe dicts continuing a prior
    generation's AV latent, plus the (width, height) it implies. Simplified
    from the fork's version: no static_time (that's specifically for a
    spatial/world reference, not "continue a real prior clip"), no aug/
    noise-augmentation strength override."""
    ctx_samples = context_latent["samples"]
    is_av = ctx_samples.is_nested
    ctx_video = ctx_samples.tensors[0] if is_av else ctx_samples
    ctx_audio = ctx_samples.tensors[1] if is_av else None
    if ctx_video.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    ctx_t, ctx_h, ctx_w = ctx_video.shape[2], ctx_video.shape[3], ctx_video.shape[4]
    n_frames = min(context_frames, ctx_t)
    width, height = ctx_w * 16, ctx_h * 16  # inherit the source clip's canvas exactly

    keyframes = [{"kind": "context", "num_frames": n_frames,
                  "latent": ctx_video[:, :, ctx_t - n_frames:, :, :]}]
    if ctx_audio is not None:
        ctx_audio_t = ctx_audio.shape[-1]
        n_audio_frames = min(round(_context_span(n_frames)), ctx_audio_t)
        if n_audio_frames > 0:
            keyframes.append({"kind": "context_audio", "num_frames": n_audio_frames,
                              "audio_latent": ctx_audio[:, :, :, ctx_audio_t - n_audio_frames:]})
    return width, height, keyframes


def _pin_last_context_frame(vae, context_latent, width, height, resize_fn):
    """Decode context_latent's true trailing pixel frame and return a
    zero-RoPE-distance first_frame-style keyframe for it -- removes the
    ambiguity of context_frames alone only carrying whole latent frames
    (each spans 1-4 pixel frames), which otherwise can show up as the
    continuation re-playing a moment that already happened."""
    ctx_samples = context_latent["samples"]
    ctx_video = ctx_samples.tensors[0] if ctx_samples.is_nested else ctx_samples
    ctx_t = ctx_video.shape[2]
    tail = ctx_video[:, :, max(0, ctx_t - 6):, :, :]
    decoded = vae.decode(tail)
    img = resize_fn(decoded[:, -1], width, height, "disabled")
    return {"resolved_frame_index": 0, "image": img}


def _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                       ref_images, ref_videos, ref_video_audios, ref_audios):
    """Verbatim copy of stock MiniMaxH3ReferenceToVideo.execute()'s own
    inline ref-block-building logic, refactored into a standalone function so
    MiniMaxH3VideoExtendPatched can reuse it exactly -- stock doesn't expose
    this as a shared helper the way the fork does."""
    import comfy_extras.nodes_minimax_h3 as native

    CANVAS_MULTIPLE = 32
    REF_IMAGE_SHORT_EDGE = 2048
    FPS = 24
    encode_ref_audio = _resolve_encode_ref_audio(native)

    ref_items = []
    ref_blocks = []

    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = native._resize(img[:1], tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = native.adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = native._resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    return ref_items, ref_blocks


def _execute(clip, vae, context_latent, prompt, length, context_frames=2, pin_last_frame=True,
             first_frame=None, last_frame=None, audio_vae=None, ref_image_size="match", ref_images=None,
             ref_videos=None, ref_video_audios=None, ref_audios=None):
    import node_helpers
    import comfy_extras.nodes_minimax_h3 as native

    width, height, keyframes = _context_keyframes(context_latent, context_frames)
    if first_frame is not None:
        img = native._resize(first_frame[:1], width, height, "disabled")
        keyframes.append({"resolved_frame_index": 0, "image": img})
    elif pin_last_frame:
        keyframes.append(_pin_last_context_frame(vae, context_latent, width, height, native._resize))
    latent, frame_count = native._empty_av_latent(width, height, length)
    if last_frame is not None:
        # aspect-preserving cover-crop ("follower"), same convention stock's
        # own MiniMaxH3ImageToVideo uses for its last_frame -- distinct from
        # first_frame's plain stretch ("geometry anchor")
        img = native._resize(last_frame[:1], width, height, "center")
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    ref_items, ref_blocks = ([], [])
    if any((ref_images, ref_videos, ref_audios)):
        if audio_vae is None and (ref_video_audios or ref_audios):
            raise ValueError("audio_vae is required when ref_video_audios or ref_audios are supplied")
        ref_items, ref_blocks = _build_ref_blocks(vae, audio_vae, width, height, frame_count, ref_image_size,
                                                   ref_images, ref_videos, ref_video_audios, ref_audios)

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)

    for kf in keyframes:
        if "image" in kf:
            kf["latent"] = vae.encode(kf.pop("image"))

    values = {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count}
    if ref_blocks:
        values["minimax_refs"] = ref_blocks
    cond = node_helpers.conditioning_set_values(cond, values)
    return cond, latent


class MiniMaxH3VideoExtendPatched:
    """Continue a prior MiniMax H3 clip from its trailing latent frames,
    optionally alongside character/background references -- backported onto
    stock ComfyUI via patch.py. See this pack's README.md for verification
    status before trusting output from this for real work."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "context_latent": ("LATENT", {"tooltip": "AV latent output from a prior MiniMax H3 generation to continue from"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "New frame count at 24 fps for the continuation only (excludes context_frames)"}),
                "context_frames": ("INT", {"default": 2, "min": 1, "max": 64,
                                           "tooltip": "Trailing latent frames of context_latent carried over as context"}),
                "pin_last_frame": ("BOOLEAN", {"default": True,
                                               "tooltip": "Decode context_latent's true trailing pixel frame and pin it as this call's frame 0. Ignored if first_frame is connected."}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "first_frame": ("IMAGE", {"tooltip": "Hard-pin this call's frame 0 to an exact image (e.g. the prior clip's real last output frame) instead of pin_last_frame's decode"}),
                "last_frame": ("IMAGE", {"tooltip": "Pin this continuation segment's own final frame to an exact image -- e.g. to land precisely on a known next shot/keyframe instead of leaving the ending fully generated. Not part of the native fork's VideoExtend node (which has no end-anchor at all); added here since it's a natural, low-risk extension of the same PackedLayout mechanism first_frame/context already use."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "ref_images": ("IMAGE", {"tooltip": "Reference image(s) -- connect a batch (e.g. via ImageBatch) for more than one; each frame becomes its own <Picture i> reference. Not the native node's per-slot Autogrow inputs -- this is a single batched socket."}),
                "ref_audio": ("AUDIO", {"tooltip": "One standalone reference audio clip"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "model/conditioning/minimax"
    DESCRIPTION = "Continue a prior MiniMax H3 clip from its trailing latent frames (backported, see README.md)."

    def run(self, clip, vae, context_latent, prompt, length, context_frames=2, pin_last_frame=True,
            audio_vae=None, first_frame=None, last_frame=None, ref_image_size="match", ref_images=None, ref_audio=None):
        # this classic dict-based node has no equivalent to the native node's
        # Autogrow (numbered ref_image_0/1/2... slots bundled into a dict
        # before execute() ever sees them) -- confirmed 2026-08-11 that a
        # plain IMAGE socket arrives as a raw tensor, not a dict, so any()
        # over it blows up on ambiguous multi-element truthiness. Wrap
        # whatever's connected into the dict shape _execute()/_build_ref_blocks
        # actually expect, splitting a batch into one ref_image_N per frame.
        ref_images_dict = None
        if ref_images is not None:
            ref_images_dict = {f"ref_image_{i + 1}": ref_images[i:i + 1] for i in range(ref_images.shape[0])}
        ref_audios_dict = {"ref_audio_1": ref_audio} if ref_audio is not None else None

        cond, latent = _execute(
            clip, vae, context_latent, prompt, length, context_frames=context_frames,
            pin_last_frame=pin_last_frame, first_frame=first_frame, last_frame=last_frame, audio_vae=audio_vae,
            ref_image_size=ref_image_size, ref_images=ref_images_dict, ref_audios=ref_audios_dict,
        )
        return (cond, latent)


def _encode_ref_audio_for_context(audio_vae, audio):
    """Vendored from the native nodes_minimax_h3.py's module-level
    _encode_ref_audio -- see MiniMaxH3EncodeAVPatched's docstring for why."""
    import torchaudio

    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


class MiniMaxH3EncodeAVPatched:
    """VAE-encode video frames (+ optional audio) into a MiniMax H3 AV latent
    (NestedTensor pair) -- feeds MiniMaxH3VideoExtendPatched's context_latent
    input from externally-sourced footage (e.g. VHS_LoadVideo). Vendored from
    the native MiniMaxH3EncodeAV node, which isn't part of stock/public
    MiniMax H3 support either (only kat3ri/ComfyUI's fork) -- lives here
    rather than in ComfyUI-H3-Cast since it's an extend/continuation concern,
    not a cast/character one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "images": ("IMAGE", {"tooltip": "Video frames at 24 fps"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "model/latent/minimax"
    DESCRIPTION = "VAE-encode video frames (+ optional audio) into a MiniMax H3 AV latent (NestedTensor pair)."

    def run(self, vae, images, audio_vae=None, audio=None):
        import comfy.nested_tensor

        video_z = vae.encode(images[..., :3])
        if audio is None:
            return ({"samples": video_z},)
        if audio_vae is None:
            raise ValueError("audio_vae is required when audio is supplied")
        audio_z, _ = _encode_ref_audio_for_context(audio_vae, audio)
        return ({"samples": comfy.nested_tensor.NestedTensor((video_z, audio_z))},)


class _NativeShim:
    """Matches the calling convention H3CastToVideoExtend (ComfyUI-H3-Cast)
    already uses against the real native node -- a classmethod `execute`
    taking the same kwargs, returning a plain (conditioning, latent) tuple.
    Unsupported kwargs (context_strength, world_latent, etc.) are already
    dropped gracefully by the caller's own inspect.signature check before
    this is ever invoked, so this only needs to declare what it truly
    supports."""

    @classmethod
    def execute(cls, clip, vae, audio_vae, context_latent, prompt, length, context_frames=2,
                pin_last_frame=True, first_frame=None, last_frame=None, ref_image_size="match",
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        return _execute(
            clip, vae, context_latent, prompt, length, context_frames=context_frames,
            pin_last_frame=pin_last_frame, first_frame=first_frame, last_frame=last_frame, audio_vae=audio_vae,
            ref_image_size=ref_image_size, ref_images=ref_images, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios,
        )


def inject_into_native():
    """Adds MiniMaxH3VideoExtend/MiniMaxH3EncodeAV to
    comfy_extras.nodes_minimax_h3's own namespace, only if genuinely absent,
    so anything that looks either up by name there (e.g. ComfyUI-H3-Cast's
    H3CastToVideoExtend) finds a working implementation transparently --
    never overrides a real native class that's already there."""
    import comfy_extras.nodes_minimax_h3 as native
    if not hasattr(native, "MiniMaxH3VideoExtend"):
        native.MiniMaxH3VideoExtend = _NativeShim
    if not hasattr(native, "MiniMaxH3EncodeAV"):
        native.MiniMaxH3EncodeAV = MiniMaxH3EncodeAVPatched


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3VideoExtendPatched": MiniMaxH3VideoExtendPatched,
    "MiniMaxH3EncodeAVPatched": MiniMaxH3EncodeAVPatched,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3VideoExtendPatched": "MiniMax H3 Video Extend (Backported)",
    "MiniMaxH3EncodeAVPatched": "MiniMax H3 Encode AV (Backported)",
}
