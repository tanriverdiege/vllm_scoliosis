"""Thin wrappers around HuggingFace vision-language models.

Subclass `VLM` and implement `format_prompt` to support a new model family --
that is the only part that genuinely differs between VLMs.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlm_config import VLMConfig


@dataclass
class VLMOutput:
    """One generation, plus whatever diagnostics were requested.

    Attributes:
        text: The decoded reply, with the prompt echo removed.
        sequences: Generated token ids, prompt echo already sliced off.
        scores: Per-step distributions *after* logits processors (temperature,
            top-k, top-p). None unless `output_scores` was set.
        logits: Per-step *raw* model outputs, before any processing. None
            unless `output_logits` was set. Use these to reason about the
            model itself; use `scores` to reason about what sampling saw.
        transition_scores: Log-probability of each token the model actually
            chose, shape (batch, generated_len). None unless `scores` exist.
        pad_token_id: Used to drop padding from confidence statistics. In a
            batch, every sample that finishes before the longest one is padded,
            and those positions carry meaningless scores (1.0 or 0.0).
    """

    text: str
    sequences: torch.Tensor
    scores: tuple[torch.Tensor, ...] | None = None
    logits: tuple[torch.Tensor, ...] | None = None
    transition_scores: torch.Tensor | None = None
    pad_token_id: int | None = None

    def token_confidences(self, skip_padding: bool = True) -> list[tuple[int, float]]:
        """Per-token probability of the chosen token.

        Args:
            skip_padding: Drop trailing pad positions, whose scores are an
                artefact of batching rather than real model confidence.

        Returns:
            (token_id, probability) pairs, one per generated token. Empty if
            generation ran without `output_scores`.
        """
        if self.transition_scores is None:
            return []
        probs = self.transition_scores[0].exp().tolist()
        ids = self.sequences[0].tolist()
        # strict=True: a length mismatch means the slicing and the transition
        # scores disagree, which should surface rather than silently truncate.
        pairs = list(zip(ids, probs, strict=True))
        if skip_padding and self.pad_token_id is not None:
            pairs = [(i, p) for i, p in pairs if i != self.pad_token_id]
        return pairs

    def mean_confidence(self) -> float | None:
        """Mean probability across generated tokens, excluding padding.

        Returns:
            The mean, or None if scores were not collected.
        """
        pairs = self.token_confidences(skip_padding=True)
        if not pairs:
            return None
        return sum(p for _, p in pairs) / len(pairs)


class VLM:
    """Base wrapper: loading, generation, and decoding are shared."""

    def __init__(self, config: VLMConfig) -> None:
        """Load the processor and model.

        Args:
            config: Every generation and loading argument, in one object.
                Build it in code or read it from a file with
                `VLMConfig.from_yaml`; see `config.yaml` for the file form.
        """
        self.config = config
        self.device = self._pick_device(config.device)
        # fp16 only pays off on CUDA; on CPU most half-precision kernels are
        # missing or far slower than fp32. bf16 can also be used on CUDA,
        # but is not supported by all models.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name, dtype=self.dtype
        ).to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _pick_device(self, requested: str | None) -> str:
        """Resolve the torch device to run on.

        Args:
            requested: An explicit device string, or None to auto-detect.

        Returns:
            The device string to use.
        """
        if requested is not None:
            return requested
        if torch.cuda.is_available():
            return "cuda"
        # MPS needs macOS 14+; is_available() is False on 13.x even on Apple silicon.
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_image(self, path: str) -> Image.Image:
        """Open an image and convert it to RGB.

        Args:
            path: Filesystem path to the image.

        Returns:
            The image in RGB mode.
        """
        return Image.open(path).convert("RGB")

    def format_prompt(self, prompts: list[str], images: list[Image.Image]) -> str:
        """Build the model-specific prompt string.

        Args:
            prompts: Text segments to include, in order.
            images: Images the prompt refers to.

        Returns:
            The formatted prompt, ready to pass to the processor as `text`.

        Raises:
            NotImplementedError: Always; subclasses must override this because
                the prompt format is model-specific.
        """
        raise NotImplementedError("Subclasses must implement format_prompt.")

    def __call__(self, images_paths: list[str], prompts: list[str]) -> VLMOutput:
        """Run the model on one or more images.

        Args:
            images_paths: Paths to the images to show the model.
            prompts: Text segments, applied after the images so that the
                question can attend to them under causal masking.

        Returns:
            The reply plus any requested diagnostics.
        """
        images = [self.load_image(path) for path in images_paths]
        chat = self.format_prompt(prompts=prompts, images=images)
        inputs = self.processor(text=chat, images=images, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            # do_sample=False is argmax over the softmaxed logits at each step
            # (deterministic); do_sample=True draws from the distribution.
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                return_dict_in_generate=self.config.return_dict_in_generate,
                output_scores=self.config.output_scores,
                output_logits=self.config.output_logits,
            )

        # With return_dict_in_generate the result is a ModelOutput, not a
        # tensor, so the token ids live under .sequences. Narrow on the actual
        # type rather than on the flag: generation_config can switch to the
        # dict form on its own, and this stays correct if it does.
        if isinstance(generated, torch.Tensor):
            sequences: torch.Tensor = generated
            scores = None
            logits = None
        else:
            sequences = generated.sequences
            scores = getattr(generated, "scores", None)
            logits = getattr(generated, "logits", None)

        # Transition scores must be computed against the FULL sequences, before
        # the prompt echo is sliced off -- the method derives the generated
        # span itself from the input length.
        transition_scores = None
        if scores is not None:
            transition_scores = self.model.compute_transition_scores(
                sequences, scores, normalize_logits=True
            )

        # Decoder-only models echo the prompt in their output; encoder-decoder
        # models (BLIP-2 + flan-t5, for instance) do not. Slicing by input
        # length is robust for the former and wrong for the latter.
        if not self.model.config.is_encoder_decoder:
            sequences = sequences[:, inputs["input_ids"].shape[1] :]

        decoded: list[str] = self.processor.batch_decode(
            sequences, skip_special_tokens=True
        )
        return VLMOutput(
            text=decoded[0].strip(),
            sequences=sequences,
            scores=scores,
            logits=logits,
            transition_scores=transition_scores,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

    def batch(
        self,
        images_paths: list[list[str]],
        prompts: list[list[str]],
    ) -> list[VLMOutput]:
        """Run several independent image/prompt pairs in one forward pass.

        Each entry of `images_paths` and `prompts` is one sample. For the
        common one-image-one-question case, pass `[["a.png"], ["b.png"]]` and
        `[["Q1"], ["Q2"]]`.

        Args:
            images_paths: Per-sample lists of image paths.
            prompts: Per-sample lists of text segments.

        Returns:
            One VLMOutput per sample, in input order.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(images_paths) != len(prompts):
            raise ValueError(
                f"got {len(images_paths)} image groups but {len(prompts)} prompt "
                "groups; they must correspond one-to-one"
            )

        # Nested list is required: a flat list[Image] is read as ONE prompt
        # holding several images, not as a batch.
        batch_images = [[self.load_image(p) for p in paths] for paths in images_paths]
        chats = [
            self.format_prompt(prompts=texts, images=imgs)
            for texts, imgs in zip(prompts, batch_images, strict=True)
        ]

        # Decoder-only generation must pad on the LEFT. With right padding the
        # model continues from pad tokens and emits garbage. The tokenizer
        # defaults to "right", so override it and restore afterwards rather
        # than mutating shared state for the rest of the process.
        tokenizer = self.processor.tokenizer
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = self.processor(
                text=chats, images=batch_images, return_tensors="pt", padding=True
            ).to(self.device)
        finally:
            tokenizer.padding_side = previous_side

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                return_dict_in_generate=self.config.return_dict_in_generate,
                output_scores=self.config.output_scores,
                output_logits=self.config.output_logits,
            )

        if isinstance(generated, torch.Tensor):
            sequences: torch.Tensor = generated
            scores = None
            logits = None
        else:
            sequences = generated.sequences
            scores = getattr(generated, "scores", None)
            logits = getattr(generated, "logits", None)

        transition_scores = None
        if scores is not None:
            transition_scores = self.model.compute_transition_scores(
                sequences, scores, normalize_logits=True
            )

        # Left padding makes every row's prompt the same length, so one slice
        # point is correct for the whole batch.
        if not self.model.config.is_encoder_decoder:
            sequences = sequences[:, inputs["input_ids"].shape[1] :]

        decoded: list[str] = self.processor.batch_decode(
            sequences, skip_special_tokens=True
        )
        return [
            VLMOutput(
                text=text.strip(),
                sequences=sequences[i : i + 1],
                scores=tuple(s[i : i + 1] for s in scores) if scores else None,
                logits=tuple(g[i : i + 1] for g in logits) if logits else None,
                transition_scores=(
                    None if transition_scores is None else transition_scores[i : i + 1]
                ),
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            for i, text in enumerate(decoded)
        ]


class HuggingFaceVLM(VLM):
    """Models whose processor exposes a chat template (SmolVLM, Qwen-VL, ...)."""

    def format_prompt(self, prompts: list[str], images: list[Image.Image]) -> str:
        """Render the chat template for these images and text segments.

        One image entry is emitted per image, so the number of `<image>`
        placeholders always matches the number of images the processor
        encodes. Text follows the images so it can attend to them.

        Args:
            prompts: Text segments; joined with spaces, since the template
                concatenates adjacent text blocks with no separator.
            images: Images to reference.

        Returns:
            The rendered prompt string.
        """
        content: list[dict[str, str]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": " ".join(prompts)})

        messages = [{"role": "user", "content": content}]
        chat: str = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return chat


if __name__ == "__main__":
    # The below is a simple unit test to verify that the VLM class works as expected.
    vlm = HuggingFaceVLM(VLMConfig.from_yaml("configs/base_config.yaml"))

    prompts = [
        ["What do you see in the image ?"],
        ["What do you see in the image ?"],
        ["What do you see in the image ?"],
    ]
    image_paths = [
        ["../data/example_image1.png"],
        ["../data/example_image2.png"],
        ["../data/example_image3.png"],
    ]

    # Single inference
    single_out = vlm(images_paths=image_paths[0], prompts=prompts[0])

    # Batched inference
    batched_out = vlm.batch(images_paths=image_paths, prompts=prompts)

    print("*****Single inference output******")
    print(single_out.text)
    print("*****Batched inference output (X-Ray)******")
    print(batched_out[0].text)
    print("*****Batched inference output (Cat)******")
    print(batched_out[1].text)
    print("*****Batched inference output (Dog)******")
    print(batched_out[2].text)
