from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face generative model with litert-torch."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task",
        default="text_generation",
        help="litert-torch export task, for example text_generation or image_text_to_text.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prefill-lengths")
    parser.add_argument("--cache-length", type=int)
    parser.add_argument("--quantization-recipe")
    parser.add_argument("--externalize-embedder", action="store_true")
    parser.add_argument("--single-token-embedder", action="store_true")
    parser.add_argument("--split-cache", action="store_true")
    parser.add_argument("--cache-implementation")
    parser.add_argument("--auto-model-override")
    parser.add_argument("--use-jinja-template", action="store_true")
    parser.add_argument("--bundle-litert-lm", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--export-vision-encoder", action="store_true")
    parser.add_argument("--vision-encoder-quantization-recipe")
    parser.add_argument("--litert-lm-model-type")
    parser.add_argument("--litert-lm-metadata")
    parser.add_argument("--lightweight-conversion", action="store_true")
    args = parser.parse_args()

    if args.auto_model_override:
        import transformers

        getattr(transformers, args.auto_model_override)

    from litert_torch.generative.export_hf import export as export_module
    if args.auto_model_override:
        from litert_torch.generative.export_hf.core import export_lib

        getattr(export_lib.transformers, args.auto_model_override)
    if args.trust_remote_code:
        from litert_torch.generative.export_hf.core import export_lib

        _patch_trust_remote_code(export_lib.transformers.AutoImageProcessor)
        _patch_trust_remote_code(export_lib.transformers.AutoTokenizer)

    export_fn = getattr(export_module, "export", export_module)
    export_fn(
        model=args.model,
        output_dir=args.output_dir,
        task=args.task,
        trust_remote_code=args.trust_remote_code,
        prefill_lengths=_int_list(args.prefill_lengths),
        cache_length=args.cache_length,
        quantization_recipe=args.quantization_recipe,
        externalize_embedder=args.externalize_embedder or None,
        single_token_embedder=args.single_token_embedder or None,
        split_cache=args.split_cache or None,
        cache_implementation=args.cache_implementation,
        auto_model_override=args.auto_model_override,
        use_jinja_template=args.use_jinja_template or None,
        bundle_litert_lm=args.bundle_litert_lm or None,
        experimental_use_mixed_precision=args.mixed_precision or None,
        export_vision_encoder=args.export_vision_encoder or None,
        vision_encoder_quantization_recipe=args.vision_encoder_quantization_recipe,
        litert_lm_model_type_override=args.litert_lm_model_type,
        litert_lm_llm_metadata_override=args.litert_lm_metadata,
        experimental_lightweight_conversion=args.lightweight_conversion,
    )


def _int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _patch_trust_remote_code(auto_cls: object) -> None:
    original = getattr(auto_cls, "from_pretrained")
    if getattr(original, "_dmf_trust_remote_code", False):
        return

    def from_pretrained_with_trust(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("trust_remote_code", True)
        return original(*args, **kwargs)

    from_pretrained_with_trust._dmf_trust_remote_code = True  # type: ignore[attr-defined]
    setattr(auto_cls, "from_pretrained", from_pretrained_with_trust)


if __name__ == "__main__":
    main()
