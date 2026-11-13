"""Run SAM 3 text prompts locally and return one binary region per instance."""


class Sam3Concepts:
    def __init__(self, checkpoint):
        import torch
        from transformers import Sam3Config, Sam3Model, Sam3Processor

        if not (checkpoint / "config.json").is_file():
            raise ValueError(f"Thiếu checkpoint SAM 3 tại {checkpoint}.")
        config = Sam3Config.from_pretrained(checkpoint, local_files_only=True)
        self.model, loading = Sam3Model.from_pretrained(
            checkpoint, config=config, local_files_only=True, use_safetensors=True, output_loading_info=True
        )
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError("Checkpoint SAM 3 thiếu trọng số cho chế độ nhận diện bằng mô tả.")
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(checkpoint, local_files_only=True)
        self.torch = torch
        self.text_features = {}

    def predict(self, image, prompts, score_threshold=0.5):
        inputs = self.processor(images=image, return_tensors="pt")
        with self.torch.inference_mode():
            vision = self.model.get_vision_features(pixel_values=inputs.pixel_values.to(dtype=self.model.dtype))
            for prompt in prompts:
                if prompt not in self.text_features:
                    text = self.processor(text=prompt, return_tensors="pt")
                    self.text_features[prompt] = (self.model.get_text_features(
                        input_ids=text.input_ids, attention_mask=text.attention_mask, return_dict=True
                    ), text.attention_mask)
                features, attention_mask = self.text_features[prompt]
                outputs = self.model(vision_embeds=vision, text_embeds=features, attention_mask=attention_mask)
                result = self.processor.post_process_instance_segmentation(
                    outputs, threshold=score_threshold, mask_threshold=0.5, target_sizes=inputs.original_sizes.tolist()
                )[0]
                yield prompt, result["masks"].cpu().numpy()
