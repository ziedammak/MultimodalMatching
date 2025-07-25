import torch
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

input_ids = torch.randint(0, 77, (1, 77))
pixel_values = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    (input_ids, pixel_values),
    "clip_model.onnx",
    input_names=["input_ids", "pixel_values"],
    output_names=["text_embeds", "image_embeds"],
    dynamic_axes={"input_ids": {0: "batch"}, "pixel_values": {0: "batch"}},
    opset_version=16
)
print("Exported to clip_model.onnx")
