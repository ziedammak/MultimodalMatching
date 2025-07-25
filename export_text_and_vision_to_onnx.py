import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

# Text Encoder 
print("Exporting text encoder (MiniLM)...")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model.eval()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
dummy_input = tokenizer(["It is amazing!"], return_tensors="pt", padding="max_length", truncation=True, max_length=32)
input_ids = dummy_input["input_ids"]

torch.onnx.export(
    text_model,
    input_ids,
    "text_encoder.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch"}},
    opset_version=16
)

# Vision Encoder
print("Exporting vision encoder (CLIP Vision Tower)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Add projection layer to vision encoder
class VisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values)
        pooled_output = outputs[1]  # Pooler output
        return self.visual_projection(pooled_output)

vision_model = VisionWrapper(clip_model)
vision_model.eval()
dummy_vision = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    vision_model,
    dummy_vision,
    "vision_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["embedding"],
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=16
)

print(" Export complete: text_encoder.onnx and vision_encoder.onnx")