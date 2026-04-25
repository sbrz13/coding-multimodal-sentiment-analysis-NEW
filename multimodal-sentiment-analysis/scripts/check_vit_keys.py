from transformers import AutoModel

vit = AutoModel.from_pretrained("google/vit-base-patch16-224")
print("ViT state_dict keys:")
for key in sorted(vit.state_dict().keys()):
    print(f"  {key}")
