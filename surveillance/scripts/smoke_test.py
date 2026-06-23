import asyncio
from PIL import Image

from services.embedding.core.config import Settings
from services.embedding.services.clip_model import CLIPEmbedder


async def main():
    settings = Settings(CLIP_MODEL_NAME="openai/clip-vit-base-patch32", CLIP_DEVICE="cpu")
    embedder = CLIPEmbedder(settings)
    embedder.load()  # downloads the model from Hugging Face on first run

    # any local image works — swap in a real path
    Image.new("RGB", (224, 224), color="red").save("/tmp/sample.jpg")
    vector = await embedder.embed_image("/tmp/sample.jpg")

    print(f"vector length: {len(vector)}")
    print(f"L2 norm (should be ~1.0): {sum(v * v for v in vector) ** 0.5:.6f}")


asyncio.run(main())