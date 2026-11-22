# Multimodal Search & Style Transfer for Yoga Poses

Image search over yoga poses with pre-trained CNN embeddings, a Qdrant vector database and a
reranker, plus a VGG-19 neural style-transfer study — built as a staged, config-driven pipeline.

> Work in progress — see [docs/learning-guide.md](docs/learning-guide.md) for the step-by-step build.

```bash
make install
python run_pipeline.py      # runs the stages switched on in config.yaml
```
