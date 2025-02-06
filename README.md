# VADE
Video Anomaly Detection with Embeddings

## Dependencies

This project requires Python>=3.12 and the packages listed in `requirements.txt`.

### Creating a Virtual Environment (Recommended)

```bash
conda create -n vade python=3.12
conda activate vade
pip install -r requirements.txt
```

### Running VADE
```bash
# Train VADE_CNN classification model (will utilize pre-generated frame descriptions if present)
python vade.py --train --dataset <ped2|avenue>

# Test VADE_CNN classification model (can be ran after training)
python vade.py --test --dataset <ped2|avenue>
```