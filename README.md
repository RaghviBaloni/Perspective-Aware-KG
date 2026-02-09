# Perspective Enriched Social Media Arguments

Research codebase for constructing and evaluating perspective-aware knowledge graphs and classifiers that incorporate perspective annotations into TACO dataset.

## Requirements
- Python 3.10+
- Recommended: create and activate a virtual environment, then install dependencies:

    '''
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    '''

## Quick start
- Preprocess data:
    '''
    python src/data_preprocessing.py
    python src/data_stats.py
    '''

- Annotate the TACO dataset with perspective labels
    '''
    python src/annotation_pipeline.py
    python src/csv_conversion.py
    '''

- Evaluate CV on perspective enriched TACO 
    '''
    src/perspective_classifier.ipynb
    '''

- Build knowledge graph:
    '''
    pyhton bert_pers_embs.py
    python src/inductive_kg_construct.py
    '''

- Train / evaluate graph classifier:
    '''
    python src/inductive_graph classifier --model MLP/SAGE/GCN/GAT
    '''

- Train and evaluate vanilla TACO fine-tuned embeddings
    '''
    python src/baseline_embs.py
    python src/baseline_mlp.py
    '''

## Repository layout
- `data/` — raw and processed datasets, annotations, embeddings, outputs.  
- `results/` — experiment outputs (JSON).  
- `logs/` — run logs.  
- `src/` — main scripts and modules.  
- `requirements.txt` — Python dependencies.  

## Results & logs
- Check `results/` for experiment outputs and `logs/` for run logs and debugging output.

## Development notes
- Edit `src/config.py` to adjust local paths and settings before running scripts.  
- Recommended: create an isolated venv and install via `requirements.txt`.  
