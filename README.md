# GenreGraph: Discovering Hidden Genre Communities in Netflix Shows

This project builds a graph-based machine learning model that analyzes Netflix shows to discover hidden genre communities. It combines natural language processing (NLP), graph theory, and clustering to create a content-based recommendation engine.

## Overview

Genres on streaming platforms are often broad or inconsistent. This project aims to uncover hidden relationships between shows using:

- Semantic similarity from show descriptions
- Connections through shared cast and directors
- Graph embeddings (Node2Vec) and clustering

It also includes a recommendation engine that suggests similar shows based on graph structure and content similarity.

## Workflow

### 1. Data Preparation
- Dataset: [Netflix Movies and TV Shows on Kaggle](https://www.kaggle.com/datasets/muhammadtahir194/netflix-movies-and-tv-shows-dataset)
- Cleaned to remove rows without cast or description
- Cast and genres are converted to lists

### 2. Text Embeddings
- Descriptions are embedded using `all-MiniLM-L6-v2` from Sentence Transformers
- Cosine similarity is used to compare show content

### 3. Graph Construction
- Each show is a node
- Edges are added between nodes that:
  - Have similar descriptions
  - Share actors or directors
- Similarity scores and metadata are stored as edge attributes

### 4. Graph Embeddings (Node2Vec)
- Learns vector representations of shows based on graph structure
- Captures both direct and indirect relationships

### 5. Clustering
- KMeans is used to group similar shows based on embeddings
- Each show is assigned to a cluster (interpreted as a genre community)

### 6. Recommendation Engine
- Given a show, finds others in the same cluster
- Ranks them by cosine similarity in the embedding space

## Example Usage
recommend("Stranger Things", top_k=5)

## Files
- netflix_cleaned.csv: Cleaned dataset
- description_embeddings.npy: Text embeddings from BERT
- graph_embeddings.npy: Node2Vec embeddings
- clustered_netflix.csv: Final dataset with cluster assignments
- recommend.py: Script to recommend similar shows
