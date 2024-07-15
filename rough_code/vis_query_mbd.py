import pacmap
import numpy as np
import plotly.express as px
import pandas as pd


def plot_knowledge_base(docs_processed, prompt, KNOWLEDGE_VECTOR_DATABASE):

    embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)

    embeddings_2d = [list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))] + [query_vector]

    # fit the data (The index of transformed data corresponds to the index of the original data)
    documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")
    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": docs_processed[i].metadata["source"].split("/")[1],
                "extract": docs_processed[i].page_content[:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(docs_processed))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "User query",
                "extract": prompt,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )

    # visualize the embedding
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,)
    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),)
    fig.update_layout(
        legend_title_text="<b>Chunk source</b>",
        title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",)
    fig.show()