from app.retrieval import retrieve


QUERIES = [
    "What is overfitting in machine learning?",
    "How does backpropagation work?",
    "Explain gradient descent",
    "What is a transformer model?",
    "Difference between supervised and unsupervised learning",
    "What is retrieval-augmented generation?",
    "How does k-means clustering work?",
    "What is a word embedding?",

    # Outside our knowledge base
    "Who won the 2010 FIFA World Cup?",
    "What is the recipe for butter chicken?",
]


for q in QUERIES:
    print("=" * 80)
    print("Q:", q)

    for r in retrieve(q, k=3):
        print(
            f"  [{r['similarity']:.3f}] "
            f"{r['source']} "
            f"(chunk {r['chunk_index']})"
        )

        print(
            f"          {r['text'][:120]}..."
        )