from sentence_transformers import CrossEncoder

model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")

query = "What is the capital of China?"
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)
print(scores)
# [ 7.625 -11.375]

rankings = model.rank(query, documents)
print(rankings)
# [{'corpus_id': 0, 'score': 7.625}, {'corpus_id': 1, 'score': -11.375}]
