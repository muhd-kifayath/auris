from openai import OpenAI
import os
from typing import List, Dict
import json
import numpy as np
from vector_store import VectorStore
from vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class QuestionGenerator:
    def __init__(self, model="gpt-4o-mini", retriever=None):
        self.model = model
        self.retriever = retriever

    def rag_query(self, topic: str):
        prompt = f"""
Generate a one-sentence summary of the topic: {topic}.
The summary should be concise and informative, suitable for a NEET exam context.
This will be used to retrieve relevant documents from a vector database.
Return the output as a single string.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    def generate_questions(self, topic: str, context: str, count: int = 5) -> List[Dict]:
        prompt = f"""
Using the context provided, generate {count} NEET-style multiple choice questions from previous years on the topic: {topic}.
Each question must include:
- 1 question
- 4 options (A to D)
- The correct option
- A short 2-line explanation

Context:
{context}

Return the output as a list of dictionaries like:
[
{{
"question": "What is the function of mitochondria?",
"options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
"answer": "C",
"explanation": "Mitochondria are the powerhouse of the cell. They produce ATP."
}},
...]

Do not include any other text or explanations.
Make sure to format the OUTPUT as LIST OF DICTIONARIES ONLY.
"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content

# Example usage
if __name__ == "__main__":

    topics = ["Biological classification", "Evolution", "Human Reproduction"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer = Vectorizer()

    retriever = VectorStore()
    gpt = QuestionGenerator(retriever=retriever)

    os.makedirs("evaluation", exist_ok=True)

    for topic in topics:
        print(f"\n--- Generating questions for topic: {topic} ---")
        summary = gpt.rag_query(topic)

        topic_slug = topic.lower().replace(" ", "_")

        # Without RAG
        questions_no_rag = gpt.generate_questions(topic, context="")
        questions_no_rag = eval(questions_no_rag.strip().strip("```json").strip("```"))
        with open(f"evaluation/{topic_slug}_without_rag.json", "w") as f:
            json.dump(questions_no_rag, f, indent=2)

        # With RAG
        query_embedding = np.array([vectorizer.model.encode(summary)], dtype=np.float32)
        query_embedding = normalize(query_embedding, axis=1)
        context = retriever.search(query_embedding, top_k=3)
        questions_with_rag = gpt.generate_questions(topic, context=context)
        questions_with_rag = eval(questions_with_rag.strip().strip("```json").strip("```"))
        with open(f"evaluation/{topic_slug}_with_rag.json", "w") as f:
            json.dump(questions_with_rag, f, indent=2)
