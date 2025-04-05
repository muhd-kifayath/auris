import openai

class GPTHelper:
    def __init__(self, api_key):
        self.api_key = api_key

    def ask_gpt(self, query, retrieved_docs):
        """Send query + retrieved documents to GPT and return a refined answer."""
        context = "\n\n".join(retrieved_docs)
        prompt = f"Based on the following documents, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            api_key=self.api_key,
        )
        return response["choices"][0]["message"]["content"]
