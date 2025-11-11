from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
import google.generativeai as genai


class Generator:
    """
    Gestisce la generazione della risposta usando il modello Gemini 2.0 Flash via API.

    Non serve scaricare nulla in locale:
    basta avere la chiave API impostata in GOOGLE_API_KEY.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Inizializza il client di Gemini.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "❌ GOOGLE_API_KEY non trovata. "
                "Imposta la variabile d'ambiente prima di usare il Generator."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def _build_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Costruisce un prompt chiaro e controllato per Gemini.
        """
        context = "\n\n---\n\n".join(context_chunks)

        prompt = (
            "Sei un assistente IA esperto nel DMV (motorizzazione civile).\n"
            "Devi rispondere in modo accurato e conciso alla domanda dell'utente, "
            "basandoti ESCLUSIVAMENTE sul contesto fornito. "
            "Se il contesto non contiene la risposta, spiega che non hai informazioni sufficienti.\n\n"
            f"Contesto:\n{context}\n\n"
            f"Domanda: {query}\n\n"
            "Risposta:"
        )

        return prompt

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Genera una risposta usando Gemini (modello via API).
        """
        prompt = self._build_prompt(query, context_chunks)

        # Chiamata al modello Gemini
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,     # equilibrio tra creatività e aderenza al contesto
                max_output_tokens=256,
                top_p=0.9,
                top_k=40,
            ),
        )

        # Estraggo il testo
        if response and response.text:
            return response.text.strip()
        else:
            return "⚠️ Nessuna risposta generata dal modello."


if __name__ == "__main__":
    # Esempio dimostrativo
    generator = Generator()

    # Simuliamo un retrieval di contesti dal DMV
    context_chunks = [
        "To update your address, you can log into the DMV online portal and submit a change of address form.",
        "Alternatively, you can visit a local DMV office with proof of your new residence.",
    ]

    query = "How do I update my address with the DMV?"
    answer = generator.generate_answer(query, context_chunks)

    print("🔎 Domanda:", query)
    print("\n📚 Contesto usato:")
    for c in context_chunks:
        print("-", c)
    print("\n🤖 Risposta generata da Gemini:")
    print(answer)
