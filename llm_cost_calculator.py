from typing import Dict, Tuple
import tiktoken
from dataclasses import dataclass

@dataclass
class LLMProvider:
    name: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    model: str

class LLMCostCalculator:
    def __init__(self):
        # Fiyatlar USD cinsinden, 1000 token başına
        self.providers = {
            "gemini-2.0-flash": LLMProvider(
                name="Google Gemini 2.0 Flash",
                input_cost_per_1k_tokens=0.00025,  # $0.00025 per 1K input tokens
                output_cost_per_1k_tokens=0.0005,  # $0.0005 per 1K output tokens
                model="gemini-2.0-flash"
            ),
            "gemini-1.5-flash-latest": LLMProvider(
                name="Google Gemini 1.5 Flash Latest",
                input_cost_per_1k_tokens=0.0001,   # $0.0001 per 1K input tokens
                output_cost_per_1k_tokens=0.0002,  # $0.0002 per 1K output tokens
                model="gemini-1.5-flash-latest"
            )
        }
        
        # Gemini modelleri için tokenizer
        self.encoders = {
            "gemini-2.0-flash": tiktoken.encoding_for_model("gpt-3.5-turbo"),  # Gemini için GPT-3.5 tokenizer'ı kullanıyoruz
            "gemini-1.5-flash-latest": tiktoken.encoding_for_model("gpt-3.5-turbo")  # Gemini için GPT-3.5 tokenizer'ı kullanıyoruz
        }

    def count_tokens(self, text: str, model: str) -> int:
        """Metindeki token sayısını hesaplar"""
        try:
            encoder = self.encoders[model]
            return len(encoder.encode(text))
        except KeyError:
            # Eğer model için encoder bulunamazsa, varsayılan olarak GPT-3.5 encoder'ını kullan
            return len(self.encoders["gemini-1.5-flash-latest"].encode(text))

    def calculate_cost(self, prompt: str, response: str, model: str) -> Dict[str, float]:
        """Prompt ve response için maliyeti hesaplar"""
        if model not in self.providers:
            raise ValueError(f"Desteklenmeyen model: {model}")

        provider = self.providers[model]
        
        # Token sayılarını hesapla
        prompt_tokens = self.count_tokens(prompt, model)
        response_tokens = self.count_tokens(response, model)
        
        # Maliyetleri hesapla
        prompt_cost = (prompt_tokens / 1000) * provider.input_cost_per_1k_tokens
        response_cost = (response_tokens / 1000) * provider.output_cost_per_1k_tokens
        total_cost = prompt_cost + response_cost
        
        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "prompt_cost": prompt_cost,
            "response_cost": response_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }

    def get_provider_info(self, model: str) -> Dict[str, str]:
        """Model sağlayıcısı hakkında bilgi döndürür"""
        if model not in self.providers:
            raise ValueError(f"Desteklenmeyen model: {model}")
            
        provider = self.providers[model]
        return {
            "name": provider.name,
            "model": provider.model,
            "input_cost_per_1k_tokens": provider.input_cost_per_1k_tokens,
            "output_cost_per_1k_tokens": provider.output_cost_per_1k_tokens
        }

# Kullanım örneği
if __name__ == "__main__":
    calculator = LLMCostCalculator()
    
    # Test için örnek prompt ve response
    test_prompt = "Merhaba, nasılsın?"
    test_response = "Merhaba! Ben bir yapay zeka asistanıyım. Size nasıl yardımcı olabilirim?"
    
    # Gemini 2.0 Flash için maliyet hesaplama
    cost_info = calculator.calculate_cost(test_prompt, test_response, "gemini-2.0-flash")
    print("Gemini 2.0 Flash Maliyet Bilgisi:", cost_info)
    
    # Gemini 1.5 Flash Latest için maliyet hesaplama
    cost_info = calculator.calculate_cost(test_prompt, test_response, "gemini-1.5-flash-latest")
    print("Gemini 1.5 Flash Latest Maliyet Bilgisi:", cost_info) 