import torch
import torch.nn as nn
from model import GPT2, GPT2Config
import argparse


class TextGenerator:
    def __init__(self, model_path, vocab_file=None):
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
        print(f"Using device: {self.device}")

        # These are the same texts used during training
        texts = [
            "hello world",
            "how are you",
            "nice to meet you",
            "have a great day",
            "what is your name",
            "the weather is nice",
            "i love programming",
            "python is awesome",
            "deep learning is fun",
            "goodbye see you later",
        ]

        self.tokenizer = CharacterTokenizer(texts)
        self.config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            block_size=32,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.1,
            bias=True,
        )

        self.model = GPT2(self.config)
        # Add weights_only=True for safe loading
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self, prompt, max_length=32, temperature=0.8, top_k=50, top_p=0.9, num_samples=1
    ):
        generated_texts = []
        for _ in range(num_samples):
            with torch.no_grad():
                tokens = self.tokenizer.encode(prompt)
                tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

                for _ in range(max_length - len(prompt)):
                    # Get the model's predictions
                    logits = self.model(tokens)
                    next_token_logits = logits[0, -1, :] / temperature

                    # Apply top-k filtering - modified to handle vocab size
                    top_k = min(
                        top_k, self.tokenizer.vocab_size
                    )  # Ensure top_k doesn't exceed vocab size
                    if top_k > 0:
                        values, _ = torch.topk(next_token_logits, top_k)
                        min_value = values[-1]
                        next_token_logits[next_token_logits < min_value] = float("-inf")

                    # Apply softmax to get probabilities
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Ensure no NaN or inf values and proper probability distribution
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        # Fall back to uniform distribution over vocabulary
                        probs = torch.ones(
                            self.tokenizer.vocab_size, device=self.device
                        )
                        probs = probs / probs.sum()

                    # Sample from the filtered distribution
                    try:
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    except RuntimeError:
                        # If sampling fails, fall back to argmax
                        next_token = torch.argmax(probs).item()

                    if next_token >= self.tokenizer.vocab_size or next_token < 0:
                        # If we somehow get an invalid token, break
                        break

                    # Add the predicted token to our sequence
                    tokens = torch.cat(
                        [tokens, torch.tensor([[next_token]]).to(self.device)], dim=1
                    )

                    if len(tokens[0]) >= max_length:
                        break

                # Decode the generated sequence
                generated_text = self.tokenizer.decode(tokens[0].cpu().numpy())
                generated_texts.append(generated_text)

        return generated_texts


class CharacterTokenizer:
    def __init__(self, texts):
        self.chars = sorted(list(set("".join(texts))))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, indices):
        return "".join(
            [self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using trained GPT-2 model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--prompt", type=str, default="hello", help="Text prompt to start generation"
    )
    parser.add_argument(
        "--max_length", type=int, default=32, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering value")
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to generate"
    )

    args = parser.parse_args()

    try:
        generator = TextGenerator(args.model_path)

        generated_texts = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            num_samples=args.num_samples,
        )

        print(f"\nPrompt: '{args.prompt}'")
        print("\nGenerated samples:")
        for i, text in enumerate(generated_texts, 1):
            print(f"{i}. {text}")

    except Exception as e:
        print(f"Error during text generation: {str(e)}")


if __name__ == "__main__":
    main()
