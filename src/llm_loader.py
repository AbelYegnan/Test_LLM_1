# src/llm_loader.py

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class LLMWrapper:

    def __init__(self, hf_id: str, label: str, load_in_4bit: bool = False):
        if not HAS_TORCH:
            raise ImportError("torch et transformers sont requis.")

        self.hf_id  = hf_id
        self.label  = label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Chargement {label} sur {self.device}...")

        # Tokenizer — sans trust_remote_code pour Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Modèle — chargement minimal et stable
        self.model = AutoModelForCausalLM.from_pretrained(hf_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"     {label} prêt.")

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def get_log_probs(self, system: str, user: str):
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        log_probs = []
        for step_idx, score in enumerate(output.scores):
            log_prob_dist = torch.log_softmax(score[0], dim=-1)
            token_id = output.sequences[0][
                inputs["input_ids"].shape[1] + step_idx
            ]
            log_probs.append(log_prob_dist[token_id].item())

        return torch.tensor(log_probs)


def load_llm(llm_config: dict) -> LLMWrapper:
    return LLMWrapper(
        hf_id        = llm_config["hf_id"],
        label        = llm_config["label"],
        load_in_4bit = llm_config.get("load_in_4bit", False),
    )