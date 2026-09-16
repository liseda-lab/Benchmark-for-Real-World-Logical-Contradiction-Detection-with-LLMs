import re
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from huggingface_wrapper import HFLocalModel
from openai_wrapper import OpenAIModel

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

class ContradictionEvaluator:

    def __init__(self, model_name: str, model_wrapper: str, api_key=None,
                 thinking: bool = False, language: str = "en", prompt_version: str = "simple"):

        self.wrapper_type = model_wrapper
        self.thinking = thinking
        self.language = language
        self.prompt_version = prompt_version
        self.prompt_templates = self.load_prompt_templates(language)

        if model_wrapper == "openai":
            self.model = OpenAIModel(model_name, api_key)
        elif model_wrapper == "huggingface":
            self.model = HFLocalModel(model_name)
        else:
            raise ValueError("Invalid model wrapper")

    def load_prompt_templates(self, language: str) -> dict:
        path = os.path.join(PROMPTS_DIR, f"{language}.json")
        fallback = os.path.join(PROMPTS_DIR, "en.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        elif os.path.exists(fallback):
            print(f"Warning: No prompt file for '{language}', falling back to 'en'.")
            with open(fallback) as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No prompt templates found for language '{language}' and no 'en' fallback.")
    
    # def build_cot_prompt_batch_v1(self, list_of_statements_lists: list[list[str]]):

    #     prompts = []
    #     for statements in list_of_statements_lists:

    #         prompt = (
    #             "ROLE: You are an expert in logical reasoning and you're tasked on detecting contradictions."
    #             "CONTRADICTION TAXONOMY:This contradictions can have different levels of explicitness. Some contradictions are direct and explicit, while others are more subtle and require deeper reasoning to identify."
    #             "Also, the contradictions come in different logical forms and are represented as sets of statements."
    #             # "Some contradictions are hierarchical, where one statement implies another, while others are non-hierarchical, where the statements are independent but still contradict each other.\n"
    #             "When explicitly presented, the contradiction is defined by three statements. The first two being possibly contradictory, and the third statement providing necessary context to identify the contradiction."
    #             "When implicitly presented, the contradiction is defined by two statements that can be contradictory on their own, requiring you to use your background knowledge to identify it.\n"
    #             "STEP-BY-STEP PROCESS:"
    #             "Follow the step-by-step reasoning process to determine whether the statements define a contradiction:"
    #             "1. Analyze the first statement and identify its key components."
    #             "2. In a small paragraph explain your rational for the decision."
    #             "3. Repeat the process for the second and third statements."
    #             "4. For the final statement analyze it and determine how it relates to the first two statements."
    #             "5 Determine if the block of statements define a contradiction or not, and explain your reasoning in a small paragraph."
    #             "LOGIC:  A contradiction occurs when the statements cannot all be true at the same time.\n"
    #             # "OUTPUT: Provide a clear answer of 'Yes' if the statements define a contradiction, or 'No' if they do not define a contradiction.\
    #             "OUTPUT: Provide a JSON object with the following structure: {Statement1: <rational>, Statement2: <rational>, Context: <rational>, Contradiction: <yes or no>}\n"
    #             "Now, given the following sets of statements, determine whether they define a contradiction.\n"
    #         )

    #         for i, s in enumerate(statements, 1):
    #             if i < 3:
    #                 prompt += f"Statement {i}: {s}\n"
    #             else:
    #                 prompt += f"Context: {s}\n"
            
    #         prompts.append(prompt)
        
    #     return prompts


    def build_prompt_batch(self, list_of_statements_lists: list[list[str]]):
        template = self.prompt_templates[self.prompt_version]
        system_prompt = template["system"]
        stmt_label = template["statement_label"]
        ctx_label = template["context_label"]

        prompts = []
        for statements in list_of_statements_lists:
            prompt = system_prompt
            for i, s in enumerate(statements, 1):
                if i < 3:
                    prompt += f"{stmt_label} {i}: {s}\n"
                else:
                    prompt += f"{ctx_label}: {s}\n"
            prompts.append(prompt)
        return prompts

    def parse_output(self, output: str) -> str:
        template = self.prompt_templates[self.prompt_version]
        contradiction_key = template["contradiction_key"]
        verdict_map = template["verdict_map"]

        clean = re.sub(r"```(?:json)?", "", output).strip()

        try:
            start = clean.index("{")
            depth = 0
            for i, ch in enumerate(clean[start:], start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        parsed = json.loads(clean[start:i+1])
                        # Case-insensitive key lookup
                        verdict = next(
                            (str(v).strip().lower() for k, v in parsed.items()
                            if k.lower() == contradiction_key.lower()),
                            ""
                        )
                        if verdict in verdict_map:
                            return verdict_map[verdict]
                        break
        except (ValueError, json.JSONDecodeError):
            pass

        # Regex fallback using the localized verdicts
        clean = clean.lower()
        for local_verdict, normalized in verdict_map.items():
            if re.search(rf'"({re.escape(contradiction_key.lower())})\s*":\s*"{re.escape(local_verdict)}"', clean):
                return normalized

        # Last resort: scan for any verdict word in the text
        for local_verdict, normalized in verdict_map.items():
            if re.search(rf'\b{re.escape(local_verdict)}\b', clean):
                return normalized

        return "unknown"

    
    def evaluate(self, dataset: list[dict], batch_size: int = 16):

        if self.wrapper_type == "huggingface":
            return self.evaluate_batched(dataset, batch_size)
        else:
            return self.evaluate_sequential(dataset)
    

    def evaluate_sequential(self, dataset):

        y_true = []
        y_pred = []
        raw_outputs = []

        for sample in dataset:
            prompt = self.build_prompt_batch([sample["statements"]])[0]
            raw_output = self.model.generate(prompt, enable_thinking=self.thinking)
            prediction = self.parse_output(raw_output)

            y_true.append(sample["label"])
            y_pred.append(prediction)
            raw_outputs.append(raw_output)

        metrics = self.compute_metrics(y_true, y_pred)

        return metrics, y_pred, raw_outputs


    def evaluate_batched(self, dataset, batch_size):

        y_true = []
        y_pred = []
        raw_outputs = []

        for i in range(0, len(dataset), batch_size):

            batch = dataset[i:i+batch_size]

            batch_statements = [sample["statements"] for sample in batch]
            prompts = self.build_prompt_batch(batch_statements)

            raw_batch_outputs = self.model.generate(prompts, enable_thinking=self.thinking)

            for sample, raw_output in zip(batch, raw_batch_outputs):
                prediction = self.parse_output(raw_output)

                y_true.append(sample["label"])
                y_pred.append(prediction)
                raw_outputs.append(raw_output)

        metrics = self.compute_metrics(y_true, y_pred)

        return metrics, y_pred, raw_outputs


    def compute_metrics(self, y_true, y_pred):

        valid_pairs = []

        for t, p in zip(y_true, y_pred):

            p_str = str(p).strip().lower()

            if p_str in ["yes", "no"]:

                # ✅ Ground truth is already 0/1
                t_val = int(t)

                # ✅ Convert prediction
                p_val = 1 if p_str == "yes" else 0

                valid_pairs.append((t_val, p_val))

        if not valid_pairs:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall_pos": 0.0,
                "recall_neg": 0.0,
                "f1_macro": 0.0,
                "unknown_rate": 1.0,
                "confusion_matrix": [[0, 0], [0, 0]]
            }

        y_true_f, y_pred_f = zip(*valid_pairs)

        return {
            "accuracy": accuracy_score(y_true_f, y_pred_f),
            "precision": precision_score(y_true_f, y_pred_f, zero_division=0),
            "recall_pos": recall_score(y_true_f, y_pred_f, zero_division=0),
            "recall_neg": recall_score(y_true_f, y_pred_f, zero_division=0, pos_label=0),
            "f1_macro": f1_score(y_true_f, y_pred_f, zero_division=0, average="macro"),
            "unknown_rate": list(map(str, y_pred)).count("unknown") / len(y_pred),
            "confusion_matrix": confusion_matrix(y_true_f, y_pred_f, labels=[0, 1])
        }