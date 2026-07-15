from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import tqdm

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
input_file = "wiki-treaties.csv"

# prepare the model input

def clean_note(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, notes: list[str]) -> list[str]:
    results = []

    system_prompt = """
    You are an information extraction assistant.

    Extract only the name of the treaty, convention, agreement, pact, protocol, or similar legal instrument referred to in the input.

    Rules:
    - Remove any leading "Also known as".
    - Return only the treaty name.
    - Do not explain your answer.
    - Do not add punctuation or extra text.
    - If refers to an abbreviation, extract it.
    - If there's more than one way to name the treaty, extract them using semicolon.
    """
    for note in tqdm(notes):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"Extract the treaty name from the following text:\n\n{note}"}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        torch.set_grad_enabled(False)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=40,
            do_sample=False,
            temperature=None,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        output = output.strip()
        results.append(output)
    return results

def clean_title(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, titles: list[str]) -> list[str]:
    results = []

    system_prompt = """
        You are an information extraction assistant.

        Erase the year of the name of the treaty, convention, agreement, pact, protocol, or similar legal instrument referred to in the input.
        If the year is enclosed in parenthesis, remove the parenthesis.

        Rules:
        - Remove (YEAR).
        - Remove ,YEAR.
        - Remove of YEAR.
        - Remove the preceding article 'the'
        - Do not explain your answer.
        """

    for title in tqdm(titles):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"Extract the treaty name from the following text, erase the year, keep only the treaty title:\n\n{title}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        torch.set_grad_enabled(False)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=40,
            do_sample=False,
            temperature=None,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        output = output.strip()
        results.append(output)

    return results


def main():
    df = pd.read_csv(input_file)
    inputs_df = df.dropna(subset=["note"]).copy()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cleaned_notes = clean_note(model, tokenizer, inputs_df["note"].tolist())

    # Create a new column in the original dataframe
    df.loc[inputs_df.index, "cleaned_note"] = cleaned_notes

    inputs_df = df.copy()

    cleaned_titles = clean_title(model, tokenizer, inputs_df['name'].tolist())

    df.loc[inputs_df.index, "cleaned_title"] = cleaned_titles

    print(f"\n{'=' * 72}")
    print(f"Cleaned and reformatted notes and treaties titles:")
    print()
    print(df.head(10))
    print(f"{'=' * 72}")

    df.to_csv("./wiki-treaties_formatted.csv", index=False)
    print("Saved!")
    print(f"{'=' * 72}")
if __name__ == "__main__":
    main()
