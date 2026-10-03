import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

TARGET_MODELS = {
    "claude-opus-4-20250514": "Claude",
    "gemini-2.5-flash": "Gemini",
    "chatgpt-4o-latest-20250326": "ChatGPT",
}

SAMPLES_PER_MODEL = 700
MIN_WORDS = 50
MAX_WORDS = 800

collected = {m: [] for m in TARGET_MODELS}


# Filtering the responses so it will be a 1 turn interaction with no prior history.
def extract_strict_single_turn(row, model_id):

    # Filter out known sessions
    full_conv = row.get("full_conversation")
    if isinstance(full_conv, list) and len(full_conv) > 2:
        return None, None

    meta = row.get("conv_metadata")
    if isinstance(meta, dict) and meta.get("turns", 1) > 1:
        return None, None

    # Match conversation side
    if row.get("model_a") == model_id:
        conv = row.get("conversation_a")
    elif row.get("model_b") == model_id:
        conv = row.get("conversation_b")
    else:
        return None, None

    # Making sure it is strictly 1 user prompt and 1 assistant reply
    if not isinstance(conv, list) or len(conv) != 2:
        return None, None

    turn1, turn2 = conv[0], conv[1]
    if (
        not isinstance(turn1, dict)
        or not isinstance(turn2, dict)
        or turn1.get("role") != "user"
        or turn2.get("role") != "assistant"
    ):
        return None, None

    prompt = turn1.get("content")
    resp = turn2.get("content")

    # Clean structured text blocks if present
    if isinstance(prompt, list):
        prompt = " ".join(
            b.get("text", "")
            for b in prompt
            if isinstance(b, dict) and "text" in b
        )
    if isinstance(resp, list):
        resp = " ".join(
            b.get("text", "")
            for b in resp
            if isinstance(b, dict) and "text" in b
        )

    if isinstance(prompt, str) and isinstance(resp, str):
        return prompt.strip(), resp.strip()

    return None, None


dataset = load_dataset(
    "lmarena-ai/arena-human-preference-140k", split="train", streaming=True
)

for row in dataset:
    # Stop once all 3 models reach the 700
    if all(len(collected[m]) >= SAMPLES_PER_MODEL for m in TARGET_MODELS):
        break

    if row.get("language") != "en":
        continue

    for m_id in TARGET_MODELS:
        if len(collected[m_id]) < SAMPLES_PER_MODEL and (
            row.get("model_a") == m_id or row.get("model_b") == m_id
        ):
            prompt, resp = extract_strict_single_turn(row, m_id)
            if not prompt or not resp:
                continue

            word_count = len(resp.split())
            if not (MIN_WORDS <= word_count <= MAX_WORDS):
                continue

            collected[m_id].append(
                {
                    "LLM_name": TARGET_MODELS[m_id],
                    "LLM_Input": prompt,
                    "LLM_output": resp,
                }
            )

# Assemble full dataframe
all_rows = []
for m_id, rows in collected.items():
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)

counts = df["LLM_name"].value_counts()
print(counts)

# Check assertion
assert all(
    cnt == SAMPLES_PER_MODEL for cnt in counts
), f"Error: Not all models reached {SAMPLES_PER_MODEL} samples!"

# Stratified split: 70% Train, 15% Val, 15% Test
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["LLM_name"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["LLM_name"], random_state=42
)

# Export CSV files
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)