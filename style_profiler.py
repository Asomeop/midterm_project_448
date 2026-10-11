import re
import string
import pandas as pd

def extract_linguistic_features(df, text_col="LLM_output"):
    # use a predefined list of discourse markers, which are linguistic terms or phrases that help organize speech
    # we will focus on a small group of these discourse markers: hedges, boosters, transitions, personal pronouns, and direct addresses to the recipient
    DISCOURSE_MARKERS = {
        "hedges": [
            "generally", "typically", "often", "somewhat", "relatively", 
            "largely", "potentially", "arguably", "tends to", "in general"
        ],
        "boosters": [
            "clearly", "definitely", "essential", "crucial", "fundamental", 
            "undoubtedly", "critical", "vital", "must", "always"
        ],
        "transitions": [
            "however", "furthermore", "moreover", "additionally", "in addition", 
            "consequently", "specifically", "in conclusion", "therefore", "overall"
        ],
        "personal_pronouns": ["i", "me", "my", "we", "our", "us"],
        "direct_address": ["you", "your", "yours"]
    }
    results = []

    for _, row in df.iterrows():
        raw_text = str(row[text_col])
        clean_text = raw_text.lower()

        # structural characteristics
        words = re.findall(r"\b[a-z']+\b", clean_text)
        word_count = len(words)
        char_count = len(raw_text)
        denom = word_count + 1e-6
        newlines = raw_text.count("\n")
        bullets = len(re.findall(r"^[\s]*[-*•\d+\.]\s+", raw_text, re.MULTILINE))
        has_code = 1 if "```" in raw_text else 0
        punct_count = sum(1 for c in raw_text if c in string.punctuation)
        punct_ratio = punct_count / (char_count + 1e-6)
        unique_words = set(words)
        ttr = len(unique_words) / denom  # type-token ratio, used to distinguish lexical diversity

        # linguistic characteristics
        hedges = sum(clean_text.count(h) for h in DISCOURSE_MARKERS["hedges"])
        boosters = sum(clean_text.count(b) for b in DISCOURSE_MARKERS["boosters"])
        transitions = sum(clean_text.count(t) for t in DISCOURSE_MARKERS["transitions"])
        first_person = sum(words.count(p) for p in DISCOURSE_MARKERS["personal_pronouns"])
        direct_address = sum(words.count(p) for p in DISCOURSE_MARKERS["direct_address"])
        modals = len(re.findall(r"\b(can|could|would|should|might|may|will|shall)\b", clean_text))

        results.append({
            "model": row["LLM_name"],
            "avg_word_count": word_count,
            "newlines": newlines,
            "bullets": bullets,
            "code_block_pct": has_code * 100.0,
            "punct_ratio": punct_ratio,
            "ttr": ttr,
            "hedges_per_1k": (hedges / denom) * 1000,
            "boosters_per_1k": (boosters / denom) * 1000,
            "transitions_per_1k": (transitions / denom) * 1000,
            "first_person_per_1k": (first_person / denom) * 1000,
            "direct_address_per_1k": (direct_address / denom) * 1000,
            "modals_per_1k": (modals / denom) * 1000,
        })

    profile_df = pd.DataFrame(results)
    summary = profile_df.groupby("model").mean().round(2)
    print("Structural & Linguistic Profile of LLM Outputs\n")
    print(summary)
    return summary

if __name__ == "__main__":
    test_df = pd.read_csv("test.csv")
    summary = extract_linguistic_features(test_df)
    summary.to_csv("lingustic_profiling_summary.csv")
    print("\nSaved profiling metrics to 'lingustic_profiling_summary.csv'")