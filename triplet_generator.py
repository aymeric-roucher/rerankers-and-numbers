import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PREVENT_ROUNDING = True

TRIPLET_PROMPT = f"""
You are to help training of a rereanker model.
For this, we need triplets of sentences: one refereence sentence referring to a number but int he qualitative form, then a pir of sentences that are either close or very far from the reference sentence.

For instance the triplet could be:
reference: "The firm's operating profit was 101 million dollars"
close: "The firm's operating profit was 90 million dollars"
far: "The firm's operating profit was 152 million dollars"

or

reference: "Spending got out of control"
close: "Net revenue including expenses was -18,000 dollars"
far: "Net revenue including expenses was 152 thousand dollars"

A current painpoint of rerankers is that they encode by tokens, so "-100" and "100" are closer for them than "100" and "153", you should make triplets that play on this.
Make it diverse: topic could be enterprise, government spending, children's plays, whatever.
Instead of always using generic elements like "The" as the first word, use made-up proper names, it should look like a real sentence.
{"Do not round your numbers too much with zeros, rather make it 31,304 than 31,000." if PREVENT_ROUNDING else ""}
Make 10 of these triplets.

Now go on! Generate this in the shape of a JSON object with the following fields:
- reference: the reference sentence
- close: the close sentence
- far: the far sentence
"""


def generate_triplet() -> dict[str, str] | None:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": TRIPLET_PROMPT,
            }
        ],
        response_format={"type": "json_object"},
    )
    try:
        output = response.choices[0].message.content
        parsed_output = json.loads(output)
        return parsed_output
    except Exception:
        return None


if __name__ == "__main__":
    for _ in tqdm(range(10)):
        triplet = generate_triplet()
        if isinstance(triplet, dict) and "triplets" in triplet:
            triplets = triplet["triplets"]
        else:
            triplets = [triplet]

        if triplets is None:
            continue

        for triplet in triplets:
            with open("triplets.jsonl", "a") as f:
                f.write(json.dumps(triplet) + "\n")
            time.sleep(0.01)
