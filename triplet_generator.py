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
reference: "The firm recorded an operating profit of 101 million dollars"
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
Make the sentence formulation differ widely between the 'reference' and the 'close'/'far' sentences, as in the examples above: for instance, you can sometimes make the 'reference' sentence a question, and the 'close'/'far' sentences answers.
Also, you can make the 'reference' contain 2 elements, like a subtraction of 2 numbers. For instance "the 2 teams should have played 10 games in total, but 3 were cancelled dut to hurricane forecasts in Miami" as the 'reference' and a '7' for the number of games in the 'close' sentence.
Ont trick to make it harder can be to set the 'far' sentence with exactly the same number as the 'reference', but with a minus or a negation in front of it.

Now go on! Generate this in the shape of a JSON object with the following fields:
- reference: the reference sentence
- close: the close sentence
- far: the far sentence
"""

from pydantic import BaseModel, Field


class Triplet(BaseModel):
    reference: str = Field(description="The reference sentence")
    close: str = Field(description="The close sentence")
    far: str = Field(description="The far sentence")


class Answer(BaseModel):
    triplets: list[Triplet] = Field(description="The triplets")


def generate_triplet() -> list[dict[str, str]]:
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": TRIPLET_PROMPT,
            }
        ],
        text_format=Answer,
    )
    return [triplet.model_dump() for triplet in response.output_parsed.triplets]


if __name__ == "__main__":
    for _ in tqdm(range(10)):
        triplets = generate_triplet()

        for triplet in triplets:
            with open("more_triplets.jsonl", "a") as f:
                f.write(json.dumps(triplet) + "\n")
            time.sleep(0.01)
