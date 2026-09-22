import os
import pandas as pd
from transformers import pipeline

# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

print("=" * 60)
print("WEEK 6 - PROMPT ENGINEERING")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD GPT-2 MODEL
# --------------------------------------------------

print("\nLoading GPT-2...")

generator = pipeline(
    "text-generation",
    model="gpt2"
)

print("GPT-2 loaded successfully!")


# --------------------------------------------------
# 2. REAL-WORLD PROBLEMS
# --------------------------------------------------

problems = [
    {
        "problem_id": 1,
        "domain": "Education",
        "problem": (
            "A student has an exam in 7 days and needs to prepare "
            "for five subjects. Create a simple study plan."
        )
    },

    {
        "problem_id": 2,
        "domain": "Healthcare",
        "problem": (
            "A person has a mild headache and wants general information "
            "about safe first steps they can consider."
        )
    },

    {
        "problem_id": 3,
        "domain": "Finance",
        "problem": (
            "A college student earns Rs. 10,000 per month and wants "
            "to create a simple monthly budget."
        )
    },

    {
        "problem_id": 4,
        "domain": "Customer Support",
        "problem": (
            "A customer received a damaged product and wants to request "
            "a replacement from the company."
        )
    },

    {
        "problem_id": 5,
        "domain": "Programming",
        "problem": (
            "A beginner gets a Python TypeError and wants to understand "
            "how to identify and fix the problem."
        )
    }
]


# Display the problems

print("\nReal-world problems:")

for item in problems:
    print(
        f"\n{item['problem_id']}. "
        f"{item['domain']}: {item['problem']}"
    )
# --------------------------------------------------
# 3. PROMPT ENGINEERING TECHNIQUES
# --------------------------------------------------

def zero_shot_prompt(problem):
    return f"""
Answer the following question clearly and simply.

Question:
{problem}
"""


def one_shot_prompt(problem):
    return f"""
Here is an example:

Question:
How can a student prepare for an exam?

Answer:
Create a study schedule, divide the subjects into smaller topics,
review important concepts, practice questions, and take short breaks.

Now answer the following question in a similar way.

Question:
{problem}
"""


def few_shot_prompt(problem):
    return f"""
Example 1:

Question:
How can a student prepare for an exam?

Answer:
Make a timetable, divide the subjects, revise regularly,
and practice questions.

Example 2:

Question:
How can someone manage their monthly expenses?

Answer:
List income and expenses, prioritize essential expenses,
set a savings target, and track spending.

Now answer this question:

Question:
{problem}
"""


def role_prompt(problem):
    return f"""
You are an experienced professional who gives practical,
clear and beginner-friendly advice.

Your task is to answer the following problem.

Problem:
{problem}

Give a useful and easy-to-understand answer.
"""


def reasoning_prompt(problem):
    return f"""
Analyze the problem using these steps:

1. Identify the main problem.
2. Identify the important factors.
3. Suggest practical options.
4. Give a final recommendation.

Problem:
{problem}

Keep the answer clear and concise.
"""
# --------------------------------------------------
# 4. GENERATE 25 PROMPTS
# --------------------------------------------------

prompt_records = []

for item in problems:

    problem = item["problem"]

    techniques = {
        "Zero-shot": zero_shot_prompt(problem),
        "One-shot": one_shot_prompt(problem),
        "Few-shot": few_shot_prompt(problem),
        "Role Prompting": role_prompt(problem),
        "Structured Reasoning": reasoning_prompt(problem)
    }

    for technique, prompt in techniques.items():

        prompt_records.append({
            "problem_id": item["problem_id"],
            "domain": item["domain"],
            "technique": technique,
            "problem": problem,
            "prompt": prompt.strip()
        })


# Convert to DataFrame

prompts_df = pd.DataFrame(prompt_records)

# Save CSV

prompts_df.to_csv(
    "dataset/prompts.csv",
    index=False
)

print("\n" + "=" * 60)
print("PROMPT GENERATION COMPLETED")
print("=" * 60)

print(f"Total prompts generated: {len(prompts_df)}")

print("\nPrompts saved to:")
print("dataset/prompts.csv")
# --------------------------------------------------
# 5. GENERATE GPT-2 RESPONSES
# --------------------------------------------------

print("\n" + "=" * 60)
print("GENERATING GPT-2 RESPONSES")
print("=" * 60)

response_records = []

for index, row in prompts_df.iterrows():

    print(
        f"Generating response "
        f"{index + 1}/{len(prompts_df)} - "
        f"{row['domain']} - {row['technique']}"
    )

    result = generator(
        row["prompt"],
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    generated_text = result[0]["generated_text"]

    # Remove the original prompt from the generated result
    response = generated_text[len(row["prompt"]):].strip()

    response_records.append({
        "problem_id": row["problem_id"],
        "domain": row["domain"],
        "technique": row["technique"],
        "problem": row["problem"],
        "response": response
    })


# Convert responses into DataFrame

responses_df = pd.DataFrame(response_records)


# Save responses

responses_df.to_csv(
    "dataset/generated_responses.csv",
    index=False
)


print("\n" + "=" * 60)
print("RESPONSE GENERATION COMPLETED")
print("=" * 60)

print(f"Total responses generated: {len(responses_df)}")

print("\nResponses saved to:")
print("dataset/generated_responses.csv")
# --------------------------------------------------
# 6. COMPARE PROMPTING TECHNIQUES
# --------------------------------------------------

print("\n" + "=" * 60)
print("CREATING PROMPT COMPARISON")
print("=" * 60)

comparison_records = []

for index, row in responses_df.iterrows():

    response = str(row["response"])

    comparison_records.append({
        "problem_id": row["problem_id"],
        "domain": row["domain"],
        "technique": row["technique"],
        "response_length": len(response),
        "word_count": len(response.split())
    })


comparison_df = pd.DataFrame(comparison_records)

comparison_df.to_csv(
    "dataset/prompt_comparison.csv",
    index=False
)


print("\nPrompt comparison completed!")

print(f"Total comparison records: {len(comparison_df)}")

print("\nComparison saved to:")
print("dataset/prompt_comparison.csv")
# --------------------------------------------------
# 7. BASIC RESPONSE EVALUATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATING RESPONSES")
print("=" * 60)


def evaluate_response(response, problem):
    """
    Basic heuristic evaluation.

    This is not a human evaluation or factuality test.
    It provides simple measurable indicators for the project.
    """

    response = str(response).strip()

    words = response.split()
    word_count = len(words)

    # -----------------------------
    # Clarity
    # -----------------------------

    if word_count >= 40:
        clarity = 9
    elif word_count >= 25:
        clarity = 8
    elif word_count >= 15:
        clarity = 7
    elif word_count >= 8:
        clarity = 6
    else:
        clarity = 5

    # -----------------------------
    # Completeness
    # -----------------------------

    if word_count >= 50:
        completeness = 9
    elif word_count >= 30:
        completeness = 8
    elif word_count >= 15:
        completeness = 7
    elif word_count >= 8:
        completeness = 6
    else:
        completeness = 5

    # -----------------------------
    # Relevance
    # -----------------------------

    problem_words = set(problem.lower().split())
    response_words = set(response.lower().split())

    common_words = problem_words.intersection(response_words)

    if len(common_words) >= 5:
        relevance = 9
    elif len(common_words) >= 3:
        relevance = 8
    elif len(common_words) >= 2:
        relevance = 7
    elif len(common_words) >= 1:
        relevance = 6
    else:
        relevance = 5

    # -----------------------------
    # Overall score
    # -----------------------------

    overall_score = round(
        (clarity + completeness + relevance) / 3,
        2
    )

    return clarity, completeness, relevance, overall_score


evaluation_records = []


for index, row in responses_df.iterrows():

    clarity, completeness, relevance, overall_score = evaluate_response(
        row["response"],
        row["problem"]
    )

    evaluation_records.append({
        "problem_id": row["problem_id"],
        "domain": row["domain"],
        "technique": row["technique"],
        "clarity": clarity,
        "completeness": completeness,
        "relevance": relevance,
        "overall_score": overall_score
    })


evaluation_df = pd.DataFrame(evaluation_records)


# Save evaluation results

evaluation_df.to_csv(
    "dataset/evaluation_results.csv",
    index=False
)


print("\nEvaluation completed!")

print(f"Total evaluated responses: {len(evaluation_df)}")

print("\nEvaluation results saved to:")
print("dataset/evaluation_results.csv")