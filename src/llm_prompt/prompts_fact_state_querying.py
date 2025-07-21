# src/llm_prompt/prompts_fact_state_querying.py

prompts_fact_state_querying = {
    'CoT': {
        'prompt': """# Task: Determine the truth state of the query based on the given ASP program.

## Understanding the ASP Program

You are given an Answer Set Programming (ASP) program consisting of facts, rules, and a query. Your goal is to determine if the `[query]` is **True**, **False**, or **Unknown** based on the provided `[facts]` and `[rules]`.

*   **Facts:** Basic statements considered true, ending with a period (`.`). Example: `bird(tweety).` means "bird(tweety) is true."
    *   Strongly negated facts like `-fly(penguin).` mean "It is explicitly known that fly(penguin) is false."
*   **Rules:** Define relationships and conclusions in the form `Head :- Body.`. This means "If the `Body` is true, then the `Head` is true."
    *   The `Body` can contain literals (like `p(X)`) and negated literals.
    *   **Default Negation (`not`):** `not p(X)` means "It cannot be proven that p(X) is true." or "There is no evidence that p(X) is true."
    *   **Strong Negation (`-`):** `-p(X)` means "It is explicitly known that p(X) is false." This is a stronger claim than `not p(X)`.
    *   **Integrity Constraints:** Rules with an empty head (e.g., `:- p(X), q(X).`) mean the body cannot be true simultaneously. If the body becomes true based on the facts and other rules, it indicates a contradiction in the potential answer set.
*   **Query:** A statement ending with `?`. You need to determine its truth value based on the logical consequences of the facts and rules.

## How to Determine the Query State

1.  **Apply the rules** to the facts iteratively to derive all possible conclusions (the stable model or answer set). Pay close attention to how `not` and `-` affect the derivability of literals.
2.  **Check the query against the derived conclusions:**
    *   **True:** The query (e.g., `p(a)`) is present in the derived conclusions/stable model.
    *   **False:** The strong negation of the query (e.g., `-p(a)`) is present in the derived conclusions/stable model.
    *   **Unknown:** Neither the query nor its strong negation is present in the derived conclusions/stable model. This typically happens if the information is insufficient to prove either case.

## Examples

**Example 1: Query is True**

*   **(Symbolic Program)**
    *   Facts: `bird(tweety).`
    *   Rules: `fly(X) :- bird(X), not -fly(X).`
    *   Query: `fly(tweety)?`

*   **(Textualized Program)**
    *   Facts:
        *   `bird(tweety)` is true.
    *   Rules:
        *   If `bird(X)` is true and there is no evidence that `fly(X)` is explicitly false, then `fly(X)` is true.
    *   Query: The query is `fly(tweety)`.

*   **(Reasoning)**
    1.  We are given the fact "`bird(tweety)` is true" (symbolically `bird(tweety).`).
    2.  Consider the rule: "If `bird(X)` is true and there is no evidence that `fly(X)` is explicitly false, then `fly(X)` is true." (symbolically `fly(X) :- bird(X), not -fly(X).`). We apply this for `X = tweety`.
    3.  The first condition `bird(tweety)` is true holds (from the fact).
    4.  The second condition is "there is no evidence that `fly(tweety)` is explicitly false" (symbolically `not -fly(tweety)`). We check if `-fly(tweety)` ("`fly(tweety)` is explicitly false") can be derived. It cannot be derived from the program.
    5.  Therefore, the condition `not -fly(tweety)` holds.
    6.  Since both conditions in the rule body are satisfied, the rule allows us to derive the head: "`fly(tweety)` is true" (symbolically `fly(tweety)`).

*   **(Conclusion)**
    *   Because `fly(tweety)` ("`fly(tweety)` is true") is derivable and belongs to the stable model, the state of the query `fly(tweety)?` is **True**.

**Example 2: Query is False (Derived)**

*   **(Symbolic Program)**
    *   Facts: `penguin(pingu).`, `bird(pingu).`
    *   Rules:
        *   `fly(X) :- bird(X), not -fly(X).`
        *   `-fly(X) :- penguin(X).`
    *   Query: `fly(pingu)?`

*   **(Textualized Program)**
    *   Facts:
        *   `penguin(pingu)` is true.
        *   `bird(pingu)` is true.
    *   Rules:
        *   If `bird(X)` is true and there is no evidence that `fly(X)` is explicitly false, then `fly(X)` is true.
        *   If `penguin(X)` is true, then `fly(X)` is explicitly false.
    *   Query: The query is `fly(pingu)`.

*   **(Reasoning)**
    1.  We are given the fact "`penguin(pingu)` is true" (symbolically `penguin(pingu).`).
    2.  Consider the second rule: "If `penguin(X)` is true, then `fly(X)` is explicitly false." (symbolically `-fly(X) :- penguin(X).`).
    3.  Applying this rule with `X = pingu`, since the condition "`penguin(pingu)` is true" holds, we derive the head: "`fly(pingu)` is explicitly false" (symbolically `-fly(pingu)`).

*   **(Conclusion)**
    *   Because `-fly(pingu)` ("`fly(pingu)` is explicitly false") is derivable and belongs to the stable model, the state of the query `fly(pingu)?` is **False**.

**Example 3: Query is Unknown**

*   **(Symbolic Program)**
    *   Facts: `bird(polly).`
    *   Rules: `fly(X) :- bird(X), can_fly(X).`
    *   Query: `fly(polly)?`

*   **(Textualized Program)**
    *   Facts:
        *   `bird(polly)` is true.
    *   Rules:
        *   If `bird(X)` is true and `can_fly(X)` is true, then `fly(X)` is true.
    *   Query: The query is `fly(polly)`.

*   **(Reasoning)**
    1.  We are given the fact "`bird(polly)` is true" (symbolically `bird(polly).`).
    2.  Consider the rule: "If `bird(X)` is true and `can_fly(X)` is true, then `fly(X)` is true." (symbolically `fly(X) :- bird(X), can_fly(X).`). We apply this for `X = polly`.
    3.  The first condition `bird(polly)` is true holds.
    4.  The second condition is "`can_fly(polly)` is true" (symbolically `can_fly(polly)`). We check if this can be derived from the program.
    5.  There are no facts or other rules that allow us to derive `can_fly(polly)`. Therefore, the body of the rule cannot be satisfied, and we cannot derive `fly(polly)` using this rule.
    6.  We also check if the strong negation `-fly(polly)` ("`fly(polly)` is explicitly false") can be derived. There are no facts or rules allowing this derivation.

*   **(Conclusion)**
    *   Since neither `fly(polly)` ("`fly(polly)` is true") nor `-fly(polly)` ("`fly(polly)` is explicitly false") can be derived from the program, the state of the query `fly(polly)?` is **Unknown**.

## Input Program

[facts]:
{% for fact in facts %}
fact {{ loop.index }}: {{ fact }}
{% endfor %}

[rules]:
{% for rule in rules %}
rule {{ loop.index }}: {{ rule }}
{% endfor %}

[query]
{{target_query}}

## Your Answer

Based on the facts and rules, analyze the logical consequences step-by-step. Explain your reasoning process, showing how the rules are applied to the facts to derive conclusions. Finally, state clearly whether the query is **True**, **False**, or **Unknown**.

**Reasoning:**
[Explain your step-by-step derivation here]

**Final Answer:** [True/False/Unknown]
""",
        'variables': ['facts', 'rules', 'query']
    }
}
