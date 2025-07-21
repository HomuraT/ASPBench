"""
Module containing prompts for generating Answer Sets for ASP programs.
"""
from typing import Dict, List

prompts_answer_set_generation: Dict[str, Dict[str, str | List[str]]] = {
    'Generation': {
        'prompt': """# Task: Generate a valid Answer Set (Stable Model) for the given ASP program.

## Understanding the Task

You are given an Answer Set Programming (ASP) program consisting of facts and rules. Your task is to find *one* valid Answer Set (also known as a Stable Model) for this program.

## Key ASP Concepts Recap

*   **Facts:** Ground atoms assumed to be true (e.g., `p(a).`).
*   **Rules:** Statements of the form `Head :- Body.` ("If `Body` is true, `Head` must be true").
    *   `Body` can contain positive literals (`q(X)`), strongly negated literals (`-r(X)`), and default-negated literals (`not s(X)`).
    *   **Default Negation (`not`):** `not p` holds if `p` cannot be derived.
    *   **Strong Negation (`-`):** `-p` means `p` is explicitly false.
*   **Answer Set (Stable Model):** A set of ground literals `A` that is:
    1.  **Consistent:** Does not contain `p` and `-p` simultaneously.
    2.  **Stable:** `A` is the minimal classical model of the program's reduct `P^A` (formed by simplifying rules based on `A`). Essentially, everything in `A` must be derivable from the simplified program, and nothing more.

## Goal

Your goal is to output a single set of ground literals that constitutes a valid Answer Set for the program defined by `[facts]` and `[rules]`. There might be multiple possible Answer Sets; you only need to provide one.

## Examples

**Example 1:**

*   **(Symbolic Program)**
    *   Facts: `p(a).`
    *   Rules: `q(X) :- p(X), not r(X).`
*   **(Textualized Program)**
    *   Facts:
        *   `p(a)` is true.
    *   Rules:
        *   If `p(X)` is true and there is no evidence that `r(X)` is true, then `q(X)` is true.
*   Input:
    *   `[facts]`: `p(a).`
    *   `[rules]`: `q(X) :- p(X), not r(X).`
*   Possible Output (Generated Answer Set):
    *   Symbolic: `{ p(a), q(a) }` (Because `p(a)` is a fact, and `r(a)` cannot be derived, so `not r(a)` holds, making `q(a)` derivable).
    *   Textualized: `p(a)` is true. `q(a)` is true.

**Example 2:**

*   **(Symbolic Program)**
    *   Facts: (None)
    *   Rules: `a :- not b.`, `b :- not a.`
*   **(Textualized Program)**
    *   Facts:
        *   (None)
    *   Rules:
        *   If there is no evidence that `b` is true, then `a` is true.
        *   If there is no evidence that `a` is true, then `b` is true.
*   Input:
    *   `[facts]`: (None)
    *   `[rules]`: `a :- not b.`, `b :- not a.`
*   Possible Output (Generated Answer Set):
    *   Symbolic: `{ a }` (This is one of the two possible stable models. `{ b }` is the other. You only need to provide one.)
    *   Textualized: `a` is true.

**Example 3:**

*   **(Symbolic Program)**
    *   Facts: `p(a).`, `-q(b).`
    *   Rules: `r(X) :- p(X), not q(X).`
*   **(Textualized Program)**
    *   Facts:
        *   `p(a)` is true.
        *   `q(b)` is explicitly false.
    *   Rules:
        *   If `p(X)` is true and there is no evidence that `q(X)` is true, then `r(X)` is true.
*   Input:
    *   `[facts]`: `p(a).`, `-q(b).`
    *   `[rules]`: `r(X) :- p(X), not q(X).`
*   Possible Output (Generated Answer Set):
    *   Symbolic: `{ p(a), -q(b), r(a) }` (Because `p(a)` and `-q(b)` are facts, and `q(a)` cannot be derived, `not q(a)` holds, making `r(a)` derivable).
    *   Textualized: `p(a)` is true. `q(b)` is explicitly false. `r(a)` is true.

## Input

[facts]:
{% for fact in facts %}
fact {{ loop.index }}: {{ fact }}
{% endfor %}

[rules]:
{% for rule in rules %}
rule {{ loop.index }}: {{ rule }}
{% endfor %}

## Your Answer

Generate *one* set of ground literals that represents a valid Answer Set for the provided ASP program.

**Generated Answer Set:**
{ [Provide the comma-separated list of literals in the answer set here] }
""",
        'variables': ['facts', 'rules']
    }
} 