# src/llm_prompt/prompts_answer_set_selection.py

prompts_answer_set_decision = {
    'CoT': {
        'prompt': """# Task: Determine if the Candidate Answer Set is a valid Answer Set (Stable Model) for the given ASP program.

## Understanding the Task

You are given an Answer Set Programming (ASP) program (consisting of facts and rules) and a specific set of literals called the 'Candidate Answer Set'. Your task is to rigorously check if this candidate set qualifies as a true Answer Set (also known as a Stable Model) for the program.

## Key ASP Concepts

*   **Facts:** Ground atoms assumed to be true (e.g., `p(a).`).
*   **Rules:** Statements of the form `Head :- Body.` meaning "If `Body` is true, then `Head` must be true."
    *   `Body` can contain positive literals (e.g., `q(X)`), strongly negated literals (e.g., `-r(X)`), and default-negated literals (e.g., `not s(X)`).
    *   **Default Negation (`not`):** `not p` holds if `p` cannot be derived.
    *   **Strong Negation (`-`):** `-p` means `p` is explicitly false. It's a different literal from `p`.
*   **Answer Set (Stable Model):** A set of ground literals `A` is an answer set if it satisfies two main conditions:
    1.  **Consistency:** `A` does not contain a literal `p` and its strong negation `-p` simultaneously.
    2.  **Stability (Gelfond-Lifschitz Reduct):** Let `P` be the ASP program. The reduct `P^A` is obtained by:
        a. Removing every rule from `P` where a default-negated literal `not q` appears in the body, and `q` is present in the candidate set `A`.
        b. Removing all default-negated literals `not q` from the bodies of the remaining rules.
        The candidate set `A` is an answer set if and only if `A` is the *minimal classical model* of the resulting reduct `P^A` (which is now a program without default negation). This essentially means that every literal in `A` must be derivable from the facts and the simplified rules (`P^A`), and `A` contains nothing more than what is derivable.

## Verification Steps

To determine if the `[candidate_answer_set]` is a valid answer set for the program defined by `[facts]` and `[rules]`:

1.  **Consistency Check:** Verify that the `[candidate_answer_set]` does not contain both `p` and `-p` for any ground atom `p`. If it does, it's not a valid answer set.
2.  **Stability Check (using the Reduct concept):**
    a.  Construct the reduct `P^A` based on the original program `P` and the `[candidate_answer_set]` `A`:
        i.  Iterate through each rule in `[rules]`.
        ii. If a rule contains `not q` in its body and `q` *is* in `A`, discard the rule entirely.
        iii. If a rule survives step (ii), remove all `not q` literals from its body. Keep the head and the remaining positive/strongly-negated body literals.
    b.  Consider the reduct `P^A` (the simplified rules from step 2.a.iii) along with the original `[facts]`.
    c.  Determine the minimal classical model of `P^A` plus `[facts]`. This means finding the smallest set of ground literals that includes all `[facts]` and is closed under the rules in `P^A`.
    d.  Compare this minimal model with the `[candidate_answer_set]` `A`. They must be identical. If they are, `A` is a stable model. Otherwise, it is not.

## Examples

**Example 1: Valid Answer Set**

*   **(Symbolic Program)** `P`:
    *   Facts: (None)
    *   Rules: `a :- not b.`
*   **(Textualized Program)**
    *   Facts:
        *   (None)
    *   Rules:
        *   If there is no evidence that `b` is true, then `a` is true.
*   Candidate Answer Set `A`: `{a}`

1.  **Consistency Check:** `{a}` is consistent (no `p` and `-p`). **OK.**
2.  **Stability Check:**
    a.  Construct Reduct `P^A`:
        *   Rule `a :- not b.`: Since `b` is *not* in `A = {a}`, the rule is kept, and `not b` is removed. The rule becomes `a :- .` (or simply `a.`).
        *   Reduct `P^A` is: `a.`
    b.  Minimal classical model of `P^A`: The smallest set closed under the rule `a.` is `{a}`.
    c.  Compare: Minimal model `{a}` is identical to the candidate set `A = {a}`. **OK.**
*   **Verdict:** Yes, `{a}` is a valid answer set.

**Example 2: Invalid Answer Set (Fails Stability)**

*   **(Symbolic Program)** `P`:
    *   Facts: (None)
    *   Rules: `p :- not q.`, `q :- not p.`
*   **(Textualized Program)**
    *   Facts:
        *   (None)
    *   Rules:
        *   If there is no evidence that `q` is true, then `p` is true.
        *   If there is no evidence that `p` is true, then `q` is true.
*   Candidate Answer Set `A`: `{p, q}`

1.  **Consistency Check:** `{p, q}` is consistent. **OK.**
2.  **Stability Check:**
    a.  Construct Reduct `P^A`:
        *   Rule `p :- not q.`: Since `q` *is* in `A = {p, q}`, discard this rule.
        *   Rule `q :- not p.`: Since `p` *is* in `A = {p, q}`, discard this rule.
        *   Reduct `P^A` is empty (contains no rules).
    b.  Minimal classical model of the empty reduct `P^A`: The minimal model is `{}` (the empty set).
    c.  Compare: Minimal model `{}` is **not** identical to the candidate set `A = {p, q}`. **FAIL.**
*   **Verdict:** No, `{p, q}` is not a valid answer set for this program.

## Input

[facts]:
{% for fact in facts %}
fact {{ loop.index }}: {{ fact }}
{% endfor %}

[rules]:
{% for rule in rules %}
rule {{ loop.index }}: {{ rule }}
{% endfor %}

[candidate_answer_set]:
{ {% for literal in candidate_answer_set %}{{ literal }}{% if not loop.last %}, {% endif %}{% endfor %} }

## Your Answer

Perform the Consistency and Stability checks step-by-step based on the definitions above. Explain your reasoning clearly, detailing the construction of the reduct and the comparison of its minimal model with the candidate set. Finally, state clearly whether the `[candidate_answer_set]` **is** or **is not** a valid answer set.

**Reasoning:**
[Explain your step-by-step verification here, including consistency check, reduct construction, minimal model derivation, and comparison]

**Final Answer:** [Yes/No]
""",
        'variables': ['facts', 'rules', 'candidate_answer_set']
    }
} 