prompts_textulization = {
    'textulization': {
        'prompt': """# Task: Convert the given Answer Set Programming (ASP) program into clear, unambiguous natural language text.

## About Answer Set Programming (ASP) and Input Format Transformation

ASP is a declarative programming paradigm. You will receive symbolic items (facts, rules, queries).
*   **Facts (Potentially Transformed):** Represent basic truths, ending with a period (`.`).
    *   Positive facts remain unchanged: `bird(tweety).`
    *   **Strongly negated facts are transformed:** Original `-fact(a).` becomes `<strong negation> fact(a).` for clarity.
*   **Rules (Transformed Format):** Rules have been pre-processed for clarity. They appear in the format:
  `Original Rule. :: [if] Processed Body [then] Processed Head`
    *   `Original Rule.` is the standard ASP rule.
    *   `::` separates the original rule from its transformed representation.
    *   `[if] Processed Body` contains the conditions (rule body).
    *   `[then] Processed Head` contains the conclusion (rule head).
    *   **The core logic remains: If the conditions in `[if]` are met, then the conclusion in `[then]` holds.**
*   **Negation Markers:** To avoid ambiguity, negations within the `[if]` and `[then]` parts are explicitly marked:
    *   **`<default negation>`:** Replaces the standard `not`. It means "it cannot be assumed that..." or "there is no evidence that...".
    *   **`<strong negation>`:** Replaces the standard `-`. It means "it is explicitly known that ... is not true" or "it is definitively false that...".
    *   **Example Transformation:**
        *   Original: `-p(X) :- q(X), not -r(X).`
        *   Transformed Input: `-p(X) :- q(X), not -r(X). :: [if] q(X), <default negation> <strong negation> r(X) [then] <strong negation> p(X)`
*   **Literals:** Basic components in the `[if]` and `[then]` parts, possibly preceded by negation markers.
*   **Query:** A question ending with `?`. Example: `fly(tweety)?` means "Is it true that Tweety flies?". Queries are not transformed.

## Input Format

You will receive the ASP program structured as follows:

[facts]:
fact 1: ...
fact 2: ...
...

[rules]:
rule 1: ...
rule 2: ...
...

[query]
...

## Your Task: Textualization

Translate the provided ASP facts, rules, and query into natural language text, strictly adhering to the following guidelines, paying attention to examples for different predicate arities (0-ary, 1-ary, 2-ary, 2+-ary):

1.  **Express Facts:** Translate each fact into a simple declarative sentence, preserving all arguments and their order.
    *   Example (0-ary): `system_ready.` -> "The system is ready."
    *   Example (0-ary, Strong Negation): `-alarm_active.` -> `<strong negation> alarm_active.` -> "alarm_active is explicitly false."
    *   Example (1-ary): `bird(tweety).` -> "bird(tweety) is true." or "The predicate bird applies to tweety."
    *   Example (1-ary, Strong Negation): `-fly(penguin).` -> `<strong negation> fly(penguin).` -> "fly(penguin) is explicitly false."
    *   Example (2-ary): `connected(routerA, switchB).` -> "connected(routerA, switchB) is true."
    *   Example (2-ary, Strong Negation): `-parent(adam, cain).` -> `<strong negation> parent(adam, cain).` -> "parent(adam, cain) is explicitly false."
    *   Example (2+-ary, e.g., 3-ary): `route(packet1, routerA, port3).` -> "route(packet1, routerA, port3) is true."
    *   Example (2+-ary, Strong Negation): `-owns(personX, carY, year2023).` -> `<strong negation> owns(personX, carY, year2023).` -> "owns(personX, carY, year2023) is explicitly false."

2.  **Express Rules (Using Transformed Format):** Rules are provided in the format `Original Rule. :: [if] Processed Body [then] Processed Head`. Translate them into a conditional statement ("If [body conditions], then [head conclusion].").
    *   Focus on the `[if]` and `[then]` parts for the translation logic.
    *   Translate the `Processed Body` as the condition(s) and the `Processed Head` as the conclusion(s).
    *   Preserve all predicates, constants, variables, and argument structures exactly as they appear within the `[if]` and `[then]` parts.
    *   Translate the explicit negation markers (`<default negation>`, `<strong negation>`) according to Guideline 5.
    *   **Example (Rule with Negations):**
        *   Original Rule: `-p(X) :- q(X), not -r(X).`
        *   Transformed Input: `-p(X) :- q(X), not -r(X). :: [if] q(X), <default negation> <strong negation> r(X) [then] <strong negation> p(X)`
        *   Translation: "If q(X) holds and there is no evidence that r(X) is explicitly false, then p(X) is explicitly false."
    *   **Disjunction in Head:** If the `[then]` part contains `|` (disjunction), translate it as "either ... or ...". Example: `a(X) | b(X) :- c(X). :: [if] c(X) [then] a(X) | b(X)` -> "If c(X) is true, then either a(X) is true or b(X) is true."
    *   **Integrity Constraints:** These rules have an empty head and signify conditions that cannot hold true simultaneously. They are represented as `:- Body. :: [if] Processed Body [then] <false>`.
        *   Example: `:- owns(Person, car), minor(Person). :: [if] owns(Person, car), minor(Person) [then] <false>`
        *   Translation: "It cannot be the case that both owns(Person, car) is true and minor(Person) is true." or "It is impossible for a minor person to own a car."
    *   **Comparison Constraints:** Comparisons (like `>`, `<`, `=`, `!=`, `>=`, `<=`) can appear in the rule body. They are translated directly.
        *   Example: `high_temp(Sensor) :- temp(Sensor, Value), Value > 100. :: [if] temp(Sensor, Value), Value > 100 [then] high_temp(Sensor)`
        *   Translation: "If temp(Sensor, Value) is true and Value is greater than 100, then high_temp(Sensor) is true."

3.  **Preserve Predicates, Constants, and Arity:** Use the **exact same** predicate names, constant names, and variable names from the symbolic code (found within facts or the `[if]` / `[then]` parts of rules). **Do not change them or use synonyms.** Ensure **all** arguments appear in the translation in the **exact same order**. This applies **even if arguments are identical**. The structure and number of arguments (arity) must be preserved.
    *   Example (0-ary): `error.` must be translated reflecting 'error', e.g., "An error occurred."
    *   Example (1-ary): `active(process1).` -> "active(process1) is true."
    *   Example (2-ary, repetition): `relation(A, A).` -> "relation(A, A) is true."
    *   Example (2-ary, strong negation, repetition): `-connected(PortA, PortA).` -> `<strong negation> connected(PortA, PortA).` -> "connected(PortA, PortA) is explicitly false."
    *   Example (2+-ary, e.g., 3-ary): `transfer(AccountX, AccountY, Amount100).` -> "transfer(AccountX, AccountY, Amount100) is true."
    *   Example (2+-ary, e.g., 3-ary, repetition): `between(PointA, PointB, PointA).` -> "between(PointA, PointB, PointA) is true."
    *   **Correct Examples (Emphasis on Argument Preservation):**
        *   `relation(A, A).` must be translated preserving both arguments, like "relation(A, A) is true." Merging them ("A relates to itself") is **wrong**.
        *   `-connected(PortA, PortA).` must be translated preserving both arguments, like `-connected(PortA, PortA).` -> `<strong negation> connected(PortA, PortA).` -> "connected(PortA, PortA) is explicitly false." Omitting one ("PortA is not connected itself.") is **wrong**.
        *   `between(PointA, PointB, PointA).` must be translated preserving all arguments in order, like "between(PointA, PointB, PointA) is true." Changing order or structure ("PointA is between PointB.") is **wrong**.
    *   **General Incorrect Practices:** Omitting arguments, merging arguments (e.g., `loves(R, R)` -> "R loves himself"), or changing order/structure are **wrong**.

4.  **Consistency:** Ensure the same predicates and constants are translated **consistently** throughout the text, regardless of arity.
    *   Example (0-ary): If `shutdown.` is translated as "A shutdown occurred.", use this phrasing consistently.
    *   Example (1-ary): If `bird(tweety).` is "Tweety is a bird," use "Tweety" and "bird" consistently.
    *   Example (2-ary): If `parent(adam, cain).` is "adam is the parent of cain.", use "parent", "adam", "cain" consistently in other translations involving them.
    *   Example (2+-ary): If `assign(task1, userA, projectX).` is "task1 is assigned to userA for projectX.", maintain this phrasing style for other `assign` facts/rules.

5.  **Translate Explicit Negation Markers:** Translate the `<default negation>` and `<strong negation>` markers consistently, following the `Original -> Transformed -> Translation` pattern where applicable.
    *   **`<strong negation>`:** Translate as explicit falsehood. Corresponds to original `-p(...)`.
        *   Example (0-ary): `-safe.` -> `<strong negation> safe.` -> "safe is explicitly false."
        *   Example (1-ary): `-fly(penguin).` -> `<strong negation> fly(penguin).` -> "fly(penguin) is explicitly false."
        *   Example (2-ary): `-connected(serverA, serverB).` -> `<strong negation> connected(serverA, serverB).` -> "connected(serverA, serverB) is explicitly false."
        *   Example (2+-ary): `-authorized(userX, actionY, resourceZ).` -> `<strong negation> authorized(userX, actionY, resourceZ).` -> "authorized(userX, actionY, resourceZ) is explicitly false."
        *   Use phrases like "**... is explicitly false**".
    *   **`<default negation>`:** Translate as failure to prove / lack of evidence. Corresponds to original `not p(...)`.
        *   Example (in rule body): `... :- ..., not error_detected.` -> `... :: [if] ..., <default negation> error_detected [then] ...` -> "If ... and there is no evidence that error_detected is true, then ..."
        *   Example (in rule body): `fly(X) :- bird(X), not broken_wing(X).` -> `fly(X) :- bird(X), not broken_wing(X). :: [if] bird(X), <default negation> broken_wing(X) [then] fly(X)` -> "If bird(X) is true and there is no evidence that broken_wing(X) is true, then fly(X) is true."
        *   Example (in rule body): `eligible(P, L) :- citizen(P, C), not has_loan(P, L).` -> `eligible(P, L) :- citizen(P, C), not has_loan(P, L). :: [if] citizen(P, C), <default negation> has_loan(P, L) [then] eligible(P, L)` -> "If citizen(P, C) is true and there is no evidence that has_loan(P, L) is true, then eligible(P, L) is true."
        *   Example (in rule body with underscore): `can_assign(...) :- task(T), ..., not assigned(T, _, _).` -> `... :: [if] task(T), ..., <default negation> assigned(T, _, _) [then] can_assign(...)` -> "... and there is no evidence that assigned(T, _, _) is true for any user and project..."
        *   Use phrases like "**there is no evidence that ... is true**".
    *   **Combination (`<default negation> <strong negation> p(...)`):** Translate as lack of proof for strong negation. This corresponds to the original ASP pattern `not -p(...)`.
        *   Example (0-ary): `not -system_stable.` -> `<default negation> <strong negation> system_stable.` -> "There is no evidence that system_stable is explicitly false."
        *   Example (1-ary): `not -blocked(userX).` -> `<default negation> <strong negation> blocked(userX).` -> "There is no evidence that blocked(userX) is explicitly false."
        *   Example (2-ary): `not -married(personA, personB).` -> `<default negation> <strong negation> married(personA, personB).` -> "There is no evidence that married(personA, personB) is explicitly false."
        *   Example (2+-ary): `not -conflict(meeting1, roomA, timeT).` -> `<default negation> <strong negation> conflict(meeting1, roomA, timeT).` -> "There is no evidence that conflict(meeting1, roomA, timeT) is explicitly false."

6.  **Reduce Ambiguity:** Structure sentences clearly. Pay attention to the implicit universal quantification of variables in rules (as seen in the `[if]` and `[then]` parts).
    *   Example (2-ary): `ancestor(X, Y) :- parent(X, Y).` implies "For any X and Y, if X is the parent of Y, then X is the ancestor of Y."
    *   Example (2+-ary): `[if] parent(X, Y), parent(Y, Z) [then] grandparent(X, Z)` implies "For any X, Y, and Z, if X is the parent of Y and Y is the parent of Z, then X is the grandparent of Z."

7.  **Handle the Query:** Queries are provided in their original format (ending with `?`). Phrase the query as a question or statement to be verified, preserving its structure and arity. Translate any negations (`-` or `not`) directly as they appear in the query.
    *   Example (0-ary): `alarm?` -> "Is alarm true?"
    *   Example (1-ary): `fly(tweety)?` -> "Is fly(tweety) true?"
    *   Example (2-ary): `connected(routerA, switchB)?` -> "Is connected(routerA, switchB) true?"
    *   Example (2-ary, negation): `-parent(adam, cain)?` -> "Is parent(adam, cain) explicitly false?"
    *   Example (2+-ary): `route(packet1, routerA, port3)?` -> "Is route(packet1, routerA, port3) true?"

8.  **Rule Generalization:** The transformed rule format already presents the rule generally using variables. Translate it directly based on the `[if]` and `[then]` parts.

9.  **No Direct Copying:** The output must be a natural language translation, not a copy of the symbolic item (whether original or transformed).

10. **Symbolic Nature of Predicates:** Remember predicate names are symbolic labels, regardless of arity (0, 1, 2, 2+). Their meaning comes from the program's logic, not the word itself. Translate them consistently based on the name.
    *   Example (0-ary): `flagged.` might just mean a condition is met. Translate as "flagged is true."
    *   Example (1-ary): `red(blockA).` means the predicate 'red' applies to 'blockA'. Translate as "red(blockA) is true."
    *   Example (2-ary): `press(A, B)` represents the symbolic relationship 'press' between A and B. Translate as "press(A, B) is true."
    *   Example (2+-ary, e.g., 3-ary): `link(Node1, Node2, ProtocolA)` represents the symbolic relationship 'link' involving Node1, Node2, and ProtocolA. Translate as "link(Node1, Node2, ProtocolA) is true."

## Output Format

Present the textualized result clearly.

---

Now, textualize the symbolic items listed below. Remember that rules and strongly negated facts are presented in a transformed format using `:: [if]...[then]...` for rules and `<strong negation>` for facts/rule parts. Translate positive facts and queries directly. Translate transformed items based on the explicit markers (`<default negation>`, `<strong negation>`) and rule structure (`[if]...[then]...`).

[Symbolic Items to Textualize]:
{# Render the list of symbolic items passed in the context #}
{% if render_context.symbolic_items %}
  {% for item in render_context.symbolic_items %}
    $ {{ item }} $ 
  {% else %}
    (No symbolic items provided for textualization)
  {% endfor %}
{% else %}
(No symbolic items provided in the context)
{% endif %}
""",
        # Update variables list to reflect the key used in the template logic
        'variables': ['render_context'] # render_context contains symbolic_items
    },
}
