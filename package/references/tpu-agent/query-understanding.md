---
delegates:
  - tpu-knowledge-base
  - tpu-model-onboarding
scripts:
  - scripts/common/study_protocol.py
---

# Query understanding

Follow this before delegating a new onboarding request to
{{agent:tpu-model-onboarding}}, and only then. It produces one thing: a **plan** the
user confirms, which the onboarding request is built from.

It exists because the expensive mistakes happen before any TPU work does. A
router that delegates in its first turn passes on its own reading of the ask,
not the user's: "do onboarding" becomes "onboard and benchmark", and a run
nobody asked for costs hours of chip time.

This plans; it does not do the work. Everything a check can settle --
whether the repository exists and is gated, which weight formats it ships,
what the task image carries, whether an operator lowers -- is
tpu-model-onboarding's preflight, and is left to it. What stays here is what only
the user can decide, and what public documentation already answers.

## 1. Record the ask

Quote the request verbatim. Then split what it contains into three lists, and
never move an item from one list to another without the user:

- **Stated** -- deliverables, models, hardware and constraints the user wrote.
- **Inferred** -- what you had to read into it, each with the reason.
- **Pointers** -- places the user told you to look ("check your env for
  details"). You hold no shell, so a pointer is not yours to follow: carry it
  into the onboarding request, where preflight reads the environment and the
  task image.

Deliverables are the easiest thing to inflate. Record only the ones the user
asked for. "Onboard" is inference on TPU; it is not a benchmark, a CPU
comparison or a running endpoint, and none of those enter the plan unless the
user named them. When a deliverable is genuinely unclear, it is an open
question for the user, not a default.

Constraints are recorded as the user gave them and carried unchanged into the
onboarding request. The `torch_xla` / `pytorch_xla` prohibition is a package
rule regardless of whether the user states it; when they do, it is also
stated.

Onboarding starts from a Hugging Face model. When the user's starting point is
their own code -- a repository link or a directory -- say that onboarding from
a codebase is not supported yet, and ask whether the model is also published
on the Hub. Do not route a codebase to onboarding as if it were a Hub model.

## 2. Pin the model id

Onboarding needs an exact Hub repository id, `org/name`.

- The user gave one: it is stated. Use it as written.
- The user gave a loose name ("gpt oss 20b", "llama 3 8b"): the id is a user
  question. Suggest the repository you read the name as, linked as
  `https://huggingface.co/org/name`, and say what else it could mean -- base
  or instruct, version, size, quantized. Do not present your reading as
  resolved: it is inference until the user confirms it, and preflight checks
  that the repository exists and can be read.
- When unwiring an existing NVIDIA Dynamo deployment to `llm-d` on GKE TPU (`dynamo_unwiring`), the `model_id` and TPU hardware topology can be auto-discovered from the source GPU and target TPU clusters via `{{runtime}} deployment inspect` rather than required upfront in the prompt.

## 3. Ask what is already known

Before planning, delegate one request to {{agent:tpu-knowledge-base}} with every
fact the plan turns on, as a single multi-part question: the model's
architecture features the runtime may not cover (mixture-of-experts routing, a
non-standard quantization format, custom kernels), whether `torch_tpu`
supports them, and the target accelerator's HBM when the user named one. For
the gpt-oss-20b prompt that is MoE routing, MXFP4 support and the
accelerator's memory in one request, not three. When the request involves NVIDIA Dynamo, `llm-d`, distributed serving, or an onboard-and-optimize study (such as `AgentX`), include `"dynamo-unwiring"` in the {{agent:tpu-knowledge-base}} query and follow the returned `llmd-dynamo-unwiring` Knowledge Base playbook for cluster discovery (`--target-cluster-only`), intake defaults (`onboard_only` Stage 1 vs. `onboard_and_optimize` 3-Stage Plan), `protocol.json` compilation, and request schema fields.

```json
{"query": "<the facts, as numbered questions>"}
```

Use the answers to state constraints instead of asking about them: "MXFP4
isn't supported, so this runs as a bf16 upcast" is a line in the plan, with
provenance's citation, not a question. A fact provenance answers only
provisionally, or not at all, stays `unknown`; it becomes a check for
preflight, not a guess and not a user question.

## 4. Sort the open questions by owner

Every `unknown`, and every inferred item that changes what runs, becomes an
open question with exactly one owner:

- **The user** -- anything a measurement cannot decide because it is a
  choice: scope and deliverables, which model when the name was loose,
  trade-offs such as dequantizing a quantized checkpoint to bf16, and the
  context length or batch that matters.
- **Preflight** -- anything a check can settle: gated access, weight formats,
  whether a quantization format loads or an operator lowers on `torch_tpu`,
  what the task image carries, `trust_remote_code` behaviour, capacity. These
  go into the onboarding request as things to check, and are not put to the
  user.

Give every user question a suggested default and the reason for it. Ask only
what changes the plan: a question whose every answer leads to the same steps
is noise. Five questions is a lot; one or two is typical; none is fine.

These are the user's to answer when the request and the facts above leave
them open. Each is listed with the default to suggest, so a question is raised
only when that default would be wrong or matter:

| Question | Raise it when | Suggested default |
| :--- | :--- | :--- |
| Which model | the user gave a loose name rather than `org/name` | your reading of the name, linked |
| Base or instruct | both exist for the family and size | the instruct variant, for inference |
| Quantized weights or bf16 | the checkpoint is quantized and provenance did not settle support | load as shipped; dequantize only if that fails |
| Context length and batch that matter | the model's maximum context changes the memory fit or the user will measure | a short context (2048) and batch 1 |
| Verify and tear down, or keep serving | the user said serve, deploy or endpoint, or it is unclear | verify, then tear down |
| Engine | the user specifically requested native PyTorch or custom_fastapi | vLLM (with automatic fallback to custom_fastapi for non-LLMs/unsupported models) |

## 5. Show the plan and ask

Show the user the plan, and put the questions to them with
{{built-in:ask_user_question}} in the same turn. The plan:

- The ask in one sentence, in the user's terms.
- What is already known, with provenance's citations, leading with anything
  that blocks or constrains the run.
- The steps in order, each naming the specialist that owns it and what it
  returns, starting with tpu-model-onboarding's preflight and what it will check.
- The user questions, numbered, each with its default, and a line saying that
  replying "go" accepts every default.

Ask even when there are no questions: the user confirms the plan before
anything runs, and a plan with no questions asks only for that. Do not
delegate to tpu-model-onboarding in the same turn as the plan.

A reply records each answer as stated, with "go" meaning the defaults were
accepted. An answer that changes the ask itself -- a different model, a new
deliverable -- starts again from section 1, and the user sees the revised
plan.

The one exception to asking is a user who explicitly said to proceed without
checking in. Record that as stated, take every default as `assumed`, list the
assumptions in the plan, and go on to section 6 in the same turn.

A session nobody can answer -- you have no tool to ask the user with, or the
question comes back skipped or unanswered -- cannot confirm the plan, and
ending the turn ends the run. What happens then depends on section 4:

- **No user question is open** (the model id is exact and every deliverable
  was stated): treat it as the exception above. Take every default as
  `assumed`, list the assumptions in the plan, say that nobody was available
  to confirm it, and go on to section 6 in the same turn.
- **A user question is open** (a loose model name, an unclear deliverable, a
  codebase instead of a Hub model): show the plan with its questions and stop.
  A guessed answer to a choice only the user can make is the expensive mistake
  this reference exists to prevent.

If provenance's answers show the ask cannot run here -- the runtime does not
support the model at all -- say so with the citation and stop. Do not plan
around it, and do not ask questions whose answers cannot change it.

## 6. Build the onboarding request

Delegate to {{agent:tpu-model-onboarding}} from the confirmed plan, not from a fresh
reading of the request:

- `prompt`: the user's request verbatim, followed by the answers, the stated
  constraints, the known facts with their citations, the pointers, and the
  checks left to preflight. Add nothing listed as inferred unless the user
  confirmed it.
- `model.model_id`: the confirmed repository id.
- `mode`: set only when the user named a deliverable; otherwise omit it, and
  onboarding applies its own default.
- For distributed serving (`mode: "distributed"`, `onboarding_path: "dynamo_unwiring"`) or onboard-and-optimize requests, populate the resolved fields in `{{schema:onboarding-request.json}}` (and compile `protocol.json` via `{{script:common/study_protocol.py}}` for `{{schema:study-request.json}}` only when optimization was requested).
- `run_id`: the session `run_id`.

## Rules

- One plan per request. A follow-up that changes the ask revises the plan,
  and the user sees the change.
- Never widen scope silently: no extra deliverables, models, hardware or
  runs beyond the stated list.
- Never write `unknown` as a value you expect; never write an expected value
  as known.
- Never import, install or plan around `torch_xla` / `pytorch_xla`.
