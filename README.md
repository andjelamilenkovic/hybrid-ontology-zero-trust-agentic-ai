# Hybrid Ontology-Grounded Zero Trust Access Control for Agentic AI Systems

Companion code, ontology, and benchmark material for the paper:

> A. Milenkovic and M. Tosic, "Hybrid Ontology-Grounded Zero Trust Access Control
> for Agentic AI Systems," *Mobile Networks and Applications* (MONET), Special Issue
> "Security and Efficiency in Agentic AI" (under review).

It builds on our earlier formal model:

> A. Milenkovic and M. Tosic, "Zero Trust Security Models for Agentic Systems in 6G
> Networks," *Facta Universitatis, Series: Working and Living Environmental Protection*,
> vol. 23, no. 1, pp. 17–36, 2026. https://doi.org/10.22190/WLEP260522002M

## What this is

We encode a Zero Trust access-decision model for agentic AI systems as an OWL 2
ontology and compare six ways of computing the decision:

| Config | Description |
|--------|-------------|
| **A** | Rule-based reference implementation of the decision function `D(q)` |
| **B** | LLM with no context (model weights only) |
| **C** | LLM given the policy as prose |
| **D** | LLM given the policy as the OWL 2 ontology |
| **E** | Hybrid: authentication, trust, workflow validity and least privilege resolved in code; the LLM is asked only whether the action is semantically consistent with the declared goal |
| **F** | Team of specialized agents (trust / workflow / semantic) with a deterministic aggregator |

The decision function is:

```
D(q) = Allow  ⟺  Auth(e)=1 ∧ T(e,c,l) ≥ τ ∧ WF(q)=1 ∧ Sem(e,a,r,w,g,c)=1 ∧ LP(e,a,r,w,g)=1
```

with trust threshold τ = 0.6.

Configurations are evaluated on three workflows (20 scenarios each): **clinical
monitoring**, **insurance pre-authorization and claims**, and **6G edge
orchestration**, with two local models (Llama 3.2 and Mistral).

## Repository structure

```
ontology/     OWL 2 ontology (zerotrust6g) in Turtle and RDF/XML — Online Resource 1
scripts/      Evaluation scripts; prompts and benchmark scenarios are defined inside them — Online Resources 2–3
results/      Raw terminal output of the runs reported in the paper
```

### `ontology/`
- `zerotrust6g.ttl` — ontology in Turtle (developed in Protégé, reasoned over with HermiT)
- `zerotrust6g.owl` — same ontology in RDF/XML

### `scripts/`
- `config_a_rule_based.py` — Configuration A: rule-based `D(q)`, five-scenario proof of concept (Section 6)

**`scripts/clinical/`** — clinical-monitoring workflow
- `configs_a_f_clinical_benchmark_llama.py` — all six configurations (A–F), Llama 3.2
- `configs_a_f_clinical_benchmark_mistral.py` — all six configurations (A–F), Mistral
- `configs_a_d_clinical_latency.py` — A–D latency/accuracy over three runs
- `config_f_agent_team.py` — Configuration F (specialized agent team), standalone
- `config_f_diagnostic.py` — per-agent signal diagnostics for Configuration F

**`scripts/insurance/`** — insurance pre-authorization and claims workflow
- `configs_a_f_insurance_benchmark_llama.py` — all six configurations (A–F), Llama 3.2
- `configs_a_f_insurance_benchmark_mistral.py` — all six configurations (A–F), Mistral
- `configs_a_d_insurance_benchmark.py` — A–D, both models, per-category breakdown

**`scripts/6g_edge/`** — 6G edge orchestration workflow
- `configs_a_f_6g_edge_benchmark_llama.py` — all six configurations (A–F), Llama 3.2
- `configs_a_f_6g_edge_benchmark_mistral.py` — all six configurations (A–F), Mistral

**`scripts/ablation/`** — context-scoping ablation
- `ablation_d_full_vs_selective.py` — full ontology vs request-scoped slice (Mistral, insurance)

### `results/`
Raw terminal output of the runs reported in the paper.

## Requirements

- Python 3.9+
- `requests` (only for the LLM configurations): `pip install -r requirements.txt`
- The proof of concept (`config_a_rule_based.py`) uses only the standard library.
- The LLM configurations use [Ollama](https://ollama.com) running locally, with the
  models pulled beforehand:
  ```
  ollama pull llama3.2
  ollama pull mistral
  ```
  Models are queried at temperature 0. Experiments in the paper were run on a single
  laptop (CPU only).

## Running

```bash
# Configuration A — rule-based proof of concept (no dependencies)
python scripts/config_a_rule_based.py

# A full six-configuration benchmark (requires Ollama running)
python scripts/clinical/configs_a_f_clinical_benchmark_llama.py
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
