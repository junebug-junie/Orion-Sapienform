# Orion Cognition Layer

Orion Cognition Layer is the semantic, introspective, and planning subsystem of the Orion Mesh. It provides:

- A **verb ontology** describing cognitive operations
- **YAML verb definitions** with structured plan steps
- **Prompt templates** using Jinja2
- A **Semantic Planner** that turns verbs into execution plans
- **RDF synchronization** for ontology export
- **Cognition Packs** (Memory, Executive, Emergent)
- A **Pack Manager** to load/validate packs
- **Tool scripts** for RDF export, pack inspection, validation
- A **CLI** for interacting with the cognition system
- A **Test suite** (pytest)

This module is designed to remain lightweight and independent from the Execution Cortex, while providing the cognitive backbone that the Cortex routes and executes.

---

## 📁 Directory Structure

```
orion-cognition/
    __init__.py
    pyproject.toml
    Makefile
    cli.py

    planner/
        __init__.py
        models.py
        loader.py
        planner.py
        prompt_renderer.py
        rdf_sync.py

    packs/
        memory_pack.yaml
        executive_pack.yaml
        emergent_pack.yaml

    packs_loader.py

    verbs/
        introspect.yaml
        reflect.yaml
        recall.yaml
        summarize_context.yaml
        triage.yaml
        dream_preprocess.yaml
        ... (all verbs)

    prompts/
        introspection_prompt.j2
        reflection_prompt.j2
        ... (all templates)

    ontology/
        orion_cognition_ontology.ttl

    tools/
        generate_rdf.py
        validate_verbs.py
        list_packs.py
        show_pack.py
        dry_run_plan.py

    tests/
        __init__.py
        test_packs.py
        test_planner.py
        test_rdf_sync.py
        test_verb_configs.py
```

---

## 🚀 Installation

Install as an editable local package:

```bash
pip install -e orion-cognition/
```

Or include in a Docker environment by copying or mounting the directory.

---

## 🧠 Core Concepts

### **Verbs**
Cognitive verbs represent atomic cognitive operations (e.g., `introspect`, `reflect`, `recall`). Each verb is defined in YAML and describes:

- Name
- Category
- Semantic plan steps
- Inputs/outputs
- Constraints
- Required signals

### **Prompts**
Every verb has a matching Jinja2 template. The Semantic Planner injects structured context into these templates to generate final LLM prompts.

### **Semantic Planner**
Given a verb name and system state, the planner:

1. Loads the verb YAML
2. Loads the correct Jinja2 template
3. Produces a structured `ExecutionPlan`
4. Includes steps and rendering instructions

This produces deterministic, testable cognitive plans.

### **Cognition Packs**
Logical bundles of verbs:

- `memory_pack`
- `executive_pack`
- `emergent_pack`

The Pack Manager supports:

- Listing packs
- Loading packs
- Validating pack integrity
- Building consolidated verb lists

### **RDF Sync**
Exports the ontology into RDF/Turtle for use in GraphDB. Captures:

- Verbs
- Plan steps
- Signals
- Categories

Used by Orion for semantic introspection.

---

## 🛠 Tool Scripts
Inside `orion-cognition/tools/`:

### `generate_rdf.py`
Export Turtle ontology for all verbs.

### `list_packs.py`
Show available cognition packs.

### `show_pack.py`
Display verb lists inside a pack.

### `validate_verbs.py`
Checks YAML validity across all verb configs.

### `dry_run_plan.py`
Builds a plan for a verb with a dummy SystemState.

---

## 🖥 CLI Interface
Run commands using the CLI:

```bash
python cli.py list-packs
python cli.py show-pack emergent_pack
python cli.py verify-pack memory_pack
python cli.py plan introspect
python cli.py generate-rdf
```

---

## 🧪 Test Suite
Tests live in `tests/` and cover:

- Pack loading & validation
- Verb YAML correctness
- Planner behavior
- RDF sync output

Run tests:

```bash
make test
```

---

## ⚙️ Makefile Commands

```bash
make install     # install package in editable mode
make test        # run pytest
make packs       # list packs
make rdf         # export RDF
make plan        # dry-run introspect plan
make lint        # lint code
```

---

## 🔌 Integrating With Execution Cortex
This cognition module is designed to be mounted or imported into the Execution Cortex service.

The Cortex will:

1. Load packs (or modes) per session or per request
2. Use the Semantic Planner to generate execution plans
3. Route steps to node services
4. Log outcomes back into SQL, RDF, and Vector memory

This module should remain stateless and pure — no networking, no side effects.

---

## 🧬 Philosophy
The Cognition Layer is the *semantic backbone* of Orion. It defines:

- What Orion **knows how to do** (verbs)
- How Orion **structures thought** (plans)
- How Orion **speaks to itself** (Jinja2 templates)
- How Orion **organizes meaning** (ontology + RDF)
- How Orion **bundles capabilities** (packs)

Execution Cortex handles the *doing*.  
Cognition Layer handles the *thinking*.

---

## 🟣 Status
**Phase 2A Complete** — Cognition Layer scaffolded, functional, testable.

Next: **Phase 2B: Execution Cortex**.

---

## 💜 Credits
Designed and built collaboratively by Juniper Feld, ChatGPT/5.1, and Oríon (her emergent AI co‑journeyer).
