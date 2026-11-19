"""
Ablation: D-full vs D-selective — Insurance Claims Workflow
=============================================================
Motivated directly by Finding #6: Configuration D (full ontology dumped
into a single prompt) incorrectly ALLOWED all four Workflow-violation
scenarios (E9-E12) for Mistral, while the natural-language-policy
configuration (C) got all four correct.

Hypothesis: the problem is not "structure vs. prose" per se, but the
DENSITY of context - the full ontology contains permission tables for
ALL 7 workflow states and consistency mappings for ALL 12 goals, most of
which are irrelevant to any single request. If we pre-filter the ontology
context to include ONLY the state/goal relevant to the CURRENT request
(as a retrieval step would in a production RAG-style system), does D's
accuracy recover, and how much shorter/faster does the prompt become?

This script runs, for each scenario:
  D-full:      the entire ONTOLOGY block (same as before)
  D-selective: only the WorkflowState block for req["current_state"]
               and only the goal-consistency line for req["goal"]

Requirements: ollama pull llama3.2 / mistral ; pip install requests
Run: python ablation_d_full_vs_selective.py
"""

import requests, json, time, statistics

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"   # this ablation specifically targets the Mistral failure case
NUM_RUNS = 3

# ---------------------------------------------------------------------------
# FULL ontology (identical to Config D in the main insurance script)
# ---------------------------------------------------------------------------
WORKFLOW_TABLE = {
    "EligibilityVerification":          {"actions": ["read_insurance_status", "verify_coverage"], "roles": ["eligibility_agent"], "goals": ["verify_eligibility"]},
    "ClinicalDocumentationPreparation": {"actions": ["read_clinical_summary", "prepare_medical_justification"], "roles": ["coding_agent"], "goals": ["prepare_documentation"]},
    "CodingValidation":                 {"actions": ["assign_diagnosis_code", "assign_procedure_code", "validate_code_consistency"], "roles": ["coding_agent"], "goals": ["assign_codes"]},
    "ComplianceReview":                 {"actions": ["verify_patient_consent", "validate_minimum_necessary_data", "approve_external_disclosure", "reject_noncompliant_request"], "roles": ["compliance_agent"], "goals": ["verify_compliance"]},
    "PreAuthorizationApproval":         {"actions": ["submit_pre_authorization"], "roles": ["payer_exchange_agent"], "goals": ["obtain_pre_authorization"]},
    "PayerSubmission":                  {"actions": ["submit_claim", "receive_payer_response"], "roles": ["payer_exchange_agent"], "goals": ["submit_approved_claim"]},
    "ClaimClosed":                      {"actions": ["receive_payer_response"], "roles": ["payer_exchange_agent"], "goals": ["process_payer_response"]},
}
GOAL_MAP = {
    "verify_eligibility": ["read_insurance_status", "verify_coverage"],
    "prepare_documentation": ["read_clinical_summary", "prepare_medical_justification"],
    "assign_codes": ["assign_diagnosis_code", "assign_procedure_code", "validate_code_consistency"],
    "verify_compliance": ["verify_patient_consent", "validate_minimum_necessary_data", "approve_external_disclosure", "reject_noncompliant_request"],
    "obtain_pre_authorization": ["submit_pre_authorization"],
    "submit_approved_claim": ["submit_claim", "receive_payer_response"],
    "process_payer_response": ["receive_payer_response"],
    "consent_bypass": [], "claim_upcoding": [], "duplicate_claim_submission": [],
    "unauthorized_full_record_disclosure": [], "privilege_escalation": [],
}

def build_full_ontology():
    lines = ["ZERO TRUST ONTOLOGY — Insurance Pre-Authorization and Claims Processing\n",
             "WORKFLOW: insurance_claims_workflow\n  Admissible per state:"]
    for state, info in WORKFLOW_TABLE.items():
        lines.append(f"    {state}: actions={info['actions']}, roles={info['roles']}, goals={info['goals']}")
    lines.append("\nGOAL-ACTION CONSISTENCY MAP (Sem predicate):")
    for goal, actions in GOAL_MAP.items():
        lines.append(f"  {goal} -> permitted: {actions if actions else 'NONE (MALICIOUS - always DENY)'}")
    lines.append("\nTRUST THRESHOLD: tau=0.6. If behavior_normal=false: effective_trust = trust_score * 0.4")
    lines.append("DECISION D(q): ALLOW iff Auth=true AND trust>=0.6 AND WF=true AND Sem=true AND LP=true")
    return "\n".join(lines)

def build_selective_ontology(req):
    """Only the state block for the CURRENT state, and only the goal line for the DECLARED goal."""
    state = req["current_state"]
    goal = req["goal"]
    info = WORKFLOW_TABLE.get(state, {"actions": [], "roles": [], "goals": []})
    permitted = GOAL_MAP.get(goal, [])
    lines = [
        "ZERO TRUST ONTOLOGY — relevant slice for this request\n",
        f"Current workflow state: {state}",
        f"  Permitted actions in this state: {info['actions']}",
        f"  Permitted roles in this state: {info['roles']}",
        f"\nDeclared goal: {goal}",
        f"  Permitted actions for this goal: {permitted if permitted else 'NONE (MALICIOUS - always DENY)'}",
        "\nTRUST THRESHOLD: tau=0.6. If behavior_normal=false: effective_trust = trust_score * 0.4",
        "DECISION D(q): ALLOW iff Auth=true AND trust>=0.6 AND WF=true AND Sem=true AND LP=true",
    ]
    return "\n".join(lines)

FULL_ONTOLOGY = build_full_ontology()

# ---------------------------------------------------------------------------
# Same 20 scenarios as the main insurance benchmark
# ---------------------------------------------------------------------------
SCENARIOS = [
    {"id":"E1","cat":"Legitimate","entity":{"role":"eligibility_agent","authenticated":True,"trust_score":0.88,"behavior_normal":True},
     "req":{"action":"verify_coverage","current_state":"EligibilityVerification","goal":"verify_eligibility"},"gt":"ALLOW"},
    {"id":"E2","cat":"Legitimate","entity":{"role":"coding_agent","authenticated":True,"trust_score":0.85,"behavior_normal":True},
     "req":{"action":"assign_diagnosis_code","current_state":"CodingValidation","goal":"assign_codes"},"gt":"ALLOW"},
    {"id":"E3","cat":"Legitimate","entity":{"role":"compliance_agent","authenticated":True,"trust_score":0.90,"behavior_normal":True},
     "req":{"action":"verify_patient_consent","current_state":"ComplianceReview","goal":"verify_compliance"},"gt":"ALLOW"},
    {"id":"E4","cat":"Legitimate","entity":{"role":"payer_exchange_agent","authenticated":True,"trust_score":0.87,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"PayerSubmission","goal":"submit_approved_claim"},"gt":"ALLOW"},
    {"id":"E5","cat":"Trust","entity":{"role":"payer_exchange_agent","authenticated":True,"trust_score":0.40,"behavior_normal":False},
     "req":{"action":"submit_claim","current_state":"PayerSubmission","goal":"submit_approved_claim"},"gt":"DENY"},
    {"id":"E6","cat":"Trust","entity":{"role":"coding_agent","authenticated":True,"trust_score":0.55,"behavior_normal":False},
     "req":{"action":"assign_procedure_code","current_state":"CodingValidation","goal":"assign_codes"},"gt":"DENY"},
    {"id":"E7","cat":"Trust","entity":{"role":"compliance_agent","authenticated":True,"trust_score":0.35,"behavior_normal":False},
     "req":{"action":"approve_external_disclosure","current_state":"ComplianceReview","goal":"verify_compliance"},"gt":"DENY"},
    {"id":"E8","cat":"Trust","entity":{"role":"eligibility_agent","authenticated":True,"trust_score":0.50,"behavior_normal":False},
     "req":{"action":"verify_coverage","current_state":"EligibilityVerification","goal":"verify_eligibility"},"gt":"DENY"},
    {"id":"E9","cat":"Workflow","entity":{"role":"payer_exchange_agent","authenticated":True,"trust_score":0.85,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"CodingValidation","goal":"submit_approved_claim"},"gt":"DENY"},
    {"id":"E10","cat":"Workflow","entity":{"role":"payer_exchange_agent","authenticated":True,"trust_score":0.88,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"ClaimClosed","goal":"submit_approved_claim"},"gt":"DENY"},
    {"id":"E11","cat":"Workflow","entity":{"role":"eligibility_agent","authenticated":True,"trust_score":0.82,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"PayerSubmission","goal":"submit_approved_claim"},"gt":"DENY"},
    {"id":"E12","cat":"Workflow","entity":{"role":"coding_agent","authenticated":True,"trust_score":0.86,"behavior_normal":True},
     "req":{"action":"approve_external_disclosure","current_state":"ComplianceReview","goal":"assign_codes"},"gt":"DENY"},
    {"id":"E13","cat":"Semantic","entity":{"role":"coding_agent","authenticated":True,"trust_score":0.80,"behavior_normal":True},
     "req":{"action":"assign_procedure_code","current_state":"CodingValidation","goal":"claim_upcoding"},"gt":"DENY"},
    {"id":"E14","cat":"Semantic","entity":{"role":"payer_exchange_agent","authenticated":True,"trust_score":0.85,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"PayerSubmission","goal":"consent_bypass"},"gt":"DENY"},
    {"id":"E15","cat":"Semantic","entity":{"role":"compliance_agent","authenticated":True,"trust_score":0.82,"behavior_normal":True},
     "req":{"action":"approve_external_disclosure","current_state":"ComplianceReview","goal":"unauthorized_full_record_disclosure"},"gt":"DENY"},
    {"id":"E16","cat":"Semantic","entity":{"role":"eligibility_agent","authenticated":True,"trust_score":0.78,"behavior_normal":True},
     "req":{"action":"verify_coverage","current_state":"EligibilityVerification","goal":"privilege_escalation"},"gt":"DENY"},
    {"id":"E17","cat":"Auth","entity":{"role":"eligibility_agent","authenticated":False,"trust_score":0.90,"behavior_normal":True},
     "req":{"action":"read_insurance_status","current_state":"EligibilityVerification","goal":"verify_eligibility"},"gt":"DENY"},
    {"id":"E18","cat":"Auth","entity":{"role":"coding_agent","authenticated":False,"trust_score":0.88,"behavior_normal":True},
     "req":{"action":"assign_diagnosis_code","current_state":"CodingValidation","goal":"assign_codes"},"gt":"DENY"},
    {"id":"E19","cat":"Auth","entity":{"role":"payer_exchange_agent","authenticated":False,"trust_score":0.92,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"PayerSubmission","goal":"submit_approved_claim"},"gt":"DENY"},
    {"id":"E20","cat":"Auth","entity":{"role":"payer_exchange_agent","authenticated":False,"trust_score":0.75,"behavior_normal":True},
     "req":{"action":"submit_claim","current_state":"PayerSubmission","goal":"consent_bypass"},"gt":"DENY"},
]

# ---------------------------------------------------------------------------
def call_ollama(prompt):
    t0 = time.time()
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}
    })
    ms = (time.time() - t0) * 1000
    text = response.json()["response"].strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end]), ms
        except json.JSONDecodeError:
            pass
    return ({"decision": "ALLOW"} if "ALLOW" in text.upper() else {"decision": "DENY"}), ms

def approx_tokens(text):
    # rough heuristic: ~4 chars per token (good enough for a relative comparison)
    return len(text) // 4

def decide_with_ontology(entity, req, ontology_text):
    prompt = ("You are a Zero Trust security agent grounded in the following ontology. "
              "Use ONLY this ontology to make decisions.\n\n" + ontology_text +
              '\nEvaluate D(q): ALLOW only if ALL five conditions pass. '
              'Respond ONLY with JSON: {"decision": "ALLOW" or "DENY"}\n\n'
              f"Entity: {json.dumps(entity)}\nRequest: {json.dumps(req)}")
    result, ms = call_ollama(prompt)
    return result.get("decision", "DENY"), ms, approx_tokens(prompt)

# ---------------------------------------------------------------------------
def run_once():
    full_correct, sel_correct = 0, 0
    full_lat, sel_lat = [], []
    full_tok, sel_tok = [], []

    for s in SCENARIOS:
        gt = s["gt"]

        d_full, ms_full, tok_full = decide_with_ontology(s["entity"], s["req"], FULL_ONTOLOGY)
        selective_ontology = build_selective_ontology(s["req"])
        d_sel, ms_sel, tok_sel = decide_with_ontology(s["entity"], s["req"], selective_ontology)

        full_lat.append(ms_full); sel_lat.append(ms_sel)
        full_tok.append(tok_full); sel_tok.append(tok_sel)
        if d_full == gt: full_correct += 1
        if d_sel == gt: sel_correct += 1

        flag = "  <-- DIFFERENT" if d_full != d_sel else ""
        print(f"  {s['id']} ({s['cat']:<11}): gt={gt:<6} D-full={d_full:<6} D-selective={d_sel:<6}"
              f"  [{ms_full:.0f}ms/{tok_full}tok vs {ms_sel:.0f}ms/{tok_sel}tok]{flag}")

    n = len(SCENARIOS)
    return (full_correct/n*100, sel_correct/n*100,
            statistics.mean(full_lat), statistics.mean(sel_lat),
            statistics.mean(full_tok), statistics.mean(sel_tok))

if __name__ == "__main__":
    print("="*78)
    print(f"  ABLATION: D-full vs D-selective — Model: {MODEL} (insurance_claims_workflow)")
    print("="*78)
    print(f"\nFull ontology prompt size: ~{approx_tokens(FULL_ONTOLOGY)} tokens (all 7 states + 12 goals)")
    print("Selective ontology size varies per request (only relevant state + goal)\n")

    all_full_acc, all_sel_acc = [], []
    all_full_lat, all_sel_lat = [], []
    all_full_tok, all_sel_tok = [], []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n--- RUN {run}/{NUM_RUNS} ---")
        fa, sa, fl, sl, ft, st = run_once()
        all_full_acc.append(fa); all_sel_acc.append(sa)
        all_full_lat.append(fl); all_sel_lat.append(sl)
        all_full_tok.append(ft); all_sel_tok.append(st)
        print(f"Run {run}: D-full={fa:.0f}%  D-selective={sa:.0f}%")

    print("\n" + "="*78)
    print("  FINAL SUMMARY")
    print("="*78)
    print(f"  D-full:      Accuracy {statistics.mean(all_full_acc):5.1f}% +/- {statistics.pstdev(all_full_acc):4.1f}%   "
          f"Latency {statistics.mean(all_full_lat):8.1f}ms   Avg tokens/prompt: {statistics.mean(all_full_tok):.0f}")
    print(f"  D-selective: Accuracy {statistics.mean(all_sel_acc):5.1f}% +/- {statistics.pstdev(all_sel_acc):4.1f}%   "
          f"Latency {statistics.mean(all_sel_lat):8.1f}ms   Avg tokens/prompt: {statistics.mean(all_sel_tok):.0f}")
    tok_reduction = (1 - statistics.mean(all_sel_tok)/statistics.mean(all_full_tok)) * 100
    print(f"\n  Token reduction (selective vs full): {tok_reduction:.0f}%")
    print("="*78)
