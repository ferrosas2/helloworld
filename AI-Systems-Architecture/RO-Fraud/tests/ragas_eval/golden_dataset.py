"""
Golden dataset for RAGAS evaluation of the RO-Fraud RAG pipeline.

Each entry represents one complete RAG trace:
  - question:     the incoming claim text sent to the system
  - contexts:     the chunks that Vertex AI Vector Search would retrieve
  - answer:       the expected LLM response (ground truth for scoring)
  - ground_truth: a concise factual reference answer

These are representative of real fraud patterns seen in the pipeline's
training data (all PII already redacted, matching the offline pipeline's
regex scrubbing step).
"""

GOLDEN_DATASET = [
    {
        "question": (
            "Claim C-TEST-001: Customer reports car stolen from parking lot. "
            "No police report filed. Claim amount: $45,000."
        ),
        "contexts": [
            (
                "Claim Context: Stolen art piece, no police report filed. "
                "Contact: [REDACTED_PHONE].\n"
                "Resolution Notes: Repeat offender, identical claim filed 2 years ago across state lines."
            ),
            (
                "Claim Context: Ph: [REDACTED_PHONE]. SSN: [REDACTED_SSN]. "
                "Claim for stolen laptop from car.\n"
                "Resolution Notes: Multiple inconsistencies in police report. Suspected serial fraudster."
            ),
        ],
        "answer": (
            '{"fraud_probability_score": 0.85, '
            '"risk_factors": ["No police report filed for a high-value stolen vehicle", '
            '"Pattern matches historical repeat offender with identical no-report claims", '
            '"High claim amount unsupported by documentation"], '
            '"executive_summary": "The claim mirrors historical fraud patterns where '
            'stolen asset claims are filed without police documentation. A repeat offender '
            'profile was identified with an identical cross-state claim."}'
        ),
        "ground_truth": (
            "High fraud probability. No police report for a $45,000 stolen vehicle directly "
            "matches historical fraudulent claim patterns. Repeat offender indicators present."
        ),
    },
    {
        "question": (
            "Claim C-TEST-002: Customer claims $50,000 water damage to property. "
            "Incident date: July 12. Weather records show no precipitation that day."
        ),
        "contexts": [
            (
                "Claim Context: SSN [REDACTED_SSN]. Claimed $50k for water damage "
                "but weather was sunny.\n"
                "Resolution Notes: Weather data contradicts claim completely. Fabricated event."
            ),
        ],
        "answer": (
            '{"fraud_probability_score": 0.92, '
            '"risk_factors": ["Weather records directly contradict the claimed damage event", '
            '"Claim amount matches historical fabricated water damage case", '
            '"No corroborating evidence provided"], '
            '"executive_summary": "Weather data for the incident date shows no precipitation, '
            'directly contradicting the water damage claim. This matches a confirmed fraudulent '
            'case in the historical database where identical contradictions were found."}'
        ),
        "ground_truth": (
            "Very high fraud probability. Weather records contradict the event date. "
            "Matches confirmed historical fraud case with identical weather contradiction pattern."
        ),
    },
    {
        "question": (
            "Claim C-TEST-003: Customer reports minor fender bender. "
            "Provides repair estimates and photos. Claim amount: $1,200."
        ),
        "contexts": [
            (
                "Claim Context: Customer reported minor fender bender. No PII.\n"
                "Resolution Notes: Standard processing, approved. Valid repair estimates."
            ),
        ],
        "answer": (
            '{"fraud_probability_score": 0.08, '
            '"risk_factors": [], '
            '"executive_summary": "The claim closely matches a historically approved '
            'legitimate fender bender case. Documentation provided is consistent with '
            'the claim type and amount. Low fraud risk."}'
        ),
        "ground_truth": (
            "Low fraud probability. Claim matches historical legitimate fender bender pattern "
            "with proper documentation. Standard processing recommended."
        ),
    },
]
