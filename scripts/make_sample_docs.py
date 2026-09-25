"""Generate the sample document set (fictional company "Northwind Cloud") into data/sample.

    uv run python -m scripts.make_sample_docs
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"

# Each PDF is a list of pages; each page is a list of (heading, body) sections.
PDFS = {
    "incident_response_runbook.pdf": [
        [
            ("Northwind Cloud - Incident Response Runbook (v3.2, effective January 2026)", ""),
            ("1. Purpose",
             "This runbook defines how Northwind Cloud detects, classifies, escalates and resolves production "
             "incidents. It applies to every engineer on the on-call rotation and to the Incident Commander role."),
            ("2. Severity levels",
             "SEV1: complete outage of a customer-facing service, data loss, or a confirmed security breach. "
             "Acknowledge within 5 minutes and mitigate within 1 hour. "
             "SEV2: major degradation affecting more than 10% of customers or a single enterprise customer. "
             "Acknowledge within 15 minutes and mitigate within 4 hours. "
             "SEV3: minor degradation with a workaround available. Acknowledge within 1 business hour. "
             "SEV4: cosmetic issues and internal tooling problems, handled through the normal backlog."),
        ],
        [
            ("3. Escalation",
             "For a SEV1 incident the primary on-call engineer must page the secondary on-call engineer, the "
             "Incident Commander on duty and the VP of Engineering through PagerDuty. If the primary on-call "
             "engineer does not acknowledge within 5 minutes, PagerDuty automatically escalates to the secondary. "
             "For a SEV2 incident, page the secondary on-call and the Incident Commander."),
            ("4. Communication",
             "The Incident Commander opens a dedicated Slack channel named #inc-YYYYMMDD-short-name. "
             "For SEV1 and SEV2 incidents the status page must be updated within 15 minutes of declaration and "
             "then at least every 30 minutes until resolution. Customer Support is notified in #support-escalations."),
            ("5. Closing an incident",
             "An incident is closed only after metrics have been stable for 30 minutes. A blameless postmortem is "
             "mandatory for every SEV1 and SEV2 incident and must be published within 5 business days."),
        ],
    ],
    "deployment_policy.pdf": [
        [
            ("Northwind Cloud - Production Deployment Policy", ""),
            ("Deployment windows",
             "Production deployments are allowed Monday to Thursday between 09:00 and 16:00 IST. Deployments on "
             "Friday after 12:00 IST, on weekends and on public holidays are not permitted unless they are an "
             "emergency fix for an active SEV1 or SEV2 incident approved by the Incident Commander."),
            ("Change freeze",
             "A company-wide change freeze applies from 20 December to 3 January every year, and during the 48 "
             "hours before a major customer launch. Only emergency fixes may be deployed during a freeze."),
            ("Approvals",
             "Every production change requires one approving code review from an engineer who did not author the "
             "change, a green CI pipeline, and a linked ticket. Database schema migrations additionally require "
             "approval from a member of the Data Platform team."),
        ],
        [
            ("Rollouts and rollback",
             "All services are deployed with a canary release: 5% of traffic for 15 minutes, then 50% for 15 "
             "minutes, then 100%. The deploy is automatically rolled back if the error rate rises above 1% or p99 "
             "latency increases by more than 20% compared with the previous version. Every change must have a "
             "tested rollback plan; destructive migrations must be split into expand and contract steps."),
        ],
    ],
    "postmortem_2026_03_db_outage.pdf": [
        [
            ("Postmortem: Orders database outage, 14 March 2026 (SEV1)", ""),
            ("Summary",
             "On 14 March 2026 the orders API was unavailable for 47 minutes, from 10:12 to 10:59 IST. "
             "Approximately 18,000 checkout requests failed. No data was lost."),
            ("Root cause",
             "A schema migration added an index to the orders table without using CONCURRENTLY. The migration "
             "took an exclusive lock on the table, which blocked all writes. Connection pools on the orders API "
             "filled up with waiting connections and health checks began failing, which caused the load balancer "
             "to remove every instance. The migration had been approved but had not been tested against a "
             "production-sized dataset."),
            ("Timeline",
             "10:12 migration started by the deploy pipeline. 10:15 alerts fired for checkout error rate. "
             "10:19 SEV1 declared and Incident Commander assigned. 10:41 the migration was identified as the cause "
             "and cancelled. 10:59 error rates returned to normal."),
        ],
        [
            ("Action items",
             "1. Add a CI lint rule that rejects CREATE INDEX without CONCURRENTLY on large tables (owner: Data "
             "Platform, due 31 March 2026). 2. Run all migrations against a staging copy with production-sized "
             "data before approval (owner: Release Engineering, due 15 April 2026). 3. Set lock_timeout to 5 "
             "seconds for migration sessions (owner: Data Platform, due 21 March 2026). 4. Add a dashboard panel "
             "for database lock waits (owner: SRE, due 31 March 2026)."),
            ("What went well",
             "Alerting fired within 3 minutes and the status page was updated within 10 minutes of declaration."),
        ],
    ],
}

TEXT_DOC = ("access_security_policy.txt", """Northwind Cloud - Access and Secrets Security Policy

Multi-factor authentication
MFA is mandatory for all employees on email, SSO, the AWS console, GitHub and PagerDuty. Hardware security keys are required for anyone with production administrator access.

Credentials and secrets
Long-lived AWS access keys must be rotated at least every 90 days; engineers should prefer short-lived credentials through AWS IAM Identity Center. Secrets must be stored in AWS Secrets Manager or SSM Parameter Store and must never be committed to Git. A secret that is committed by mistake must be treated as compromised and rotated within 4 hours.

Production access
Standing access to production is limited to the SRE team. Other engineers request just-in-time access through the access portal; grants expire after 8 hours and every grant is logged for audit.

Offboarding
All access for a departing employee must be revoked within 24 hours of their last working day. Their manager is responsible for filing the offboarding ticket.

Reporting
Suspected security incidents must be reported immediately in #security and are treated as SEV1 until triaged.
""")


def _latin1(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(pages, path: Path) -> None:
    pdf = FPDF()
    pdf.set_margins(20, 20)
    for page in pages:
        pdf.add_page()
        for heading, body in page:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, _latin1(heading), new_x="LMARGIN", new_y="NEXT")
            if body:
                pdf.set_font("Helvetica", size=11)
                pdf.multi_cell(0, 6, _latin1(body), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
    pdf.output(str(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in PDFS.items():
        build_pdf(pages, OUT / name)
    (OUT / TEXT_DOC[0]).write_text(TEXT_DOC[1])
    print(f"wrote {len(PDFS) + 1} documents to {OUT}")


if __name__ == "__main__":
    main()
