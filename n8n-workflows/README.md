# n8n workflow backups

Version-controlled exports of the production n8n workflows for the Chatwoot chatbot pipeline, kept here purely as a rollback/diff reference — n8n itself (`n8n.irrihub.com`) is the source of truth, not this directory.

Chatwoot API tokens in these files are replaced with `REDACTED_CHATWOOT_TOKEN` before committing. If you ever need to re-import one of these into n8n, you must manually paste the real tokens back into the relevant HTTP Request nodes' `api_access_token` header parameters first — a redacted import will fail auth against Chatwoot.

- `irriHubRagChatwoot01.json` — the live RAG assistant workflow (webhook path `irrihub-chatwoot-v2`), exported 2026-09-21 as a pre-change baseline before adding presence-aware escalation routing.
- `add-create-crm-lead-node.mjs` — adds the `Create CRM Lead` node to the JSON above. Idempotent; re-run it after any fresh export from n8n.

## Applying the `Create CRM Lead` node

Because n8n is the source of truth, editing the JSON here changes nothing on its own. The node is in the export so it can be reviewed in a diff and rebuilt after a re-export, but **it must still be created in the n8n editor** (or the workflow re-imported) before any lead reaches the CRM.

What it does: on the escalate branch, in parallel with the existing reply/label/assign chain, POST to `/api/leads/chatwoot` on the CRM.

```
Check If Escalate ─true─┬─> Send AI Reply (Escalated) -> Label -> Assign
                        └─> Create CRM Lead
```

Parallel, not in front: a CRM outage must never delay the farmer's reply or stop a human being assigned. The node continues on error, with 3 retries, for the same reason.

Two things it needs:

1. **`CHATWOOT_LEAD_WEBHOOK_TOKEN` in n8n's environment**, matching the value in the CRM's `.env`. The endpoint returns 401 otherwise.
2. **The CRM reachable at `http://irrihub-app`** from the n8n container. Adjust the URL if they are not on a shared Docker network.

`full_name`, `phone` and `email` come from `crm_payload` on the RAG response — populated when the assistant recognises contact details in a message — falling back to the Chatwoot contact's own fields. All three are optional: the CRM fills in whatever arrives on a later escalation, deduplicating on the Chatwoot contact id, so an anonymous first escalation followed by a name is one party, not two.
