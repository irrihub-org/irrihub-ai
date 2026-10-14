/**
 * Adds the "Create CRM Lead" node to irriHubRagChatwoot01.json.
 *
 * Written as a script rather than hand-edited JSON so the change is
 * reviewable, repeatable, and can be re-run after the workflow is
 * re-exported from n8n (it is idempotent — running twice is a no-op).
 *
 * Where it sits: on the escalate branch, in PARALLEL with the existing
 * reply/label/assign chain, not in front of it. A CRM outage must never
 * delay the farmer's reply or stop a human being assigned. Losing a lead
 * record is recoverable; leaving a farmer unanswered mid-conversation is
 * not.
 *
 *   Check If Escalate ─true─┬─> Send AI Reply (Escalated) -> Label -> Assign
 *                           └─> Create CRM Lead            (this node)
 *
 * Usage:  node add-create-crm-lead-node.mjs
 */

import { readFileSync, writeFileSync } from 'fs';

const FILE = new URL('./irriHubRagChatwoot01.json', import.meta.url);
const NODE_NAME = 'Create CRM Lead';

const doc = JSON.parse(readFileSync(FILE, 'utf8'));
const wf = Array.isArray(doc) ? doc[0] : doc;

if (wf.nodes.some((n) => n.name === NODE_NAME)) {
  console.log(`"${NODE_NAME}" already present — nothing to do.`);
  process.exit(0);
}

const escalateReply = wf.nodes.find((n) => n.name === 'Send AI Reply (Escalated)');
if (!escalateReply) throw new Error('Send AI Reply (Escalated) not found');

// Sits below the escalation chain so the parallel branch is visible on
// the canvas rather than overlapping it.
const [x, y] = escalateReply.position;

wf.nodes.push({
  parameters: {
    method: 'POST',
    url: 'http://irrihub-app/api/leads/chatwoot',
    sendHeaders: true,
    headerParameters: {
      parameters: [
        {
          name: 'Authorization',
          // Set CHATWOOT_LEAD_WEBHOOK_TOKEN in n8n's environment. The
          // endpoint returns 401 if this does not match the value in the
          // CRM's .env.
          value: '=Bearer {{ $env.CHATWOOT_LEAD_WEBHOOK_TOKEN }}',
        },
        { name: 'Content-Type', value: 'application/json' },
      ],
    },
    sendBody: true,
    specifyBody: 'json',
    // contact_id identifies the person across conversations — the CRM
    // dedups on (source, external_id), so repeat escalations update one
    // party instead of creating duplicates. It comes from the Chatwoot
    // conversation meta rather than the message body.
    //
    // full_name / phone / email come from crm_payload, which the RAG
    // service populates when it recognises contact details in a message.
    // They are absent on most escalations and the CRM treats each as
    // optional, filling in whatever arrives later.
    jsonBody: [
      '={',
      '  "contact_id": "{{ $(\'Squize Webhook Data\').item.json.data.meta?.sender?.id ?? $(\'Squize Webhook Data\').item.json.data.conv_id }}",',
      '  "conversation_id": "{{ $(\'Squize Webhook Data\').item.json.data.conv_id }}",',
      '  "full_name": {{ JSON.stringify($json.crm_payload?.full_name ?? $(\'Squize Webhook Data\').item.json.data.meta?.sender?.name ?? null) }},',
      '  "phone": {{ JSON.stringify($json.crm_payload?.phone ?? $(\'Squize Webhook Data\').item.json.data.meta?.sender?.phone_number ?? null) }},',
      '  "email": {{ JSON.stringify($json.crm_payload?.email ?? $(\'Squize Webhook Data\').item.json.data.meta?.sender?.email ?? null) }},',
      '  "actor_type": "farmer",',
      '  "language": {{ JSON.stringify($json.language ?? "en") }},',
      '  "reason": {{ JSON.stringify($json.category === "lead_capture" ? "Visitor provided contact details in chat" : "Escalated to a human agent by the AI assistant") }}',
      '}',
    ].join('\n'),
    options: {},
  },
  type: 'n8n-nodes-base.httpRequest',
  typeVersion: 4.2,
  position: [x + 272, y + 208],
  id: 'f1a7c3e2-9b4d-4c61-8a2f-6d5e0b7c1a93',
  name: NODE_NAME,
  // A CRM outage must not break the escalation the farmer is waiting on.
  onError: 'continueRegularOutput',
  retryOnFail: true,
  maxTries: 3,
  waitBetweenTries: 2000,
});

// Append to the escalate branch (index 0 = true) alongside the existing
// reply node, rather than replacing it.
const escalateTrue = wf.connections['Check If Escalate'].main[0];
escalateTrue.push({ node: NODE_NAME, type: 'main', index: 0 });

wf.connections[NODE_NAME] = { main: [[]] };

writeFileSync(FILE, JSON.stringify(doc, null, 2) + '\n');
console.log(`Added "${NODE_NAME}" on the escalate branch, parallel to the reply chain.`);
