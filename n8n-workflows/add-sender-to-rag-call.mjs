/**
 * Adds the `sender` block to the "Call IrriHub RAG /chat" node's body.
 *
 * Without it the assistant cannot know what Chatwoot already holds about
 * the contact, so it asks every visitor for a name even when a pre-chat
 * form already collected one — and, worse, cannot tell whether it has a
 * phone number, which is the detail that makes a lead callable.
 *
 * ChatRequest.sender has existed on the model all along; nothing ever
 * populated it.
 *
 * Idempotent. Usage:  node add-sender-to-rag-call.mjs
 */

import { readFileSync, writeFileSync } from 'fs';

const FILE = new URL('./irriHubRagChatwoot01.json', import.meta.url);
const NODE = 'Call IrriHub RAG /chat';

const doc = JSON.parse(readFileSync(FILE, 'utf8'));
const wf = Array.isArray(doc) ? doc[0] : doc;

const node = wf.nodes.find((n) => n.name === NODE);
if (!node) throw new Error(`${NODE} not found`);

if (node.parameters.jsonBody.includes('"sender"')) {
  console.log('sender block already present — nothing to do.');
  process.exit(0);
}

const SENDER = [
  '  "sender": {',
  '    "name": {{ JSON.stringify($(\'Squize Webhook Data\').item.json.data.meta?.sender?.name ?? null) }},',
  '    "email": {{ JSON.stringify($(\'Squize Webhook Data\').item.json.data.meta?.sender?.email ?? null) }},',
  '    "phone": {{ JSON.stringify($(\'Squize Webhook Data\').item.json.data.meta?.sender?.phone_number ?? null) }}',
  '  }',
].join('\n');

// Append as the last member of the JSON object.
node.parameters.jsonBody = node.parameters.jsonBody.replace(
  /\n\}$/,
  `,\n${SENDER}\n}`,
);

writeFileSync(FILE, JSON.stringify(doc, null, 2) + '\n');
console.log(`Added the sender block to "${NODE}".`);
