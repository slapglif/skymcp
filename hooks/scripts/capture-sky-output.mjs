#!/usr/bin/env node
/**
 * PostToolUse hook: capture-sky-output
 *
 * After any Bash tool invocation, inspect the command and output for SkyPilot
 * operations (launch, exec, serve). Extract job IDs, cluster names, and
 * endpoints, then persist them to /tmp/skymcp-state.json for cross-session
 * tracking.
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";

const STATE_FILE = "/tmp/skymcp-state.json";

const SKY_COMMAND_PATTERNS = [
  "sky launch",
  "sky jobs launch",
  "sky exec",
  "sky serve up",
];

function readState() {
  if (!existsSync(STATE_FILE)) {
    return { tracked_jobs: [], tracked_clusters: [], tracked_services: [] };
  }
  try {
    return JSON.parse(readFileSync(STATE_FILE, "utf-8"));
  } catch {
    return { tracked_jobs: [], tracked_clusters: [], tracked_services: [] };
  }
}

function writeState(state) {
  writeFileSync(STATE_FILE, JSON.stringify(state, null, 2), "utf-8");
}

function parseJobId(output) {
  const patterns = [
    /Job ID:\s*(\d+)/i,
    /Managed job ID:\s*(\d+)/i,
    /job_id=(\d+)/i,
    /Job submitted\.\s*Job ID:\s*(\d+)/i,
  ];
  for (const pattern of patterns) {
    const match = output.match(pattern);
    if (match) return match[1];
  }
  return null;
}

function parseClusterName(output) {
  const patterns = [
    /Cluster:\s*([\w-]+)/i,
    /Launching on cluster\s+'?([\w-]+)'?/i,
    /cluster_name=\s*'?([\w-]+)'?/i,
    /Cluster ([\w-]+) is up/i,
  ];
  for (const pattern of patterns) {
    const match = output.match(pattern);
    if (match) return match[1];
  }
  return null;
}

function parseEndpoint(output) {
  const patterns = [
    /https?:\/\/[\d.]+:\d+/,
    /Endpoint:\s*(https?:\/\/[^\s]+)/i,
    /Replica.*URL:\s*(https?:\/\/[^\s]+)/i,
  ];
  for (const pattern of patterns) {
    const match = output.match(pattern);
    if (match) return match[0].replace(/^Endpoint:\s*/i, "").replace(/^Replica.*URL:\s*/i, "");
  }
  return null;
}

function parseServiceName(command) {
  const match = command.match(/sky serve up\s+\S+\s+-n\s+([\w-]+)/);
  if (match) return match[1];
  return null;
}

function parseClusterFromCommand(command) {
  const match = command.match(/(?:sky launch|sky exec)\s+(?:\S+\s+)?-c\s+([\w-]+)/);
  if (match) return match[1];
  return null;
}

async function main() {
  let input = "";
  try {
    input = readFileSync("/dev/stdin", "utf-8");
  } catch {
    process.exit(0);
  }

  let hookData;
  try {
    hookData = JSON.parse(input);
  } catch {
    process.exit(0);
  }

  const command = hookData?.tool_input?.command || "";
  const output = hookData?.tool_output || "";

  const isSkyCommand = SKY_COMMAND_PATTERNS.some((pattern) =>
    command.includes(pattern)
  );
  if (!isSkyCommand) {
    process.exit(0);
  }

  const state = readState();
  const timestamp = new Date().toISOString();

  const jobId = parseJobId(output);
  const clusterName = parseClusterName(output) || parseClusterFromCommand(command);
  const endpoint = parseEndpoint(output);
  const serviceName = parseServiceName(command);

  if (jobId) {
    const exists = state.tracked_jobs.some((j) => j.job_id === jobId);
    if (!exists) {
      state.tracked_jobs.push({
        job_id: jobId,
        cluster: clusterName || null,
        command: command.slice(0, 200),
        endpoint: endpoint || null,
        launched_at: timestamp,
        status: "submitted",
      });
    }
  }

  if (clusterName) {
    const idx = state.tracked_clusters.findIndex(
      (c) => c.name === clusterName
    );
    const entry = {
      name: clusterName,
      last_command: command.slice(0, 200),
      endpoint: endpoint || null,
      updated_at: timestamp,
    };
    if (idx >= 0) {
      state.tracked_clusters[idx] = {
        ...state.tracked_clusters[idx],
        ...entry,
      };
    } else {
      state.tracked_clusters.push({ ...entry, created_at: timestamp });
    }
  }

  if (serviceName) {
    const exists = state.tracked_services.some((s) => s.name === serviceName);
    if (!exists) {
      state.tracked_services.push({
        name: serviceName,
        endpoint: endpoint || null,
        launched_at: timestamp,
      });
    }
  }

  // Trim to last 50 entries per category to prevent unbounded growth
  state.tracked_jobs = state.tracked_jobs.slice(-50);
  state.tracked_clusters = state.tracked_clusters.slice(-50);
  state.tracked_services = state.tracked_services.slice(-50);

  writeState(state);

  // Return empty -- do not block the tool
  process.exit(0);
}

main();
