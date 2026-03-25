#!/usr/bin/env node
/**
 * SessionStart hook: load-sky-context
 *
 * On session start, query SkyPilot for active clusters, running jobs, and
 * services. Output a brief context summary so the agent knows what cloud
 * resources are live.
 */

import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";

const STATE_FILE = "/tmp/skymcp-state.json";

function tryExec(cmd) {
  try {
    return execSync(cmd, {
      encoding: "utf-8",
      timeout: 8000,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
  } catch {
    return "";
  }
}

function parseClusters(statusOutput) {
  if (!statusOutput) return [];
  const clusters = [];
  const lines = statusOutput.split("\n");
  for (const line of lines) {
    // sky status table rows contain cluster info after the header
    // Format varies but typically: NAME  LAUNCHED  RESOURCES  STATUS  ...
    const trimmed = line.trim();
    if (
      !trimmed ||
      trimmed.startsWith("NAME") ||
      trimmed.startsWith("-") ||
      trimmed.startsWith("Cluster") ||
      trimmed.startsWith("No cluster") ||
      trimmed.startsWith("To show")
    ) {
      continue;
    }
    const parts = trimmed.split(/\s{2,}/);
    if (parts.length >= 3) {
      clusters.push({
        name: parts[0],
        resources: parts.find((p) => /[A-Z]\d+/.test(p)) || parts[2],
        status: parts.find(
          (p) =>
            p === "UP" ||
            p === "STOPPED" ||
            p === "INIT" ||
            p === "PROVISIONING"
        ) || "UNKNOWN",
      });
    }
  }
  return clusters;
}

function parseJobs(queueOutput) {
  if (!queueOutput) return [];
  const jobs = [];
  const lines = queueOutput.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (
      !trimmed ||
      trimmed.startsWith("ID") ||
      trimmed.startsWith("-") ||
      trimmed.startsWith("No managed") ||
      trimmed.startsWith("To show")
    ) {
      continue;
    }
    const parts = trimmed.split(/\s{2,}/);
    if (parts.length >= 3) {
      const id = parts[0];
      const name = parts[1] || "";
      const status =
        parts.find(
          (p) =>
            p === "RUNNING" ||
            p === "PENDING" ||
            p === "SUCCEEDED" ||
            p === "FAILED" ||
            p === "RECOVERING" ||
            p === "SUBMITTED" ||
            p === "STARTING" ||
            p === "CANCELLING"
        ) || "UNKNOWN";
      if (
        status === "RUNNING" ||
        status === "PENDING" ||
        status === "RECOVERING" ||
        status === "SUBMITTED" ||
        status === "STARTING"
      ) {
        jobs.push({ id, name, status });
      }
    }
  }
  return jobs;
}

function parseServices(serveOutput) {
  if (!serveOutput) return [];
  const services = [];
  const lines = serveOutput.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (
      !trimmed ||
      trimmed.startsWith("NAME") ||
      trimmed.startsWith("-") ||
      trimmed.startsWith("No service")
    ) {
      continue;
    }
    const parts = trimmed.split(/\s{2,}/);
    if (parts.length >= 2) {
      services.push({ name: parts[0], status: parts[1] || "UNKNOWN" });
    }
  }
  return services;
}

function estimateCost(clusters) {
  // Rough hourly cost estimates per GPU type (spot prices)
  const gpuCosts = {
    H100: 3.5,
    A100: 1.8,
    "A100-80GB": 2.2,
    L4: 0.4,
    T4: 0.35,
    V100: 0.8,
    A10G: 0.75,
    L40: 1.2,
    L40S: 1.3,
  };

  let total = 0;
  for (const cluster of clusters) {
    if (cluster.status !== "UP") continue;
    const res = cluster.resources || "";
    for (const [gpu, cost] of Object.entries(gpuCosts)) {
      if (res.includes(gpu)) {
        const countMatch = res.match(new RegExp(`${gpu}:(\\d+)`));
        const count = countMatch ? parseInt(countMatch[1]) : 1;
        total += cost * count;
        break;
      }
    }
  }
  return total;
}

function loadTrackedState() {
  if (!existsSync(STATE_FILE)) return null;
  try {
    return JSON.parse(readFileSync(STATE_FILE, "utf-8"));
  } catch {
    return null;
  }
}

function main() {
  const statusOutput = tryExec("sky status 2>/dev/null");
  const jobsOutput = tryExec("sky jobs queue 2>/dev/null");
  const serveOutput = tryExec("sky serve status 2>/dev/null");
  const trackedState = loadTrackedState();

  const clusters = parseClusters(statusOutput);
  const activeClusters = clusters.filter(
    (c) => c.status === "UP" || c.status === "INIT" || c.status === "PROVISIONING"
  );
  const jobs = parseJobs(jobsOutput);
  const services = parseServices(serveOutput);

  const parts = [];

  if (activeClusters.length > 0) {
    const clusterDescs = activeClusters
      .map((c) => `${c.name} on ${c.resources}`)
      .join(", ");
    parts.push(
      `${activeClusters.length} active cluster${activeClusters.length > 1 ? "s" : ""} (${clusterDescs})`
    );
  }

  if (jobs.length > 0) {
    const runningJobs = jobs.filter((j) => j.status === "RUNNING");
    const pendingJobs = jobs.filter(
      (j) => j.status !== "RUNNING"
    );
    const jobDescs = [];
    if (runningJobs.length > 0) {
      jobDescs.push(
        `${runningJobs.length} running (${runningJobs.map((j) => j.name || `#${j.id}`).join(", ")})`
      );
    }
    if (pendingJobs.length > 0) {
      jobDescs.push(`${pendingJobs.length} pending`);
    }
    parts.push(`jobs: ${jobDescs.join(", ")}`);
  }

  if (services.length > 0) {
    parts.push(
      `${services.length} service${services.length > 1 ? "s" : ""} (${services.map((s) => s.name).join(", ")})`
    );
  }

  const estCost = estimateCost(activeClusters);

  if (trackedState) {
    const recentJobs = (trackedState.tracked_jobs || []).filter((j) => {
      const age = Date.now() - new Date(j.launched_at).getTime();
      return age < 24 * 60 * 60 * 1000; // last 24h
    });
    if (recentJobs.length > 0 && jobs.length === 0) {
      parts.push(
        `${recentJobs.length} tracked job${recentJobs.length > 1 ? "s" : ""} from previous sessions`
      );
    }
  }

  if (parts.length === 0) {
    // Nothing active -- stay silent
    process.exit(0);
  }

  let message = `SkyPilot: ${parts.join(", ")}`;
  if (estCost > 0) {
    message += `, est. cost $${estCost.toFixed(2)}/hr`;
  }

  // Output the context message for Claude Code
  process.stdout.write(message + "\n");
}

main();
