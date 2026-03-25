#!/usr/bin/env node
/**
 * SkyMCP MCP Server
 *
 * Wraps SkyPilot CLI commands as structured MCP tools, providing
 * Claude Code with typed access to cluster management, job orchestration,
 * cost reporting, and GPU discovery.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { execSync } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
function runCommand(cmd, timeoutMs = 60_000) {
    try {
        const stdout = execSync(cmd, {
            encoding: "utf-8",
            timeout: timeoutMs,
            stdio: ["pipe", "pipe", "pipe"],
            maxBuffer: 10 * 1024 * 1024,
        });
        return { stdout: stdout.trim(), stderr: "", exitCode: 0 };
    }
    catch (err) {
        const e = err;
        return {
            stdout: (e.stdout || "").toString().trim(),
            stderr: (e.stderr || "").toString().trim(),
            exitCode: e.status ?? 1,
        };
    }
}
function tryParseJson(text) {
    try {
        return JSON.parse(text);
    }
    catch {
        return null;
    }
}
// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------
const server = new McpServer({
    name: "skypilot",
    version: "0.1.0",
});
// ---------------------------------------------------------------------------
// Tool: sky_status
// ---------------------------------------------------------------------------
server.tool("sky_status", "List all SkyPilot clusters, managed jobs, and services with their current status", async () => {
    const clusterResult = runCommand("sky status --refresh 2>&1", 30_000);
    const jobsResult = runCommand("sky jobs queue 2>&1", 15_000);
    const serveResult = runCommand("sky serve status 2>&1", 15_000);
    const response = {
        clusters: {
            raw: clusterResult.stdout,
            exitCode: clusterResult.exitCode,
        },
        managed_jobs: {
            raw: jobsResult.stdout,
            exitCode: jobsResult.exitCode,
        },
        services: {
            raw: serveResult.stdout,
            exitCode: serveResult.exitCode,
        },
    };
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify(response, null, 2),
            },
        ],
    };
});
// ---------------------------------------------------------------------------
// Tool: sky_launch
// ---------------------------------------------------------------------------
server.tool("sky_launch", "Launch a SkyPilot cluster or managed job from a YAML spec", {
    yaml_path: z
        .string()
        .optional()
        .describe("Absolute path to a SkyPilot YAML file"),
    yaml_content: z
        .string()
        .optional()
        .describe("Inline YAML content (will be written to a temp file). Use this OR yaml_path, not both."),
    cluster_name: z
        .string()
        .optional()
        .describe("Cluster name for sky launch -c. If omitted, runs as managed job via sky jobs launch."),
    extra_flags: z
        .string()
        .optional()
        .describe("Additional CLI flags (e.g. '--use-spot --idle-minutes-to-autostop 30')"),
    env_vars: z
        .string()
        .optional()
        .describe('JSON-encoded key-value pairs for --env flags, e.g. \'{"WANDB_API_KEY":"abc","HF_TOKEN":"xyz"}\''),
    dryrun: z
        .boolean()
        .optional()
        .describe("If true, add --dryrun flag to show what would happen"),
}, async ({ yaml_path, yaml_content, cluster_name, extra_flags, env_vars: env_vars_json, dryrun }) => {
    const env_vars = env_vars_json
        ? JSON.parse(env_vars_json)
        : undefined;
    if (!yaml_path && !yaml_content) {
        return {
            content: [
                {
                    type: "text",
                    text: JSON.stringify({
                        error: "Either yaml_path or yaml_content is required",
                    }),
                },
            ],
            isError: true,
        };
    }
    let resolvedPath = yaml_path;
    // Write inline YAML to temp file
    if (yaml_content) {
        const tmpDir = mkdtempSync(join(tmpdir(), "skymcp-"));
        resolvedPath = join(tmpDir, "task.yaml");
        writeFileSync(resolvedPath, yaml_content, "utf-8");
    }
    // Build command
    const isManaged = !cluster_name;
    let cmd;
    if (isManaged) {
        cmd = `sky jobs launch ${resolvedPath} -y`;
    }
    else {
        cmd = `sky launch ${resolvedPath} -c ${cluster_name} -y`;
    }
    if (extra_flags) {
        cmd += ` ${extra_flags}`;
    }
    if (env_vars) {
        for (const [key, val] of Object.entries(env_vars)) {
            cmd += ` --env ${key}=${val}`;
        }
    }
    if (dryrun) {
        cmd += " --dryrun";
    }
    const result = runCommand(cmd, 120_000);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    command: cmd,
                    stdout: result.stdout,
                    stderr: result.stderr,
                    exitCode: result.exitCode,
                }, null, 2),
            },
        ],
        isError: result.exitCode !== 0,
    };
});
// ---------------------------------------------------------------------------
// Tool: sky_logs
// ---------------------------------------------------------------------------
server.tool("sky_logs", "Retrieve recent logs from a SkyPilot cluster or managed job", {
    job_id: z
        .string()
        .optional()
        .describe("Managed job ID (for sky jobs logs)"),
    cluster_name: z
        .string()
        .optional()
        .describe("Cluster name (for sky logs)"),
    tail: z
        .number()
        .optional()
        .describe("Number of tail lines (default 100)"),
}, async ({ job_id, cluster_name, tail }) => {
    if (!job_id && !cluster_name) {
        return {
            content: [
                {
                    type: "text",
                    text: JSON.stringify({
                        error: "Either job_id (for managed jobs) or cluster_name (for clusters) is required",
                    }),
                },
            ],
            isError: true,
        };
    }
    const tailLines = tail || 100;
    let cmd;
    if (job_id) {
        // Managed job logs -- stream the last N lines
        cmd = `sky jobs logs ${job_id} 2>&1 | tail -n ${tailLines}`;
    }
    else {
        cmd = `sky logs ${cluster_name} 2>&1 | tail -n ${tailLines}`;
    }
    const result = runCommand(cmd, 30_000);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    source: job_id ? `managed_job:${job_id}` : `cluster:${cluster_name}`,
                    lines: tailLines,
                    logs: result.stdout,
                    exitCode: result.exitCode,
                }, null, 2),
            },
        ],
    };
});
// ---------------------------------------------------------------------------
// Tool: sky_down
// ---------------------------------------------------------------------------
server.tool("sky_down", "Tear down a SkyPilot cluster, cancel a managed job, or take down a service", {
    cluster_name: z
        .string()
        .optional()
        .describe("Cluster name to tear down (sky down)"),
    job_id: z
        .string()
        .optional()
        .describe("Managed job ID to cancel (sky jobs cancel)"),
    service_name: z
        .string()
        .optional()
        .describe("Service name to tear down (sky serve down)"),
    purge: z
        .boolean()
        .optional()
        .describe("Force purge even if teardown fails"),
}, async ({ cluster_name, job_id, service_name, purge }) => {
    if (!cluster_name && !job_id && !service_name) {
        return {
            content: [
                {
                    type: "text",
                    text: JSON.stringify({
                        error: "One of cluster_name, job_id, or service_name is required",
                    }),
                },
            ],
            isError: true,
        };
    }
    const results = [];
    if (cluster_name) {
        const purgeFlag = purge ? " --purge" : "";
        const cmd = `sky down ${cluster_name} -y${purgeFlag}`;
        const result = runCommand(cmd, 60_000);
        results.push({
            command: cmd,
            stdout: result.stdout || result.stderr,
            exitCode: result.exitCode,
        });
    }
    if (job_id) {
        const cmd = `sky jobs cancel ${job_id} -y`;
        const result = runCommand(cmd, 30_000);
        results.push({
            command: cmd,
            stdout: result.stdout || result.stderr,
            exitCode: result.exitCode,
        });
    }
    if (service_name) {
        const purgeFlag = purge ? " --purge" : "";
        const cmd = `sky serve down ${service_name} -y${purgeFlag}`;
        const result = runCommand(cmd, 60_000);
        results.push({
            command: cmd,
            stdout: result.stdout || result.stderr,
            exitCode: result.exitCode,
        });
    }
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify(results, null, 2),
            },
        ],
        isError: results.some((r) => r.exitCode !== 0),
    };
});
// ---------------------------------------------------------------------------
// Tool: sky_cost
// ---------------------------------------------------------------------------
server.tool("sky_cost", "Show SkyPilot cost report for all managed resources", async () => {
    const result = runCommand("sky cost-report 2>&1", 15_000);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    report: result.stdout,
                    exitCode: result.exitCode,
                }, null, 2),
            },
        ],
    };
});
// ---------------------------------------------------------------------------
// Tool: sky_gpus
// ---------------------------------------------------------------------------
server.tool("sky_gpus", "List GPU availability and pricing across clouds", {
    gpu_type: z
        .string()
        .optional()
        .describe("GPU type with optional count, e.g. 'H100:8', 'A100', 'L4:4'. Omit for all GPUs."),
    cloud: z
        .string()
        .optional()
        .describe("Filter by cloud provider (aws, gcp, azure, lambda, etc.)"),
}, async ({ gpu_type, cloud }) => {
    let cmd = "sky gpus list";
    if (gpu_type) {
        cmd += ` ${gpu_type}`;
    }
    if (cloud) {
        cmd += ` --cloud ${cloud}`;
    }
    cmd += " 2>&1";
    const result = runCommand(cmd, 15_000);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    query: gpu_type || "all",
                    cloud: cloud || "all",
                    pricing: result.stdout,
                    exitCode: result.exitCode,
                }, null, 2),
            },
        ],
    };
});
// ---------------------------------------------------------------------------
// Tool: sky_check
// ---------------------------------------------------------------------------
server.tool("sky_check", "Check which cloud credentials are configured and valid for SkyPilot", async () => {
    const result = runCommand("sky check 2>&1", 30_000);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    credentials: result.stdout,
                    exitCode: result.exitCode,
                }, null, 2),
            },
        ],
    };
});
// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}
main().catch((error) => {
    console.error("SkyMCP server failed to start:", error);
    process.exit(1);
});
//# sourceMappingURL=index.js.map