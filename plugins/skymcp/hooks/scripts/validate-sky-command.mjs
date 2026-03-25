#!/usr/bin/env node
/**
 * PreToolUse hook: validate-sky-command
 *
 * Before Bash execution, inspect the command for SkyPilot operations that
 * could be dangerous or suboptimal:
 *  - "sky down" without confirmation: warn about cluster teardown
 *  - Expensive GPU launch without --use-spot: suggest spot instances
 *  - Launch without --idle-minutes-to-autostop: suggest autostop
 */

import { readFileSync } from "node:fs";

const EXPENSIVE_GPUS = ["H100", "A100", "A100-80GB", "L40S", "L40", "MI300X"];

function main() {
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
  const warnings = [];

  // Check for "sky down" without explicit confirmation
  if (/sky\s+down\b/.test(command) || /sky\s+jobs\s+cancel\b/.test(command) || /sky\s+serve\s+down\b/.test(command)) {
    const hasYesFlag = /\s-y\b/.test(command) || /\s--yes\b/.test(command);
    if (!hasYesFlag) {
      warnings.push(
        "About to tear down cluster/cancel job. Confirm this is intentional. Add -y to skip prompt."
      );
    }
  }

  // Check for expensive GPU launch without --use-spot
  if (/sky\s+(launch|jobs\s+launch)\b/.test(command)) {
    const hasSpotFlag = /--use-spot/.test(command);
    const hasExpensiveGpu = EXPENSIVE_GPUS.some((gpu) =>
      command.includes(gpu)
    );

    // Also check if the YAML content might specify use_spot already
    // (we can only check the command itself here)
    if (hasExpensiveGpu && !hasSpotFlag) {
      warnings.push(
        `Launching expensive GPU without --use-spot. Consider adding --use-spot for 60-90% cost savings. (Skip if YAML already sets use_spot: true)`
      );
    }

    // Check for missing autostop
    const hasAutostop = /--idle-minutes-to-autostop/.test(command);
    const hasDown = /--down\b/.test(command);
    const isManagedJob = /sky\s+jobs\s+launch/.test(command);
    // Managed jobs auto-clean up, so autostop is not needed
    if (!hasAutostop && !hasDown && !isManagedJob) {
      warnings.push(
        "No --idle-minutes-to-autostop set. Consider adding --idle-minutes-to-autostop 30 to avoid runaway costs."
      );
    }
  }

  if (warnings.length > 0) {
    const message = warnings.join("\n");
    process.stdout.write(message + "\n");
  }

  process.exit(0);
}

main();
