# Logs

Development session records and outcomes for the SMPL-X Toolbox project.

## Contents

This directory contains chronological records of development sessions, implementation attempts, and their outcomes. Each log documents what was attempted, what worked, what didn't, and lessons learned.

## Document Types

### Session Logs
- **Implementation Sessions** - Records of feature development work
- **Bug Fix Sessions** - Debugging and problem resolution attempts
- **Research Sessions** - Investigation of technologies, algorithms, or approaches
- **Optimization Sessions** - Performance tuning and algorithm improvement work
- **Integration Sessions** - Connecting components or external systems

### Outcome Categories
- **Success Logs** - Completed implementations with working solutions
- **Failed Logs** - Unsuccessful attempts with analysis of why they failed
- **Partial Logs** - Incomplete work with progress made and next steps
- **Blocked Logs** - Work stopped due to external dependencies or issues

## Naming Convention

Use date prefix with descriptive outcome:
```
YYYY-MM-DD_[task-description]-[outcome].md
```

Examples:
- `2025-09-04_project-structure-setup-complete.md`
- `2025-09-05_core-model-implementation-success.md`
- `2025-09-06_optimization-algorithm-failed.md`
- `2025-09-07_visualization-integration-partial.md`
- `2025-09-08_dependency-upgrade-blocked.md`

## Document Format

Each log should include a HEADER section and structured content:

```markdown
# [Task/Feature Name] - [Outcome]

## HEADER
- **Purpose**: [What was being attempted]
- **Status**: [Completed/Failed/Partial/Blocked]
- **Date**: [YYYY-MM-DD]
- **Dependencies**: [What this work depended on]
- **Target**: [Who this affects - developers, users, AI assistants]

## Objective
[What was the goal of this session]

## Approach
[What strategy or method was used]

## Implementation
[What was actually done - commands run, code written, decisions made]

## Results
[What happened - success metrics, error messages, partial progress]

## Issues Encountered
[Problems that arose and how they were addressed]

## Lessons Learned
[Key insights, things to remember, approaches to avoid/favor]

## Next Steps
[What should be done next, if anything]

## References
[Links to related code, documentation, external resources]
```

## Purpose

Development logs serve to:
- **Track progress** over time and across sessions
- **Preserve knowledge** about what works and what doesn't
- **Avoid repeating mistakes** by documenting failed approaches
- **Provide context** for future development decisions
- **Share experience** between team members and AI assistants
- **Debug issues** by reviewing implementation history

## Usage Guidelines

### When to Create Logs
- At the end of significant development sessions
- When completing features or major components
- After failed attempts that provided valuable insights
- When discovering important workarounds or solutions
- After research sessions that inform future decisions

### What to Include
- **Concrete details** - specific commands, error messages, code snippets
- **Decision rationale** - why certain approaches were chosen
- **Performance data** - timing, memory usage, success rates
- **External factors** - library versions, environment issues
- **Emotional context** - frustration points, surprise discoveries

### What to Avoid
- Vague descriptions without actionable details
- Blame or negative commentary about tools/libraries
- Sensitive information like API keys or passwords
- Excessive code dumps without explanation

## Log Categories

### Success Logs
Document completed work with:
- Final implementation details
- Performance characteristics
- Integration points
- Testing results
- Documentation updates

### Failed Logs
Capture unsuccessful attempts with:
- Detailed error analysis
- Approaches attempted
- Why each approach failed
- Alternative strategies to try
- Resources that might help

### Research Logs
Record investigation sessions with:
- Questions being explored
- Sources consulted
- Findings and conclusions
- Implications for the project
- Recommended next actions

## Integration with Other Context

Logs connect to:
- **Tasks** - Document progress on specific work items
- **Plans** - Show how implementation deviates from or confirms plans
- **Summaries** - Provide raw material for knowledge consolidation
- **Hints** - Generate practical guidance from real experience
- **Design** - Validate or challenge architectural decisions

## Maintenance

- Review logs periodically to extract patterns and insights
- Archive very old logs that are no longer relevant
- Cross-reference related logs when working on similar problems
- Use logs to update project summaries and knowledge base
- Share relevant logs when onboarding new team members or AI assistants
