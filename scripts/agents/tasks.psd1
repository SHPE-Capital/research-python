# Task graph for orchestrate.ps1, taken from docs/handoff/README.md (All packets).
#
# Order matters: when several tasks are ready at once, the ones listed first
# launch first. Tasks that unblock the most come first in each wave.
#
# Fields
#   Id      task ID; branch is <name>/<Id>, worktree is <root>\<Id>
#   Agent   implementing agent from .claude/agents/
#   Wave    0 = lead setup; every task with Wave > 0 also waits for Gate
#   Deps    hard dependencies that must be merged into the base ref first
#   Kind    packet (default): docs/handoff/tasks/<Id>.md
#           stub: no packet; spec is the docs/tasks.md row and TODO(<Id>) stub
#           migration, deps: Wave 0 lead steps described in docs/handoff/README.md
#
# Done tasks (C0, D1-D5) and platform-repo tasks (P6-P9) are not listed.
# Differences from the README: C1c waits for T1 so triple_barrier ships in the
# same PR; M3 waits for M1a ("M1a or M1b"); C1n waits for the main Wave 2 tasks.
@{
    Gate  = @('MIG', 'DEPS', 'C1a', 'C1h', 'C1o', 'C1l')

    Tasks = @(
        # Wave 0: lead setup, one after another
        @{ Id = 'MIG';   Agent = 'repo-architect';           Wave = 0; Deps = @();                 Kind = 'migration' }
        @{ Id = 'C1a';   Agent = 'repo-architect';           Wave = 0; Deps = @('MIG') }
        @{ Id = 'DEPS';  Agent = 'repo-architect';           Wave = 0; Deps = @('MIG');            Kind = 'deps' }
        @{ Id = 'C1o';   Agent = 'repo-architect';           Wave = 0; Deps = @('C1a') }
        @{ Id = 'C1h';   Agent = 'repo-architect';           Wave = 0; Deps = @('C1a') }
        @{ Id = 'C1l';   Agent = 'repo-architect';           Wave = 0; Deps = @('C1a', 'DEPS') }

        # Wave 1, batch A: unblock the most
        @{ Id = 'T1';    Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'C1d';   Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'C1b';   Agent = 'repo-architect';           Wave = 1; Deps = @() }
        @{ Id = 'SS2';   Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'SS3';   Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'E1';    Agent = 'research-analyst';         Wave = 1; Deps = @() }
        @{ Id = 'V1';    Agent = 'repo-architect';           Wave = 1; Deps = @() }

        # Wave 1, batch B
        @{ Id = 'T2';    Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'T4';    Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'T5';    Agent = 'trade-management-analyst'; Wave = 1; Deps = @() }
        @{ Id = 'F2';    Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'F3';    Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'F1';    Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'F4';    Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'F5';    Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'C1e';   Agent = 'repo-architect';           Wave = 1; Deps = @() }
        @{ Id = 'C1k';   Agent = 'repo-architect';           Wave = 1; Deps = @() }
        @{ Id = 'C1g';   Agent = 'repo-architect';           Wave = 1; Deps = @() }
        @{ Id = 'M1a';   Agent = 'research-analyst';         Wave = 1; Deps = @();                 Kind = 'stub' }
        @{ Id = 'M1b';   Agent = 'research-analyst';         Wave = 1; Deps = @();                 Kind = 'stub' }
        @{ Id = 'M1c';   Agent = 'research-analyst';         Wave = 1; Deps = @();                 Kind = 'stub' }
        @{ Id = 'L1';    Agent = 'feature-engineer';         Wave = 1; Deps = @();                 Kind = 'stub' }
        @{ Id = 'E2';    Agent = 'research-analyst';         Wave = 1; Deps = @() }
        @{ Id = 'W0';    Agent = 'workshop-author';          Wave = 1; Deps = @() }
        @{ Id = 'T3';    Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'SM2';   Agent = 'strategy-builder';         Wave = 1; Deps = @() }
        @{ Id = 'M2';    Agent = 'research-analyst';         Wave = 1; Deps = @();                 Kind = 'stub' }
        @{ Id = 'C1c';   Agent = 'repo-architect';           Wave = 1; Deps = @('T1') }
        @{ Id = 'M3';    Agent = 'platform-parity';          Wave = 1; Deps = @('M1a') }
        @{ Id = 'F6';    Agent = 'feature-engineer';         Wave = 1; Deps = @() }
        @{ Id = 'D6';    Agent = 'repo-architect';           Wave = 1; Deps = @();                 Kind = 'stub' }

        # Wave 2
        @{ Id = 'T7';    Agent = 'platform-parity';          Wave = 2; Deps = @('T1', 'T2', 'T4', 'T5') }
        @{ Id = 'T6';    Agent = 'trade-management-analyst'; Wave = 2; Deps = @('T1', 'E1') }
        @{ Id = 'SM1';   Agent = 'strategy-builder';         Wave = 2; Deps = @('C1b', 'T1') }
        @{ Id = 'SS4';   Agent = 'strategy-builder';         Wave = 2; Deps = @('SS2', 'SS3', 'T1') }
        @{ Id = 'SR1';   Agent = 'strategy-builder';         Wave = 2; Deps = @('T1') }
        @{ Id = 'SR2';   Agent = 'strategy-builder';         Wave = 2; Deps = @('T1', 'F2', 'F3') }
        @{ Id = 'F7-F8'; Agent = 'feature-engineer';         Wave = 2; Deps = @('C1d', 'SS2', 'SS3') }
        @{ Id = 'T9';    Agent = 'strategy-builder';         Wave = 2; Deps = @('C1d', 'T5') }
        @{ Id = 'SS1';   Agent = 'strategy-builder';         Wave = 2; Deps = @('SS2', 'SS3') }
        @{ Id = 'SM3';   Agent = 'strategy-builder';         Wave = 2; Deps = @('C1d', 'C1e') }
        @{ Id = 'L2';    Agent = 'strategy-builder';         Wave = 2; Deps = @('T1', 'C1c') }
        @{ Id = 'C1f';   Agent = 'repo-architect';           Wave = 2; Deps = @('V1') }
        @{ Id = 'C1i';   Agent = 'research-analyst';         Wave = 2; Deps = @('E1') }
        @{ Id = 'C1j';   Agent = 'repo-architect';           Wave = 2; Deps = @('C1k') }
        @{ Id = 'C1m';   Agent = 'repo-architect';           Wave = 2; Deps = @('C1g') }

        # Wave 3 (W1 and W2 usually unblock early)
        @{ Id = 'W1';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0') }
        @{ Id = 'W2';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'M1a', 'M1b', 'M1c', 'F1', 'F2', 'F3', 'F5', 'V1') }
        @{ Id = 'T8';    Agent = 'repo-architect';           Wave = 3; Deps = @('C1j', 'T7') }
        @{ Id = 'SR3';   Agent = 'strategy-builder';         Wave = 3; Deps = @('F7-F8', 'T1') }
        @{ Id = 'W3';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'C1b', 'C1d', 'C1e', 'L1', 'V1', 'E1', 'M1b', 'SM3') }
        @{ Id = 'W4';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'SM1', 'F1', 'F2', 'F3', 'F4', 'F7-F8', 'T1', 'C1b', 'C1c', 'M1a', 'M1c', 'E1', 'E2') }
        @{ Id = 'W5';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'F3', 'C1e', 'C1c', 'C1f', 'M1b', 'E2') }
        @{ Id = 'W6';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'T1', 'C1b', 'E2', 'M2', 'T6', 'T5') }
        @{ Id = 'W7';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'SS1', 'SS2', 'SS3', 'SS4', 'F7-F8', 'T1', 'T5', 'E2', 'M2') }
        @{ Id = 'W8';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'C1d', 'C1e', 'F3', 'F7-F8', 'T4', 'T5', 'E1', 'E2', 'T9') }
        @{ Id = 'W9';    Agent = 'workshop-author';          Wave = 3; Deps = @('W0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'SM1', 'SR1', 'SR2', 'SS3', 'SS4', 'F3', 'E2', 'M2') }
        @{ Id = 'C1n';   Agent = 'repo-architect';           Wave = 3; Deps = @('C1f', 'C1i', 'C1j', 'C1m', 'T7', 'T9') }
    )
}
