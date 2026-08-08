# Aethelis Player Client

This directory contains the browser-runnable, player-facing client only.

## Boundary

- `src/api/`: typed HTTP contracts and the local Product API adapter.
- `src/app/`: client-side orchestration and reconstructable UI state.
- `src/features/`: player-task views such as New Game, timelines, and settings.
- `src/game/`: the narrow imperative PixiJS presentation boundary.
- `src/styles/`: visual tokens, base rules, and responsive product layout.

World rules, authorization, save truth, timeline lineage, commands, governance, snapshots,
and content publication remain in the Python backend under `src/aethelis/`. The frontend
must not import backend modules, write the database, or infer hidden WorldState.

## Commands

```powershell
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend lint
pnpm --dir frontend test
pnpm --dir frontend build
pnpm --dir frontend dev
```

If `pnpm` is not registered as a PowerShell command, use the equivalent Corepack form:

```powershell
corepack pnpm --dir frontend install --frozen-lockfile
```

For the complete local stack, run `scripts/dev.ps1` from the repository root after applying
the database migrations. The client expects the Product API at
`http://127.0.0.1:8000/api/v1` and is served from the exact allowed origin
`http://localhost:5173`.

`scripts/dev.ps1` writes API, Worker, and frontend logs to `tmp/dev/`. Use
`scripts/dev.ps1 -ExitAfterReady` for a bounded startup health check that cleans up all
validation processes after both HTTP endpoints respond.
