# Node.js tooling

The Streamlit applications use Python. The repository also has Node.js packages
for `neon.ts` and launches browser MCP servers from `subagents.py`.

Use pnpm **10.26.2**, pinned in `package.json`, for Node.js dependencies. With
Node.js and Corepack installed, enable the pnpm shim:

```sh
corepack enable pnpm
pnpm --version
pnpm install --frozen-lockfile
```

If Corepack is unavailable, install that pnpm version using an option in the
[official pnpm installation guide](https://pnpm.io/10.x/installation).

Commit `pnpm-lock.yaml` when changing dependencies. Use `pnpm add <package>` or
`pnpm remove <package>` to update dependencies and the lockfile together.

The browser MCP servers require Node.js, Chrome, and `pnpm` on the Streamlit
process's PATH. Their launch commands are:

```sh
pnpm dlx @playwright/mcp@latest
pnpm dlx chrome-devtools-mcp@latest
```

These fetch and run the MCP packages on demand; their existing `@latest` version
selection is retained. See the [pnpm dlx documentation](https://pnpm.io/10.x/cli/dlx).
