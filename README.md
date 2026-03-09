## `pes_noma`

`pes_noma` is the Noma-maintained fork of the original `pes` extension.  
The upstream/original `pes` repository is primarily maintained by Ricardo and Jesus. This fork exists so the Noma team can:

- ship Noma-specific adjustments faster (without blocking on upstream release cadence)
- experiment safely on our own branches
- still **track upstream changes** and regularly pull them into this fork

## Repository notes

- This repo lives under `noma_backend_local/extensions/`.
- Main Python package code is under `package/pes_noma/` (e.g. `package/pes_noma/handlers/`).
- PRs should follow the template in `.github/pull_request_template.md`.

# !!!
# EVERYTHIN BELOW WAS NOT VALIDATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!

## Upstream sync: updating this fork from the original `pes` repo

The recommended workflow is **PR-based**: pull upstream changes into a dedicated “sync” branch in this fork, then merge via PR into our `main` (and/or other long-lived branches). This keeps the update auditable and conflict resolution reviewable.

### 1) One-time setup: add the upstream remote

In your local clone of `pes_noma`:

```bash
# Verify your current remotes (origin should point to this fork)
git remote -v

# Add upstream (the original pes repository)
git remote add upstream <UPSTREAM_PES_GIT_URL>

# Example (SSH):
# git remote add upstream git@github.com:<org-or-user>/pes.git
```

Validate:

```bash
git remote -v
```

### 2) Fetch upstream updates

```bash
git fetch upstream --prune --tags
```

### 3) Create a sync branch in the fork

Start from your fork’s `main` (or whichever branch you want to update):

```bash
git checkout main
git pull --ff-only origin main

# Create a dedicated sync branch (date is optional but useful)
git checkout -b sync/upstream-$(date +%Y%m%d)
```

### 4) Integrate upstream changes (choose ONE approach)

#### Option A (recommended): merge upstream into the sync branch

This preserves a clear boundary between “upstream updates” and “fork-only changes”.

```bash
# If upstream's default branch is main:
git merge --no-ff upstream/main

# If upstream uses master:
# git merge --no-ff upstream/master
```

If there are conflicts, resolve them, then:

```bash
git add -A
git commit
```

#### Option B: rebase onto upstream

Use this only if your team prefers a linear history and you understand the implications (rebasing rewrites commits).

```bash
git rebase upstream/main
```

### 5) Push the sync branch and open a PR

```bash
git push -u origin HEAD
```

Open a PR from `sync/upstream-YYYYMMDD` → `main` in this fork, and in the PR:

- summarize upstream changes (link commits/releases if available)
- call out conflicts and how they were resolved
- include any follow-up work needed for Noma compatibility

### 6) Keeping long-lived feature branches up to date

If you have an active branch and want the latest upstream changes, do **not** merge upstream directly into many branches independently (it creates repeated conflict work). Prefer:

- update `main` via the sync PR flow above
- then update your feature branch from fork `main`:

```bash
git checkout my-feature
git fetch origin --prune
git merge origin/main

# or (if your branch is private and you prefer linear history)
# git rebase origin/main
```

## Best practices (fork maintenance)

- **Keep “upstream sync” isolated**: do upstream updates in a `sync/*` branch and land via PR.
- **Avoid rewriting shared history**: don’t rebase `main` (or any shared branch).
- **Document fork-only deltas**: when you add Noma-specific behavior, explain it in the PR so future upstream syncs are easier.
- **Sync frequently**: smaller, frequent upstream merges are easier than large, infrequent ones.
