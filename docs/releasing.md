# Releasing

The version is derived entirely from git tags — there is no version number to edit in code. Publishing a release is a five-step process.

---

## 1. Make sure main is ready

All intended changes merged, CI green.

---

## 2. Tag the commit

Tags follow [Semantic Versioning](https://semver.org): `vMAJOR.MINOR.PATCH`.

```bash
git checkout main
git pull
git tag v0.2.0
git push origin v0.2.0
```

**Guidance on which number to increment:**

| Bump | When |
|---|---|
| `PATCH` | Bug fixes, dependency updates, no API change |
| `MINOR` | New tool, new engine option, new middleware hook |
| `MAJOR` | Breaking change to an existing method signature or return type |

---

## 3. Publish a GitHub release

1. Go to `https://github.com/nanacnote/keel/releases/new`
2. Choose the tag you just pushed (`v0.2.0`) from the *Choose a tag* dropdown
3. Set the title to the tag name (`v0.2.0`)
4. Write release notes (see [below](#writing-release-notes))
5. Click **Publish release**

Publishing the release fires the `publish.yml` workflow automatically. Do not use *Save as draft* — that does not trigger the workflow.

---

## 4. Verify the workflow

Go to `https://github.com/nanacnote/keel/actions` and confirm the *Publish* run completes successfully. On success, two files are attached to the release:

- `keel-0.2.0-py3-none-any.whl` — the wheel (preferred for installs)
- `keel-0.2.0.tar.gz` — the source distribution

---

## 5. Share the install command

Once the assets are attached, consumers can install the release with:

```bash
pip install https://github.com/nanacnote/keel/releases/download/v0.2.0/keel-0.2.0-py3-none-any.whl
```

Or via git:

```bash
pip install "keel @ git+https://github.com/nanacnote/keel.git@v0.2.0"
```

---

## Writing release notes

Keep notes short and consumer-focused. A useful structure:

```markdown
### What's new
- `Engine` now accepts a `retries` argument for configuring the correction loop (#12)

### Changed
- `MiddlewareChain` now executes hooks in registration order, not reverse

### Fixed
- `Dispatcher` raised `KeyError` on unknown tool names instead of a clean `ToolNotFoundError`
```

Omit internal refactors, dependency bumps, and CI changes unless they affect consumers.

---

## Fixing a bad release

You cannot overwrite a published GitHub release's assets via the workflow. If the build was wrong:

1. Delete the release on GitHub (this does not delete the tag)
2. Delete the tag locally and remotely:
   ```bash
   git tag -d v0.2.0
   git push origin --delete v0.2.0
   ```
3. Fix the issue, re-tag, and publish a new release from step 2

