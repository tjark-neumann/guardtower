# Publishing guardtower

Two one-time setups, then releasing is "create a GitHub Release."

## 1. Push the source to GitHub

```bash
# from the project root
git init
git add .
git commit -m "guardtower 0.3.0"
git branch -M main
git remote add origin https://github.com/tjark-neumann/guardtower.git
git push -u origin main
```

The repo URLs (`pyproject.toml`) and the `LICENSE` copyright are already set to
`tjark-neumann` / `Tjark Neumann`. Just create the empty repo at
https://github.com/tjark-neumann/guardtower first, then run the commands above.

The `tests` workflow runs automatically on every push/PR.

## 2. Set up PyPI Trusted Publishing (no API token needed)

1. Create a PyPI account at https://pypi.org/account/register/.
2. Go to https://pypi.org/manage/account/publishing/ and add a
   **pending publisher** with:
   - PyPI project name: `guardtower`
   - Owner: `tjark-neumann`
   - Repository: `guardtower`
   - Workflow name: `publish.yml`
   - Environment: `pypi`
3. (Optional) In the GitHub repo: Settings → Environments → create `pypi`.

## 3. Release (this is the recurring step)

```bash
git tag v0.3.0
git push origin v0.3.0
```

Then on GitHub: Releases → Draft a new release → choose tag `v0.3.0` →
Publish. The `publish` workflow builds the wheel + sdist and uploads them to
PyPI automatically. Within a minute:

```bash
pip install guardtower
```

For later versions: bump `version` in `pyproject.toml` and
`__version__` in `src/guardtower/__init__.py`, commit, tag `vX.Y.Z`, release.

## Manual publish (alternative, if you skip Trusted Publishing)

```bash
pip install build twine
python -m build
twine upload dist/*        # prompts for a PyPI API token
```
