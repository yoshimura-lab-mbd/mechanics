# Commit Command Rules

When the user says `commit`:
- Commit only the most recent work (the current logical task).
- Write a clear commit message that matches the task.
- Do not include unrelated modified files.

When the user says `commit all`:
- Commit all currently modified files.
- Split into multiple commits by logical work unit whenever possible.
- Use clear commit messages for each commit.

If there are unclear or potentially user-owned changes:
- You may ask the user before committing those files.
- Prefer confirming intent when file ownership or purpose is ambiguous.
