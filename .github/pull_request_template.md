<!--
HOW TO USE THIS TEMPLATE 
1. Fill in all sections (something is better than nothing)
2. Run `python noma_scripts/pr_preparation_scripts.py` locally
3. Delete these instruction comments (what begins with `<!--`)  before submitting if desired (including this block)
4. Click on PREVIEW on the top bar to see how it will look for the reviewer

WHY IT MATTERS: 
- Consistency: Ensures every PR looks the same and provides the same level of detail. 
- Safety: Forces a check for security flaws (API keys) and code hygiene (debug prints) via the scripts. 
- Context: Helps reviewers understand *what* changed and *where* (dependencies), speeding up the approval process.
-->

## 📌 Description
<!--Objective summary of the change and if there is no task/ticket, explain the necessity of what was done-->

## 🧠 Context
- **Type:** [Feature | Bugfix | Refactor | Chore | Hotfix] <!-- Choose one-->
- **Impacted Module/Flow:** <!--Ex: Checkout, Login, Payment API-->
- **Dependencies on another task/PR:** <!-- Is there a dependency on another task/PR? (Yes/No)-->
<!--
Chore: Technical maintenance tasks that do not alter user functionality (ex: updating libraries, configuring tools, code cleanup).
Hotfix: Urgent correction for a critical error already in production, applied immediately (bypassing the normal development flow).
-->

## 📦 Repository Versions
<!-- Copy and paste the output of the script "pr_preparation_scripts.py" located at https://github.com/LucasToscanou/noma_scripts
Example:
| Repo       | Branch                                        | Commit  | Commit Message
|------------|-----------------------------------------------|---------|--------------------------------------------------
| renglo-api | main                                          | 2eabef1 | CLA
| renglo-lib | main                                          | 53db6af | Merge branch 'main' of https://github.com/...
| wss        | main                                          | 69902e2 | Merge branch 'main' of github.com:renglo/wss
| schd       | main                                          | f9ba187 | Implement safe parsing function for JSON a...
| enerclave  | main                                          | 36adafb | initial commit
| noma       | Completing-the-flow-for-the-quote_train-ac... | 54c2160 | TESTING_SCRIPT
| pes        | main                                          | b68d392 | Reactivate verification, Append tool init ...
| data       | main                                          | b206183 | Refactor imports
| system     | main                                          | 3291879 | CLA docs
| console    | main                                          | fe01a5c | Refactor ChatWidgetPlanPreview to return e...
| NOMA       | main                                          | 07e74101 | fix: small fix landing page
-->

```bash
<PASTE THE TABLE HERE>
```

## 🔗 Related Links
- **Jira/Issue:** <!--[LINK]-->
- **Docs:** <!--[LINK]-->

## 🧪 Testing & Validation
- **Environment:** [Dev | Staging | Local] <!-- Choose one-->
- **Org:** <!-- Org used -->
<!--
Local: The environment on the developer's own machine (localhost), where code is created and tested in isolation.
Dev: Shared environment where the team integrates new code; usually unstable and used for initial technical tests.
Staging: Faithful replica of the production environment (with similar data), used for final and safe validation before the official release.
-->
- **Tested Scenarios:**  
    1. <Scenario A>
    2. <Scenario B>

- **Steps to Reproduce:**
<!-- Clear instructions for the reviewer/tester to validate -->
    1. Step 1
    2. Step 2

## 📸 Evidence
<!-- Screenshots, videos, logs or GIFs --><!-- On GitHub you can copy and paste the image/gif and it will upload it and generate the link here automatically-->
![Evidence 1](https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif)

## ⚠️ Attention Points / Risks
<!-- List critical changes that require caution here, for example:
 - Database migrations (Schema changes)
 - New Environment Variables (ENVs) required
 - Breaking Changes (Loss of compatibility with previous versions)
 - Scripts that need to run during deploy
--> 
 - Attention point A
 - Attention point B

## ✅ Checklist
- [ ] Code follows project standards (Lint/Prettier)
- [ ] Unit/integration tests added or updated
- [ ] Ran `pr_preparation_scripts.py` locally (Security & Debug checks passed)
- [ ] Self-review performed (no console.log or commented-out code)
- [ ] Technical documentation updated (if necessary)
<!-- 
Place an "x" inside the brackets, ex:
- [x] Code follows project standards (Lint/Prettier)
- [ ] Unit/integration tests added or updated
- [ ] Self-review performed (no console.log or commented-out code)
- [ ] Technical documentation updated (if necessary)
-->