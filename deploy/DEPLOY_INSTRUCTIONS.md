# Deploy to gerritsxd.com/2kmer

## Files to upload

Upload the contents of `deploy/2kmer/` to your web server so they appear at:

**https://gerritsxd.com/2kmer/**

### Option A: FTP/SFTP

1. Connect to your hosting via FTP (e.g. FileZilla, WinSCP)
2. Navigate to your web root (often `public_html`, `www`, or `htdocs`)
3. Create folder `2kmer` if it doesn't exist
4. Upload `index.html`, `gametheory.html`, and `timeseries.html` into `2kmer/`

Result: **https://gerritsxd.com/2kmer/** will show the hemicycle visualization. Game Theory and Voting Dynamics are linked from the nav.

### Option B: cPanel File Manager

1. Log in to cPanel
2. Open **File Manager**
3. Go to `public_html` (or your document root)
4. Create folder `2kmer`
5. Upload `index.html`, `gametheory.html`, and `timeseries.html` into `2kmer/`

### Option C: Git / SSH (if you have shell access)

```bash
# From your project folder — upload all deploy files
scp -r deploy/2kmer/* user@gerritsxd.com:~/public_html/2kmer/
```

---

## What's included

- `index.html` — Self-contained hemicycle visualization (no external dependencies)
- `gametheory.html` — Strategic landscape + payoff breakdown (5 components: policy, coalition, electoral, reciprocity, discipline)
- `timeseries.html` — Voting dynamics over time
- All data is embedded; no API calls or JSON files needed

---

## Security reminder

If you shared login credentials in chat, **change your password immediately** for any affected accounts.
