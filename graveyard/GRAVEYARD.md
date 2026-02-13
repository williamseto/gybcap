# Graveyard - Archived Code Repository

This directory contains deprecated, superseded, or experimental code that is no longer actively used in the project but is preserved for historical reference.

## Purpose

The graveyard serves three key functions:

1. **Preservation**: Maintains git history and allows restoration if needed
2. **Organization**: Separates active code from deprecated implementations
3. **Documentation**: Provides clear migration paths and deprecation reasons

## Directory Structure

```
graveyard/
├── GRAVEYARD.md           # This file - comprehensive documentation
├── 2026-02-feb/           # February 2026 cleanup
│   ├── README.md          # Migration summary
│   ├── deprecated_strategies/  # Old monolithic strategy files
│   │   ├── test_bo_retest.py   (837 lines) - Superseded by strategies/
│   │   ├── pred_util.py        (976 lines) - Ported to strategies/reversal/tcn_model.py
│   │   └── test_pipe_real.py   (238 lines) - Old prediction pipeline
│   └── root_scripts/      # Orphaned test scripts
│       ├── test_vp.py, test_time.py, test_stats.py, test_kmeans.py
│       ├── test_sac.py, test_time_seconds.py, scratch.py
│       ├── ovn_touch.py, analyze_ib_stats.py, stats_util.py, test.py
│       └── (11 files, 1,469 lines total)
├── util/                  # February 2026 refactor
│   ├── test_client.py     # Old monolithic real-time client
│   ├── strategy_util.py   # Old strategy implementations
│   └── client_util.py     # Old data fetch utilities
├── sandbox/               # Pre-2026 experiments
│   └── (Various archived experiments)
├── root_scripts/          # Previous cleanup
│   └── (Earlier archived scripts)
└── vp/                    # Old VP pipeline
    └── (Legacy volume profile code)
```

## Migration History

### February 2026 Cleanup (2026-02-12)

**Context**: Following the modular `strategies/` package refactor, removed dead code with zero active imports.

**Files Archived**: 14 files (3,520 lines)

#### Category A: Deprecated Strategies (3 files, 2,051 lines)
- `sandbox/test_bo_retest.py` → `2026-02-feb/deprecated_strategies/`
  - **Reason**: Already marked DEPRECATED (line 32), replaced by `strategies/` package
  - **Replaced by**: `strategies.breakout.BreakoutRetestStrategy`, `strategies.reversion.ReversionStrategy`
  - **Last active**: 2026-02-03

- `sandbox/pred_util.py` → `2026-02-feb/deprecated_strategies/`
  - **Reason**: TCN models ported to `strategies/reversal/tcn_model.py`, no active imports
  - **Replaced by**: `strategies.reversal.tcn_model.TCN`, `strategies.reversal.causal_model`
  - **Last active**: 2025-07-01

- `sandbox/test_pipe_real.py` → `2026-02-feb/deprecated_strategies/`
  - **Reason**: Deprecated prediction pipeline (imports pred_util)
  - **Replaced by**: `strategies.reversal.causal_trainer`, `strategies.realtime` package
  - **Last active**: 2025-07-01

#### Category B: Root-Level Test Files (11 files, 1,469 lines)
Orphaned standalone scripts moved to `2026-02-feb/root_scripts/`:
- `test_vp.py`, `test_time.py`, `test_stats.py`, `test_kmeans.py`
- `test_sac.py`, `test_time_seconds.py`, `scratch.py`
- `ovn_touch.py`, `analyze_ib_stats.py`, `stats_util.py`, `test.py`

**Reason**: Zero active imports, standalone exploration scripts with no dependencies.

**Validation Results**:
- ✅ Pre-migration: 56 tests (2 failures unrelated to migration)
- ✅ Import analysis: Zero references to archived files in active code
- ✅ Git history: All moves preserve full git history via `git mv`

### February 2026 Refactor (2026-02-03)

**Context**: Replaced monolithic `util/` with modular `strategies/realtime/` package.

**Files Archived**: 3 files from `util/` directory
- `util/test_client.py` → `graveyard/util/`
- `util/strategy_util.py` → `graveyard/util/`
- `util/client_util.py` → `graveyard/util/`

**Replaced by**: `strategies.realtime` package with plug-and-play architecture.

## Deprecation Policy

### When to Archive Code

Archive code when it meets ANY of these criteria:

1. **Zero Active Imports**: No active files import or reference it
2. **Superseded Implementation**: Functionality replaced by newer code
3. **Orphaned Experiments**: Standalone scripts with no dependencies
4. **Marked DEPRECATED**: Code explicitly marked as deprecated in comments

### How to Archive Code

1. **Use git mv**: Preserves full git history
   ```bash
   git mv old_file.py graveyard/YYYY-MM-mmm/category/
   ```

2. **Add Deprecation Header**: Prepend to moved files
   ```python
   """
   ARCHIVED: [Month Year]
   REASON: [Specific reason]
   REPLACED BY: [New location or N/A]
   LAST ACTIVE: [git commit hash and date]

   [Original docstring...]
   """
   ```

3. **Update Documentation**: Add entry to this file and category README.md

4. **Validate**: Run test suite and import analysis to ensure no breakage

### Files to NEVER Archive

- Any file imported by active code
- Core library modules (`strategies/`, `gex/`, `vp/`)
- Active experiment scripts (e.g., `sandbox/train_causal.py`)
- Test suites (`tests/`)
- Configuration files

## Restoration Procedures

### Restore a Single File

```bash
# From git history (recommended - gets original location)
git log --all --full-history -- graveyard/path/to/file.py
git checkout <commit-hash> -- path/to/file.py

# From graveyard (if still exists)
git mv graveyard/path/to/file.py original/path/
```

### Restore Entire Migration

```bash
# Find migration commit
git log --oneline graveyard/

# Revert specific commit
git revert <migration-commit-hash>

# Or reset to before migration (destructive)
git reset --hard <commit-before-migration>
```

### Reference Archived Code

Archived code is accessible for reference without restoration:

```python
# Read archived implementation
with open('graveyard/2026-02-feb/deprecated_strategies/test_bo_retest.py') as f:
    legacy_code = f.read()
```

## Active vs Archived Code

### Active Code (DO NOT MOVE)
- `strategies/` - Current modular strategy implementation
- `gex/` - Gamma exposure utilities (preserved per user request)
- `vp/` - Volume profile and Dalton classifier
- `sandbox/train_causal.py` - V3 causal zone prediction (active experiment)
- `sandbox/analyze_v3_structural.py` - Active analysis
- `tests/` - Test suite (56 tests)
- `CLAUDE.md` - Project documentation

### Archived Code (IN GRAVEYARD)
- `graveyard/2026-02-feb/` - February 2026 cleanup
- `graveyard/util/` - Old monolithic real-time client
- `graveyard/sandbox/` - Pre-2026 experiments
- `graveyard/root_scripts/` - Previous cleanup
- `graveyard/vp/` - Old VP pipeline

## Validation Checklist

Before archiving code, verify:

- [ ] Zero active imports (grep for imports across codebase)
- [ ] Test suite passes (pytest tests/ -v)
- [ ] Git history preserved (used git mv, not rm + add)
- [ ] Deprecation headers added to moved files
- [ ] Documentation updated (GRAVEYARD.md + category README.md)
- [ ] Safety branch created for easy rollback

## Statistics

### Total Archived (as of 2026-02-12)
- **Files**: 29+ files
- **Lines**: 7,500+ lines of code
- **Categories**: 5 major cleanup phases
- **Git History**: 100% preserved via git mv

### Impact
- **Reduced codebase**: Fewer files to index and navigate
- **Clearer structure**: Active vs archived code distinction
- **Faster IDE**: Reduced indexing overhead
- **Better documentation**: Clear migration paths and history

## Related Documentation

- `/Users/wseto/gybcap/CLAUDE.md` - Project instructions (graveyard section)
- `graveyard/2026-02-feb/README.md` - February 2026 cleanup details
- `strategies/README.md` - Current modular strategy documentation
- `strategies/realtime/README.md` - Real-time engine documentation

## Questions?

For questions about archived code:
1. Check this file for migration context
2. Check category-specific README.md files
3. Use `git log --all --full-history -- graveyard/path/to/file.py` to see history
4. Consult CLAUDE.md for active code references
