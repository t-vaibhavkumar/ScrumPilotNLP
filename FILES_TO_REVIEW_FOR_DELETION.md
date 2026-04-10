# Files to Review for Deletion

**Context Transfer Complete** - Phase 6 production enhancements are done (8/10 production ready).

This document identifies files that can be safely deleted to clean up the repository.

---

## Summary

- **Test Scripts**: 5 files (safe to delete, functionality covered by main pipelines)
- **Experimental Files**: 2 files (safe to delete, experiments complete)
- **Empty Directories**: 1 directory (safe to delete, placeholder only)
- **Keep**: main.py, chrome_bot_profile/ (needed for Phase 7)

---

## Files Safe to Delete

### 1. Test Scripts (5 files)

These are standalone test scripts that were used during development. The functionality is now integrated into the main pipelines.

```
backend/tests/test_backlog_extractor.py
backend/tests/test_grooming_extractor.py
backend/tests/test_epic_decomposer.py
backend/tests/test_wsjf_calculator.py
backend/tests/test_jira_creator.py
```

**Reason**: 
- Each agent has `if __name__ == "__main__"` blocks for testing
- Main pipelines (`backlog_pipeline.py`, `scrum_pipeline.py`) provide end-to-end testing
- These test files are redundant

**Impact**: None - functionality preserved in main agents

---

### 2. Agent Test File (1 file)

```
backend/agents/test_scrum_agents.py
```

**Reason**: 
- Standalone test for scrum agents
- Functionality covered by `scrum_pipeline.py`

**Impact**: None

---

### 3. Experimental Files (2 files)

```
experiments/diarizaion_test.py
experiments/PyAudioWPatchTest.py
```

**Reason**: 
- Experimental audio/diarization tests
- Functionality integrated into `backend/speech/` modules
- No longer needed

**Impact**: None - experiments complete

---

### 4. Empty Frontend Directory (1 directory)

```
frontend/
```

**Contents**: Only `.gitkeep` file

**Reason**: 
- Placeholder directory with no actual frontend code
- Not used in current implementation

**Impact**: None - can be recreated if needed later

---

## Files to KEEP

### Keep: main.py
**Reason**: Entry point for Phase 7 (Google Meet Bot Integration)

### Keep: chrome_bot_profile/
**Reason**: Chrome profile data needed for Phase 7 Meet bot automation

### Keep: backend/tools/test_jira.py
**Reason**: Useful diagnostic tool for Jira connectivity testing

---

## Deletion Options

### Option A: Delete All (Recommended)
**Delete**: 5 test scripts + 1 agent test + 2 experiments + frontend/
**Total**: 8 files + 1 directory

```bash
# Test scripts
rm backend/tests/test_backlog_extractor.py
rm backend/tests/test_grooming_extractor.py
rm backend/tests/test_epic_decomposer.py
rm backend/tests/test_wsjf_calculator.py
rm backend/tests/test_jira_creator.py

# Agent test
rm backend/agents/test_scrum_agents.py

# Experiments
rm experiments/diarizaion_test.py
rm experiments/PyAudioWPatchTest.py
rmdir experiments

# Empty frontend
rm frontend/.gitkeep
rmdir frontend
```

---

### Option B: Conservative (Keep Tests)
**Delete**: 2 experiments + frontend/
**Total**: 2 files + 1 directory

```bash
# Experiments
rm experiments/diarizaion_test.py
rm experiments/PyAudioWPatchTest.py
rmdir experiments

# Empty frontend
rm frontend/.gitkeep
rmdir frontend
```

---

### Option C: Minimal (Only Experiments)
**Delete**: 2 experiments
**Total**: 2 files

```bash
rm experiments/diarizaion_test.py
rm experiments/PyAudioWPatchTest.py
rmdir experiments
```

---

## Recommendation

**Option A (Delete All)** is recommended because:

1. Test scripts are redundant - agents have built-in test blocks
2. Main pipelines provide comprehensive end-to-end testing
3. Experiments are complete and integrated
4. Frontend directory is empty placeholder
5. Cleaner repository structure
6. No functionality loss

---

## Verification After Deletion

After deletion, verify the system still works:

```bash
# Test Jira connection
python -m backend.tools.test_jira

# Test complete pipeline
python -m backend.pipelines.backlog_pipeline

# Test scrum pipeline
python -m backend.pipelines.scrum_pipeline --dry-run
```

---

## Next Steps

1. **User confirms** which option to proceed with
2. **Execute deletions** based on user choice
3. **Verify** system functionality
4. **Update .gitignore** if needed
5. **Commit changes** with clear message

---

**Awaiting User Confirmation**: Which option would you like to proceed with?
- Option A: Delete all (8 files + 1 directory) - RECOMMENDED
- Option B: Conservative (2 files + 1 directory)
- Option C: Minimal (2 files only)
- Custom: Specify which files to delete
